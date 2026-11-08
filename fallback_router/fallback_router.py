"""
Task 4: LLM Gateway Resilient Fallback Router with Token Rate Limiting

Terminal testing:

0. Run the gateway router:
   uv run python fallback_router.py

1. Test normal routing (Primary):
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer tenant_test_key" \
    -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello fallback router!"}]}'

2. Test Primary 429 failover to Secondary:
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer tenant_test_key" \
    -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "TRIGGER_PRIMARY_429"}]}'

3. Test Primary oversize failover to Secondary:
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer tenant_test_key" \
    -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "SIMULATE_OVERSIZED_PROMPT"}]}'

4. Test missing Authentication (401 Error)
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello without auth"}]}'
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import aiosqlite
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import tiktoken
import uvicorn

# logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("fallback-router")

# ============================================================================
# configuration
# ============================================================================
DB_FILE = os.getenv("GATEWAY_DB_PATH", "gateway_rate_limit.db")  # SQLite stored on disk
RATE_LIMIT_TOKENS = 50_000   # maximum tokens allowed per sliding window per tenant
WINDOW_SECONDS = 60          # rolling window duration (seconds)
PRIMARY_TIMEOUT_MS = 3000    # primary timeout (milliseconds)
SECONDARY_TIMEOUT_MS = 5000  # backup timeout (milliseconds)

# target URLs; defaults route to in-memory FastAPI mock apps
PRIMARY_BASE_URL = os.getenv("PRIMARY_LLM_URL", "http://in-memory-primary")
SECONDARY_BASE_URL = os.getenv("SECONDARY_LLM_URL", "http://in-memory-secondary")

# initialize tiktoken tokenizer for OpenAI BPE encoding (gpt-4/gpt-3.5)
try:
    TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKEN_ENCODER = None


def estimate_prompt_tokens(messages: list) -> int:
    """
    counts (or estimates) tokens in request messages for rate limiting before sending upstream
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        # simulation to trigger rate limit errors (429s)
        if content == "SIMULATE_OVERSIZED_PROMPT":
            return 55_000

        # if available, tiktoken counts tokens; otherwise falls back to ~4 chars/token heuristic
        if TOKEN_ENCODER:
            total += len(TOKEN_ENCODER.encode(content)) + 4  # +4 for OpenAI message formatting overhead
        else:
            total += (len(content) // 4) + 4
            
    return max(total, 1)  # ensure at least 1 token


# ============================================================================
# sliding window rate limiter (on-disk SQLite)
# ============================================================================
class SqliteTokenRateLimiter:
    """
    on-disk SQLite sliding window rate limiter
    append-only log of token grants; evicts records older than the active window
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # async lock serializes transactions within this process to prevent read-modify-write race conditions
        self._lock = asyncio.Lock()

    async def init_db(self):
        """creates table and composite index for timestamp queries"""
        async with aiosqlite.connect(self.db_path) as db:
            # write-ahead logging allows concurrent readers without blocking writes
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_key TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    timestamp REAL NOT NULL
                );
                """
            )
            # multi-column index speeds up tenant window lookups and deletions
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tenant_timestamp 
                ON token_usage (tenant_key, timestamp);
                """
            )
            await db.commit()

    async def check_and_reserve(self, tenant_key: str, requested_tokens: int) -> Tuple[bool, int, int]:
        """
        checks if the token request is within the sliding window
        if true, reserves tokens immediately
        returns: (allowed, current_window_usage, remaining_tokens)
        """
        now = time.time()
        window_start = now - WINDOW_SECONDS  # cutoff timestamp; records older than this are outside window

        async with self._lock:  # acquire lock
            async with aiosqlite.connect(self.db_path) as db:
                # remove expired records outside the time window across all tenants
                await db.execute(
                    "DELETE FROM token_usage WHERE timestamp < ?;",
                    (window_start,),
                )

                # sum tokens consumed by this tenant within the active window
                async with db.execute(
                    """
                    SELECT COALESCE(SUM(tokens), 0) 
                    FROM token_usage 
                    WHERE tenant_key = ? AND timestamp >= ?;
                    """,
                    (tenant_key, window_start),
                ) as cursor:
                    row = await cursor.fetchone()
                    current_usage = row[0] if row else 0

                # check quota compliance against configured threshold
                if current_usage + requested_tokens > RATE_LIMIT_TOKENS:
                    await db.commit()  # wait for the commit to delete the expired records persists even though this request is rejected
                    return False, current_usage, max(0, RATE_LIMIT_TOKENS - current_usage)

                # insert the newly granted tokens
                await db.execute(
                    """
                    INSERT INTO token_usage (tenant_key, tokens, timestamp)
                    VALUES (?, ?, ?);
                    """,
                    (tenant_key, requested_tokens, now),
                )
                await db.commit()

                new_usage = current_usage + requested_tokens
                return True, new_usage, max(0, RATE_LIMIT_TOKENS - new_usage)


# ============================================================================
# sanitize errors
# ============================================================================
def gateway_error_response(status_code: int, message: str, error_type: str, code: Optional[str] = None) -> JSONResponse:
    """
    builds an OpenAI-compliant error payload
    hides Python stack traces, connection URLs, and internal database details
    """
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code or f"gateway_{status_code}",
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


# ============================================================================
# mock upstream providers for testing
# ============================================================================
mock_primary_app = FastAPI(title="Primary Endpoint")
mock_secondary_app = FastAPI(title="Secondary Endpoint")


@mock_primary_app.post("/v1/chat/completions")
async def mock_primary_chat(request: Request):
    """simulates primary provider behavior; including 429s and long timeouts"""
    body = await request.json()
    content = body.get("messages", [{}])[-1].get("content", "")

    # trigger mock rate limit
    if "TRIGGER_PRIMARY_429" in content:
        return JSONResponse(status_code=429, content={"error": "upstream quota exhausted"})

    # trigger mock timeout; sleeps 4.0s
    if "TRIGGER_PRIMARY_TIMEOUT" in content:
        await asyncio.sleep(4.0)

    return {
        "id": "chatcmpl-primary",
        "object": "chat.completion",
        "provider": "primary-provider",
        "model": body.get("model", "primary-default"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "response served by PRIMARY"}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }


@mock_secondary_app.post("/v1/chat/completions")
async def mock_secondary_chat(request: Request):
    """simulates secondary provider."""
    body = await request.json()
    return {
        "id": "chatcmpl-secondary",
        "object": "chat.completion",
        "provider": "secondary-backup-provider",
        "model": "secondary-fallback-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "response served by SECONDARY"}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }


# ============================================================================
# LLM gateway application
# ============================================================================
rate_limiter = SqliteTokenRateLimiter(DB_FILE)
primary_client: Optional[httpx.AsyncClient] = None
secondary_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """initializes SQLite and configures reusable HTTP clients"""
    global primary_client, secondary_client
    await rate_limiter.init_db()

    # route in-memory via ASGI transport as default, otherwise use real HTTP targets
    primary_transport = (
        httpx.ASGITransport(app=mock_primary_app) if PRIMARY_BASE_URL == "http://in-memory-primary" else None
    )
    secondary_transport = (
        httpx.ASGITransport(app=mock_secondary_app) if SECONDARY_BASE_URL == "http://in-memory-secondary" else None
    )

    # initialize long-lived HTTP client pools; reuse TCP connections
    primary_client = httpx.AsyncClient(
        transport=primary_transport,
        base_url=PRIMARY_BASE_URL,
        timeout=httpx.Timeout(PRIMARY_TIMEOUT_MS / 1000.0),  # exact 3.0s timeout per requirement
    )
    secondary_client = httpx.AsyncClient(
        transport=secondary_transport,
        base_url=SECONDARY_BASE_URL,
        timeout=httpx.Timeout(SECONDARY_TIMEOUT_MS / 1000.0),
    )

    yield  # start serving requests; returns control to FastAPI

    # cleanup connection pools on shutdown
    if primary_client:
        await primary_client.aclose()
    if secondary_client:
        await secondary_client.aclose()


gateway_app = FastAPI(title="Resilient Fallback Gateway", lifespan=lifespan)


@gateway_app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    # tenant identification and authorization
    if not authorization or not authorization.startswith("Bearer "):
        return gateway_error_response(
            status_code=401,
            message="Missing or malformed Authorization header.",
            error_type="authentication_error",
            code="invalid_api_key",
        )

    tenant_key = authorization[len("Bearer ") :].strip()  # use bearer token as unique tenant ID

    # validate the request payload
    try:
        body = await request.json()
    except Exception:
        return gateway_error_response(
            status_code=400,
            message="Payload is invalid JSON.",
            error_type="invalid_request_error",
            code="malformed_json",
        )

    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return gateway_error_response(
            status_code=400,
            message="'messages' field must be a non-empty array.",
            error_type="invalid_request_error",
            code="invalid_messages",
        )

    # rate limiting
    estimated_tokens = estimate_prompt_tokens(messages)
    allowed, current_usage, remaining = await rate_limiter.check_and_reserve(tenant_key, estimated_tokens)

    if not allowed:
        return gateway_error_response(
            status_code=429,
            message=f"Rate limit of 50,000 tokens/minute exceeded. Current usage: {current_usage}, requested: {estimated_tokens}.",
            error_type="rate_limit_error",
            code="token_rate_limit_exceeded",
        )

    # route to PRIMARY
    should_failover = False
    failover_reason = ""

    try:
        logger.info(f"Routing request to PRIMARY for tenant '{tenant_key[:8]}...'")
        resp = await primary_client.post("/v1/chat/completions", json=body)

        # failover 1: upstream (too many requests)
        if resp.status_code == 429:
            should_failover = True
            failover_reason = "primary_status_429"
            logger.warning("PRIMARY returned 429 (Too Many Requests); initiating failover")
        # failover 2: upstream server outage (5xx)
        elif resp.status_code >= 500:
            should_failover = True
            failover_reason = f"primary_status_{resp.status_code}"
            logger.warning(f"PRIMARY returned ({resp.status_code}); initiating failover")
        elif resp.status_code == 200:
            # PRIMARY succeeded; return immediately
            return JSONResponse(status_code=200, content=resp.json())
        else:
            # 4xx client errors (invalid model params...) returned directly; failover does not fix bad input
            return JSONResponse(status_code=resp.status_code, content=resp.json())

    # failover 3: 3000ms timeout exceeded
    except (httpx.TimeoutException, asyncio.TimeoutError):
        should_failover = True
        failover_reason = "primary_timeout"
        logger.warning(f"PRIMARY timed out after {PRIMARY_TIMEOUT_MS}ms; initiating failover.")
    # failover 4: network disconnect or DNS failure
    except Exception as exc:
        should_failover = True
        failover_reason = "primary_connection_error"
        logger.error(f"Primary provider connection failed: {type(exc).__name__}. Initiating failover.")

    # fallback to SECONDARY
    if should_failover:
        logger.info(f"Executing fallback to SECONDARY due to {failover_reason}")
        try:
            sec_resp = await secondary_client.post("/v1/chat/completions", json=body)
            if sec_resp.status_code == 200:
                data = sec_resp.json()
                # attach non-destructive routing telemetry so callers know failover occurred
                data["_gateway_route"] = {
                    "failover_executed": True,
                    "primary_failure": failover_reason,
                    "provider": "secondary",
                }
                return JSONResponse(status_code=200, content=data)
            else:
                # SECONDARY returns a non-200; return sanitized error without body
                logger.error(f"SECONDARY failed with status {sec_resp.status_code}")
                return gateway_error_response(
                    status_code=502,
                    message="Secondary model provider was unable to process the request.",
                    error_type="upstream_error",
                    code="secondary_provider_error",
                )
        except (httpx.TimeoutException, asyncio.TimeoutError):
            logger.error("SECONDARY timed out.")
            return gateway_error_response(
                status_code=504,
                message="All configured model providers timed out.",
                error_type="upstream_timeout",
                code="all_providers_timed_out",
            )
        except Exception:
            logger.error("SECONDARY connection failed.")
            return gateway_error_response(
                status_code=502,
                message="All configured model providers failed.",
                error_type="gateway_routing_error",
                code="upstream_unavailable",
            )

    return gateway_error_response(
        status_code=500,
        message="Unexpected gateway routing error.",
        error_type="internal_error",
    )


if __name__ == "__main__":
    uvicorn.run("fallback_router:gateway_app", host="0.0.0.0", port=8000, reload=False)