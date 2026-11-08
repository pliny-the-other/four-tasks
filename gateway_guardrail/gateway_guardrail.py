"""
Task 3: LLM Gateway Streaming Guardrail (with PII Redaction)

Terminal testing:

0. Start the guardrail proxy:
   uv run python gateway_guardrail.py

1. Test live streaming with curl (-N disables client buffering):
   curl -N -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"stream": true, "messages": [{"role": "user", "content": "Personal Information"}]}'
"""

# ============================================================================
# We're parsing the stream from the LLM : "Intercept the returning response"
# "Parse" incoming stream deltas" sounds like the client prompt?
# ============================================================================

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import re
import sys
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

# logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("gateway-guardrail")

# ============================================================================
# PII redaction
# ============================================================================
PII_PATTERNS = [
    # email addresses
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # ssns 
    re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
    # credit card numbers (13-19 digits; spaced, hyphenated, or contiguous)
    re.compile(r"\b(?:\d{4}[-\s]){3}\d{1,4}\b|\b\d{13,19}\b"),
]

# characters found in valid PII strings; `space`` handled in `while tail_start > 0`
# because spaces should only cause holds when preceded by a digit
PII_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@.-_")
MAX_LOOKAHEAD = 45  # upper bound length of above patterns


class StreamingPIIRedactor:
    """
    streams safe text immediately; holds any trailing characters that
    could be incomplete PII until looking at the next chunk
    """

    def __init__(self):
        self.buffer = ""

    def process_chunk(self, delta_text: str) -> str:
        self.buffer += delta_text
        # run the redactions immediately if there is PII in the buffer
        self._apply_redactions()

        # hold tokens if buffer is smaller than the maximum possible PII pattern length
        if len(self.buffer) <= MAX_LOOKAHEAD:
            return ""

        # look backward for PII; finds where potentially incomplete PII begins
        tail_start = len(self.buffer)
        while tail_start > 0:
            ch = self.buffer[tail_start - 1]
            if ch in PII_CHARS:
                tail_start -= 1
            # allow space only when preceded by a digit; for spaced cards/SSNs to avoid overholding
            elif ch == " " and tail_start > 1 and self.buffer[tail_start - 2].isdigit():
                tail_start -= 1
            else:
                break

        # the number of contiguous PII-like characters found above
        tail_hold_len = len(self.buffer) - tail_start
        hold_len = min(tail_hold_len, MAX_LOOKAHEAD)

        # split the buffer; what to hold, what to send to user
        split_idx = len(self.buffer) - hold_len
        to_user = self.buffer[:split_idx]
        self.buffer = self.buffer[split_idx:]
        return to_user

    def flush(self) -> str:
        """flushes the balance when the stream completes."""
        self._apply_redactions()
        remaining = self.buffer
        self.buffer = ""
        return remaining

    def _apply_redactions(self):
        for pattern in PII_PATTERNS:
            self.buffer = pattern.sub("[REDACTED]", self.buffer)


# ============================================================================
# mock LLM provider
# ============================================================================
mock_app = FastAPI(title="Mock LLM Provider")

# split redactables across chunk boundaries for simulation
MOCK_STREAM_CHUNKS = [
    "Requested ",
    "Personal Information",
    ":\nEmail: jean-paul",
    ".sartre",
    "@example",
    ".com",
    "\nSSN: 123",
    "-45-",
    "6789\nCreditCard: ",
    "1111",
    "-2222-3333-",
    "4444\nBut ",
    "you cannot",
    " see it!"
]

@mock_app.post("/v1/chat/completions")
async def mock_completions(request: Request):
    async def sse_generator():
        for chunk in MOCK_STREAM_CHUNKS:
            await asyncio.sleep(0.04)  # simulated token latency
            data = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": chunk}, "index": 0}],
            }
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# ============================================================================
# LLM gateway proxy
# ============================================================================
# checks if LLM ENV is set
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://in-memory-llm")
LLM_API_KEY = os.getenv("LLM_API_KEY", "mock-key")

upstream_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI): # if LLM_BASE_URL is left unset, routes to in-memory mock_app
    global upstream_client
    if LLM_BASE_URL == "http://in-memory-llm":
        transport = httpx.ASGITransport(app=mock_app)
        base_url = "http://in-memory-llm"
    else:
        transport = None
        base_url = LLM_BASE_URL

    upstream_client = httpx.AsyncClient( # single reusable client
        transport=transport,
        base_url=base_url,
        timeout=60.0,
    )
    yield
    if upstream_client:
        await upstream_client.aclose()

gateway_app = FastAPI(title="LLM Gateway Guardrail", lifespan=lifespan)

async def redact_sse_stream(response: httpx.Response) -> AsyncGenerator[str, None]:
    redactor = StreamingPIIRedactor()
    try:
        async for line in response.aiter_lines(): # read line by line; do not buffer entire response (async iter)
            if not line:
                continue

            if line.startswith("data: "): # look for payload
                raw_data = line[len("data: ") :].strip()
                if raw_data == "[DONE]": # look for stream completion (data: [DONE])
                    remaining_text = redactor.flush()
                    # print(remaining_text, end="", flush=True)  # final tokens to terminal
                    if remaining_text:
                        done_chunk = {
                            "id": "chatcmpl-guardrail",
                            "object": "chat.completion.chunk",
                            "choices": [{"delta": {"content": remaining_text}, "index": 0}],
                        }
                        yield f"data: {json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                try:
                    payload = json.loads(raw_data)
                    choices = payload.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}

                    if "content" in delta and delta["content"]:
                        filtered_text = redactor.process_chunk(delta["content"]) # do the redaction
                        if filtered_text:
                            # print(filtered_text, end="", flush=True)  # redacted stream to terminal
                            delta["content"] = filtered_text
                            yield f"data: {json.dumps(payload)}\n\n"
                        # if filtered_text is empty, redactor is holding possible matches
                    else:
                        yield f"{line}\n\n" # pass through metadata deltas (role, finish_reason...)
                except json.JSONDecodeError:
                    yield f"{line}\n\n"
            else:
                yield f"{line}\n\n" # pass through non-payload (comments, event lines...)
    finally:
        await response.aclose() # ensure upstream connection is closed if client disconnects early

@gateway_app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    body = await request.json()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    req = upstream_client.build_request(
        "POST",
        "/v1/chat/completions",
        json=body,
        headers=headers,
    )
    upstream_response = await upstream_client.send(req, stream=True)

    return StreamingResponse(
        redact_sse_stream(upstream_response),
        status_code=upstream_response.status_code,
        media_type="text/event-stream",
    )

if __name__ == "__main__":
    uvicorn.run("gateway_guardrail:gateway_app", host="0.0.0.0", port=8000, reload=False)