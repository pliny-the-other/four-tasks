"""
Task 2: MCP Security Gateway Proxy

Rules:
- tools/list is forwarded transparently for all valid tokens.
- error if 'admin_set_color' accessed by anyone not an admin.
- 'admin_set_color' successful if user role is admin

Terminal testing:

0. Start server
uv run python tiny_mcp_proxy.py (Ctrl+C to quit)

1. List tools
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer oar-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'

2. 'oar-puller' calls admin_set_color (BLOCKED):
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer oar-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "admin_set_color", "arguments": {"color": "blue"}}}'

3. 'admin' calls admin_set_color (ALLOWED):
curl -s -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "admin_set_color", "arguments": {"color": "gold"}}}'
"""

from contextlib import asynccontextmanager
import logging
import sys
from typing import Any, Optional

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field
import uvicorn

UNAUTHORIZED_TOOL_CODE = -32001 #unauthorized tool error code

# logging only to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tiny-mcp-proxy")

# role mapping
TOKEN_ROLES = {
    "admin-token": "admin",
    "oar-token": "oar-puller",
    "fire-token": "fire-starter",
}

# ============================================================================
# in program mock mcp server(using ASGI)
# ============================================================================
downstream_app = FastAPI(title="Mock MCP Server")

DOWNSTREAM_TOOLS_CATALOG = [
    {
        "name": "admin_set_color",
        "description": "sets the color red, green, blue or gold; admin only", # color not enforced
        "inputSchema": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "available colors.",
                }
            },
            "required": ["color"],
        },
    }
]

@downstream_app.post("/mcp")
async def downstream_handler(request: Request):
    payload = await request.json()
    method = payload.get("method")
    req_id = payload.get("id")

    if method == "initialize":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "color-me-beautiful", "version": "1"},
                },
            }
        )

    if method == "tools/list":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": DOWNSTREAM_TOOLS_CATALOG},
            }
        )

    if method == "tools/call":
        tool_name = payload.get("params", {}).get("name")
        args = payload.get("params", {}).get("arguments", {})

        if tool_name == "admin_set_color":
            selected_color = args.get("color", "unspecified")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"System color updated successfully to '{selected_color}'.",
                            }
                        ]
                    },
                }
            )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": "Method not found"},
        }
    )

# ============================================================================
# gateway schemas
# ============================================================================

# parse the incoming request for adherence to JSON-RPC 2.0
class JsonRpcRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", pattern=r"^2\.0$")
    id: Optional[Any] = None
    method: str
    params: Optional[dict[str, Any]] = None

downstream_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global downstream_client
    downstream_client = httpx.AsyncClient( # this is the magic unbeknowst to me that allows us to not use sockets
        transport=httpx.ASGITransport(app=downstream_app),
        base_url="http://in-memory-downstream",
        timeout=10.0,
    )
    yield
    if downstream_client:
        await downstream_client.aclose()

gateway_app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)

# ============================================================================
# gateway proxy & authorization
# ============================================================================
@gateway_app.post("/mcp")
async def handle_mcp_proxy(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    # check if it looks like a bearer token
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("bad request - invalid Bearer header")
        return JSONResponse(
            status_code=401,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": UNAUTHORIZED_TOOL_CODE,
                    "message": "invalid Bearer token",
                },
            },
        )

    token = authorization.split("Bearer ", 1)[1].strip()
    role = TOKEN_ROLES.get(token)
    if not role:
        logger.warning("bad request - malformed token '%s'", token)
        return JSONResponse(
            status_code=403,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": UNAUTHORIZED_TOOL_CODE,
                    "message": "malformed token",
                },
            },
        )

    # validate the wire format
    try:
        body = await request.json()
        rpc_req = JsonRpcRequest.model_validate(body)
    except Exception as err:
        logger.warning("payload not valid JSON-RPC: %s", err)
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"parse error: {str(err)}"},
            },
        )

    # check if role = admin
    if rpc_req.method == "tools/call":
        tool_name = (rpc_req.params or {}).get("name", "")
        if tool_name.startswith("admin_") and role != "admin":
            logger.warning(
                "BLOCKED: '%s' attempted to use admin only tool '%s'",
                role,
                tool_name,
            )
            return JSONResponse(
                status_code=200,
                content={
                    "jsonrpc": "2.0",
                    "id": rpc_req.id,
                    "error": {
                        "code": UNAUTHORIZED_TOOL_CODE,
                        "message": (
                            f"unauthorized tool call: '{role}' is not permitted to use '{tool_name}' - admin required."
                        ),
                    },
                },
            )

    # forward authorized requests
    logger.info("FORWARDING: '%s' is calling '%s'", role, rpc_req.method)
    resp = await downstream_client.post(
        "/mcp",
        json=body,
        headers={"Content-Type": "application/json"},
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type="application/json",
    )

if __name__ == "__main__":
    uvicorn.run("tiny_mcp_proxy:gateway_app", host="0.0.0.0", port=8000, reload=False)