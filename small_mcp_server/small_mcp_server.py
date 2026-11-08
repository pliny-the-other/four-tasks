"""
Task 1 : MCP Server with Strict Validation & Transport Handling

Borrows liberally from https://github.com/modelcontextprotocol/quickstart-resources/blob/main/weather-server-python/weather.py

Exposes two tools:
1. get_customer_record: customer_id (CUST-XXXXX)
2. trigger_refund: customer_id (CUST-XXXXX), amount (> 0.0), reason (min_length=10)

Terminal testing:

1. handshake:
(
  echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "terminal-test", "version": "1"}}}'
  sleep 0.5
) | uv run small_mcp_server.py

2. list tools:
(
  echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "terminal-test", "version": "1"}}}'
  echo '{"jsonrpc": "2.0", "method": "notifications/initialized"}'
  echo '{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}'
  sleep 0.5
) | uv run small_mcp_server.py

3. get_customer_record:
(
  echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "terminal-test", "version": "1"}}}'
  echo '{"jsonrpc": "2.0", "method": "notifications/initialized"}'
  echo '{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-10001"}}}'
  sleep 0.5
) | uv run small_mcp_server.py

4. trigger_refund:
(
  echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "terminal-test", "version": "1"}}}'
  echo '{"jsonrpc": "2.0", "method": "notifications/initialized"}'
  echo '{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "trigger_refund", "arguments": {"customer_id": "CUST-10001", "amount": 150.0, "reason": "good customer; small amount"}}}'
  sleep 0.5
) | uv run small_mcp_server.py

5. validation failure (ErrorCode -32602):
(
  echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "terminal-test", "version": "1"}}}'
  echo '{"jsonrpc": "2.0", "method": "notifications/initialized"}'
  echo '{"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "trigger_refund", "arguments": {"customer_id": "wrong_id", "amount": -10.0, "reason": "too_short"}}}'
  sleep 0.5
) | uv run small_mcp_server.py
"""

import logging
import sys
from pydantic import BaseModel, Field, ValidationError # this is like .../validator/... in Go

from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError

INVALID_PARAMS = -32602 # invalid parameters error code

# ============================================================================
# stdout is  for JSON-RPC messages.
# stderr for server logs.
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr, # keeps it out of stdout
)
logger = logging.getLogger("small-mcp-server")

# ============================================================================
# initialize server
# ============================================================================
mcp = MCPServer("small-mcp-server")

# strongly typed customer record; no `Any`
class CustomerAccount(BaseModel):
    name: str
    email: str
    tier: str
    balance: float


TEST_DATABASE: dict[str, CustomerAccount] = {
    "CUST-10001": CustomerAccount(
        name="Jean-Paul Sartre",
        email="jps@example.com",
        tier="Gold",
        balance=2000.00,
    ),
    "CUST-20002": CustomerAccount(
        name="Immanuel Kant",
        email="ik@example.com",
        tier="Silver",
        balance=1000.00,
    ),
}

# ============================================================================
# input schemas
# ============================================================================
class CustomerRecordInput(BaseModel):
    customer_id: str = Field(
        ...,
        pattern=r"^CUST-[A-Za-z0-9]{5}$",
        description="customer ID strictly as CUST-XXXXX",
    )


class TriggerRefundInput(BaseModel):
    customer_id: str = Field(
        ...,
        pattern=r"^CUST-[A-Za-z0-9]{5}$",
        description="customer ID strictly as CUST-XXXXX",
    )
    amount: float = Field(
        ...,
        gt=0.0,
        description="refund must be a positive float",
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="reason for refund reason; minimum of 10 characters",
    )


# ============================================================================
# mcp tools
# ============================================================================
@mcp.tool()
async def get_customer_record(customer_id: str) -> str:
    """retrieve customer details using customer_id.

    Args:
        customer_id: string formatted as CUST-XXXXX.
    """
    logger.info("handling get_customer_record for ID: %s", customer_id)

    try:
        validated = CustomerRecordInput(customer_id=customer_id)
    except ValidationError as err:
        logger.warning("input validation failure: %s", err)
        raise MCPError(
            INVALID_PARAMS,
            f"invalid parameters: {err.errors()}",
        )

    record = TEST_DATABASE.get(validated.customer_id)
    if not record:
        logger.info("ID : %s : record not found", validated.customer_id)
        return f"ID : {validated.customer_id} record not found"

    return (
        f"Customer Record [{validated.customer_id}]:\n"
        f"Name: {record.name}\n"
        f"Email: {record.email}\n"
        f"Tier: {record.tier}\n"
        f"Current Balance: ${record.balance:.2f}"
    )


@mcp.tool()
async def trigger_refund(customer_id: str, amount: float, reason: str) -> str:
    """process a customer refund.

    Args:
        customer_id: string formatted as CUST-XXXXX.
        amount: positive float refund amount.
        reason: string reason; minimum length of 10 characters.
    """
    logger.info(
        "Handling trigger_refund for %s | amount: %s | reason: %s",
        customer_id,
        amount,
        reason,
    )

    try:
        validated = TriggerRefundInput(
            customer_id=customer_id,
            amount=amount,
            reason=reason,
        )
    except ValidationError as err:
        logger.warning("input validation failure: %s", err)
        raise MCPError(
            INVALID_PARAMS,
            f"invalid parameters: {err.errors()}",
        )

    cid = validated.customer_id
    if cid in TEST_DATABASE:
        TEST_DATABASE[cid].balance = round(
            TEST_DATABASE[cid].balance - validated.amount, 2
        )
        balance_display = f"${TEST_DATABASE[cid].balance:.2f}"
    else:
        balance_display = "N/A - customer does not exist"

    logger.info("refund processed for %s; new balance: %s", cid, balance_display)
    return (
        f"refund processed successfully:\n"
        f"- customer ID: {cid}\n"
        f"- refund amount: ${validated.amount:.2f}\n"
        f"- reason: {validated.reason}\n"
        f"- remaining balance: {balance_display}"
    )


# ============================================================================
# entry
# ============================================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")