"""A2A adapter: speak the envelope an Agent2Agent client already sends.

The concept note says the layer fronts "any A2A-compatible financial agent".
This is the honest version of that claim: AARAMSE presents an agent card and
accepts A2A's `message/send` JSON-RPC envelope, extracts the text parts,
repairs the query, and returns the text the caller should forward to its agent.

**What this is not.** It is not a full A2A implementation. There is no task
lifecycle, no streaming (`message/stream`), no push notifications, and no
artifact store. Those are the parts of the protocol a *destination* agent
needs; a middleware that rewrites one message and hands it back needs the
envelope and nothing else. `SUPPORTED_METHODS` is the whole surface, and the
agent card advertises exactly that, so a client can discover the limitation
rather than hit it.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "A2A_PATH",
    "AGENT_CARD_PATH",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "SUPPORTED_METHODS",
    "agent_card",
    "handle_jsonrpc",
    "text_parts",
]

logger = logging.getLogger(__name__)

AGENT_CARD_PATH = "/.well-known/agent-card.json"
A2A_PATH = "/a2a"

# The entire surface. Anything else returns "method not found" rather than a
# half-implemented lifecycle.
SUPPORTED_METHODS: Tuple[str, ...] = ("message/send",)

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def agent_card(model: str, url: str = "") -> Dict[str, Any]:
    """Describe this layer to an A2A client.

    Args:
        model: Model spec the gateway fronts, surfaced for operator clarity.
        url: Public URL of the A2A endpoint, when the deployer knows it.

    Returns:
        An agent card advertising only the methods actually implemented.
    """
    return {
        "protocolVersion": "0.3.0",
        "name": "AARAMSE over-refusal repair",
        "description": (
            "Middleware that detects when the model behind it refuses a legitimate "
            "question, repairs the refusal with a certified operator, and logs the "
            "intervention. Returns the text the caller should forward to its agent."
        ),
        "url": url,
        "version": "0.1.0",
        "provider": {"organization": "AARAMSE"},
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [{
            "id": "repair-over-refusal",
            "name": "Repair an over-refused query",
            "description": (
                "Probes the deployed model, and if it refuses, returns a repaired "
                "query that preserves the original intent. Prohibited requests are "
                "escalated, not repaired."
            ),
            "tags": ["safety", "over-refusal", "compliance"],
            "inputModes": ["text/plain"],
            "outputModes": ["text/plain"],
        }],
        "x-aaramse": {"model": model, "supportedMethods": list(SUPPORTED_METHODS)},
    }


def text_parts(message: Dict[str, Any]) -> List[str]:
    """Extract the text of every text part in an A2A message.

    Args:
        message: An A2A message object.

    Returns:
        The text of each part whose kind is "text", in order.
    """
    parts = message.get("parts")
    if not isinstance(parts, list):
        return []
    out: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "text" and isinstance(part.get("text"), str):
            out.append(part["text"])
    return out


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    """Build a JSON-RPC error response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def handle_jsonrpc(repair: Any, payload: Any) -> Dict[str, Any]:
    """Handle one A2A JSON-RPC request.

    Args:
        repair: Callable taking the query text and returning a RepairResult.
        payload: The decoded JSON-RPC request body.

    Returns:
        A JSON-RPC response object. Protocol-level problems become JSON-RPC
        errors rather than HTTP errors, which is what an A2A client expects.
    """
    if not isinstance(payload, dict):
        return _error(None, INVALID_REQUEST, "request must be a JSON object")

    request_id = payload.get("id")
    if payload.get("jsonrpc") != "2.0":
        return _error(request_id, INVALID_REQUEST, "jsonrpc must be '2.0'")

    method = payload.get("method")
    if method not in SUPPORTED_METHODS:
        return _error(
            request_id,
            METHOD_NOT_FOUND,
            f"{method!r} is not implemented; this layer supports "
            f"{', '.join(SUPPORTED_METHODS)} only",
        )

    params = payload.get("params")
    if not isinstance(params, dict):
        return _error(request_id, INVALID_PARAMS, "params must be an object")
    message = params.get("message")
    if not isinstance(message, dict):
        return _error(request_id, INVALID_PARAMS, "params.message must be an object")

    texts = text_parts(message)
    query = "\n".join(t for t in texts if t.strip()).strip()
    if not query:
        return _error(request_id, INVALID_PARAMS, "message carries no non-empty text part")

    result = repair(query)
    return _result(request_id, _message_result(message, result))


def _message_result(request_message: Dict[str, Any], result: Any) -> Dict[str, Any]:
    """Wrap a repair outcome in an A2A message.

    The returned text is what the caller forwards to its agent. It equals the
    original query unless the decision is "repaired", which keeps the contract
    identical to the plain HTTP route.
    """
    context_id = request_message.get("contextId")
    payload: Dict[str, Any] = {
        "kind": "message",
        "role": "agent",
        "messageId": str(uuid.uuid4()),
        "parts": [{"kind": "text", "text": result.rewritten}],
        "metadata": {
            "aaramse/decision": result.decision.value,
            "aaramse/program": list(result.program.names),
            "aaramse/refusalMargin": result.refusal_margin,
            "aaramse/reason": result.reason,
        },
    }
    if context_id is not None:
        payload["contextId"] = context_id
    task_id = _optional_str(request_message.get("taskId"))
    if task_id:
        payload["taskId"] = task_id
    return payload


def _optional_str(value: Any) -> Optional[str]:
    """Return the value when it is a non-empty string."""
    return value if isinstance(value, str) and value else None
