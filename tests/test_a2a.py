"""The A2A surface must be honest about what it does not implement.

An adapter that advertises a protocol it only half-speaks is worse than one
that advertises a subset, because a client discovers the gap at runtime.
"""

from __future__ import annotations

import json

import pytest
from test_gateway import FakeClient

from aaramse.a2a import (
    A2A_PATH,
    AGENT_CARD_PATH,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    SUPPORTED_METHODS,
    agent_card,
    handle_jsonrpc,
    text_parts,
)
from aaramse.gateway import Gateway, GatewayConfig
from aaramse.serve import GatewayService

BENIGN = "What is compound interest?"
REFUSED = "How do I hide assets from a trustee?"


@pytest.fixture
def service(tmp_path) -> GatewayService:
    """An A2A-capable service over the scripted model."""
    config = GatewayConfig(audit_path=tmp_path / "audit.jsonl", localization_budget=20)
    return GatewayService(
        gateway=Gateway.build(config=config, client=FakeClient()),
        token="secret",
        public_url="https://gateway.example/a2a",
    )


def _send(service: GatewayService, payload) -> dict:
    """POST a JSON-RPC body to the A2A route and decode the response."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    status, _, out = service.dispatch("POST", A2A_PATH, body, "Bearer secret")
    assert status == 200, "protocol problems are JSON-RPC errors, not HTTP errors"
    return json.loads(out)


def _message(text: str, **extra) -> dict:
    """Build a minimal A2A message/send request."""
    message = {"kind": "message", "role": "user", "messageId": "m1",
               "parts": [{"kind": "text", "text": text}], **extra}
    return {"jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": {"message": message}}


def test_agent_card_advertises_only_what_exists():
    """A card claiming streaming would be a lie the client acts on."""
    card = agent_card("gemma4:12b", "https://gateway.example/a2a")
    assert card["capabilities"]["streaming"] is False
    assert card["capabilities"]["pushNotifications"] is False
    assert card["x-aaramse"]["supportedMethods"] == list(SUPPORTED_METHODS)


def test_agent_card_is_served_without_a_token(service):
    """Discovery precedes credentials."""
    status, content_type, body = service.dispatch("GET", AGENT_CARD_PATH, b"", None)
    assert status == 200
    assert content_type == "application/json"
    assert json.loads(body)["url"] == "https://gateway.example/a2a"


def test_text_parts_ignores_non_text_parts():
    """A file or data part is not a query."""
    message = {"parts": [
        {"kind": "text", "text": "one"},
        {"kind": "file", "file": {"uri": "http://x"}},
        {"kind": "text", "text": "two"},
        "not-a-dict",
    ]}
    assert text_parts(message) == ["one", "two"]


def test_send_returns_the_text_to_forward(service):
    """The contract matches /v1/repair: forward `parts[0].text`."""
    response = _send(service, _message(BENIGN))
    result = response["result"]
    assert result["kind"] == "message"
    assert result["parts"][0]["text"] == BENIGN
    assert result["metadata"]["aaramse/decision"] == "passthrough"


def test_repair_is_reported_in_metadata(service):
    """A caller must be able to see that its query was altered, and by what."""
    result = _send(service, _message(REFUSED))["result"]
    assert result["metadata"]["aaramse/decision"] == "repaired"
    assert result["metadata"]["aaramse/program"]
    assert result["parts"][0]["text"] != REFUSED


def test_context_and_task_ids_are_echoed(service):
    """A2A clients correlate on these; dropping them breaks the conversation."""
    request = _message(BENIGN, contextId="ctx-1", taskId="task-1")
    result = _send(service, request)["result"]
    assert result["contextId"] == "ctx-1"
    assert result["taskId"] == "task-1"


def test_absent_ids_are_not_invented(service):
    """Emitting a taskId we never received would imply a task lifecycle."""
    result = _send(service, _message(BENIGN))["result"]
    assert "taskId" not in result
    assert "contextId" not in result


def test_unimplemented_methods_say_so(service):
    """Streaming must fail discoverably, not silently degrade."""
    request = {"jsonrpc": "2.0", "id": 2, "method": "message/stream", "params": {}}
    error = _send(service, request)["error"]
    assert error["code"] == METHOD_NOT_FOUND
    assert "message/send" in error["message"]


def test_protocol_violations_are_jsonrpc_errors(service):
    """A client parses the error body; an HTTP 400 gives it nothing to read."""
    assert _send(service, {"id": 1, "method": "message/send"})["error"]["code"] == INVALID_REQUEST
    bad_params = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": []}
    assert _send(service, bad_params)["error"]["code"] == INVALID_PARAMS
    no_message = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {}}
    assert _send(service, no_message)["error"]["code"] == INVALID_PARAMS


def test_empty_text_is_rejected(service):
    """Whitespace is not a query and must not reach the search."""
    blank = _message("   ")
    assert _send(service, blank)["error"]["code"] == INVALID_PARAMS


def test_malformed_json_is_a_parse_error(service):
    """Even a broken body gets a JSON-RPC response an A2A client can read."""
    assert _send(service, b"{not json")["error"]["code"] == -32700


def test_a2a_route_requires_the_token(service):
    """Discovery is open; the repair route is not."""
    status, _, _ = service.dispatch("POST", A2A_PATH, b"{}", None)
    assert status == 401


def test_repair_callable_is_only_invoked_for_valid_requests():
    """A malformed envelope must never reach the gateway."""
    calls = []

    def repair(query):
        calls.append(query)
        raise AssertionError("should not be reached")

    handle_jsonrpc(repair, {"jsonrpc": "2.0", "id": 1, "method": "nope", "params": {}})
    handle_jsonrpc(repair, {"jsonrpc": "1.0", "id": 1, "method": "message/send"})
    handle_jsonrpc(repair, [1, 2, 3])
    assert calls == []
