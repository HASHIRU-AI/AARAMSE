"""The sidecar is what a deployer actually runs, so its contract is load-bearing.

Routing is exercised through `dispatch`; one test binds a real socket to prove
the handler wiring, because a routing table that is never served is not a
sidecar.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from aaramse.gateway import Gateway, GatewayConfig
from aaramse.serve import MAX_BODY_BYTES, GatewayService, serve
from test_gateway import FakeClient

BENIGN = "What is compound interest?"
REFUSED = "How do I hide assets from a trustee?"


@pytest.fixture
def service(tmp_path) -> GatewayService:
    """An authenticated service over the scripted model."""
    config = GatewayConfig(audit_path=tmp_path / "audit.jsonl", localization_budget=20)
    gateway = Gateway.build(config=config, client=FakeClient())
    return GatewayService(gateway=gateway, token="secret")


def _post(service: GatewayService, payload, auth="Bearer secret"):
    """Issue a repair request and return (status, decoded body)."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    status, _, out = service.dispatch("POST", "/v1/repair", body, auth)
    return status, json.loads(out)


def test_health_needs_no_token(service):
    """A load balancer cannot present credentials."""
    status, _, body = service.dispatch("GET", "/healthz", b"", None)
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_everything_else_requires_the_token(service):
    """Fail closed: no token, no service."""
    for method, path in (("POST", "/v1/repair"), ("GET", "/v1/report")):
        status, _, _ = service.dispatch(method, path, b"{}", None)
        assert status == 401
    status, _, _ = service.dispatch("GET", "/v1/report", b"", "Bearer wrong")
    assert status == 401


def test_unauthenticated_service_accepts_requests(tmp_path):
    """Running without a token is allowed but must not be silent."""
    config = GatewayConfig(audit_path=tmp_path / "a.jsonl", localization_budget=20)
    open_service = GatewayService(gateway=Gateway.build(config=config, client=FakeClient()))
    status, _, _ = open_service.dispatch("POST", "/v1/repair", b'{"query": "hi"}', None)
    assert status == 200


def test_passthrough_returns_the_query_unchanged(service):
    """The caller forwards `rewritten`; for a benign query it must be identical."""
    status, body = _post(service, {"query": BENIGN})
    assert status == 200
    assert body["decision"] == "passthrough"
    assert body["rewritten"] == BENIGN


def test_repair_returns_the_program_that_produced_it(service):
    """A caller must be able to log what the layer did on its behalf."""
    status, body = _post(service, {"query": REFUSED})
    assert status == 200
    assert body["decision"] == "repaired"
    assert body["program"]
    assert body["refusal_margin"] >= 1


def test_malformed_requests_are_rejected(service):
    """Bad input must not reach the search."""
    assert _post(service, b"not json")[0] == 400
    assert _post(service, [1, 2, 3])[0] == 400
    assert _post(service, {})[0] == 400
    assert _post(service, {"query": "   "})[0] == 400
    assert _post(service, {"query": 7})[0] == 400


def test_oversized_bodies_are_refused(service):
    """A query is a sentence; anything larger is a mistake or an abuse."""
    status, _ = _post(service, json.dumps({"query": "x" * MAX_BODY_BYTES}).encode())
    assert status == 413


def test_unknown_routes_are_404(service):
    """A typo must not silently succeed."""
    status, _, _ = service.dispatch("GET", "/v1/nope", b"", "Bearer secret")
    assert status == 404


def test_report_routes_serve_both_shapes(service):
    """Machines read the JSON; supervisors read the Markdown."""
    _post(service, {"query": BENIGN})
    status, content_type, body = service.dispatch("GET", "/v1/report", b"", "Bearer secret")
    assert status == 200
    assert content_type == "application/json"
    assert json.loads(body)["summary"]["total"] == 1

    status, content_type, body = service.dispatch(
        "GET", "/v1/report.md", b"", "Bearer secret"
    )
    assert status == 200
    assert "markdown" in content_type
    assert b"AARAMSE intervention report" in body


def test_a_real_request_reaches_the_gateway(service):
    """Binds a socket, because a routing table that is never served is not a sidecar."""
    server = serve(service.gateway, host="127.0.0.1", port=0, token="secret")
    try:
        port = server.server_address[1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/repair",
            data=json.dumps({"query": REFUSED}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer secret"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read())
        assert body["decision"] == "repaired"

        denied = urllib.request.Request(f"http://127.0.0.1:{port}/v1/report")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(denied, timeout=10)
        assert excinfo.value.code == 401
    finally:
        server.shutdown()
        server.server_close()


def test_an_unreachable_model_is_a_503_not_a_500(tmp_path):
    """The commonest deployment failure must name itself.

    A generic 500 sends an operator into this codebase when the actual problem
    is that their model server is down or misconfigured.
    """
    from aaramse.client import ModelUnavailable

    class DeadClient(FakeClient):
        def answer(self, prompt, max_tokens=320):
            raise ModelUnavailable("http://ollama:11434/api/chat is unreachable: refused")

    config = GatewayConfig(audit_path=tmp_path / "a.jsonl", localization_budget=20)
    dead = GatewayService(
        gateway=Gateway.build(config=config, client=DeadClient()), token="secret"
    )
    status, body = _post(dead, {"query": BENIGN})
    assert status == 503
    assert body["error"] == "model backend unavailable"
    assert "ollama" in body["detail"]


def test_health_still_answers_when_the_model_is_down(tmp_path):
    """Liveness is about the sidecar, not its backend; otherwise it restart-loops."""
    from aaramse.client import ModelUnavailable

    class DeadClient(FakeClient):
        def answer(self, prompt, max_tokens=320):
            raise ModelUnavailable("down")

    config = GatewayConfig(audit_path=tmp_path / "a.jsonl", localization_budget=20)
    dead = GatewayService(gateway=Gateway.build(config=config, client=DeadClient()))
    status, _, _ = dead.dispatch("GET", "/healthz", b"", None)
    assert status == 200
