"""HTTP surface: the drop-in sidecar the concept note promises.

The gateway was a Python class you had to import, which is not something a
deployed agent can sit behind. This wraps it in a stdlib HTTP server so the
layer can front any agent that can make a request, in any language.

Routing is a pure function (`dispatch`) and the handler is a thin shell over
it, so the routing table is tested without binding a socket.

Two deliberate constraints:

* **One request at a time.** `RepairSearch` and `AuditLog` share mutable state,
  and the log recomputes its tail hash by reading the file, so concurrent
  appends would interleave and break the chain. A lock serialises handling.
  Throughput is not the point of this layer; an unbroken chain is.
* **Fail closed on auth.** If a token is configured, a request without it is
  refused. If none is configured the server says so loudly at startup rather
  than pretending to be protected.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple, Type

from .client import ModelUnavailable
from .gateway import Gateway

__all__ = [
    "MAX_BODY_BYTES",
    "GatewayService",
    "Response",
    "build_handler",
    "serve",
]

logger = logging.getLogger(__name__)

# A query is a sentence, not a payload. Anything larger is a mistake or an abuse.
MAX_BODY_BYTES = 64 * 1024

TOKEN_ENV = "AARAMSE_API_TOKEN"

Response = Tuple[int, str, bytes]


def _json(status: int, payload: Dict[str, Any]) -> Response:
    """Encode a JSON response."""
    return status, "application/json", json.dumps(payload, ensure_ascii=False).encode()


def _text(status: int, body: str, content_type: str = "text/plain; charset=utf-8") -> Response:
    """Encode a plain-text or Markdown response."""
    return status, content_type, body.encode()


@dataclass
class GatewayService:
    """A gateway plus the request handling around it.

    Attributes:
        gateway: The configured repair layer.
        token: Bearer token required on every non-health request. None means
            the service is unauthenticated, which is logged as a warning.
        lock: Serialises handling so the audit chain cannot interleave.
    """

    gateway: Gateway
    token: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self.token:
            logger.warning(
                "no %s configured: this gateway will accept unauthenticated requests",
                TOKEN_ENV,
            )

    def authorised(self, header: Optional[str]) -> bool:
        """Return True when the request may proceed."""
        if self.token is None:
            return True
        if not header:
            return False
        scheme, _, value = header.partition(" ")
        return scheme.lower() == "bearer" and value.strip() == self.token

    def dispatch(self, method: str, path: str, body: bytes, auth: Optional[str]) -> Response:
        """Route one request.

        Args:
            method: HTTP method.
            path: Request path, query string already stripped.
            body: Raw request body.
            auth: Value of the Authorization header, if any.

        Returns:
            Status, content type, and encoded body.
        """
        route = path.rstrip("/") or "/"
        if method == "GET" and route in ("/healthz", "/"):
            return _json(200, {"status": "ok", "model": self.gateway.config.model})
        if not self.authorised(auth):
            return _json(401, {"error": "unauthorized"})

        try:
            if method == "POST" and route == "/v1/repair":
                return self._repair(body)
            if method == "GET" and route == "/v1/report":
                with self.lock:
                    return _json(200, self.gateway.intervention_report().to_dict())
            if method == "GET" and route == "/v1/report.md":
                with self.lock:
                    return _text(
                        200, self.gateway.render_report(), "text/markdown; charset=utf-8"
                    )
        except ModelUnavailable as error:
            # The layer cannot decide anything without the model it fronts. Say
            # which, rather than making an operator read a stack trace: this is
            # what a deployer sees when their model server is down.
            logger.error("model backend unavailable: %s", error)
            return _json(503, {"error": "model backend unavailable", "detail": str(error)})
        return _json(404, {"error": "not found", "path": route})

    def _repair(self, body: bytes) -> Response:
        """Handle one repair request."""
        if len(body) > MAX_BODY_BYTES:
            return _json(413, {"error": "request body too large"})
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return _json(400, {"error": "body is not valid JSON"})
        if not isinstance(payload, dict):
            return _json(400, {"error": "body must be a JSON object"})

        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return _json(400, {"error": "'query' must be a non-empty string"})

        with self.lock:
            result = self.gateway.handle(query)
        return _json(200, {
            "decision": result.decision.value,
            # What the caller forwards to the agent. Identical to `query`
            # unless the decision is "repaired".
            "rewritten": result.rewritten,
            "program": list(result.program.names),
            "program_render": result.program.render(),
            "refusal_margin": result.refusal_margin,
            "oracle_calls": result.oracle_calls,
            "reason": result.reason,
        })



def build_handler(service: GatewayService) -> Type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one service."""

    class Handler(BaseHTTPRequestHandler):
        """Thin shell over `GatewayService.dispatch`."""

        server_version = "aaramse"
        protocol_version = "HTTP/1.1"

        def _run(self, method: str) -> None:
            """Read the request, dispatch it, and write the response."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            body = self.rfile.read(min(length, MAX_BODY_BYTES + 1)) if length else b""
            path = self.path.split("?", 1)[0]
            try:
                status, content_type, payload = service.dispatch(
                    method, path, body, self.headers.get("Authorization")
                )
            except Exception:
                logger.exception("unhandled error serving %s %s", method, path)
                status, content_type, payload = _json(500, {"error": "internal error"})
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            """Serve health and report routes."""
            self._run("GET")

        def do_POST(self) -> None:
            """Serve the repair route."""
            self._run("POST")

        def log_message(self, format: str, *args: Any) -> None:
            """Route access logs through logging instead of stderr."""
            logger.info("%s - %s", self.address_string(), format % args)

    return Handler


def serve(
    gateway: Gateway,
    host: str = "0.0.0.0",
    port: int = 8080,
    token: Optional[str] = None,
) -> ThreadingHTTPServer:
    """Build and start an HTTP server in the background.

    Args:
        gateway: The configured repair layer.
        host: Interface to bind.
        port: Port to bind; 0 selects a free one.
        token: Bearer token to require. Falls back to `AARAMSE_API_TOKEN`.

    Returns:
        The running server. Call `shutdown()` to stop it.
    """
    service = GatewayService(
        gateway=gateway,
        token=token or os.environ.get(TOKEN_ENV) or None,
    )
    server = ThreadingHTTPServer((host, port), build_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("aaramse listening on %s:%s", host, server.server_address[1])
    return server
