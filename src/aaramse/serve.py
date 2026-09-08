"""HTTP surface: the drop-in supervisory sidecar.

Provides a standard-library HTTP server in front of the Gateway, allowing any
deployed agent or application — in any programming language — to interact with
AARAMSE over standard REST endpoints.

Routing is implemented as a pure function (`dispatch`), making the HTTP routing
table unit-testable without binding network sockets.

## Key architectural constraints:

* **One request at a time per log (sequential cryptographic chaining):**
  `RepairSearch` and `AuditLog` update shared state, and the audit log extends
  its cryptographic SHA-256 hash chain sequentially on disk. Handling requests
  under a lock ensures no interleaved writes break the chain. For high-volume
  deployments, scaling is horizontal: run multiple worker gateways, each managing
  its own independent log file.
* **Repairs are asynchronous jobs, not blocking calls:**
  Testing multiple candidate repairs against a live model takes several model calls.
  `POST /v1/chat` immediately returns a `202 Accepted` status with a job ID, and
  the caller polls `/v1/chat/{id}` for status. This prevents open HTTP connections
  from timing out during longer repairs.
* **Fail-closed authentication:**
  If an API token (`AARAMSE_API_TOKEN`) is configured, incoming requests without a
  matching Bearer token are rejected immediately. If no token is set, the server
  warns clearly in startup logs rather than giving a false impression of security.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple, Type

from .client import ModelUnavailable
from .gateway import Gateway
from .providers import api_key_env_for, litellm_spec
from .ui import JobStore, attempts_of, baseline_of, console_html, trace_of

__all__ = [
    "MAX_BODY_BYTES",
    "GatewayService",
    "Response",
    "build_handler",
    "serve",
]

CONSOLE_PATHS = ("/", "/console")

logger = logging.getLogger(__name__)

# A query is a sentence, not a payload. Anything larger is a mistake or an abuse.
MAX_BODY_BYTES = 64 * 1024

TOKEN_ENV = "AARAMSE_API_TOKEN"

Response = Tuple[int, str, bytes]


def _model_slug(spec: str) -> str:
    """Filesystem-safe stem for a model spec, for its own audit chain."""
    return re.sub(r"[^a-z0-9]+", "-", spec.lower()).strip("-") or "model"


def _html(status: int, payload: bytes) -> Response:
    """Encode an HTML response."""
    return status, "text/html; charset=utf-8", payload


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
        allow_model_swap: Whether `/v1/model` may change the model or store a
            credential. True suits the local tool this console is: the key is
            the reader's own and the process is theirs. A hosted deployment
            must set it False, because there is one process environment and one
            gateway, so a key pasted by one visitor would serve the next
            visitor's turns and a swap would move the model for everyone.
        lock: Serialises handling so the audit chain cannot interleave.
        jobs: Background work the console polls on, because a repair can take
            minutes and does not fit behind a synchronous request.
    """

    gateway: Gateway
    token: Optional[str] = None
    allow_model_swap: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)
    jobs: JobStore = field(default_factory=JobStore)

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
        if method == "GET" and route == "/healthz":
            return _json(200, {"status": "ok", "model": self.gateway.config.model})
        # The console is a static page carrying no user data, so it sits
        # outside the token like the health route. Everything it calls does not.
        if method == "GET" and route in CONSOLE_PATHS:
            return _html(200, console_html())
        if not self.authorised(auth):
            return _json(401, {"error": "unauthorized"})

        try:
            if method == "POST" and route == "/v1/repair":
                return self._repair(body)
            if method == "POST" and route == "/v1/chat":
                return self._chat_start(body)
            if method == "GET" and route.startswith("/v1/chat/"):
                return self._chat_poll(route.rsplit("/", 1)[-1])
            if method == "POST" and route == "/v1/model":
                if not self.allow_model_swap:
                    return _json(403, {
                        "error": "model swapping is disabled on this deployment",
                    })
                return self._swap_model(body)
            if method == "GET" and route == "/v1/config":
                return _json(200, {
                    "model": self.gateway.config.model,
                    "deployer_name": self.gateway.config.deployer_name,
                    "authorisation_ref": self.gateway.config.authorisation_ref,
                    "system_prompt": self.gateway.config.system_prompt,
                    "operators": [op.name for op in self.gateway.operators],
                    "certificates": {
                        name: cert.to_dict()
                        for name, cert in self.gateway.certificates.items()
                    },
                    "authenticated": self.token is not None,
                    "api_key_env": api_key_env_for(self.gateway.config.model),
                    "credential_present": self._credential_present(),
                    "model_swap_allowed": self.allow_model_swap,
                    "rewriter_model": self.gateway.config.rewriter_model,
                    "rewriter_api_key_env": (
                        api_key_env_for(self.gateway.config.rewriter_model)
                        if self.gateway.config.rewriter_model else None
                    ),
                })
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

    def _credential_present(self) -> bool:
        """Return whether the current model's credential is in the environment."""
        env = api_key_env_for(self.gateway.config.model)
        return env is None or bool(os.environ.get(env, "").strip())

    def _swap_model(self, body: bytes) -> Response:
        """Point the layer at a different model, with an optional credential.

        The console is a local tool, which is the only reason this is
        acceptable: a key posted here is put in this process's environment for
        the provider to read, and is never written to disk, logged, echoed back,
        or recorded in a run. Exposing this port would be exposing a credential
        sink.

        The audit path moves with the model. A hash chain spanning two models
        describes neither of them, and `intervention_report` takes the model as
        a parameter precisely because the records do not carry one.
        """
        if len(body) > MAX_BODY_BYTES:
            return _json(413, {"error": "request body too large"})
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return _json(400, {"error": "body is not valid JSON"})
        if not isinstance(payload, dict):
            return _json(400, {"error": "body must be a JSON object"})

        base = self.gateway.config
        changes: Dict[str, Any] = {}

        # Each slot moves on its own. Sending only one leaves the other alone,
        # so a reader can change the model being repaired without silently
        # also changing the instrument that repairs it.
        for field_name, key_name in (("model", "api_key"), ("rewriter_model", "rewriter_api_key")):
            raw = payload.get(field_name)
            if raw is None:
                continue
            if not isinstance(raw, str) or not raw.strip():
                return _json(400, {"error": f"'{field_name}' must be a non-empty string"})
            try:
                spec = litellm_spec(raw.strip())
            except ValueError as error:
                # A bad spec fails here as a sentence, rather than as a 503 on
                # the reader's next turn with nothing saying which field was wrong.
                return _json(400, {"error": str(error)})
            changes[field_name] = spec
            secret = payload.get(key_name)
            env = api_key_env_for(spec)
            if isinstance(secret, str) and secret.strip() and env:
                os.environ[env] = secret.strip()

        if not changes:
            return _json(400, {"error": "name at least one of 'model' or 'rewriter_model'"})

        with self.lock:
            spec = changes.get("model", base.model)
            audit = base.audit_path.parent / (
                f"{base.audit_path.stem}_{_model_slug(spec)}{base.audit_path.suffix}"
            )
            self.gateway = Gateway.build(replace(base, audit_path=audit, **changes))
            # Jobs describe turns taken against the previous pair; keeping them
            # would let the console poll a trace and render it under a heading
            # naming models that never produced it.
            self.jobs = JobStore()

        logger.info("models now %s / rewriter %s",
                    self.gateway.config.model,
                    self.gateway.config.rewriter_model or "(same)")
        return _json(200, {
            "model": self.gateway.config.model,
            "rewriter_model": self.gateway.config.rewriter_model,
            "api_key_env": api_key_env_for(self.gateway.config.model),
            "credential_present": self._credential_present(),
            "audit_path": str(self.gateway.config.audit_path),
        })

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



    def _chat_start(self, body: bytes) -> Response:
        """Accept one chat turn and hand back a job to poll.

        A repair runs 23-26 model calls. Holding the socket open for that is
        what the deployment notes say not to do, so the work goes on a thread
        and the caller polls.
        """
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

        started = time.monotonic()
        calls_at_start = self.gateway.client.calls

        def work() -> Dict[str, Any]:
            with self.lock:
                result = self.gateway.handle(query)
                probe = self.gateway.probe
                # The reply the user receives is the one the search accepted,
                # which is the generation for whatever prompt actually ran.
                answer = getattr(probe, "answers", {}).get(result.rewritten, "")
                records = list(self.gateway.audit.read())
                audit = {
                    "index": len(records) - 1,
                    "hash": records[-1]["hash"] if records else "",
                    "prev_hash": records[-1]["prev_hash"] if records else "",
                    "chain_ok": self.gateway.audit.verify() is None,
                } if records else {}
                return trace_of(
                    result,
                    answer=answer,
                    baseline=baseline_of(probe, query),
                    audit=audit,
                    elapsed_s=time.monotonic() - started,
                    model_calls=self.gateway.client.calls - calls_at_start,
                    # Read after the search, while the operators still hold this
                    # turn's attempt; the next turn resets them.
                    attempts=attempts_of(self.gateway.operators),
                )

        job = self.jobs.start("chat", query, work, calls_at_start=calls_at_start)
        return _json(202, job.snapshot(self.gateway.client.calls))

    def _chat_poll(self, job_id: str) -> Response:
        """Report a job's progress, or its result once it has one."""
        job = self.jobs.get(job_id)
        if job is None:
            return _json(404, {"error": "unknown job", "id": job_id})
        return _json(200, job.snapshot(self.gateway.client.calls))


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
    allow_model_swap: bool = True,
) -> ThreadingHTTPServer:
    """Build and start an HTTP server in the background.

    Args:
        gateway: The configured repair layer.
        host: Interface to bind.
        port: Port to bind; 0 selects a free one.
        token: Bearer token to require. Falls back to `AARAMSE_API_TOKEN`.
        allow_model_swap: Whether `/v1/model` may change the model or store a
            credential. Set False when hosting; see `GatewayService`.

    Returns:
        The running server. Call `shutdown()` to stop it.
    """
    service = GatewayService(
        gateway=gateway,
        token=token or os.environ.get(TOKEN_ENV) or None,
        allow_model_swap=allow_model_swap,
    )
    server = ThreadingHTTPServer((host, port), build_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(
        "aaramse listening on %s:%s -- console at http://%s:%s/",
        host, server.server_address[1],
        "localhost" if host in ("0.0.0.0", "") else host, server.server_address[1],
    )
    return server
