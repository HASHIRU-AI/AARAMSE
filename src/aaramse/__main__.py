"""Command line entry point, so the container has something to run.

    python -m aaramse serve                     # run the sidecar
    python -m aaramse repair "some question"    # one query, printed as JSON
    python -m aaramse report --audit a.jsonl    # render the supervisor report

Every option also reads an environment variable, because that is how a
container is configured. Flags win over the environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Optional, Sequence

from .audit import AuditLog
from .gateway import DEFAULT_SYSTEM_PROMPT, Gateway, GatewayConfig
from .report import build_report
from .serve import TOKEN_ENV, serve

__all__ = ["build_parser", "main"]

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    """Read an environment variable, falling back to a default."""
    return os.environ.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    # Shared options live on a parent parser so they are accepted on either
    # side of the subcommand. `aaramse repair --log-level DEBUG "q"` is the
    # order a person actually types; rejecting it is a papercut.
    # SUPPRESS, not a real default. `parents=` shares one action object between
    # this parser and every subparser, and a subparser parses into a fresh
    # namespace whose defaults are then copied over the parent's -- so a real
    # default here makes `--log-level DEBUG repair q` silently revert to INFO.
    # `set_defaults` does not help: it mutates the shared action. The default is
    # resolved in `main` instead.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level", default=argparse.SUPPRESS,
        help="Python logging level (default: INFO).",
    )

    parser = argparse.ArgumentParser(
        prog="aaramse",
        description="Auditable over-refusal repair for regulated AI advice.",
        parents=[common],
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    def add_gateway_options(sub: argparse.ArgumentParser) -> None:
        """Options shared by every subcommand that builds a gateway."""
        sub.add_argument(
            "--model", default=_env("AARAMSE_MODEL", "gemma4:12b"),
            help='Model spec: "openai:gpt-5", "anthropic:claude-opus-5", or an Ollama tag.',
        )
        sub.add_argument(
            "--audit", default=_env("AARAMSE_AUDIT_PATH", "audit/gateway.jsonl"),
            help="Path to the hash-chained audit log.",
        )
        sub.add_argument(
            "--deployer", default=_env("AARAMSE_DEPLOYER", "Acme Wealth Ltd"),
            help="Authorised firm operating the gateway.",
        )
        sub.add_argument(
            "--authorisation-ref", default=_env("AARAMSE_AUTHORISATION_REF", "FRN-123456"),
            help="That firm's regulatory reference.",
        )
        sub.add_argument(
            "--system-prompt", default=_env("AARAMSE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            help="Deployment compliance instruction.",
        )
        sub.add_argument(
            "--localization-budget", type=int,
            default=int(_env("AARAMSE_LOCALIZATION_BUDGET", "8")),
            help="Delta-debugging probes per query (default: 8). The library "
                 "default is 32, which is a measurement setting; localization "
                 "dominates a repair's call burst and hosted free tiers "
                 "throttle inside a single repair at that size. Raising it "
                 "buys a finer fragment at the cost of latency and quota.",
        )
        sub.add_argument(
            "--allow-uncertified", action="store_true",
            default=_env("AARAMSE_ALLOW_UNCERTIFIED", "") == "1",
            help="Run operators that hold no passing certificate. Off by default: "
                 "an uncertified operator is excluded, not trusted.",
        )

    serve_cmd = subcommands.add_parser("serve", parents=[common], help="Run the HTTP sidecar.")
    add_gateway_options(serve_cmd)
    serve_cmd.add_argument("--host", default=_env("AARAMSE_HOST", "0.0.0.0"))
    serve_cmd.add_argument("--port", type=int, default=int(_env("AARAMSE_PORT", "8080")))

    repair_cmd = subcommands.add_parser(
        "repair", parents=[common], help="Repair one query and print the result."
    )
    add_gateway_options(repair_cmd)
    repair_cmd.add_argument("query", help="The user query to put to the layer.")

    report_cmd = subcommands.add_parser(
        "report", parents=[common], help="Render the intervention report."
    )
    report_cmd.add_argument(
        "--audit", default=_env("AARAMSE_AUDIT_PATH", "audit/gateway.jsonl"),
        help="Path to the audit log to read.",
    )
    report_cmd.add_argument("--model", default=_env("AARAMSE_MODEL", "unknown"))
    report_cmd.add_argument(
        "--json", action="store_true", help="Emit JSON instead of Markdown."
    )
    return parser


def _gateway(args: argparse.Namespace) -> Gateway:
    """Build a gateway from parsed arguments."""
    return Gateway.build(GatewayConfig(
        model=args.model,
        system_prompt=args.system_prompt,
        deployer_name=args.deployer,
        authorisation_ref=args.authorisation_ref,
        audit_path=Path(args.audit),
        localization_budget=args.localization_budget,
        require_certificates=not args.allow_uncertified,
    ))


def _shutdown_event() -> threading.Event:
    """Install SIGINT/SIGTERM handlers that request shutdown.

    The handler records the request rather than acting on it. Shutting the
    server down from inside a signal handler means the only evidence that a
    stop was actually asked for is the handler having run, and `signal.pause()`
    cannot be trusted to report that -- see `_wait_for_shutdown`.
    """
    stopping = threading.Event()

    def stop(signum: int, frame: Optional[FrameType]) -> None:
        """Record that a shutdown was requested."""
        logger.info("received signal %s; shutting down", signum)
        stopping.set()

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, stop)
    return stopping


def _wait_for_shutdown(stopping: threading.Event, poll: float = 1.0) -> None:
    """Block until shutdown is actually requested.

    This was `signal.pause()`, which returns on *any* interruption rather than
    only on a handled SIGINT/SIGTERM, and the caller treated one return as a
    stop request. LiteLLM's lazy first-call import interrupts it within about
    two seconds, so the sidecar shut itself down on the first turn -- exit code
    0, no traceback, no log line, which is the worst way for a demo to fail.
    Waiting on the flag the handler sets is immune: a spurious wakeup resumes
    the loop, and only a real request ends it.
    """
    while not stopping.wait(poll):
        pass


def _serve(args: argparse.Namespace) -> int:
    """Run the sidecar until interrupted."""
    if not os.environ.get(TOKEN_ENV):
        logger.warning(
            "%s is not set. The sidecar will answer unauthenticated requests; "
            "set it before exposing this port.", TOKEN_ENV,
        )
    server = serve(
        _gateway(args), host=args.host, port=args.port,
    )

    stopping = _shutdown_event()
    logger.info("serving on %s:%s", args.host, server.server_address[1])
    try:
        _wait_for_shutdown(stopping)
    except KeyboardInterrupt:  # pragma: no cover - depends on handler timing
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _repair(args: argparse.Namespace) -> int:
    """Put one query to the layer and print what it decided."""
    result = _gateway(args).handle(args.query)
    print(json.dumps({
        "decision": result.decision.value,
        "rewritten": result.rewritten,
        "program": list(result.program.names),
        "refusal_margin": result.refusal_margin,
        "reason": result.reason,
    }, indent=2, ensure_ascii=False))
    return 0


def _report(args: argparse.Namespace) -> int:
    """Render the report for an existing audit log."""
    path = Path(args.audit)
    if not path.exists():
        print(f"no audit log at {path}", file=sys.stderr)
        return 1
    report = build_report(AuditLog(path), model=args.model)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
          if args.json else report.render_markdown())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and run the requested subcommand.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    level = getattr(args, "log_level", None) or _env("AARAMSE_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    handlers = {"serve": _serve, "repair": _repair, "report": _report}
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
