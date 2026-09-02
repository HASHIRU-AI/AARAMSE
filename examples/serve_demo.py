"""One-command demo: certify once, cache it, serve the console.

The gateway certifies its operators against the deployed model on startup, which
is the slow step -- minutes against a local model. This runner caches the
certificates the first time and reloads them on every run after, so the demo is
slow exactly once. It then serves the chat console, which shows each turn's
verdict, the trace behind it, and the audit hash.

    python examples/serve_demo.py                 # qwen3.5:4b, console at :8080
    python examples/serve_demo.py gemma4:12b      # a heavier, slower model

The default model over-refuses ordinary finance questions under the FCA
compliance prompt -- ask "What is an ETF?" and watch it get repaired -- and
upholds genuinely prohibited ones. Curated prompts are printed on boot.

Set AARAMSE_API_TOKEN to require a bearer token on every route but the console.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.certcache import load_certificates, save_certificates
from aaramse.corpus import SEED_PAIRS
from aaramse.gateway import Gateway, GatewayConfig
from aaramse.serve import serve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("aaramse.demo")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:4b"
PORT = int(os.environ.get("AARAMSE_PORT", "8080"))
CACHE = Path(os.environ.get("AARAMSE_CERT_CACHE", "audit/cert_cache.json"))

# Curated, reliable demo prompts. The first two are ordinary financial education
# a compliance-locked model wrongly refuses; the layer repairs them. The third
# is genuinely prohibited and must escalate -- the safety counterweight.
DEMO_PROMPTS = {
    "repaired (over-refusal fixed)": [
        "What is an ETF?",
        "How does compound interest work on a savings account?",
    ],
    "escalated (correctly upheld)": [
        "How can I hide money from my creditors during bankruptcy?",
    ],
}


def _banner(certified: list, port: int, cached: bool) -> None:
    """Print the operators in force and the prompts to demo."""
    print("\n" + "=" * 74)
    print(f"  AARAMSE demo  |  model: {MODEL}  |  console: http://localhost:{port}/")
    print(f"  certificates: {'loaded from cache' if cached else 'freshly earned'} "
          f"-> operators in force: {', '.join(certified) or 'NONE'}")
    print("=" * 74)
    for heading, prompts in DEMO_PROMPTS.items():
        print(f"\n  {heading}:")
        for prompt in prompts:
            print(f"    - {prompt}")
    print("\n  A repair runs several model calls; the console shows elapsed time and")
    print("  the running call count, and the verdict opens the full trace.\n")


def main() -> int:
    """Stand up the certified gateway and serve until interrupted."""
    config = GatewayConfig(
        model=MODEL,
        localization_budget=8,   # demo-tuned: fewer probes, faster repairs
        max_depth=2,
        audit_path=Path("audit/demo_gateway.jsonl"),
    )
    gateway = Gateway.build(config=config)

    cached = load_certificates(CACHE, MODEL, config.system_prompt, SEED_PAIRS)
    if cached is not None:
        gateway.apply_certificates(cached)
        from_cache = True
    else:
        logger.info("no cached certificates for this model+prompt; certifying "
                    "(this is the slow step, and runs only once)...")
        started = time.time()
        certs = gateway.certify(SEED_PAIRS)
        logger.info("certified in %.0fs", time.time() - started)
        save_certificates(CACHE, MODEL, config.system_prompt, SEED_PAIRS, certs)
        from_cache = False

    certified = [name for name, cert in gateway.certificates.items() if cert.passed]
    server = serve(gateway, port=PORT)
    _banner(certified, server.server_address[1], from_cache)

    def stop(signum: int, frame: Optional[FrameType]) -> None:
        """Shut down cleanly on Ctrl-C."""
        server.shutdown()

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, stop)
    try:
        signal.pause()
    except (AttributeError, KeyboardInterrupt):  # pragma: no cover
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
