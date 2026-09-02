"""The third obligation, measured on a real model: leave benign queries alone.

FalseReject measures repair and OR-Bench measures leakage. Neither catches a
layer that intervenes on a query nobody refused. `tests/test_control.py` asserts
that property against the simulated boundary, where it cannot fail -- the
simulator is verified two tests earlier not to refuse the control set. This
script asserts it against a model that can.

Every FinQA control prompt must come back `PASSTHROUGH` and byte-identical. Any
other decision is a false intervention: it costs latency and a supervisor's
attention on a query that was never objected to.

    python examples/finqa_control.py                   # all 120, gemma4:12b
    python examples/finqa_control.py gemma4:12b 10     # a subset, to smoke-test

Writes `audit/finqa_control.json` plus a hash-chained `audit/finqa_control.jsonl`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.finqa import load_control
from aaramse.gateway import DEFAULT_SYSTEM_PROMPT, Gateway, GatewayConfig
from aaramse.types import Decision

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 means the whole slice
OUT = Path(os.environ.get("FINQA_OUT", "audit/finqa_control.json"))
AUDIT = OUT.with_suffix(".jsonl")
# Same env var the sidecar reads, so an experiment and a deployment are
# configured the same way. Unset means the FCA prompt the other runs used.
SYSTEM_PROMPT = os.environ.get("AARAMSE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


def _median(values: List[float]) -> float:
    """Median of a possibly empty list, rounded; 0.0 when there is nothing to take."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 1)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 1)


def main() -> int:
    """Run the control set through the gateway and report false interventions."""
    control = load_control()
    if N:
        control = control[:N]
    if AUDIT.exists():
        AUDIT.unlink()

    gateway = Gateway.build(config=GatewayConfig(
        model=MODEL, audit_path=AUDIT, system_prompt=SYSTEM_PROMPT))
    records: List[Dict[str, Any]] = []
    started = time.time()

    for index, item in enumerate(control, 1):
        began = time.time()
        result = gateway.handle(item.prompt)
        identical = result.rewritten == item.prompt
        clean = result.decision is Decision.PASSTHROUGH and identical
        records.append({
            "source_id": item.source_id,
            "question": item.question,
            "decision": result.decision.value,
            "byte_identical": identical,
            "refusal_margin": result.refusal_margin,
            "program": result.program.render(),
            "oracle_calls": result.oracle_calls,
            "elapsed_s": round(time.time() - began, 1),
        })
        print(f"[{index}/{len(control)}] {records[-1]['decision']:<12} "
              f"{records[-1]['elapsed_s']:>6.1f}s  {item.question[:58]}"
              f"{'' if clean else '  <-- FALSE INTERVENTION'}", flush=True)

    missed = [r for r in records if r["decision"] != "passthrough" or not r["byte_identical"]]
    report = gateway.report()
    summary = {
        "model": MODEL,
        "system_prompt_is_default": SYSTEM_PROMPT == DEFAULT_SYSTEM_PROMPT,
        "n": len(control),
        "false_interventions": len(missed),
        "false_intervention_rate": round(len(missed) / len(control), 4) if control else 0.0,
        "by_decision": {d: sum(1 for r in records if r["decision"] == d)
                        for d in sorted({r["decision"] for r in records})},
        "median_passthrough_s": _median(
            [r["elapsed_s"] for r in records if r["decision"] == "passthrough"]
        ),
        "elapsed_s": round(time.time() - started, 1),
        "model_calls": report["model_calls"],
        "chain_intact": report["chain_intact"],
    }
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {OUT} and {AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
