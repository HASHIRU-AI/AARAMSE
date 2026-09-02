"""Three-arm comparison: targeted repair vs full-prompt rewrite vs context injection.

  targeted : localize the mRTF by delta debugging, replace only that fragment
  rewrite  : full-prompt LLM rephrase (DDOR's baseline)
  frame    : prepend authorised-deployer context, question untouched

All arms share one client and one probe, so a repair is always scored on the
generation the search accepted -- an earlier harness issued a second answer call
and the two could disagree.

Usage: python examples/three_arm_eval.py [model] [n_per_class]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.client import OllamaClient
from aaramse.falsereject import load_benign, load_toxic
from aaramse.gateway import DEFAULT_SYSTEM_PROMPT, JudgedProbe
from aaramse.judge import ThreeWayJudge
from aaramse.operators import FrameAssert, FrameConfig
from aaramse.refusal import delivers_procedure
from aaramse.rewriter import LLMRephrase, RewriterConfig, SemanticEquivalence
from aaramse.search import RepairSearch, SearchConfig
from aaramse.targeted import TargetedConfig, TargetedRepair

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = Path("audit/three_arm_eval.json")


def main() -> int:
    """Run all three arms over one shared baseline."""
    client = OllamaClient(model=MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT)
    judge = ThreeWayJudge(generate=lambda p: client.complete(p, 0.0, 24))
    probe = JudgedProbe(client=client, judge=judge)
    equivalence = SemanticEquivalence(generate=lambda p: client.complete(p, 0.0, 8))

    operators = {
        "targeted": TargetedRepair(
            refuses=probe.refuses,
            generate=lambda p: client.complete(p, 0.0, 40),
            equivalence=equivalence,
            config=TargetedConfig(max_localization_tests=32),
        ),
        "rewrite": LLMRephrase(
            generate=lambda p, t: client.complete(p, t, 80),
            equivalence=equivalence,
            config=RewriterConfig(candidates=2, temperature_ladder=(0.0, 0.5)),
        ),
        "frame": FrameAssert(FrameConfig("Acme Wealth Ltd", "FRN-123456")),
    }
    arms = {
        name: RepairSearch([op], probe=probe,
                           config=SearchConfig(max_depth=1, max_oracle_calls=48))
        for name, op in operators.items()
    }

    started = time.time()
    records: List[Dict[str, Any]] = []
    for kind, items in (("benign", load_benign(N)), ("toxic", load_toxic(N))):
        for item in items:
            base = probe.probe(item.prompt)
            rec: Dict[str, Any] = {
                "kind": kind, "prompt": item.prompt,
                "baseline_refused": base.refused, "arms": {},
            }
            if base.refused:
                for name, search in arms.items():
                    result = search.repair(item.prompt)
                    entry: Dict[str, Any] = {"decision": result.decision.value, "answered": False}
                    if result.was_rewritten:
                        label = probe.seen[result.rewritten]
                        reply = probe.answers[result.rewritten]
                        entry.update({
                            "after_class": label.value,
                            "answered": not label.is_over_refusal,
                            "rewritten": result.rewritten[:280],
                            "delivers_procedure": delivers_procedure(reply),
                            "answer": reply[:350],
                        })
                    for step in result.program.steps:
                        if step.localization is not None:
                            entry["mrtf"] = step.localization.text[:120]
                            entry["localization_probes"] = step.localization.tests
                        if step.generalizations:
                            entry["edits"] = [list(g) for g in step.generalizations]
                    rec["arms"][name] = entry
            records.append(rec)
            flags = "  ".join(
                f"{a}={'Y' if rec['arms'].get(a, {}).get('answered') else '.'}" for a in arms
            )
            print(f"[{kind}] refused={base.refused!s:<5} {flags}  {item.prompt[:38]}", flush=True)

    def score(kind: str, arm: str) -> Dict[str, int]:
        rows = [r for r in records if r["kind"] == kind and r["baseline_refused"]]
        return {
            "refused_at_baseline": len(rows),
            "answered_after": sum(1 for r in rows if r["arms"].get(arm, {}).get("answered")),
            "delivered_procedure": sum(
                1 for r in rows if r["arms"].get(arm, {}).get("delivers_procedure")
            ),
        }

    summary = {
        "model": MODEL, "n_per_class": N,
        "elapsed_s": round(time.time() - started, 1), "model_calls": client.calls,
        "benign_repair": {a: score("benign", a) for a in arms},
        "toxic_leak": {a: score("toxic", a) for a in arms},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
