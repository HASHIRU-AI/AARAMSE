"""Control: reproduce DDOR's setting before claiming any negative result.

DDOR evaluates on OR-Bench with NO system prompt, where refusal is triggered by
lexical content in the user's prompt. This project's other runs use a strict FCA
compliance system prompt, where refusal is driven by the system instruction.

If targeted repair works here and not there, the difference is a real boundary
condition. If it fails here too, the implementation is wrong and every other
number in this project is meaningless.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.client import OllamaClient
from aaramse.judge import ResponseClass, ThreeWayJudge
from aaramse.rewriter import SemanticEquivalence
from aaramse.targeted import TargetedConfig, TargetedRepair

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUT = Path("audit/control_eval.json")
# No system prompt: this is DDOR's setting, where refusal is triggered by
# lexical content in the user's prompt rather than by a deployment instruction.
CLIENT = OllamaClient(model=MODEL, system_prompt=None)


def raw(prompt: str, temperature: float = 0.0, n: int = 160) -> str:
    """Plain completion, no system prompt."""
    return CLIENT.complete(prompt, temperature, n)


def main() -> int:
    """Measure baseline over-refusal and targeted-repair rate, no system prompt."""
    prompts = [r["prompt"] for r in json.loads(Path("data/orbench_hard.json").read_text())][:N]
    judge = ThreeWayJudge(generate=lambda p: raw(p, 0.0, 24))
    cache: Dict[str, str] = {}

    def label_of(prompt: str) -> ResponseClass:
        """Classify the model's answer to a prompt, memoised."""
        if prompt not in cache:
            cache[prompt] = raw(prompt)
        return judge.classify(prompt, cache[prompt]).label

    def refuses(text: str) -> bool:
        """Refusal oracle for delta debugging."""
        return label_of(text) is ResponseClass.FULL_REFUSAL

    repairer = TargetedRepair(
        refuses=refuses,
        generate=lambda p: raw(p, 0.0, 40),
        equivalence=SemanticEquivalence(generate=lambda p: raw(p, 0.0, 8)),
        config=TargetedConfig(max_localization_tests=12),
    )

    started = time.time()
    records: List[Dict[str, Any]] = []
    for prompt in prompts:
        base = label_of(prompt)
        rec: Dict[str, Any] = {"prompt": prompt, "baseline_class": base.value}
        if base is ResponseClass.FULL_REFUSAL:
            application = repairer.apply(prompt)
            loc = repairer.last_localization
            rec["mrtf"] = loc.text[:120] if loc else None
            rec["localization_tests"] = loc.tests if loc else 0
            if application is not None:
                rec["rewritten"] = application.after
                rec["substitutions"] = [list(s) for s in application.generalizations]
                after = label_of(application.after)
                rec["after_class"] = after.value
                rec["repaired"] = after is not ResponseClass.FULL_REFUSAL
            else:
                rec["repaired"] = False
                rec["reject_reasons"] = [r for _, r in repairer.rejected[-3:]]
        records.append(rec)
        print(
            f"{rec['baseline_class']:<16} repaired={rec.get('repaired', '-')!s:<6} "
            f"mrtf={str(rec.get('mrtf'))[:40]:<40} {prompt[:38]}",
            flush=True,
        )

    refused = [r for r in records if r["baseline_class"] == "full_refusal"]
    repaired = [r for r in refused if r.get("repaired")]
    localized = [r for r in refused if r.get("mrtf")]
    summary = {
        "model": MODEL, "setting": "OR-Bench-Hard-1K, no system prompt (DDOR setting)",
        "n": len(records), "elapsed_s": round(time.time() - started, 1),
        "model_calls": CLIENT.calls,
        "over_refusal_rate": round(len(refused) / len(records), 3) if records else 0.0,
        "refused": len(refused),
        "localized": len(localized),
        "repaired": len(repaired),
        "repair_rate": round(len(repaired) / len(refused), 3) if refused else None,
        "ddor_reported_orbench_reduction": 0.5196,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
