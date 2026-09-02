"""Why did the control set refuse? Separate compliance refusal from a broken item.

For every FinQA item the gateway did not pass through, ask the model twice:

    with    the FCA compliance system prompt  -- the deployed condition
    without any system prompt                 -- the model's own disposition

If it refuses with and answers without, the system prompt caused it: a real
over-refusal on arithmetic over a public filing. If it declines both ways, the
item is unanswerable from the table it ships with and the corpus is at fault.

    python finqa_cause.py [model]
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.client import OllamaClient
from aaramse.finqa import load_control
from aaramse.gateway import DEFAULT_SYSTEM_PROMPT
from aaramse.judge import ThreeWayJudge

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b"
RESULTS = Path("audit/finqa_control.json")
OUT = Path("audit/finqa_cause.json")


def main() -> int:
    """Re-ask every refused control item with and without the system prompt."""
    data = json.loads(RESULTS.read_text())
    refused_ids = {r["source_id"] for r in data["records"] if r["decision"] != "passthrough"}
    if not refused_ids:
        print("no refusals in the control run; nothing to explain")
        return 0

    items = [i for i in load_control() if i.source_id in refused_ids]
    framed = OllamaClient(model=MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT)
    bare = OllamaClient(model=MODEL, system_prompt=None)
    judge = ThreeWayJudge(generate=lambda p: bare.complete(p, 0.0, 24))

    rows: List[Dict[str, Any]] = []
    for item in items:
        with_sp = framed.answer(item.prompt)
        without_sp = bare.answer(item.prompt)
        j_with = judge.classify(item.prompt, with_sp).label
        j_without = judge.classify(item.prompt, without_sp).label
        cause = ("system prompt" if j_with.name == "FULL_REFUSAL" and j_without.answered
                 else "unanswerable item" if not j_without.answered
                 else "unclear")
        rows.append({
            "source_id": item.source_id,
            "question": item.question,
            "with_system_prompt": j_with.value,
            "without_system_prompt": j_without.value,
            "cause": cause,
            "reply_with": with_sp[:400],
            "reply_without": without_sp[:400],
        })
        print(f"{item.source_id}\n  with={j_with.value}  without={j_without.value}  -> {cause}")
        print(textwrap.indent(textwrap.fill(with_sp[:300], 92), "    "), flush=True)

    tally: Dict[str, int] = {}
    for row in rows:
        tally[row["cause"]] = tally.get(row["cause"], 0) + 1
    OUT.write_text(json.dumps({"model": MODEL, "tally": tally, "rows": rows}, indent=1) + "\n")
    print("\n" + json.dumps(tally, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
