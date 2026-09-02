"""Do the three refusal instruments agree? Scored on recorded replies.

    python examples/judge_agreement.py --reports DIR --model gemma4:12b --limit 60

Three instruments label refusals in this project, and the evaluations trust the
third:

* NAAMSE's numeric score, from the report;
* the rule-based patterns in `refusal.py`, which sit on the runtime path;
* the three-way judge in `judge.py`, validated on six hand labels.

Six hand labels is not a validation. This measures the judge against two
independent instruments on replies the target models already produced, so the
only model calls are the judge's own -- and they run on whatever local model
you have, not on the nineteen models NAAMSE evaluated.

Agreement does not make any instrument correct; three can be wrong together.
It is diagnostic in one direction: every disagreement is a case where at least
one instrument is wrong, and those are the items worth hand-adjudicating. The
script writes them out for exactly that.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.agreement import agree, best_threshold, confusion_by_class
from aaramse.judge import ResponseClass, ThreeWayJudge
from aaramse.naamse import NaamseRecord, load_directory
from aaramse.providers import build_client
from aaramse.refusal import _REFUSAL_RE, normalise_quotes
from aaramse.splits import split_items

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
RULE = "-" * 78


def pattern_refused(reply: str) -> bool:
    """The runtime instrument: rule-based patterns over the reply prefix."""
    return bool(_REFUSAL_RE.search(normalise_quotes(reply)[:400]))


def sample(records: Sequence[NaamseRecord], limit: int) -> Tuple[NaamseRecord, ...]:
    """Take a balanced, deterministic sample across both pattern classes.

    Balanced because a corpus that is 90% non-refusals lets any instrument look
    accurate by never firing; deterministic (content-hashed, via `splits`) so a
    rerun scores the same items.
    """
    unique = {r.prompt + "\x00" + r.model: r for r in records if r.reply.strip()}
    refused = [r for r in unique.values() if pattern_refused(r.reply)]
    answered = [r for r in unique.values() if not pattern_refused(r.reply)]

    def take(pool: List[NaamseRecord], n: int) -> List[NaamseRecord]:
        if len(pool) <= n:
            return pool
        fraction = n / len(pool)
        chosen = split_items(pool, key=lambda r: r.prompt + r.model, holdout=fraction)
        return list(chosen.evaluation)[:n]

    half = max(1, limit // 2)
    return tuple(take(refused, half) + take(answered, limit - half))


def main() -> int:
    """Score a sample with the judge and report agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, help="Directory of NAAMSE reports")
    parser.add_argument("--model", default="gemma4:12b", help="Model that runs the judge")
    parser.add_argument("--limit", type=int, default=60, help="Replies to judge")
    parser.add_argument("--polarity", default="benign", choices=("benign", "adversarial"))
    parser.add_argument("--out", default="audit/judge_agreement.json")
    args = parser.parse_args()

    reports = load_directory(Path(args.reports))
    records = [r for rep in reports for r in rep.polarity(args.polarity)]
    if not records:
        print(f"no {args.polarity} records under {args.reports}", file=sys.stderr)
        return 1
    chosen = sample(records, args.limit)

    print(RULE)
    print(f"{len(records)} {args.polarity} records; judging {len(chosen)} with {args.model}")
    print("Only the judge costs model calls; the replies are already recorded.")

    client = build_client(args.model)
    judge = ThreeWayJudge(generate=lambda p: client.complete(p, 0.0, 24))

    rows = []
    for i, record in enumerate(chosen, 1):
        verdict = judge.classify(record.prompt, record.reply)
        rows.append({
            "model": record.model,
            "prompt": record.prompt[:200],
            "reply": record.reply[:300],
            "naamse_score": record.score,
            "pattern_refused": pattern_refused(record.reply),
            "judge": verdict.label.value,
            "judge_full_refusal": verdict.label is ResponseClass.FULL_REFUSAL,
        })
        if i % 10 == 0:
            print(f"  judged {i}/{len(chosen)}")

    pattern = [r["pattern_refused"] for r in rows]
    judged = [r["judge_full_refusal"] for r in rows]
    scores = [r["naamse_score"] for r in rows]

    print(RULE)
    print("AGREEMENT")
    pat_judge = agree(pattern, judged)
    print("  " + pat_judge.summary("patterns", "judge"))
    threshold, score_judge = best_threshold(scores, judged, step=2.0)
    print(f"  NAAMSE score >= {threshold} best reproduces the judge:")
    print("  " + score_judge.summary("score", "judge"))
    score_pattern = agree([s >= (threshold or 0) for s in scores], pattern)
    print("  " + score_pattern.summary("score", "patterns"))

    print()
    print("  judge class vs pattern label:")
    for cls, (yes, no) in sorted(confusion_by_class([r["judge"] for r in rows], pattern).items()):
        print(f"    {cls:<18} patterns said refused: {yes:<4} not refused: {no}")

    disagreements = [r for r in rows if r["pattern_refused"] != r["judge_full_refusal"]]
    print()
    print(f"  {len(disagreements)} disagreements -- the adjudication queue:")
    for r in disagreements[:5]:
        print(f"    [{r['model']}] patterns={r['pattern_refused']} judge={r['judge']}")
        print(f"      {r['reply'][:110]!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model_judging": args.model,
        "polarity": args.polarity,
        "n": len(rows),
        "patterns_vs_judge": {
            "raw": pat_judge.observed, "kappa": pat_judge.kappa,
            "disagreements": pat_judge.disagreements,
        },
        "score_vs_judge": {
            "threshold": threshold, "raw": score_judge.observed, "kappa": score_judge.kappa,
        },
        "rows": rows,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
