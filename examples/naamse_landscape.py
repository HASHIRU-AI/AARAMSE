"""Measure refusal-boundary divergence across models, using recorded replies only.

    python examples/naamse_landscape.py --reports path/to/NAAMSE/results

This exists because of a constraint, not despite one: we cannot run the 19
models NAAMSE evaluated. Every number here is computed from replies those
models already produced, so the analysis costs nothing and needs no API key.

The question it answers is the one AARAMSE's certification design assumes:
**does a refusal boundary transfer between models?** If it does, an operator
certified once is certified everywhere and per-model certification is
ceremony. If it does not, a certificate is a property of (operator, model,
corpus) and has to be re-earned per deployment.

Two comparisons, and only one of them is valid:

* Across seeds -- NOT comparable. NAAMSE evolves a different prompt set per
  seed, so two models on different seeds were asked different questions. A
  league table over all models measures prompt difficulty as much as model
  behaviour, and this script refuses to print one.
* Within a seed, restricted to prompts every model in the cohort saw --
  comparable. Same questions, different models.

Labels come from `refusal.py`'s capability-aware patterns, which agree with the
three-way judge at kappa +0.83 over 60 hand-checked replies -- not from an
adjudicated judgement, and NAAMSE's "benign" labels are its own. Harness
failures (13% of the published records) are dropped by the loader before
anything is counted. Treat the output as a measurement to adjudicate, not a
finding.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.naamse import (
    NaamseReport,
    group_by_seed,
    is_generation_error,
    load_directory,
    shared_prompts,
)
from aaramse.refusal import _REFUSAL_RE, is_capability_limit, normalise_quotes

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
RULE = "-" * 78


def refused(reply: str, prefix_chars: int = 400) -> bool:
    """Apply the runtime refusal test to a recorded reply.

    Capability-aware: "I can't access your account" is the model lacking data,
    not declining a request. Counting those inflated the corpus-wide rate by a
    third (211 -> 141 of 1269 replies) before the exclusion existed.
    """
    head = normalise_quotes(reply)[:prefix_chars]
    return bool(_REFUSAL_RE.search(head)) and not is_capability_limit(head, prefix_chars)


def cohort_table(
    group: Sequence[NaamseReport], prompts: Sequence[str]
) -> Dict[str, List[bool]]:
    """Refusal vector per model over the prompts every model in the cohort saw."""
    table: Dict[str, List[bool]] = {}
    for report in group:
        index = {record.prompt: record for record in report.polarity("benign")}
        table[report.model] = [refused(index[p].reply) for p in prompts]
    return table


def health(report: NaamseReport) -> Tuple[int, int]:
    """Return (usable replies, harness failures) for a report's benign records."""
    everything = report.polarity("benign", usable_only=False)
    errors = sum(1 for r in everything if is_generation_error(r.reply))
    return len(everything) - errors, errors


def report_cohort(seed: str, group: Sequence[NaamseReport]) -> Tuple[int, int]:
    """Print one seed cohort. Returns (disagreeing prompts, refused prompts).

    Models whose replies are entirely harness failures are excluded and named.
    Leaving one in would drag the shared-prompt intersection to zero and delete
    the cohort; worse, before harness failures were detected at all, such a
    model scored 0% over-refusal and read as the most permissive in the table.
    """
    live, dead = [], []
    for report in group:
        usable, _ = health(report)
        (live if usable else dead).append(report)
    for report in dead:
        _, errors = health(report)
        print(f"\n  EXCLUDED {report.model} (seed {seed}): "
              f"{errors}/{errors} replies are harness failures; this model produced no output")

    prompts = shared_prompts(live, "benign")
    if len(live) < 2 or not prompts:
        return 0, 0

    table = cohort_table(live, prompts)
    group = live
    models = len(table)
    print(f"\nseed {seed}: {models} models on the same {len(prompts)} benign prompts")
    by_model = {r.model: health(r) for r in group}
    for model, vector in sorted(table.items(), key=lambda kv: -sum(kv[1])):
        usable, errors = by_model.get(model, (0, 0))
        flag = f"   [{errors} harness failures]" if errors else ""
        print(f"  {model:<42} {sum(vector):>3}/{len(vector):<3} "
              f"{sum(vector)/len(vector):6.1%}{flag}")

    counts = [sum(table[m][i] for m in table) for i in range(len(prompts))]
    touched = [c for c in counts if c]
    unanimous = sum(1 for c in touched if c == models)
    print(f"  prompts refused by at least one model : {len(touched)}")
    print(f"    refused by ALL {models} models{'':<10} : {unanimous}")
    print(f"    refused by SOME but not all         : {len(touched) - unanimous}")
    if touched:
        spread = Counter(touched)
        shape = ", ".join(f"{k}/{models}->{v}" for k, v in sorted(spread.items()))
        print(f"    distribution: {shape}")
    return len(touched) - unanimous, len(touched)


def main() -> int:
    """Run the analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, help="Directory of NAAMSE report JSON")
    args = parser.parse_args()

    reports = load_directory(Path(args.reports))
    if not reports:
        print(f"no readable reports under {args.reports}", file=sys.stderr)
        return 1

    records = sum(len(r.records) for r in reports)
    print(RULE)
    print(f"{len(reports)} reports, {len({r.model for r in reports})} models, {records} records")
    print("All figures below are from replies the models already produced.")

    print(RULE)
    print("COMPARABLE COHORTS (same seed, same prompts)")
    disagreeing = total_refused = comparable = 0
    for seed, group in group_by_seed(reports).items():
        if not seed:
            continue
        d, t = report_cohort(seed, group)
        if t:
            comparable += 1
        disagreeing += d
        total_refused += t

    print(RULE)
    print("VERDICT")
    if not total_refused:
        print("  No shared prompt was refused by any model; nothing to conclude.")
        return 0
    share = disagreeing / total_refused
    print(f"  Of {total_refused} shared benign prompts refused by at least one model,")
    print(f"  {disagreeing} ({share:.0%}) were refused by some models and answered by others.")
    print()
    if share >= 0.5:
        print("  Refusal boundaries do NOT transfer between models. An operator")
        print("  certified against one model is not certified against another, which")
        print("  is what makes a certificate a property of (operator, model, corpus)")
        print("  rather than of the operator.")
    else:
        print("  Boundaries largely agree on this corpus; per-model certification is")
        print("  weakly supported by these data and needs a larger shared prompt set.")
    print()
    print("  Caveats: labels are pattern-based, not adjudicated; 'benign' is NAAMSE's")
    print("  own label and at least one jailbreak template appears in a benign set;")
    print("  shared prompt counts are small. Adjudicate before quoting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
