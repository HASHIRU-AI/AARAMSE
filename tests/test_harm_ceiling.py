"""The harm gate's recall ceiling, pinned so it is not rediscovered by hand.

31% held out is the number to beat, and the obvious way to beat it does not
work. A second, deliberately more *compositional* pattern family -- verb x
object rather than whole phrases, authored against the certification fold under
the same discipline as the shipped patterns -- was measured and dropped:

    fold        shipped   with compositional patterns
    authoring   20/30     28/30   (+8)
    held out     9/29      9/29   (+0)
    benign        0/46      1/46  (a new false positive)

Eight more on the fold they were written against and none held out. That is not
a bad pattern set, it is evidence about the method: the benign corpus is
FalseReject, whose items are *engineered to look unsafe*, so lexical separation
from a genuinely unsafe corpus is close to its ceiling by construction.

These tests fail if the measured rates move. That is the point -- they are a
tripwire telling whoever moved them to update `plan/harm-gate.md`, not a claim
that the current numbers are good.
"""

from __future__ import annotations

from aaramse.falsereject import load_benign, load_toxic
from aaramse.harm import HarmScorer, PROHIBITED_THRESHOLD


def _caught(items) -> int:
    scorer = HarmScorer()
    return sum(1 for i in items if scorer.profile(i.prompt).score >= PROHIBITED_THRESHOLD)


def test_held_out_recall_is_still_the_documented_31_percent():
    """If this moves, `plan/harm-gate.md` is stale."""
    held = load_toxic(fold="evaluation")
    assert (_caught(held), len(held)) == (9, 29)


def test_held_out_benign_cost_is_still_zero():
    """Precision is the gate's one strong property; protect it."""
    held = load_benign(fold="evaluation")
    assert _caught(held) == 0


def test_the_train_to_heldout_gap_is_real_and_large():
    """67% authoring against 31% held out is the generalization gap itself."""
    cert, held = load_toxic(fold="certification"), load_toxic(fold="evaluation")
    authoring_rate = _caught(cert) / len(cert)
    heldout_rate = _caught(held) / len(held)
    assert authoring_rate > heldout_rate * 1.5, (
        "the gap closed; re-read plan/harm-gate.md before assuming this is good news"
    )


def test_most_held_out_prohibited_queries_still_reach_the_search():
    """The layer's honest weakness, stated as an executable fact.

    Two thirds of held-out prohibited queries are admitted to repair. What stops
    them is search exhaustion, not this gate.
    """
    held = load_toxic(fold="evaluation")
    admitted = len(held) - _caught(held)
    assert admitted == 20
