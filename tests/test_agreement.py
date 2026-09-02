"""Agreement statistics decide whether a label is worth trusting.

The trap these tests exist for: on a skewed corpus, two instruments that both
say "no" almost always will show high raw agreement while agreeing about
nothing. Kappa has to expose that, or the measurement flatters the judge.
"""

from __future__ import annotations

import pytest

from aaramse.agreement import agree, best_threshold, confusion_by_class


def test_perfect_agreement():
    """Identical labels are kappa 1.0."""
    result = agree([True, False, True, False], [True, False, True, False])
    assert result.observed == 1.0
    assert result.kappa == pytest.approx(1.0)
    assert result.disagreements == 0


def test_total_disagreement_is_negative():
    """Systematically opposite labels are worse than chance."""
    result = agree([True, True, False, False], [False, False, True, True])
    assert result.observed == 0.0
    assert result.kappa < 0


def test_high_raw_agreement_can_be_worthless():
    """The trap: both instruments almost always say "no".

    18/20 raw agreement looks strong and means nothing here -- the two never
    agree on a positive, which is the only case anyone cares about.
    """
    a = [True] + [False] * 19
    b = [False] * 19 + [True]
    result = agree(a, b)
    assert result.observed == pytest.approx(0.9)
    assert result.kappa < 0.0, "kappa must not reward agreement on the majority class"


def test_constant_instruments_are_not_perfect_agreement():
    """If neither instrument ever fires, kappa is undefined, not 1.0."""
    result = agree([False] * 10, [False] * 10)
    assert result.observed == 1.0
    assert result.kappa == 0.0


def test_only_a_and_only_b_are_distinguished():
    """Which instrument over-fires is the actionable part of a disagreement."""
    result = agree([True, True, False], [True, False, True])
    assert (result.both, result.only_a, result.only_b, result.neither) == (1, 1, 1, 0)


def test_mismatched_lengths_are_rejected():
    """Silent misalignment would corrupt every item after the gap."""
    with pytest.raises(ValueError, match="differ in length"):
        agree([True, False], [True])


def test_summary_names_both_instruments():
    """A results table needs to say which side over-fired."""
    text = agree([True, False], [False, False]).summary("patterns", "judge")
    assert "patterns" in text and "judge" in text and "kappa" in text


def test_threshold_sweep_recovers_a_separable_cutoff():
    """A score that cleanly separates the reference must be found."""
    scores = [0.0, 10.0, 20.0, 80.0, 90.0, 100.0]
    labels = [False, False, False, True, True, True]
    threshold, result = best_threshold(scores, labels, step=5.0)
    assert threshold is not None
    assert 20.0 < threshold <= 80.0
    assert result.kappa == pytest.approx(1.0)


def test_threshold_sweep_on_an_unrelated_score():
    """When the score means something else, kappa must stay low."""
    scores = [50.0, 10.0, 90.0, 30.0, 70.0, 20.0]
    labels = [True, True, False, False, True, False]
    _, result = best_threshold(scores, labels, step=10.0)
    assert result.kappa < 0.7


def test_threshold_sweep_handles_empty_input():
    """No data must return no threshold rather than raise."""
    threshold, result = best_threshold([], [])
    assert threshold is None
    assert result.total == 0


def test_confusion_by_class_cross_tabulates():
    """The three-way judge against a binary reference."""
    labels = ["full_refusal", "full_refusal", "partial_refusal", "full_compliance"]
    reference = [True, False, True, False]
    table = confusion_by_class(labels, reference)
    assert table["full_refusal"] == (1, 1)
    assert table["partial_refusal"] == (1, 0)
    assert table["full_compliance"] == (0, 1)
