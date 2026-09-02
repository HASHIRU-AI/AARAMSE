"""A certificate measured on its own corpus is not evidence."""

from __future__ import annotations

import pytest

from aaramse.splits import assert_disjoint, split_digest, split_items

PROMPTS = tuple(f"prompt number {i}" for i in range(200))


def _split(holdout=0.5, salt="aaramse-v1", items=PROMPTS):
    """Split plain strings."""
    return split_items(items, key=str, holdout=holdout, salt=salt)


def test_folds_partition_the_corpus():
    """Nothing may be dropped or duplicated."""
    split = _split()
    assert len(split.certification) + len(split.evaluation) == len(PROMPTS)
    assert set(split.certification) | set(split.evaluation) == set(PROMPTS)


def test_folds_are_disjoint():
    """The whole point: an item cannot certify and then score itself."""
    split = _split()
    assert_disjoint(split.certification, split.evaluation)


def test_the_split_is_reproducible_without_a_stored_seed():
    """Two runs, possibly on two machines, must agree."""
    assert _split().evaluation == _split().evaluation


def test_input_order_does_not_change_assignment():
    """Loading the corpus differently must not move items between folds."""
    forward = set(_split().evaluation)
    backward = set(_split(items=tuple(reversed(PROMPTS))).evaluation)
    assert forward == backward


def test_growing_the_corpus_does_not_reshuffle_it():
    """Adding prompts must not silently move an old one into certification.

    This is the failure that makes a longitudinal result meaningless: rerun
    after adding data and the "held-out" set now contains items the operators
    were certified on last time.
    """
    before = set(_split().evaluation)
    grown = PROMPTS + tuple(f"new prompt {i}" for i in range(50))
    after = set(_split(items=grown).evaluation)
    assert before <= after


def test_holdout_fraction_is_approximately_honoured():
    """A 50% holdout that returns 5% would quietly weaken every measurement."""
    split = _split(holdout=0.5)
    assert 0.4 <= len(split.evaluation) / len(PROMPTS) <= 0.6


def test_extreme_holdouts_are_total():
    """0 and 1 must mean nothing and everything, not almost."""
    assert _split(holdout=0.0).evaluation == ()
    assert _split(holdout=1.0).certification == ()


def test_invalid_holdout_is_rejected():
    """A typo'd fraction must fail, not silently clamp."""
    with pytest.raises(ValueError):
        _split(holdout=1.5)
    with pytest.raises(ValueError):
        _split(holdout=-0.1)


def test_duplicate_text_lands_in_one_fold():
    """A near-duplicate straddling the boundary is a silent leak."""
    items = ("same text", "same text", "other")
    split = split_items(items, key=str)
    assert not (set(split.certification) & set(split.evaluation))


def test_overlap_is_reported_with_a_usable_message():
    """The error has to say which corpus and show an example."""
    with pytest.raises(ValueError, match="circular"):
        assert_disjoint(["a", "b"], ["b", "c"], label="prohibited")


def test_a_different_salt_gives_a_different_split():
    """Deliberately re-splitting must be possible, and recorded."""
    assert set(_split(salt="v1").evaluation) != set(_split(salt="v2").evaluation)
    assert _split(salt="v2").salt == "v2"


def test_digest_identifies_the_evaluation_fold():
    """Two runs must be able to prove they scored the same held-out set."""
    assert split_digest(_split(), key=str) == split_digest(_split(), key=str)
    assert split_digest(_split(salt="v1"), key=str) != split_digest(_split(salt="v2"), key=str)


def test_key_selects_the_text_that_matters():
    """Splitting objects, not strings, is the real use."""
    items = [{"prompt": f"q{i}", "category": "x"} for i in range(100)]
    split = split_items(items, key=lambda d: d["prompt"])
    assert_disjoint(
        [d["prompt"] for d in split.certification],
        [d["prompt"] for d in split.evaluation],
    )
