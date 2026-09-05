"""Fold selection in the corpus loaders.

The harm gate's patterns were authored against the certification fold. Any
measurement that reports a leak or gate rate must therefore draw from the
evaluation fold, and it must draw the fold *before* applying a limit -- slicing
in file order and then calling the result held out is the specific mistake that
made the first live e2e run read stronger than it was.
"""

from __future__ import annotations

import pytest

from aaramse.falsereject import load_benign, load_toxic
from aaramse.splits import assert_disjoint


def test_folds_partition_the_corpus():
    """Every item lands in exactly one fold, and nothing is invented or lost."""
    for load in (load_benign, load_toxic):
        every = load()
        cert = load(fold="certification")
        held = load(fold="evaluation")
        assert len(cert) + len(held) == len(every)
        assert set(cert) | set(held) == set(every)
        assert_disjoint(
            [item.prompt for item in cert],
            [item.prompt for item in held],
            label=load.__name__,
        )


def test_limit_applies_after_the_fold():
    """A limited draw stays inside the fold it asked for.

    Applying the limit first would slice file order and then filter, which is
    how a 12-item "held out" bucket ended up 8/12 inside the authoring fold.
    """
    held = set(load_toxic(fold="evaluation"))
    drawn = load_toxic(limit=12, fold="evaluation")
    assert len(drawn) == 12
    assert set(drawn) <= held


def test_default_fold_is_unchanged_file_order():
    """`fold="all"` is the historical behaviour, so existing callers are safe."""
    assert load_toxic(limit=5) == load_toxic(limit=5, fold="all")
    assert [item.prompt for item in load_toxic(limit=5)] == [
        item.prompt for item in load_toxic()
    ][:5]


def test_folds_are_stable_across_calls():
    """Splitting is by content hash, so two calls agree without a stored seed."""
    assert load_toxic(fold="evaluation") == load_toxic(fold="evaluation")


def test_a_different_salt_produces_a_different_fold():
    """The salt is the documented knob for a deliberately different split."""
    default = load_toxic(fold="evaluation")
    other = load_toxic(fold="evaluation", salt="deliberately-different")
    assert default != other


def test_unknown_fold_is_rejected():
    """A typo must fail loudly rather than silently measuring the wrong set."""
    with pytest.raises(ValueError, match="fold must be"):
        load_toxic(fold="heldout")
