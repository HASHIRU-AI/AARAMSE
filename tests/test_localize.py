"""Delta-debugging localization: the step this project originally skipped."""

from __future__ import annotations

import pytest

from aaramse.localize import Localization, localize_mrtf, split_sentences, split_words

PROMPT = (
    "I am researching UK insolvency law for a university dissertation. "
    "How do I hide assets from a trustee? "
    "Please cite the relevant statutes."
)


def trigger_on(needle: str):
    """Build an oracle that refuses exactly when `needle` survives reduction."""
    return lambda text: needle in text.lower()


def test_localizes_the_minimal_trigger():
    """Reduction must land on the trigger, not the surrounding benign text."""
    loc = localize_mrtf(PROMPT, trigger_on("hide assets"))
    assert loc is not None
    assert loc.text == "hide assets"
    assert loc.granularity == "word"
    assert loc.reduction_ratio > 0.5


def test_is_one_minimal():
    """Removing any unit of the mRTF must stop the refusal."""
    oracle = trigger_on("hide assets")
    loc = localize_mrtf(PROMPT, oracle)
    assert loc is not None
    for i in range(len(loc.fragments)):
        without = " ".join(loc.fragments[:i] + loc.fragments[i + 1:])
        assert not oracle(without), f"removing {loc.fragments[i]!r} still refused"


def test_sentence_granularity_when_trigger_spans_a_sentence():
    """A trigger needing a whole sentence must not be over-reduced."""
    loc = localize_mrtf(PROMPT, trigger_on("dissertation"))
    assert loc is not None
    assert "dissertation" in loc.text.lower()


def test_spans_anchor_each_fragment_where_it_actually_sat():
    """The mRTF has to carry *where* it was, not only what it said.

    A word-granularity mRTF is a subsequence of common tokens, and the same
    token usually occurs elsewhere in the prompt. Matching by string picks the
    first occurrence, which is not in general the one that was localized.
    """
    prompt = "A friend told me to hide things. Can I hide money in a pension?"
    loc = localize_mrtf(prompt, lambda t: "hide" in t.lower() and "pension" in t.lower())

    assert loc is not None
    assert loc.spans, "localization carries no anchors"
    assert [prompt[start:end] for start, end in loc.spans] == list(loc.fragments)
    assert list(loc.spans) == sorted(loc.spans), "spans must be ascending"
    # Every anchor sits in the sentence that was actually refused, not in the
    # benign first sentence that happens to contain the same word.
    assert min(start for start, _ in loc.spans) >= prompt.index("Can I")


def test_returns_none_when_prompt_is_not_refused():
    """Nothing to localize means no localization, not a spurious fragment."""
    assert localize_mrtf(PROMPT, lambda _t: False) is None


def test_budget_is_respected():
    """Localization must stay affordable; a starved budget yields None."""
    calls = {"n": 0}

    def oracle(text: str) -> bool:
        calls["n"] += 1
        return "hide assets" in text.lower()

    localize_mrtf(PROMPT, oracle, max_tests=3)
    assert calls["n"] <= 3


def test_repeated_subsequences_are_cached():
    """The same candidate must never be queried twice."""
    seen = []

    def oracle(text: str) -> bool:
        seen.append(text)
        return "hide assets" in text.lower()

    localize_mrtf(PROMPT, oracle)
    assert len(seen) == len(set(seen))


def test_localization_is_deterministic():
    """A stable trigger per prompt keeps audit records comparable."""
    a = localize_mrtf(PROMPT, trigger_on("hide assets"))
    b = localize_mrtf(PROMPT, trigger_on("hide assets"))
    assert a == b


@pytest.mark.parametrize(
    "text,expected",
    [
        ("One. Two! Three?", 3),
        ("Single sentence with no terminator", 1),
        ("A. B.", 2),
    ],
)
def test_sentence_splitting(text, expected):
    """Fragment granularity drives the whole reduction."""
    assert len(split_sentences(text)) == expected


def test_word_splitting_drops_empty_tokens():
    """Whitespace noise must not become phantom fragments."""
    assert split_words("  hide   assets  ") == ("hide", "assets")


def test_reduction_ratio_handles_no_reduction():
    """A single-fragment prompt must not divide by zero."""
    loc = Localization(fragments=("a",), text="a", granularity="word", tests=1, reduced_from=0)
    assert loc.reduction_ratio == 0.0
