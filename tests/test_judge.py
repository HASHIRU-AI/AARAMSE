"""The three-way judge is the measurement instrument; it must be exercised."""

from __future__ import annotations

import pytest
from aaramse.falsereject import load_benign, load_toxic
from aaramse.judge import JUDGE_PROMPT, ResponseClass, ThreeWayJudge


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1_full_compliance", ResponseClass.FULL_COMPLIANCE),
        ("2_full_refusal", ResponseClass.FULL_REFUSAL),
        ("3_partial_refusal", ResponseClass.PARTIAL_REFUSAL),
        ("CLASS: 3_partial_refusal\n", ResponseClass.PARTIAL_REFUSAL),
        ("The answer is full_refusal.", ResponseClass.FULL_REFUSAL),
        ("**1_full_compliance**", ResponseClass.FULL_COMPLIANCE),
    ],
)
def test_judge_parses_labels(raw, expected):
    """Label extraction must tolerate the formatting models actually emit."""
    assert ThreeWayJudge(generate=lambda _: raw).classify("q", "r").label is expected


def test_unparseable_output_fails_to_partial():
    """An ambiguous judgement must not be silently scored as clean compliance."""
    judge = ThreeWayJudge(generate=lambda _: "I'm not sure what you mean")
    assert judge.classify("q", "r").label is ResponseClass.PARTIAL_REFUSAL


def test_partial_refusal_counts_as_answered_but_not_over_refusal():
    """The distinction the binary detector could not make."""
    assert ResponseClass.PARTIAL_REFUSAL.answered
    assert not ResponseClass.PARTIAL_REFUSAL.is_over_refusal
    assert ResponseClass.FULL_REFUSAL.is_over_refusal
    assert not ResponseClass.FULL_REFUSAL.answered
    assert ResponseClass.FULL_COMPLIANCE.answered


def test_judge_prompt_states_all_three_classes():
    """Comparability with published results depends on the taxonomy wording."""
    for token in ("1_full_compliance", "2_full_refusal", "3_partial_refusal"):
        assert token in JUDGE_PROMPT


def test_judgements_are_cached():
    """The judge must not re-query for a response it has already classified."""
    calls = {"n": 0}

    def gen(_: str) -> str:
        calls["n"] += 1
        return "2_full_refusal"

    judge = ThreeWayJudge(generate=gen)
    judge.classify("q", "r")
    judge.classify("q", "r")
    assert calls["n"] == 1


def test_finance_corpus_is_vendored():
    """Evaluation must be reproducible without network access."""
    benign, toxic = load_benign(), load_toxic()
    assert len(benign) >= 90
    assert len(toxic) >= 50
    assert all(p.benign for p in benign)
    assert not any(p.benign for p in toxic)
    assert all(p.prompt.strip() for p in benign + toxic)


def test_corpus_limit_is_respected():
    """Sampling must be deterministic and bounded."""
    assert len(load_benign(limit=7)) == 7
    assert load_benign(limit=7) == load_benign(limit=7)
