"""The generative rewriter must stay inside the guards, not around them."""

from __future__ import annotations

from typing import List

import pytest

from aaramse.invariants import ActionabilityScorer
from aaramse.rewriter import LLMRephrase, RewriterConfig, SemanticEquivalence

QUERY = "Should I dump my 401(k) into gold before the crash?"


def make(replies: List[str], equivalent: bool = True, **cfg):
    """Build a rewriter whose generator returns canned candidates in order."""
    seen = {"n": 0}

    def gen(_prompt: str, _temp: float) -> str:
        reply = replies[min(seen["n"], len(replies) - 1)]
        seen["n"] += 1
        return reply

    equiv = SemanticEquivalence(generate=lambda _: "YES" if equivalent else "NO")
    return LLMRephrase(
        generate=gen, equivalence=equiv, config=RewriterConfig(candidates=len(replies), **cfg)
    )


def test_unconfigured_rewriter_is_inapplicable():
    """Without model access the operator must sit out, not crash the search."""
    op = LLMRephrase()
    assert not op.configured
    assert not op.applicable(QUERY)
    assert op.apply(QUERY) is None


def test_accepts_a_clean_generalising_rewrite():
    """The intended happy path."""
    good = "What factors determine whether a retirement account is reallocated into commodities?"
    op = make([good])
    result = op.apply(QUERY)
    assert result is not None
    assert result.after == good
    assert ActionabilityScorer().score(result.after) < ActionabilityScorer().score(QUERY)


def test_strips_model_preamble_and_quotes():
    """Models wrap answers; the operator must extract the bare question."""
    op = make(['Rewritten question: "What determines commodity reallocation in retirement plans?"'])
    result = op.apply(QUERY)
    assert result is not None
    assert result.after.startswith("What determines")
    assert '"' not in result.after


def test_rejects_a_candidate_that_raises_actionability():
    """A rewrite may never move down the lattice."""
    op = make(["Tell me right now whether I should move my 401(k) into gold immediately!"])
    assert op.apply(QUERY) is None
    assert any("actionability" in reason for _, reason in op.rejected)


def test_rejects_a_materially_longer_candidate():
    """Length inflation is how a generative step smuggles in new content."""
    bloated = (
        "What factors determine whether an investor should consider reallocating a retirement "
        "account into commodities, and what are the tax implications, the custody arrangements, "
        "the historical returns, the storage costs and the counterparty risks involved in that?"
    )
    op = make([bloated], max_length_ratio=1.5)
    assert op.apply(QUERY) is None
    assert any("longer" in reason for _, reason in op.rejected)


def test_rejects_a_candidate_with_no_lexical_anchor():
    """A total reword is more likely a different question than a rephrasing."""
    op = make(["What is the weather forecast for tomorrow?"], min_similarity=0.5)
    assert op.apply(QUERY) is None
    assert op.rejected


def test_rejects_a_candidate_the_judge_calls_non_equivalent():
    """Intent equivalence is checked by judgement, not lexical overlap.

    The candidate here is lexically almost identical to the original, so every
    surface check passes; only the judgement can catch that it asks something
    different. This is the case cosine similarity could never separate.
    """
    op = make(
        ["What factors determine whether an investor dumps a 401(k) into gold after a crash?"],
        equivalent=False,
    )
    assert op.apply(QUERY) is None
    assert any("intent-equivalent" in reason for _, reason in op.rejected)


def test_generative_rewrite_discloses_dropped_domain_terms():
    """A generative rewrite cannot declare substitutions, so it must disclose them."""
    op = make(["What factors determine whether a retirement account holds commodities?"],
              min_similarity=0.1)
    result = op.apply(QUERY)
    assert result is not None
    dropped = {term for term, _ in result.generalizations}
    assert {"401k", "gold"} <= dropped


def test_generative_rewrite_requires_a_judge():
    """Without an equivalence judge the operator must refuse to act."""
    op = LLMRephrase(
        generate=lambda _p, _t: "What determines commodity allocation in retirement accounts?",
        config=RewriterConfig(candidates=1),
    )
    assert op.apply(QUERY) is None
    assert any("no equivalence judge" in reason for _, reason in op.rejected)


def test_picks_the_minimal_edit_among_admissible_candidates():
    """Smaller edits are easier to audit, so the closest passing rewrite wins."""
    near = "What factors determine whether an investor dumps a 401(k) into gold before a crash?"
    far = "What determines commodity allocation decisions?"
    op = make([far, near])
    result = op.apply(QUERY)
    assert result is not None
    assert result.after == near


def test_generation_failure_is_survivable():
    """A model outage must degrade to 'no rewrite', never propagate."""

    def boom(_prompt: str, _temp: float) -> str:
        raise RuntimeError("connection reset")

    op = LLMRephrase(generate=boom, config=RewriterConfig(candidates=2))
    assert op.apply(QUERY) is None


def test_equivalence_judge_fails_closed_on_unparseable_output():
    """An ambiguous verdict must not be read as approval."""
    checker = SemanticEquivalence(generate=lambda _: "hmm, hard to say")
    assert not checker.equivalent("a", "b")
    assert SemanticEquivalence(generate=lambda _: "YES").equivalent("a", "b")


def test_equivalence_verdicts_are_cached():
    """Judging is a model call; it must not repeat for the same pair."""
    calls = {"n": 0}

    def gen(_: str) -> str:
        calls["n"] += 1
        return "YES"

    checker = SemanticEquivalence(generate=gen)
    checker.equivalent("a", "b")
    checker.equivalent("a", "b")
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "raw,expected_end",
    [
        ("What is a wash sale?", "?"),
        ("What is a wash sale", "?"),
        ("  What is a wash sale?  \n\nExtra commentary here.", "?"),
    ],
)
def test_cleaning_always_yields_a_single_question(raw, expected_end):
    """Downstream audit assumes one question, not a paragraph."""
    cleaned = LLMRephrase._clean(raw)
    assert cleaned.endswith(expected_end)
    assert "\n" not in cleaned
