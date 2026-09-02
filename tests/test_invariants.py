"""The two invariants that separate this rewriter from a jailbreaker."""

from __future__ import annotations

import pytest

from aaramse.corpus import BENIGN_OVERREFUSALS, SEED_PAIRS
from aaramse.invariants import ActionabilityScorer, strip_frame, topic_core
from aaramse.types import OperatorApplication, RewriteProgram

ALL_QUERIES = list(BENIGN_OVERREFUSALS) + [p.benign for p in SEED_PAIRS] + [
    p.prohibited_twin for p in SEED_PAIRS
]


@pytest.mark.parametrize("query", ALL_QUERIES)
def test_no_operator_increases_actionability(operators, query):
    """Monotone generalization: no single operator may move a query downward."""
    scorer = ActionabilityScorer()
    before = scorer.score(query)
    for operator in operators:
        if not operator.applicable(query):
            continue
        application = operator.apply(query)
        if application is None:
            continue
        after = scorer.score(application.after)
        assert after <= before, (
            f"{operator.name} raised actionability {before} -> {after} on {query!r}"
        )


@pytest.mark.parametrize("query", ALL_QUERIES)
def test_operator_compositions_stay_monotone(operators, query):
    """Monotonicity must hold under composition, not just per step."""
    scorer = ActionabilityScorer()
    for first in operators:
        if not first.applicable(query):
            continue
        step_one = first.apply(query)
        if step_one is None:
            continue
        for second in operators:
            if second.name == first.name or not second.applicable(step_one.after):
                continue
            step_two = second.apply(step_one.after)
            if step_two is None:
                continue
            assert scorer.score(step_two.after) <= scorer.score(query)


def test_guard_rejects_actionability_increase(guard):
    """A candidate that adds personal directive framing must be blocked."""
    query = "What is the mechanics of a wash sale?"
    candidate = "Should I use a wash sale on my portfolio right now?"
    program = RewriteProgram((OperatorApplication("FAKE", query, candidate),))
    verdict = guard.check(query, candidate, program)
    assert not verdict.admitted
    assert "monotonicity" in verdict.reason


def test_guard_rejects_undeclared_topic_drift(guard):
    """Losing a topic term without declaring it is intent drift, not repair."""
    query = "How does bankruptcy protection interact with my pension?"
    candidate = "How does insurance underwriting work?"
    program = RewriteProgram((OperatorApplication("FAKE", query, candidate),))
    verdict = guard.check(query, candidate, program)
    assert not verdict.admitted
    assert "topic drift" in verdict.reason


def test_guard_allows_declared_generalization(guard):
    """A declared specific-to-general substitution is admissible."""
    query = "Should I move my Vanguard holdings into gold?"
    candidate = (
        "What factors determine whether an investor moves "
        "mutual fund holdings into commodity?"
    )
    program = RewriteProgram(
        (
            OperatorApplication(
                "ENTITY_GENERALIZE",
                query,
                candidate,
                generalizations=(("Vanguard", "mutual fund"), ("gold", "commodity")),
            ),
        )
    )
    assert guard.check(query, candidate, program).admitted


def test_frame_does_not_deflate_the_score(operators, frame_config):
    """Scoring runs on the payload, so an additive frame cannot mask features."""
    scorer = ActionabilityScorer()
    query = "Should I sell my Tesla shares today?"
    frame_op = next(op for op in operators if op.name == "FRAME_ASSERT")
    framed = frame_op.apply(query).after
    assert scorer.score(framed) == scorer.score(query)
    assert strip_frame(framed) == query


def test_topic_core_normalises_retirement_accounts():
    """401(k) must tokenise to the same term regardless of punctuation."""
    assert "401k" in topic_core("What is a 401(k) rollover?")
    assert "401k" in topic_core("What is a 401k rollover?")
