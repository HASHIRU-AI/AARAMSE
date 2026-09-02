"""The third obligation: do nothing to a query nobody objected to.

FalseReject measures repair and OR-Bench measures leakage. Neither catches a
layer that quietly rewrites queries the model was always going to answer. That
is a false intervention: it costs latency, it puts words in the user's mouth,
and it breaks the passthrough guarantee the whole design rests on.
"""

from __future__ import annotations

import pytest

from aaramse.budget import LeakageBudget
from aaramse.finqa import ControlPrompt, load_control, source_note
from aaramse.types import Decision

CONTROL = load_control()


def test_the_control_slice_is_present_and_plausible():
    """A missing or truncated corpus would make every test below vacuous."""
    assert len(CONTROL) == 120
    assert all(isinstance(item, ControlPrompt) for item in CONTROL)
    assert all(item.prompt.strip() for item in CONTROL)
    assert "FinQA" in source_note()


def test_control_prompts_are_self_contained():
    """A bare FinQA question is unanswerable, and "I lack the data" reads as refusal."""
    for item in CONTROL:
        assert "Question:" in item.prompt
        assert "|" in item.prompt, "the table must travel with the question"


def test_the_boundary_does_not_refuse_the_control_set(oracle):
    """If the simulator refused these, the simulator would be miscalibrated."""
    refused = [item for item in CONTROL if oracle.refuses(item.prompt)]
    assert refused == [], f"{len(refused)} benign analysis prompts were refused"


def test_the_layer_never_intervenes_on_the_control_set(search):
    """Every control prompt must come back byte-identical.

    This is the passthrough guarantee stated as a test over a real corpus
    rather than a single hand-written example.
    """
    interventions = []
    for item in CONTROL:
        result = search.repair(item.prompt)
        if result.decision is not Decision.PASSTHROUGH or result.rewritten != item.prompt:
            interventions.append((item.source_id, result.decision.value))
    assert interventions == [], f"false interventions: {interventions[:5]}"


def test_passthrough_costs_one_probe(search):
    """A query the model answers must not pay for a search it never needed."""
    result = search.repair(CONTROL[0].prompt)
    assert result.decision is Decision.PASSTHROUGH
    assert result.oracle_calls == 1
    assert result.refusal_margin == 0


def test_prohibited_prompts_are_escalated_not_rewritten(search, pairs):
    """The invariant the concept note attributed to a classifier it never had.

    Nothing here classifies intent. The guarantee comes from the boundary
    itself: the content trigger fires whatever the phrasing, no certified
    program clears it, and the search escalates rather than forcing it through.
    """
    for pair in pairs:
        result = search.repair(pair.prohibited_twin)
        assert result.decision is Decision.ESCALATED, pair.prohibited_twin
        assert result.rewritten == pair.prohibited_twin, "an escalation must not edit the query"


def test_a_zero_budget_is_satisfied_on_the_simulated_boundary(search, pairs):
    """The default budget has to be achievable, or it is theatre."""
    from aaramse.budget import measure_leakage

    verdict = measure_leakage(
        search.repair, [p.prohibited_twin for p in pairs], LeakageBudget()
    )
    assert verdict.induced_leaks == 0
    assert verdict.within_budget


@pytest.mark.parametrize("limit", [1, 5, 50])
def test_limit_returns_a_prefix(limit):
    """Evaluations subsample; the subsample must be deterministic."""
    assert load_control(limit=limit) == CONTROL[:limit]


def test_a_missing_corpus_says_how_to_rebuild_it(tmp_path):
    """The failure has to name the script, not just the path."""
    with pytest.raises(FileNotFoundError, match="build_finqa_control"):
        load_control(path=tmp_path / "absent.json")
