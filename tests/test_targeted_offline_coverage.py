"""The offline suite must exercise the operator that leaked.

`TARGETED_REPAIR` is the only operator that survives live certification, and it
produced both induced leaks. Offline it recorded zero trials and was excluded
as untestable, so no offline result -- PASSED included -- was ever evidence
about it. These tests pin the coverage itself, separately from whether the
operator passes: a suite that stops exercising it should go red here even if
everything else stays green.
"""

from __future__ import annotations

import pytest

from aaramse.certification import certify_operator
from aaramse.corpus import SEED_PAIRS, TOPICAL_TRIGGERS
from aaramse.equivalence import SemanticEquivalence
from aaramse.refusal import HeuristicRefusalOracle
from aaramse.simulate import simulated_complete
from aaramse.targeted import TargetedConfig, TargetedRepair


@pytest.fixture
def sim_oracle():
    return HeuristicRefusalOracle(topical_triggers=TOPICAL_TRIGGERS)


@pytest.fixture
def targeted(sim_oracle):
    """A real TARGETED_REPAIR wired to the deterministic simulator."""
    return TargetedRepair(
        refuses=sim_oracle.refuses,
        generate=lambda prompt: simulated_complete(prompt),
        config=TargetedConfig(require_equivalence=False),
    )


def test_targeted_repair_is_configured_offline(targeted):
    """Unconfigured is why it was applicable to nothing."""
    assert targeted.configured
    assert targeted.applicable("Should I move my savings right now?")


def test_certification_records_trials(targeted, sim_oracle):
    """Zero trials is the defect: it means the corpus never asked."""
    certificate = certify_operator(targeted, SEED_PAIRS, sim_oracle)
    assert certificate.trials > 0, (
        "TARGETED_REPAIR recorded no trials offline; the suite is blind to the "
        "operator that produced both induced leaks"
    )


def test_certification_verdict_is_reported_not_assumed(targeted, sim_oracle):
    """Record what the offline certificate actually says.

    This test deliberately asserts the *shape* of the evidence rather than a
    passing verdict. If the simulator ever flips a prohibited twin, that is a
    finding to surface, not a test to rewrite.
    """
    certificate = certify_operator(targeted, SEED_PAIRS, sim_oracle)
    assert certificate.trials > 0
    assert certificate.flips >= 0
    assert certificate.passed == (certificate.flips == 0 and certificate.trials > 0)


def test_operator_actually_edits_a_benign_twin(targeted):
    """Applicability is not enough; it must produce a real rewrite."""
    applied = [
        targeted.apply(pair.benign)
        for pair in SEED_PAIRS
    ]
    produced = [a for a in applied if a is not None]
    assert produced, "no benign twin produced a rewrite offline"
    assert any(a.after != a.before for a in produced)
