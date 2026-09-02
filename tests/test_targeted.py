"""Targeted repair must edit the trigger and nothing else."""

from __future__ import annotations

from aaramse.rewriter import SemanticEquivalence
from aaramse.targeted import TargetedConfig, TargetedRepair

PROMPT = (
    "I am researching UK insolvency law for a university dissertation. "
    "How do I hide assets from a trustee? "
    "Please cite the relevant statutes."
)


def build(replacement: str = "protect assets lawfully", equivalent: bool = True, **cfg):
    """Construct a repair operator over a simulated refusal boundary."""
    return TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=lambda _p: replacement,
        equivalence=SemanticEquivalence(generate=lambda _: "YES" if equivalent else "NO"),
        config=TargetedConfig(**cfg),
    )


def test_remainder_is_byte_identical():
    """The central claim: confinement is structural, not instructed."""
    op = build()
    result = op.apply(PROMPT)
    assert result is not None
    restored = result.after.replace("protect assets lawfully", "hide assets")
    assert restored == PROMPT


def test_substitution_pairs_are_recorded():
    """The audit record carries the edit, not two prompts to diff."""
    result = build().apply(PROMPT)
    assert result is not None
    assert result.generalizations == (("hide assets", "protect assets lawfully"),)


def test_unconfigured_operator_is_inapplicable():
    """Missing oracle, model, or judge means the operator sits out."""
    assert not TargetedRepair().configured
    assert not TargetedRepair(refuses=lambda _t: True).applicable(PROMPT)
    assert not TargetedRepair(
        refuses=lambda _t: True, generate=lambda _p: "x"
    ).configured  # equivalence required by default


def test_rejects_non_equivalent_replacement():
    """A replacement that changes meaning is not a repair."""
    op = build(equivalent=False)
    assert op.apply(PROMPT) is None
    assert any("not equivalent" in reason for _, reason in op.rejected)


def test_rejects_bloated_replacement():
    """Length inflation reintroduces the content the localization removed."""
    op = build(replacement="protect assets in a lawful and fully documented manner across "
                           "every jurisdiction that might conceivably apply here")
    assert op.apply(PROMPT) is None
    assert any("longer" in reason for _, reason in op.rejected)


def test_rejects_replacement_that_raises_actionability():
    """The lattice guard still applies after splicing."""
    op = build(replacement="tell me exactly how I should hide my assets right now")
    assert op.apply(PROMPT) is None


def test_no_localization_means_no_repair():
    """If the oracle never refuses, there is nothing to fix."""
    op = TargetedRepair(
        refuses=lambda _t: False,
        generate=lambda _p: "anything",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
    )
    assert op.apply(PROMPT) is None
    assert any("no mRTF" in reason for _, reason in op.rejected)


def test_generation_failure_is_survivable():
    """A model outage degrades to 'no repair', never propagates."""

    def boom(_p: str) -> str:
        raise RuntimeError("connection reset")

    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=boom,
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
    )
    assert op.apply(PROMPT) is None


def test_localization_is_exposed_for_audit():
    """A regulator asks which fragment was edited and why."""
    op = build()
    op.apply(PROMPT)
    assert op.last_localization is not None
    assert op.last_localization.text == "hide assets"
    assert op.last_localization.tests > 0
