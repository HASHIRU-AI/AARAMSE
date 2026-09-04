"""Targeted repair must edit the trigger and nothing else."""

from __future__ import annotations

from aaramse.equivalence import SemanticEquivalence
from aaramse.fidelity import MeaningFidelity
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


# --- meaning fidelity: whole-prompt equivalence and best-of-k ranking ---


def cycling(*replacements: str):
    """Generator double returning each replacement once, then repeating the last."""
    remaining = list(replacements)

    def generate(_p: str) -> str:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return generate


def test_equivalence_is_judged_on_the_whole_prompt():
    """Fragment-level equivalence cannot see the question it is embedded in.

    `certain -> specific` passes any fragment-level check; whether the *question*
    still means the same thing is only answerable with the prompt in hand.
    """
    seen: list[str] = []

    def judge(prompt: str) -> str:
        seen.append(prompt)
        return "YES"

    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=lambda _p: "protect assets lawfully",
        equivalence=SemanticEquivalence(generate=judge),
    )
    op.apply(PROMPT)
    assert seen, "equivalence judge was never consulted"
    assert "university dissertation" in seen[0]


def _fidelity_judge(poor: str):
    """Score any candidate containing `poor` down on both judged dimensions.

    Ranking has to be exercised on the *judged* dimensions. A candidate that
    scores low deterministically -- by dropping a domain term -- never reaches
    ranking, because IntentGuard rejects it first (see the drift test below).
    """

    def generate(prompt: str) -> str:
        if poor in prompt:
            return "ANSWER_TYPE: NO\nANSWERABLE: NO"
        return "ANSWER_TYPE: YES\nANSWERABLE: YES"

    return MeaningFidelity(generate=generate)


def test_best_of_k_prefers_the_higher_fidelity_candidate():
    """Given two admissible replacements, the one that keeps more meaning wins.

    Both candidates clear every guard, so the only thing that can separate them
    is the fidelity ranking. Drop the ranking and the first one drawn ships.
    """
    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=cycling("obscure assets", "protect assets lawfully"),
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        fidelity=_fidelity_judge("obscure assets"),
        config=TargetedConfig(candidates=2),
    )
    result = op.apply(PROMPT)
    assert result is not None
    assert "protect assets lawfully" in result.after
    # The loser must have been a live option, not one a guard had already killed.
    assert not any("obscure assets" in text for text, _ in op.rejected)


def test_domain_term_substitution_is_declared_not_drift():
    """The operator's self-check now sees the substitutions it just made.

    `_reject_reason` used to hand IntentGuard a program with no generalizations
    attached, so `_declared_losses` was empty and swapping any domain term read
    as undeclared topic drift -- while the identical check at the search level,
    which does see them, admitted the same candidate. The operator's own guard
    was strictly stricter than the one the search would apply.
    """
    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=lambda _p: "conceal holdings",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        fidelity=MeaningFidelity(),
    )
    result = op.apply(PROMPT)

    assert result is not None, "a declared domain-term swap must be admissible"
    assert result.generalizations == (("hide assets", "conceal holdings"),)
    assert not any("undeclared loss" in reason for _, reason in op.rejected)


def test_declared_subject_loss_is_still_priced():
    """Declaring a loss makes it legal, not invisible.

    IntentGuard admits the swap above; `MeaningFidelity` still records the lost
    subject term, which lowers the score and so makes best-of-k prefer a rewrite
    that keeps it. Blocking and pricing are different jobs and stay separate.
    """
    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=lambda _p: "conceal holdings",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        fidelity=MeaningFidelity(),
    )
    result = op.apply(PROMPT)

    assert result is not None
    assert result.fidelity is not None
    assert "assets" in result.fidelity.lost_terms
    assert result.fidelity.score < 1.0


def test_edit_lands_on_the_localized_occurrence_not_the_first_match():
    """Confinement is a claim about *which* characters changed.

    The mRTF here is the non-contiguous ("hide", "pension?") inside the second
    sentence, and "hide" also occurs in the benign first one. Splicing by string
    match rewrote the first sentence -- byte-identity held, and the layer still
    edited text delta debugging had cleared.
    """
    prompt = "A friend told me to hide things. Can I hide money in a pension?"
    op = TargetedRepair(
        refuses=lambda t: "hide" in t.lower() and "pension" in t.lower(),
        generate=lambda _p: "shelter",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        fidelity=MeaningFidelity(),
    )
    result = op.apply(prompt)

    assert result is not None
    assert result.after.startswith("A friend told me to hide things."), (
        f"the untouched sentence was edited: {result.after!r}"
    )


def test_replacement_dropping_a_quantity_is_rejected():
    """A dropped figure changes the answer, so it never reaches the model.

    The figure has to sit *inside* the localized fragment to be at risk at all:
    confinement already guarantees everything outside it survives byte-identical.
    """
    prompt = "How do I hide $50,000 from a trustee?"
    op = TargetedRepair(
        refuses=lambda t: "hide" in t.lower() and "50,000" in t,
        generate=lambda _p: "protect holdings",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        fidelity=MeaningFidelity(),
    )
    assert op.apply(prompt) is None
    assert any("quantities" in reason for _, reason in op.rejected)


def test_replacement_dropping_a_negation_is_rejected():
    """Losing 'without' inverts the question it was scoping."""
    prompt = "How do I move assets without notifying a trustee?"
    op = TargetedRepair(
        refuses=lambda t: "without notifying" in t.lower(),
        generate=lambda _p: "while notifying",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        fidelity=MeaningFidelity(),
    )
    assert op.apply(prompt) is None
    assert any("negation" in reason for _, reason in op.rejected)


def test_fidelity_report_rides_along_for_the_audit_log():
    """A supervisor reads which dimension the repair cost, not a bare score."""
    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=lambda _p: "protect assets lawfully",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        fidelity=MeaningFidelity(),
    )
    result = op.apply(PROMPT)
    assert result is not None
    assert result.fidelity is not None
    assert result.fidelity.score == 1.0


def test_best_of_k_asks_a_distinguishable_question_each_sample():
    """k identical requests return one cached string k times.

    Every deployed client memoises on (prompt, temperature, max_tokens) and the
    gateway asks at temperature 0, so sending the same fragment prompt k times
    drew one candidate and best-of-k silently collapsed to k=1 against every
    real model. Only the fake generators in this file, which ignore the prompt,
    ever saw more than one. The samples must therefore differ as *prompts*.
    """
    asked: list[str] = []

    def generate(prompt: str) -> str:
        asked.append(prompt)
        return f"protect assets lawfully {len(asked)}"

    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=generate,
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        config=TargetedConfig(candidates=3),
    )
    op.apply(PROMPT)

    assert len(asked) == 3
    assert len(set(asked)) == 3, "samples are indistinguishable to a caching client"
    # Sample 0 must be the unmodified instruction, so a k=1 run is unchanged.
    assert sum(1 for prompt in asked if "Prefer" not in prompt) == 1


def test_candidates_are_capped_at_the_available_nudges():
    """Asking for more samples than the nudge set can vary sends duplicates."""
    asked: list[str] = []
    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=lambda p: asked.append(p) or "protect assets lawfully",
        equivalence=SemanticEquivalence(generate=lambda _: "YES"),
        config=TargetedConfig(candidates=9),
    )
    op.apply(PROMPT)

    assert len(asked) == 3, "capped to 1 + len(sample_nudges)"


def test_identical_candidates_are_generated_once_each_but_judged_once():
    """A deterministic model returns k copies; only one needs judging."""
    judged: list[str] = []
    op = TargetedRepair(
        refuses=lambda t: "hide assets" in t.lower(),
        generate=lambda _p: "protect assets lawfully",
        equivalence=SemanticEquivalence(generate=lambda p: judged.append(p) or "YES"),
        fidelity=MeaningFidelity(),
        config=TargetedConfig(candidates=3),
    )
    assert op.apply(PROMPT) is not None
    assert len(judged) == 1
