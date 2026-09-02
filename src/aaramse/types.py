"""Immutable data structures for the AARAMSE rewrite stage.

Every structure here is frozen: once the gateway has emitted an audit record,
nothing downstream can mutate the evidence a supervisor would rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from .localize import Localization

__all__ = [
    "ActionabilityProfile",
    "Decision",
    "OperatorApplication",
    "OperatorKind",
    "RepairResult",
    "RewriteProgram",
    "SearchDiagnostics",
]


class OperatorKind(str, Enum):
    """How an operator acts on the pragmatic layer of a query.

    SUBTRACTIVE operators strip pragmatic surface features (person marking,
    imperative mood, named entities, urgency). ADDITIVE operators supply
    deployer-side regulatory context without touching the question itself.
    Neither class may add specificity or actionability.
    """

    SUBTRACTIVE = "subtractive"
    ADDITIVE = "additive"


class Decision(str, Enum):
    """Terminal state of a repair attempt."""

    PASSTHROUGH = "passthrough"  # the model never refused; nothing was rewritten
    REPAIRED = "repaired"  # a certified program cleared the refusal
    ESCALATED = "escalated"  # search exhausted; the refusal is content-driven and upheld
    BLOCKED = "blocked"  # an invariant violation was caught; candidate discarded


@dataclass(frozen=True)
class ActionabilityProfile:
    """Position of a query on the actionability lattice.

    Attributes:
        score: Sum of matched feature weights. Higher means more actionable.
        features: Matched feature names paired with their weights.
    """

    score: float
    features: Tuple[Tuple[str, float], ...]

    def as_dict(self) -> Dict[str, float]:
        """Return the matched features as a plain mapping."""
        return dict(self.features)


@dataclass(frozen=True)
class OperatorApplication:
    """One step of a rewrite program, with everything the audit log needs.

    Attributes:
        operator: Registered operator name.
        before: Text the operator received.
        after: Text the operator produced.
        generalizations: Specific term -> general term substitutions performed.
        dropped: Clauses removed from the query, verbatim.
        localization: The delta-debugging result, when the operator localized a
            refusal trigger before editing. This is the explainability record a
            supervisor reads: which fragment was changed, and at what cost.
    """

    operator: str
    before: str
    after: str
    generalizations: Tuple[Tuple[str, str], ...] = ()
    dropped: Tuple[str, ...] = ()
    localization: Optional["Localization"] = None


@dataclass(frozen=True)
class RewriteProgram:
    """A composition of operator applications.

    The program, not the output string, is the auditable artifact: it is
    symbolic, replayable, and diffable across queries.
    """

    steps: Tuple[OperatorApplication, ...] = ()

    @property
    def names(self) -> Tuple[str, ...]:
        """Return the operator names in application order."""
        return tuple(step.operator for step in self.steps)

    @property
    def length(self) -> int:
        """Return the program length, i.e. the refusal margin."""
        return len(self.steps)

    def render(self) -> str:
        """Render the program in composition notation for human review."""
        if not self.steps:
            return "IDENTITY"
        return " o ".join(self.names)

    def generalizations(self) -> Tuple[Tuple[str, str], ...]:
        """Return every generalization declared across all steps."""
        return tuple(g for step in self.steps for g in step.generalizations)

    def dropped(self) -> Tuple[str, ...]:
        """Return every clause dropped across all steps."""
        return tuple(d for step in self.steps for d in step.dropped)


@dataclass(frozen=True)
class SearchDiagnostics:
    """Why a search failed, recorded so an escalation is forensically legible.

    Set on escalation results only. The three counts discriminate the failure
    classes a triage needs to tell apart:

    - ``candidates_generated == 0`` -- no operator produced an applicable
      candidate; the search never had anything to test (localization/operator
      coverage gap).
    - ``blocked_candidates`` non-empty with ``probed_refused == 0`` -- candidates
      were generated but every one was rejected by an invariant before it ever
      reached the model (the guard is the binding constraint).
    - ``probed_refused > 0`` -- candidates cleared the guard and were put to the
      model, which still refused (the replacements were too weak to flip it).

    Attributes:
        candidates_generated: Distinct rewrite candidates enumerated.
        probed_refused: Candidates that passed the guard but the model refused.
        blocked_candidates: ``"<program>: <invariant reason>"`` for each candidate
            an invariant rejected before it reached the model.
    """

    candidates_generated: int
    probed_refused: int
    blocked_candidates: Tuple[str, ...] = ()

    @property
    def failure_class(self) -> str:
        """Label the dominant failure the counts imply.

        Authoritative definition of the triage classes, so no downstream
        analysis re-derives them inconsistently. Budget exhaustion is not
        distinguishable from these counts alone -- it is read from the result's
        ``reason`` -- so a caller that needs that case must check the reason
        first and treat this label as the fallback.

        Returns:
            ``"no_candidate"`` -- nothing applicable was generated.
            ``"guard_blocked"`` -- candidates were generated but only invariant
                rejections occurred; none reached the model.
            ``"model_upheld"`` -- at least one candidate reached the model and
                the refusal held.
        """
        if self.candidates_generated == 0:
            return "no_candidate"
        if self.probed_refused == 0 and self.blocked_candidates:
            return "guard_blocked"
        return "model_upheld"


@dataclass(frozen=True)
class RepairResult:
    """Outcome of a bounded repair search over one query."""

    query: str
    rewritten: str
    program: RewriteProgram
    decision: Decision
    refusal_margin: int
    actionability_before: ActionabilityProfile
    actionability_after: ActionabilityProfile
    oracle_calls: int
    search_space: int
    reason: str = ""
    diagnostics: Optional[SearchDiagnostics] = None

    @property
    def was_rewritten(self) -> bool:
        """Return True when the query reaching the model differs from the original."""
        return self.decision is Decision.REPAIRED
