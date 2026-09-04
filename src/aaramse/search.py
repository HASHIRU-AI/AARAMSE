"""Bounded shortest-program search: repair cost as the regulatory signal.

Breadth-first search over operator programs returns the *shortest* program that
clears the refusal. Its length is the refusal margin, a calibrated scalar that
separates the two cases a supervisor cares about:

* margin 0            -- no over-refusal occurred
* margin small (1-2)  -- pragmatic over-refusal, repaired and logged
* search exhausted    -- the refusal survived full generalization, so it was
                         content-driven and correct; the query escalates

Escalation is the point. The layer never "tries harder": depth k and the closed
operator set cap its total optimization power at |O|^k enumerable candidates.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Protocol, Sequence, Set, Tuple

from .fidelity import AnswerCheck
from .invariants import ActionabilityScorer, IntentGuard
from .operators.base import RewriteOperator
from .refusal import CountingOracle, RefusalOracle, ResponseProbe
from .types import Decision, RepairResult, RewriteProgram, SearchDiagnostics

__all__ = ["Realizer", "RepairSearch", "SearchConfig", "search_space_size"]

logger = logging.getLogger(__name__)


class _NullOracle:
    """Placeholder for the oracle slot when a probe supersedes it."""

    def refuses(self, prompt: str) -> bool:
        """Never reached: the probe path returns before consulting the oracle."""
        raise RuntimeError("probe path must not consult the oracle")


def search_space_size(n_operators: int, max_depth: int) -> int:
    """Return the number of candidate programs the layer may ever consider.

    Operators are idempotent and never repeat within a program, so the bound is
    the number of ordered selections without replacement.
    """
    total = 0
    running = 1
    for depth in range(1, max_depth + 1):
        if depth > n_operators:
            break
        running *= n_operators - depth + 1
        total += running
    return total


@dataclass(frozen=True)
class SearchConfig:
    """Bounds on the repair search.

    Attributes:
        max_depth: Maximum program length k.
        max_oracle_calls: Hard cap on distinct boundary probes per query.
        require_certificates: Refuse to search with uncertified operators.
        abort_on_content_delivery: Escalate the whole query the moment a candidate
            elicits step-by-step assistance. Continuing to search past that point is
            the jailbreak loop this design exists to avoid.
    """

    max_depth: int = 3
    max_oracle_calls: int = 64
    require_certificates: bool = True
    abort_on_content_delivery: bool = True


class Realizer(Protocol):
    """Optional surface-form smoother applied after a program succeeds."""

    def realize(self, text: str) -> str:
        """Return a more fluent rendering of the same question."""
        ...


@dataclass
class RepairSearch:
    """Searches for the shortest admissible repair program for a query.

    Attributes:
        operators: The closed operator algebra to search over.
        oracle: Refusal boundary to probe. Optional when `probe` is supplied,
            which supersedes it; exactly one of the two must be present.
        guard: Invariant checker applied to every candidate.
        config: Search bounds.
        realizer: Optional smoother; its output is re-checked by the guard and
            discarded if it fails, so it can never widen the safety envelope.
        probe: Optional content-aware probe. When supplied it replaces the bare
            refusal check, so acceptance depends on what the model actually handed
            over rather than on whether refusal boilerplate disappeared.
        answer_check: Optional check that the reply to a rewrite still answers the
            question the user asked. Without it, a rewrite that clears the
            boundary by asking something easier scores as a repair. Requires a
            probe, which is what supplies the reply text.
    """

    operators: Sequence[RewriteOperator]
    oracle: Optional[RefusalOracle] = None
    guard: IntentGuard = field(default_factory=IntentGuard)
    config: SearchConfig = field(default_factory=SearchConfig)
    realizer: Optional[Realizer] = None
    scorer: ActionabilityScorer = field(default_factory=ActionabilityScorer)
    probe: Optional[ResponseProbe] = None
    answer_check: Optional[AnswerCheck] = None

    def __post_init__(self) -> None:
        """Reject a search that has no way to observe the refusal boundary."""
        if self.oracle is None and self.probe is None:
            raise ValueError("RepairSearch needs either an oracle or a probe")

    def repair(self, query: str) -> RepairResult:
        """Find the shortest program clearing the refusal, or escalate.

        Args:
            query: The incoming user query, verbatim.

        Returns:
            A RepairResult recording the decision, the program, the refusal
            margin, and the oracle budget consumed.
        """
        counter = CountingOracle(self.oracle) if self.oracle is not None else CountingOracle(
            _NullOracle()
        )
        before = self.scorer.profile(query)
        space = search_space_size(len(self.operators), self.config.max_depth)

        if self.probe is not None:
            baseline = self.probe.probe(query)
            counter.calls += 1
            if not baseline.refused:
                return self._result(
                    query, query, RewriteProgram(), Decision.PASSTHROUGH, before, before,
                    counter.calls, space, "no refusal observed",
                )
        elif not counter.refuses(query):
            return self._result(
                query, query, RewriteProgram(), Decision.PASSTHROUGH, before, before,
                counter.calls, space, "no refusal observed",
            )

        frontier: Deque[Tuple[str, RewriteProgram]] = deque([(query, RewriteProgram())])
        seen: Set[str] = {query}
        blocked: List[str] = []
        probed_refused = 0
        answered_other = 0

        def diagnostics() -> SearchDiagnostics:
            """Snapshot the failure counters for an escalation result."""
            return SearchDiagnostics(
                candidates_generated=len(seen) - 1,
                probed_refused=probed_refused,
                blocked_candidates=tuple(blocked),
                answered_other=answered_other,
            )

        while frontier:
            text, program = frontier.popleft()
            if program.length >= self.config.max_depth:
                continue

            for operator in self.operators:
                if operator.name in program.names:
                    continue  # operators are idempotent; repeating cannot help
                if not operator.applicable(text):
                    continue

                application = operator.apply(text)
                if application is None:
                    continue

                candidate = application.after
                if candidate in seen:
                    continue
                seen.add(candidate)

                extended = RewriteProgram((*program.steps, application))
                verdict = self.guard.check(query, candidate, extended)
                if not verdict.admitted:
                    blocked.append(f"{extended.render()}: {verdict.reason}")
                    logger.debug("blocked candidate %s (%s)", extended.render(), verdict.reason)
                    continue

                if counter.calls >= self.config.max_oracle_calls:
                    return self._escalate(
                        query, before, counter, space, "oracle budget exhausted",
                        diagnostics(),
                    )

                if self.probe is not None:
                    response = self.probe.probe(candidate)
                    counter.calls += 1
                    if response.refused:
                        probed_refused += 1
                        frontier.append((candidate, extended))
                        continue
                    if response.delivers_procedure and self.config.abort_on_content_delivery:
                        # The rewrite elicited the assistance the refusal was
                        # withholding. Searching on would be a jailbreak loop.
                        return self._escalate(
                            query, before, counter, space,
                            f"candidate {extended.render()} elicited step-by-step "
                            "assistance; repair abandoned",
                            diagnostics(),
                        )
                    # Clearing the refusal is not the goal; answering the user's
                    # question is. A rewrite that got a reply to some easier
                    # question passes every other check here.
                    if self.answer_check is not None and not self.answer_check.answers(
                        query, response.text
                    ):
                        answered_other += 1
                        logger.debug(
                            "candidate %s answered a different question", extended.render()
                        )
                        continue
                    final = self._realize(query, candidate, extended)
                    return self._result(
                        query, final, extended, Decision.REPAIRED, before,
                        self.scorer.profile(final), counter.calls, space,
                        f"cleared at depth {extended.length}",
                    )

                if not counter.refuses(candidate):
                    final = self._realize(query, candidate, extended)
                    return self._result(
                        query, final, extended, Decision.REPAIRED, before,
                        self.scorer.profile(final), counter.calls, space,
                        f"cleared at depth {extended.length}",
                    )

                probed_refused += 1
                frontier.append((candidate, extended))

        reason = "search exhausted; refusal is content-driven"
        if blocked:
            reason += f"; {len(blocked)} candidate(s) blocked by invariants"
        if answered_other:
            reason += f"; {answered_other} candidate(s) answered a different question"
        return self._escalate(query, before, counter, space, reason, diagnostics())

    def _realize(self, query: str, candidate: str, program: RewriteProgram) -> str:
        """Apply the optional realizer, keeping its output only if it still passes."""
        if self.realizer is None:
            return candidate
        smoothed = self.realizer.realize(candidate)
        if smoothed == candidate:
            return candidate
        if not self.guard.check(query, smoothed, program).admitted:
            logger.warning("realizer output rejected by guard; keeping deterministic rewrite")
            return candidate
        if self._still_refused(smoothed):
            logger.warning("realizer output re-triggered refusal; keeping deterministic rewrite")
            return candidate
        return smoothed

    def _still_refused(self, text: str) -> bool:
        """Re-check a candidate against whichever observer this search has."""
        if self.probe is not None:
            return self.probe.probe(text).refused
        if self.oracle is not None:
            return self.oracle.refuses(text)
        raise RuntimeError("no oracle or probe configured")

    def _escalate(
        self, query: str, before, counter: CountingOracle, space: int, reason: str,
        diagnostics: Optional[SearchDiagnostics] = None,
    ) -> RepairResult:
        """Build the escalation result for an unrepairable refusal."""
        return self._result(
            query, query, RewriteProgram(), Decision.ESCALATED, before, before,
            counter.calls, space, reason, diagnostics,
        )

    @staticmethod
    def _result(
        query: str, rewritten: str, program: RewriteProgram, decision: Decision,
        before, after, calls: int, space: int, reason: str,
        diagnostics: Optional[SearchDiagnostics] = None,
    ) -> RepairResult:
        """Assemble an immutable repair result."""
        return RepairResult(
            query=query,
            rewritten=rewritten,
            program=program,
            decision=decision,
            refusal_margin=program.length,
            actionability_before=before,
            actionability_after=after,
            oracle_calls=calls,
            search_space=space,
            reason=reason,
            diagnostics=diagnostics,
        )
