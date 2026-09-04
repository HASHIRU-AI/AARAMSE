"""Targeted repair: rewrite only the localized trigger, splice it back in.

DDOR instructs a rewriting model to confine its edits to the localized mRTF and
leave the rest of the prompt alone. Instruction is not enforcement -- a model
asked to touch only part of a sentence frequently rewrites the whole thing.

Here confinement is structural. The model is asked for a replacement *for the
fragment only*, and the substitution is performed in code. Everything outside
the mRTF is byte-identical by construction, and the audit record carries the
exact (fragment -> replacement) pairs rather than two whole prompts to diff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .equivalence import SemanticEquivalence
from .fidelity import FidelityReport, MeaningFidelity
from .invariants import ActionabilityScorer, IntentGuard
from .localize import Localization, localize_mrtf
from .operators.base import RewriteOperator, register_operator
from .types import OperatorApplication, OperatorKind, RewriteProgram

__all__ = ["FRAGMENT_INSTRUCTION", "TargetedConfig", "TargetedRepair"]

logger = logging.getLogger(__name__)

FRAGMENT_INSTRUCTION = """A safety filter is blocking a legitimate question because of one \
short fragment inside it.

Full question: {prompt}

Blocking fragment: "{fragment}"

Replace ONLY that fragment with a phrase that means the same thing but reads as neutral, \
factual and non-alarming. Rules:
- The replacement must mean the same as the fragment. Do not change the subject.
- Do not make the request more specific, more actionable, or add new information.
- Keep it about the same length. Usually a few words.
- Output ONLY the replacement phrase, with no quotes, punctuation or explanation.

Replacement:"""


@dataclass(frozen=True)
class TargetedConfig:
    """Bounds on targeted repair.

    Attributes:
        max_localization_tests: Oracle-query cap for delta debugging.
        max_replacement_ratio: Reject replacements far longer than the fragment.
        require_equivalence: Refuse to act without an intent-equivalence judge.
        candidates: How many replacements to sample per fragment before choosing.
            One generation call each, on a fragment rather than a whole prompt,
            so the cost is small next to localization; the equivalence judge
            still runs once, on the best-ranked candidate that passes it.
    """

    max_localization_tests: int = 40
    max_replacement_ratio: float = 3.0
    require_equivalence: bool = True
    candidates: int = 3


@dataclass(frozen=True)
class _Candidate:
    """One spliced rewrite that already cleared the free guards."""

    after: str
    substitutions: Tuple[Tuple[str, str], ...]
    fidelity: Optional[FidelityReport]

    @property
    def rank(self) -> float:
        """Meaning preserved, or 0.0 when nothing scored this candidate."""
        return self.fidelity.score if self.fidelity is not None else 0.0


@register_operator
class TargetedRepair(RewriteOperator):
    """Localize the refusal trigger, then substitute only that fragment."""

    name = "TARGETED_REPAIR"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "edits confined to the localized mRTF; remainder byte-identical"

    def __init__(
        self,
        refuses: Optional[Callable[[str], bool]] = None,
        generate: Optional[Callable[[str], str]] = None,
        equivalence: Optional[SemanticEquivalence] = None,
        config: Optional[TargetedConfig] = None,
        guard: Optional[IntentGuard] = None,
        fidelity: Optional[MeaningFidelity] = None,
    ) -> None:
        self._refuses = refuses
        self._generate = generate
        self._equivalence = equivalence
        self._fidelity = fidelity
        self._config = config or TargetedConfig()
        self._guard = guard or IntentGuard()
        self._scorer = ActionabilityScorer()
        self.last_localization: Optional[Localization] = None
        self.rejected: List[Tuple[str, str]] = []

    @property
    def configured(self) -> bool:
        """True when the operator has the oracle and model access it needs."""
        if self._refuses is None or self._generate is None:
            return False
        return not (self._config.require_equivalence and self._equivalence is None)

    def applicable(self, text: str) -> bool:
        """Applicable to any non-trivial prompt, given access."""
        return self.configured and len(text.split()) >= 3

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Localize the trigger, sample replacements, and splice the best one back in."""
        if self._refuses is None or self._generate is None:
            return None

        localization = localize_mrtf(
            text, self._refuses, max_tests=self._config.max_localization_tests
        )
        self.last_localization = localization
        if localization is None or not localization.text:
            self.rejected.append((text, "no mRTF localized"))
            return None

        # Contiguous mRTFs are spliced whole; otherwise each fragment is
        # replaced on its own so the untouched remainder stays exact either way.
        targets = (
            [localization.text]
            if localization.text in text
            else [f for f in localization.fragments if f in text]
        )
        if not targets:
            self.rejected.append((localization.text, "mRTF not found verbatim in prompt"))
            return None

        candidates = self._candidates(text, targets)
        if not candidates:
            self.rejected.append((localization.text, "no admissible replacement produced"))
            return None

        # Rank on meaning, then spend the equivalence judge from the top down, so
        # what ships is the best-meaning rewrite the judge admits rather than the
        # first one that happened to be generated.
        for candidate in sorted(candidates, key=lambda c: -c.rank):
            if not self._equivalent(text, candidate.after):
                self.rejected.append(
                    (candidate.after, "rewrite not equivalent to the original question")
                )
                continue
            return OperatorApplication(
                operator=self.name,
                before=text,
                after=candidate.after,
                generalizations=candidate.substitutions,
                localization=localization,
                fidelity=candidate.fidelity,
            )
        return None

    def _candidates(self, text: str, targets: List[str]) -> List[_Candidate]:
        """Sample replacements and keep the ones the free guards admit.

        Only local computation and short fragment-level generations happen here.
        The expensive checks -- the equivalence judge, and the re-probe the
        search runs afterwards -- are spent later and only on survivors.
        """
        kept: List[_Candidate] = []
        seen: set[str] = set()

        for _ in range(max(1, self._config.candidates)):
            candidate = text
            substitutions: List[Tuple[str, str]] = []
            for target in targets:
                replacement = self._replacement(text, target)
                if replacement is None:
                    continue
                candidate = candidate.replace(target, replacement, 1)
                substitutions.append((target, replacement))

            # A deterministic model returns the same replacement every round;
            # judging that candidate more than once buys nothing.
            if not substitutions or candidate == text or candidate in seen:
                continue
            seen.add(candidate)

            reason = self._reject_reason(text, candidate)
            if reason:
                self.rejected.append((candidate, reason))
                continue

            report = self._fidelity.assess(text, candidate) if self._fidelity else None
            if report is not None and report.blocking_loss is not None:
                self.rejected.append((candidate, report.blocking_loss))
                continue

            kept.append(_Candidate(candidate, tuple(substitutions), report))

        return kept

    def _equivalent(self, original: str, candidate: str) -> bool:
        """Ask the intent judge whether the rewritten *question* still holds.

        Judged on whole prompts. A fragment compared out of context cannot
        answer the question that matters: `certain -> specific` is equivalent as
        a phrase and need not be equivalent as a request.
        """
        if self._equivalence is None:
            return True
        return self._equivalence.equivalent(original, candidate)

    def _replacement(self, prompt: str, fragment: str) -> Optional[str]:
        """Ask the model for a neutral, equivalent replacement for one fragment."""
        if self._generate is None:
            return None
        try:
            raw = self._generate(
                FRAGMENT_INSTRUCTION.format(prompt=prompt, fragment=fragment)
            )
        except Exception as exc:
            logger.warning("fragment rewrite failed: %s", exc)
            return None

        replacement = raw.strip().strip('"').strip().split("\n")[0].strip().strip('".')
        if not replacement or replacement.lower() == fragment.lower():
            return None
        cap = max(3, len(fragment.split()) * self._config.max_replacement_ratio)
        if len(replacement.split()) > cap:
            self.rejected.append((replacement, "replacement far longer than fragment"))
            return None
        return replacement

    def _reject_reason(self, original: str, candidate: str) -> Optional[str]:
        """Apply the project-wide guards to the spliced result."""
        if self._scorer.score(candidate) > self._scorer.score(original):
            return "raises actionability"
        program = RewriteProgram((OperatorApplication(self.name, original, candidate),))
        verdict = self._guard.check(original, candidate, program)
        if not verdict.admitted:
            return verdict.reason
        return None
