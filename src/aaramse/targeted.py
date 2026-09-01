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
    """

    max_localization_tests: int = 40
    max_replacement_ratio: float = 3.0
    require_equivalence: bool = True


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
    ) -> None:
        self._refuses = refuses
        self._generate = generate
        self._equivalence = equivalence
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
        """Localize the trigger, replace it, and splice the result back in."""
        if self._refuses is None or self._generate is None:
            return None

        localization = localize_mrtf(
            text, self._refuses, max_tests=self._config.max_localization_tests
        )
        self.last_localization = localization
        if localization is None or not localization.text:
            self.rejected.append((text, "no mRTF localized"))
            return None

        candidate = text
        substitutions: List[Tuple[str, str]] = []

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

        for target in targets:
            replacement = self._replacement(text, target)
            if replacement is None:
                continue
            candidate = candidate.replace(target, replacement, 1)
            substitutions.append((target, replacement))

        if not substitutions or candidate == text:
            self.rejected.append((localization.text, "no admissible replacement produced"))
            return None

        reason = self._reject_reason(text, candidate)
        if reason:
            self.rejected.append((candidate, reason))
            return None

        return OperatorApplication(
            operator=self.name,
            before=text,
            after=candidate,
            generalizations=tuple(substitutions),
            localization=localization,
        )

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
        equiv = self._equivalence
        if equiv is not None and not equiv.equivalent(fragment, replacement):
            self.rejected.append((replacement, "fragment replacement not equivalent"))
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
