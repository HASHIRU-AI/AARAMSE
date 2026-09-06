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
from itertools import pairwise
from typing import Callable, List, Optional, Sequence, Tuple

from .equivalence import SemanticEquivalence
from .fidelity import FidelityReport, MeaningFidelity
from .invariants import ActionabilityScorer, IntentGuard
from .localize import Localization, localize_mrtf
from .operators.base import RewriteOperator, register_operator
from .refusal import _REFUSAL_RE, normalise_quotes
from .types import OperatorApplication, OperatorKind, RewriteProgram

__all__ = [
    "FRAGMENT_INSTRUCTION",
    "NEUTRAL_FRAGMENT_INSTRUCTION",
    "SAMPLE_NUDGES",
    "TargetedConfig",
    "TargetedRepair",
]

logger = logging.getLogger(__name__)

FRAGMENT_INSTRUCTION = """A safety filter is blocking a legitimate question because of one \
short fragment inside it.

Full question: {prompt}

Blocking fragment: "{fragment}"

Replace ONLY that fragment with a phrase that means the same thing but reads as neutral, \
factual and non-alarming. Rules:
- The replacement must mean the same as the fragment. Do not change the subject.
- Do not make the request more specific, more actionable, or add new information.
- Keep it about the same length. Usually a few words.{nudge}
- Output ONLY the replacement phrase, with no quotes, punctuation or explanation.

Replacement:"""

# Fallback wording, used only when the rewriter declines the instruction above.
#
# FRAGMENT_INSTRUCTION opens by saying a safety filter is blocking the question,
# which is true and is what makes the task legible. Some models read it as a
# request to help evade a filter and refuse: muse-spark-1.2 answered "I can't
# help with circumventing safety systems", and produced "statutory definition"
# for the same fragment the moment the framing was dropped. So the obstacle was
# the instruction, not the model.
#
# The original wording is still asked first, so a model that complies with it is
# unaffected and every measurement taken on such a model stands. This is only
# reached where the current code produced no candidate at all.
NEUTRAL_FRAGMENT_INSTRUCTION = """Rewrite one short fragment of a question so the \
question reads as a request for general, factual information.

Full question: {prompt}

Fragment to rewrite: "{fragment}"

Rules:
- The replacement must mean the same as the fragment. Do not change the subject.
- Do not make the request more specific, more actionable, or add new information.
- Keep it about the same length. Usually a few words.{nudge}
- Output ONLY the replacement phrase, with no quotes, punctuation or explanation.

Replacement:"""


# Extra rule appended for the second and later samples of a fragment.
#
# Best-of-k needs k *different* candidates or it has nothing to rank, and asking
# the same question k times does not produce them: every deployed client
# memoises on (prompt, temperature, max_tokens) and the gateway asks at
# temperature 0, so k identical requests return one cached string k times and k
# silently collapses to 1. Only the unit tests, whose fake generators ignore the
# prompt, ever saw more than one candidate.
#
# Varying the *instruction* fixes that where raising the temperature would not:
# it changes the cache key, it changes the generation, and it keeps the run
# reproducible -- a property this project relies on everywhere else. Sample 0 is
# the unmodified instruction, so a k=1 run is byte-identical to the old
# behaviour and no published number moves.
SAMPLE_NUDGES: Tuple[str, ...] = (
    "\n- Prefer plain, everyday wording over technical vocabulary.",
    "\n- Prefer the precise technical or legal term for this, where one exists.",
)


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
            Capped at ``1 + len(sample_nudges)``, because sample i > 0 is drawn
            by appending nudge i-1 and there is nothing to draw beyond them.
        sample_nudges: Instruction variations that make samples 2..k differ from
            sample 1. See `SAMPLE_NUDGES` for why the variation lives here
            rather than in the sampling temperature.
    """

    max_localization_tests: int = 40
    max_replacement_ratio: float = 3.0
    require_equivalence: bool = True
    candidates: int = 3
    sample_nudges: Tuple[str, ...] = SAMPLE_NUDGES

    @property
    def effective_candidates(self) -> int:
        """How many distinct samples this configuration can actually draw."""
        return max(1, min(self.candidates, 1 + len(self.sample_nudges)))


def _splice(text: str, edits: Sequence[Tuple[Tuple[int, int], str]]) -> str:
    """Replace each character span with its replacement, in one forward pass.

    Spans come from the localizer, so they are ascending and non-overlapping and
    a single pass is exact. Every character outside them is copied verbatim,
    which is what makes confinement structural rather than instructed.

    Args:
        text: The prompt as received.
        edits: (span, replacement) pairs, in ascending span order.

    Returns:
        The spliced prompt.
    """
    out: List[str] = []
    cursor = 0
    for (start, end), replacement in edits:
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


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
        self.last_proposal: Optional[OperatorApplication] = None
        self.rejected: List[Tuple[str, str]] = []

    def reset(self) -> None:
        """Forget the previous turn's attempt.

        The operator outlives a turn but its record of what it tried must not:
        a console rendering the attempt would otherwise show a fragment
        localized for someone else's question, and a long-lived server would
        accumulate every rejection it ever made.
        """
        self.last_localization = None
        self.last_proposal = None
        self.rejected = []

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
        # Kept only when it found something. The search applies this operator
        # again at depth 2, against text a frame has already been prepended to,
        # where localization usually fails -- and overwriting a successful
        # result with that None makes the console report no fragment on a turn
        # where one was found and edited.
        if localization is not None and localization.text:
            self.last_localization = localization
        if localization is None or not localization.text:
            self.rejected.append((text, "no mRTF localized"))
            return None

        targets = self._targets(text, localization)
        if not targets:
            self.rejected.append((localization.text, "mRTF could not be anchored in the prompt"))
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
            application = OperatorApplication(
                operator=self.name,
                before=text,
                after=candidate.after,
                generalizations=candidate.substitutions,
                localization=localization,
                fidelity=candidate.fidelity,
            )
            # Kept whether or not the search goes on to use it. A rewrite the
            # model still refused is discarded by the search and would leave no
            # record at all, and the substitution is the thing a reader wants
            # to see -- it is the difference between this method and a prefix.
            self.last_proposal = application
            return application
        return None

    @staticmethod
    def _targets(text: str, localization: Localization) -> List[Tuple[int, int]]:
        """Choose the character spans to replace.

        Contiguous mRTFs are spliced whole -- asking the model to replace the
        phrase "hide assets" gets a phrase back, while asking it for "hide" and
        "assets" separately gets two words that need not compose. Fragments
        separated by real text are replaced individually, so the material
        between them stays byte-identical.

        Args:
            text: The prompt as received.
            localization: The delta-debugging result, with its anchors.

        Returns:
            Ascending, non-overlapping spans, or `[]` when the mRTF could not
            be anchored at all.
        """
        spans = localization.spans
        if not spans:
            return []
        if len(spans) > 1 and all(
            not text[left[1]:right[0]].strip() for left, right in pairwise(spans)
        ):
            return [(spans[0][0], spans[-1][1])]
        return list(spans)

    def _candidates(self, text: str, targets: List[Tuple[int, int]]) -> List[_Candidate]:
        """Sample replacements and keep the ones the free guards admit.

        Only local computation and short fragment-level generations happen here.
        The expensive checks -- the equivalence judge, and the re-probe the
        search runs afterwards -- are spent later and only on survivors.
        """
        kept: List[_Candidate] = []
        seen: set[str] = set()

        if self._config.candidates > self._config.effective_candidates:
            logger.info(
                "candidates=%d exceeds the %d distinct samples this nudge set can draw; "
                "capping", self._config.candidates, self._config.effective_candidates,
            )

        for sample in range(self._config.effective_candidates):
            edits: List[Tuple[Tuple[int, int], str]] = []
            substitutions: List[Tuple[str, str]] = []
            for span in targets:
                fragment = text[span[0]:span[1]]
                replacement = self._replacement(text, fragment, sample)
                if replacement is None:
                    continue
                edits.append((span, replacement))
                substitutions.append((fragment, replacement))
            candidate = _splice(text, edits)

            # A deterministic model returns the same replacement every round;
            # judging that candidate more than once buys nothing.
            if not substitutions or candidate == text or candidate in seen:
                continue
            seen.add(candidate)

            reason = self._reject_reason(text, candidate, tuple(substitutions))
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

    def _replacement(self, prompt: str, fragment: str, sample: int = 0) -> Optional[str]:
        """Ask the model for a neutral, equivalent replacement for one fragment.

        Args:
            prompt: The whole question, for context.
            fragment: The localized mRTF text to replace.
            sample: Which of the k draws this is. Sample 0 asks with the
                unmodified instruction; later samples append a nudge, which is
                what makes them distinguishable to a memoising client.
        """
        if self._generate is None:
            return None
        nudge = self._config.sample_nudges[sample - 1] if sample else ""
        for template in (FRAGMENT_INSTRUCTION, NEUTRAL_FRAGMENT_INSTRUCTION):
            try:
                raw = self._generate(
                    template.format(prompt=prompt, fragment=fragment, nudge=nudge)
                )
            except Exception as exc:
                logger.warning("fragment rewrite failed: %s", exc)
                return None

            # A rewriter that declined is not a rewrite that was too long. The
            # length guard was catching these and reporting them as such, which
            # tells a reader the proposal was oversized when none was made.
            if _REFUSAL_RE.search(normalise_quotes(raw)):
                self.rejected.append(
                    (raw.strip()[:120], "rewriter declined the instruction")
                )
                continue

            replacement = raw.strip().strip('"').strip().split("\n")[0].strip().strip('".')
            if not replacement or replacement.lower() == fragment.lower():
                # Recorded rather than dropped. This branch was silent, so a
                # rewriter that echoed the fragment back was indistinguishable
                # in the trace from one that was never asked.
                self.rejected.append((
                    raw.strip()[:120] or "(empty)",
                    "replacement empty or unchanged from the fragment",
                ))
                return None
            cap = max(3, len(fragment.split()) * self._config.max_replacement_ratio)
            if len(replacement.split()) > cap:
                self.rejected.append((replacement, "replacement far longer than fragment"))
                return None
            return replacement
        return None

    def _reject_reason(
        self,
        original: str,
        candidate: str,
        substitutions: Tuple[Tuple[str, str], ...] = (),
    ) -> Optional[str]:
        """Apply the project-wide guards to the spliced result.

        The substitutions are passed through to `IntentGuard` as the program's
        declared generalizations. Without them `_declared_losses` is empty, so
        every swap of a domain term reads as undeclared topic drift and the
        operator can never replace one -- while the identical check at the
        search level, which does see them, admits the same candidate. That
        asymmetry meant the operator's own guard was strictly stricter than the
        one the search would apply, for no stated reason.

        Declaring a loss makes it legal, not invisible: `MeaningFidelity`
        records it under `lost_terms`, which lowers the candidate's score and so
        makes best-of-k prefer a rewrite that keeps the subject word.

        Args:
            original: The prompt as received.
            candidate: The spliced rewrite.
            substitutions: The (fragment -> replacement) pairs just applied.

        Returns:
            Why the candidate is inadmissible, or None.
        """
        if self._scorer.score(candidate) > self._scorer.score(original):
            return "raises actionability"
        program = RewriteProgram((
            OperatorApplication(
                operator=self.name,
                before=original,
                after=candidate,
                generalizations=substitutions,
            ),
        ))
        verdict = self._guard.check(original, candidate, program)
        if not verdict.admitted:
            return verdict.reason
        return None
