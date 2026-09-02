"""Delta-debugging localization of the minimal Refusal-Triggering Fragment.

This is the step the project skipped. Following Zhou et al. (DDOR), a refused
prompt is reduced to a 1-minimal fragment whose presence alone still triggers
refusal, and repair then edits *only* that fragment. DDOR reports 51.96%
over-refusal reduction on OR-Bench and 86.41% on XSTest with this pipeline, and
measures full-prompt rewriting -- what this project built first -- as a baseline
that repairs slightly more but loses 7.01% semantic similarity by replacing
benign content along with the trigger.

Definition (1-minimal refusal-triggering fragment). For a prompt split into
fragments P = {f1..fn}, S is an mRTF when concat(S) still triggers refusal and
removing any single unit of S stops it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Localization", "localize_mrtf", "split_sentences", "split_words"]

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def split_sentences(text: str) -> Tuple[str, ...]:
    """Split a prompt into sentence-level fragments."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return tuple(parts) if parts else (text.strip(),)


def split_words(text: str) -> Tuple[str, ...]:
    """Split a fragment into whitespace-delimited word fragments."""
    return tuple(w for w in _WS_RE.split(text.strip()) if w)


@dataclass(frozen=True)
class Localization:
    """The localized trigger and the cost of finding it.

    Attributes:
        fragments: The mRTF, in original order.
        text: The mRTF rendered back to a string.
        granularity: "sentence" or "word", whichever the reduction reached.
        tests: Oracle queries consumed.
        reduced_from: Fragment count of the original prompt at that granularity.
    """

    fragments: Tuple[str, ...]
    text: str
    granularity: str
    tests: int
    reduced_from: int

    @property
    def reduction_ratio(self) -> float:
        """Fraction of the prompt eliminated during reduction."""
        if not self.reduced_from:
            return 0.0
        return 1.0 - (len(self.fragments) / self.reduced_from)


class _BudgetExceeded(Exception):
    """Raised internally when the oracle-query budget runs out."""


@dataclass
class _Tester:
    """Caches and counts refusal tests over fragment subsequences."""

    refuses: Callable[[str], bool]
    joiner: str
    budget: int
    tests: int = 0
    cache: Optional[Dict[str, bool]] = None

    def __post_init__(self) -> None:
        # Shared across reduction stages: the word stage re-tests texts the
        # sentence stage already resolved, and each test is a model call.
        if self.cache is None:
            self.cache = {}

    def __call__(self, units: Sequence[str]) -> bool:
        """Return True when this subsequence still triggers a refusal."""
        text = self.joiner.join(units).strip()
        if not text:
            return False
        assert self.cache is not None
        if text in self.cache:
            return self.cache[text]
        if self.tests >= self.budget:
            raise _BudgetExceeded
        self.tests += 1
        verdict = self.refuses(text)
        self.cache[text] = verdict
        return verdict


def _chunks(units: Sequence[str], n: int) -> List[Tuple[int, int]]:
    """Partition indices into n contiguous blocks."""
    size = max(1, len(units) // n)
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < len(units):
        end = min(start + size, len(units))
        spans.append((start, end))
        start = end
    return spans


def _ddmin(units: Sequence[str], tester: _Tester) -> Tuple[str, ...]:
    """Reduce a failing sequence to a 1-minimal one by complement testing.

    At each round the sequence is partitioned into n blocks; removing a block
    and still seeing a refusal means the removed part was irrelevant, so the
    remainder becomes the new candidate and the partition is coarsened. If no
    single removal preserves the refusal, the partition is refined instead.
    """
    current = list(units)
    n = 2
    while len(current) >= 2:
        reduced = False
        for start, end in _chunks(current, n):
            complement = current[:start] + current[end:]
            if not complement:
                continue
            if tester(complement):
                current = complement
                n = max(n - 1, 2)
                reduced = True
                break
        if not reduced:
            if n >= len(current):
                break
            n = min(n * 2, len(current))
    return tuple(current)


def localize_mrtf(
    prompt: str,
    refuses: Callable[[str], bool],
    max_tests: int = 60,
    word_stage: bool = True,
) -> Optional[Localization]:
    """Localize the minimal refusal-triggering fragment of a refused prompt.

    Args:
        prompt: A prompt the target model refuses.
        refuses: Oracle returning True when a text still triggers refusal.
        max_tests: Hard cap on oracle queries, so localization stays affordable.
        word_stage: Refine to word granularity once reduction reaches one sentence.

    Returns:
        The localization, or None when the prompt cannot be reduced (for example
        the oracle does not actually refuse it, or the budget ran out first).
    """
    sentences = split_sentences(prompt)
    sentence_tester = _Tester(refuses=refuses, joiner=" ", budget=max_tests)

    try:
        if not sentence_tester(sentences):
            logger.debug("prompt is not refused; nothing to localize")
            return None
        reduced = _ddmin(sentences, sentence_tester)
    except _BudgetExceeded:
        logger.warning("localization budget exhausted at sentence stage")
        return None

    granularity = "sentence"
    fragments = reduced
    reduced_from = len(sentences)
    tests = sentence_tester.tests

    if word_stage and len(reduced) == 1:
        words = split_words(reduced[0])
        if len(words) > 2:
            word_tester = _Tester(
                refuses=refuses,
                joiner=" ",
                budget=max(0, max_tests - tests),
                cache=sentence_tester.cache,
            )
            try:
                if word_tester(words):
                    fragments = _ddmin(words, word_tester)
                    granularity = "word"
                    reduced_from = len(words)
            except _BudgetExceeded:
                logger.info("budget exhausted at word stage; keeping sentence-level mRTF")
            tests += word_tester.tests

    return Localization(
        fragments=fragments,
        text=" ".join(fragments).strip(),
        granularity=granularity,
        tests=tests,
        reduced_from=reduced_from,
    )
