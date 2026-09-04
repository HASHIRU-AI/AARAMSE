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
        spans: Half-open character ranges of each fragment in the original
            prompt, in order. This is what lets a repair edit the fragment that
            was actually localized rather than the first textual match of it:
            a word-granularity mRTF containing a common token ("before", "my",
            "to") occurs many times, and matching by string picks whichever
            comes first. Empty when the fragments could not be anchored, which
            a caller must treat as "no anchor" rather than as position zero.
    """

    fragments: Tuple[str, ...]
    text: str
    granularity: str
    tests: int
    reduced_from: int
    spans: Tuple[Tuple[int, int], ...] = ()

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


def _unit_spans(text: str, units: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    """Locate every unit in `text`, in order, as a half-open character span.

    Exact rather than heuristic, because `units` is the *complete* ordered
    decomposition of `text`: scanning forward from the previous match, a
    repeated word can only be found at its own occurrence, since every earlier
    occurrence was already consumed by an earlier unit.

    Args:
        text: The string the units were split from.
        units: That split, complete and in order.

    Returns:
        One span per unit, or `()` when any unit could not be located -- so an
        unanchorable localization is detectable instead of silently wrong.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for unit in units:
        start = text.find(unit, cursor)
        if start < 0:
            logger.debug("unit %r not found after offset %d; dropping anchors", unit, cursor)
            return ()
        spans.append((start, start + len(unit)))
        cursor = start + len(unit)
    return tuple(spans)


def _chunks(units: Sequence[object], n: int) -> List[Tuple[int, int]]:
    """Partition indices into n contiguous blocks.

    Takes the sequence only to read its length, so it is deliberately untyped
    in its element: reduction partitions index lists, not the units themselves.
    """
    size = max(1, len(units) // n)
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < len(units):
        end = min(start + size, len(units))
        spans.append((start, end))
        start = end
    return spans


def _ddmin(units: Sequence[str], tester: _Tester) -> Tuple[int, ...]:
    """Reduce a failing sequence to a 1-minimal one by complement testing.

    At each round the sequence is partitioned into n blocks; removing a block
    and still seeing a refusal means the removed part was irrelevant, so the
    remainder becomes the new candidate and the partition is coarsened. If no
    single removal preserves the refusal, the partition is refined instead.

    Reduction carries *indices* rather than the strings themselves, so the
    caller can recover where each surviving fragment sat in the original text.
    Reducing over the strings loses that, and it cannot be recovered afterwards
    by searching for them.

    Returns:
        Ascending indices into `units` forming the 1-minimal subsequence.
    """
    current = list(range(len(units)))
    n = 2
    while len(current) >= 2:
        reduced = False
        for start, end in _chunks(current, n):
            complement = current[:start] + current[end:]
            if not complement:
                continue
            if tester([units[index] for index in complement]):
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
    sentence_spans = _unit_spans(prompt, sentences)
    sentence_tester = _Tester(refuses=refuses, joiner=" ", budget=max_tests)

    try:
        if not sentence_tester(sentences):
            logger.debug("prompt is not refused; nothing to localize")
            return None
        kept = _ddmin(sentences, sentence_tester)
    except _BudgetExceeded:
        logger.warning("localization budget exhausted at sentence stage")
        return None

    granularity = "sentence"
    fragments = tuple(sentences[index] for index in kept)
    spans = tuple(sentence_spans[index] for index in kept) if sentence_spans else ()
    reduced_from = len(sentences)
    tests = sentence_tester.tests

    if word_stage and len(kept) == 1:
        sentence = sentences[kept[0]]
        words = split_words(sentence)
        if len(words) > 2:
            word_tester = _Tester(
                refuses=refuses,
                joiner=" ",
                budget=max(0, max_tests - tests),
                cache=sentence_tester.cache,
            )
            try:
                if word_tester(words):
                    word_kept = _ddmin(words, word_tester)
                    fragments = tuple(words[index] for index in word_kept)
                    granularity = "word"
                    reduced_from = len(words)
                    # Word spans are relative to the sentence; shift them into
                    # prompt coordinates so every span means the same thing.
                    word_spans = _unit_spans(sentence, words)
                    if word_spans and sentence_spans:
                        offset = sentence_spans[kept[0]][0]
                        spans = tuple(
                            (offset + word_spans[index][0], offset + word_spans[index][1])
                            for index in word_kept
                        )
                    else:
                        spans = ()
            except _BudgetExceeded:
                logger.info("budget exhausted at word stage; keeping sentence-level mRTF")
            tests += word_tester.tests

    return Localization(
        fragments=fragments,
        text=" ".join(fragments).strip(),
        granularity=granularity,
        tests=tests,
        reduced_from=reduced_from,
        spans=spans,
    )
