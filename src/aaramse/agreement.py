"""Inter-instrument agreement, for deciding whether a label means anything.

Three instruments assign refusal labels in this project and none of them is
ground truth: NAAMSE's numeric score, the rule-based patterns in `refusal.py`,
and the LLM judge in `judge.py`. The judge is the one the evaluations trust, and
its validation to date is six hand labels.

Agreement between instruments is not proof any of them is right -- three
instruments can be wrong together. It is diagnostic in one direction only:
where they disagree, at least one is wrong, and those cases are exactly the
ones worth adjudicating by hand. Raw agreement is reported alongside Cohen's
kappa because raw agreement flatters any pair on a skewed corpus: two
instruments that both say "not refused" 90% of the time agree 90% of the time
by doing nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

__all__ = ["Agreement", "agree", "best_threshold"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Agreement:
    """Agreement between two binary instruments over the same items.

    Attributes:
        both: Items both called positive.
        neither: Items both called negative.
        only_a: Items only the first called positive.
        only_b: Items only the second called positive.
    """

    both: int
    neither: int
    only_a: int
    only_b: int

    @property
    def total(self) -> int:
        """Number of items compared."""
        return self.both + self.neither + self.only_a + self.only_b

    @property
    def observed(self) -> float:
        """Fraction of items the two instruments labelled the same way."""
        return (self.both + self.neither) / self.total if self.total else 0.0

    @property
    def expected(self) -> float:
        """Agreement expected from the marginals alone, if the two were independent."""
        n = self.total
        if not n:
            return 0.0
        a_pos, b_pos = (self.both + self.only_a) / n, (self.both + self.only_b) / n
        return a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    @property
    def kappa(self) -> float:
        """Cohen's kappa: agreement above what the marginals already explain.

        1.0 is perfect, 0.0 is chance. Negative means the two instruments
        disagree more often than independent labelling would predict, which
        usually means one of them is measuring something else entirely.
        """
        expected = self.expected
        if expected >= 1.0:
            # Both instruments were constant; kappa is undefined, and reporting
            # 1.0 would present "neither ever fired" as perfect agreement.
            return 0.0
        return (self.observed - expected) / (1 - expected)

    @property
    def disagreements(self) -> int:
        """Items where the two instruments differ; the adjudication queue."""
        return self.only_a + self.only_b

    def summary(self, name_a: str = "A", name_b: str = "B") -> str:
        """One line for a results table."""
        return (
            f"{name_a} vs {name_b}: raw {self.observed:.1%}, kappa {self.kappa:+.3f}, "
            f"{self.disagreements}/{self.total} disagree "
            f"(only {name_a}: {self.only_a}, only {name_b}: {self.only_b})"
        )


def agree(a: Sequence[bool], b: Sequence[bool]) -> Agreement:
    """Compare two binary label sequences item by item.

    Args:
        a: Labels from the first instrument.
        b: Labels from the second, in the same order.

    Returns:
        The confusion between them.

    Raises:
        ValueError: When the sequences differ in length, which would silently
            misalign every item after the first missing one.
    """
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} vs {len(b)}")
    both = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    neither = sum(1 for x, y in zip(a, b, strict=True) if not x and not y)
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return Agreement(both=both, neither=neither, only_a=only_a, only_b=only_b)


def best_threshold(
    scores: Sequence[float], labels: Sequence[bool], step: float = 1.0
) -> Tuple[Optional[float], Agreement]:
    """Find the score cutoff that best reproduces a reference label.

    Useful for asking what NAAMSE's continuous score means in the vocabulary of
    a refusal decision. A high kappa says the score is close to a relabelling
    of the reference; a low one says it is measuring something else, and the
    two should not be substituted for each other.

    Args:
        scores: Continuous scores from one instrument.
        labels: Binary labels from the reference instrument.
        step: Granularity of the threshold sweep.

    Returns:
        The best threshold and its agreement. The threshold is None when there
        is nothing to sweep.
    """
    if not scores or len(scores) != len(labels):
        return None, Agreement(0, 0, 0, 0)
    low, high = min(scores), max(scores)
    best: Tuple[Optional[float], Agreement] = (None, Agreement(0, 0, 0, 0))
    candidate = low
    while candidate <= high + step:
        predicted = [s >= candidate for s in scores]
        result = agree(predicted, list(labels))
        if best[0] is None or result.kappa > best[1].kappa:
            best = (candidate, result)
        candidate += step
    return best


def confusion_by_class(
    labels: Sequence[str], reference: Sequence[bool]
) -> Dict[str, Tuple[int, int]]:
    """Cross-tabulate a multi-class label against a binary reference.

    Args:
        labels: Class per item, e.g. the three-way judge's verdicts.
        reference: Binary label per item, in the same order.

    Returns:
        Class name to (reference true, reference false).

    Raises:
        ValueError: When the sequences differ in length.
    """
    table: Dict[str, Tuple[int, int]] = {}
    for label, flag in zip(labels, reference, strict=True):
        yes, no = table.get(label, (0, 0))
        table[label] = (yes + int(flag), no + int(not flag))
    return table
