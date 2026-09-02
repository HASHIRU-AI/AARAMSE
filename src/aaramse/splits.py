"""Held-out splits, so a certificate is not evidence about its own training set.

The concept note promises "a held-out set of genuinely prohibited advice
requests to measure safety preservation". Nothing enforced that: the same
prompts certified the operators and then scored them, which makes the leak
figure a measure of memorisation rather than of safety.

Splitting is by SHA-256 of the prompt text, not by shuffling, for three
reasons. It is reproducible without storing a seed. It is stable when the
corpus grows, so adding prompts does not reshuffle the existing ones and
silently move an item from evaluation into certification. And it is
order-independent, so two runs that load the corpus differently still agree.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar

__all__ = ["Split", "assert_disjoint", "split_digest", "split_items"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Resolution of the hash bucket. 10_000 buckets makes a holdout fraction
# accurate to one part in ten thousand, which is finer than any corpus here.
_BUCKETS = 10_000


def _bucket(text: str, salt: str) -> int:
    """Map text to a stable bucket in [0, _BUCKETS)."""
    digest = hashlib.sha256(f"{salt}\x00{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _BUCKETS


@dataclass(frozen=True)
class Split:
    """A corpus divided into a certification fold and an evaluation fold.

    Attributes:
        certification: Items an operator may be certified against.
        evaluation: Items reserved for measuring the certified system.
        salt: The salt used, recorded so a split can be reproduced exactly.
        holdout: Fraction routed to evaluation.
    """

    certification: Tuple[T, ...]  # type: ignore[valid-type]
    evaluation: Tuple[T, ...]  # type: ignore[valid-type]
    salt: str
    holdout: float

    @property
    def sizes(self) -> Tuple[int, int]:
        """Return (certification, evaluation) counts."""
        return len(self.certification), len(self.evaluation)


def split_items(
    items: Iterable[T],
    key: Callable[[T], str],
    holdout: float = 0.5,
    salt: str = "aaramse-v1",
) -> Split:
    """Split a corpus deterministically by content hash.

    Args:
        items: The corpus.
        key: Extracts the text that identifies an item. Two items with the same
            text always land in the same fold, which is what stops a near
            duplicate leaking across the boundary.
        holdout: Fraction to reserve for evaluation, in [0, 1].
        salt: Changes the assignment. Change it only to produce a deliberately
            different split, and record which one you used.

    Returns:
        The split, with both folds in input order.

    Raises:
        ValueError: When `holdout` is outside [0, 1].
    """
    if not 0.0 <= holdout <= 1.0:
        raise ValueError(f"holdout must be in [0, 1], got {holdout}")
    threshold = holdout * _BUCKETS
    certification: List[T] = []
    evaluation: List[T] = []
    for item in items:
        target = evaluation if _bucket(key(item), salt) < threshold else certification
        target.append(item)
    split = Split(tuple(certification), tuple(evaluation), salt=salt, holdout=holdout)
    logger.info(
        "split corpus: %d for certification, %d held out (salt=%s)",
        *split.sizes, salt,
    )
    return split


def assert_disjoint(
    certification: Sequence[str], evaluation: Sequence[str], label: str = "corpus"
) -> None:
    """Fail loudly when the two folds share an item.

    Args:
        certification: Texts used to certify.
        evaluation: Texts used to measure.
        label: Named in the error, so a caller knows which corpus overlapped.

    Raises:
        ValueError: When any text appears in both folds.
    """
    overlap = set(certification) & set(evaluation)
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            f"{label}: {len(overlap)} prompts appear in both the certification and "
            f"evaluation folds, so any measurement over them is circular. "
            f"First: {sample}"
        )


def split_digest(split: Split, key: Callable[[T], str]) -> str:
    """Return a stable digest identifying a split.

    Recorded alongside results so a reviewer can tell whether two runs scored
    the same held-out set.
    """
    payload = "\x00".join(sorted(key(item) for item in split.evaluation))
    return hashlib.sha256(f"{split.salt}|{split.holdout}|{payload}".encode()).hexdigest()[:16]
