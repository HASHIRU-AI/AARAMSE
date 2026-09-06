"""Loaders for the finance slice of FalseReject and OR-Bench-toxic.

XSTest turned out to be saturated on current models -- gemma4:12b over-refused
0 of 24 of its safe prompts -- so it cannot measure a mitigation. FalseReject
(Amazon Science, COLM 2025) supplies benign prompts deliberately engineered to
*look* unsafe, which still elicit over-refusal, and OR-Bench-toxic supplies
genuinely unsafe prompts for the safety side.

Both are filtered to finance-relevant items and vendored so evaluation is
reproducible offline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .splits import split_items

__all__ = ["DEFAULT_CORPUS_PATH", "FinancePrompt", "load_benign", "load_toxic"]

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "falsereject_finance.json"


@dataclass(frozen=True)
class FinancePrompt:
    """One corpus item.

    Attributes:
        prompt: The prompt text, verbatim from the source dataset.
        category: Source-provided category label.
        benign: True for FalseReject items, False for OR-Bench-toxic items.
    """

    prompt: str
    category: str
    benign: bool


def _load(path: Optional[Path] = None) -> dict:
    """Read the vendored corpus file."""
    source = Path(path or DEFAULT_CORPUS_PATH)
    if not source.exists():
        raise FileNotFoundError(
            f"corpus not found at {source}; regenerate it from FalseReject and OR-Bench."
        )
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fold(
    items: Sequence[FinancePrompt], fold: str, salt: str
) -> Tuple[FinancePrompt, ...]:
    """Restrict a corpus to one fold of the deterministic split.

    Splitting here rather than at the call site is deliberate. The harm gate's
    patterns were authored against the certification fold, so a measurement that
    slices the corpus in file order silently mixes seen and unseen items and
    reports a number stronger than it is -- which is exactly what happened to
    the first live run of `examples/e2e_smoke.py`.

    Args:
        items: The loaded corpus, in file order.
        fold: One of "all", "certification", "evaluation".
        salt: Split salt, passed through so a fold can be reproduced.

    Returns:
        The requested fold, in input order.

    Raises:
        ValueError: When `fold` is not one of the three accepted names.
    """
    if fold == "all":
        return tuple(items)
    if fold not in ("certification", "evaluation"):
        raise ValueError(
            f"fold must be 'all', 'certification' or 'evaluation', got {fold!r}"
        )
    split = split_items(items, key=lambda item: item.prompt, salt=salt)
    return tuple(getattr(split, fold))


def load_benign(
    limit: Optional[int] = None,
    path: Optional[Path] = None,
    fold: str = "all",
    salt: str = "aaramse-v1",
) -> Tuple[FinancePrompt, ...]:
    """Benign finance prompts that models commonly over-refuse.

    Args:
        limit: Keep at most this many items, applied *after* the fold so a
            limited draw stays inside the requested fold.
        path: Override the vendored corpus location.
        fold: "all" (default, file order), "certification", or "evaluation".
        salt: Split salt, ignored when `fold` is "all".

    Returns:
        The selected prompts, in corpus order.
    """
    rows: List[dict] = _load(path)["benign"]
    items = [FinancePrompt(r["prompt"], r["category"], benign=True) for r in rows]
    selected = _fold(items, fold, salt)
    return tuple(selected[:limit] if limit else selected)


def load_toxic(
    limit: Optional[int] = None,
    path: Optional[Path] = None,
    fold: str = "all",
    salt: str = "aaramse-v1",
) -> Tuple[FinancePrompt, ...]:
    """Genuinely unsafe finance prompts, used to measure leakage.

    Args:
        limit: Keep at most this many items, applied *after* the fold so a
            limited draw stays inside the requested fold.
        path: Override the vendored corpus location.
        fold: "all" (default, file order), "certification", or "evaluation".
            Pass "evaluation" for any measurement that reports a leak or gate
            rate; the harm patterns saw the certification fold.
        salt: Split salt, ignored when `fold` is "all".

    Returns:
        The selected prompts, in corpus order.
    """
    rows: List[dict] = _load(path)["toxic"]
    items = [FinancePrompt(r["prompt"], str(r["category"]), benign=False) for r in rows]
    selected = _fold(items, fold, salt)
    return tuple(selected[:limit] if limit else selected)
