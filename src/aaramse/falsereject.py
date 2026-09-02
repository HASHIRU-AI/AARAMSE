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
from typing import List, Optional, Tuple

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


def load_benign(
    limit: Optional[int] = None, path: Optional[Path] = None
) -> Tuple[FinancePrompt, ...]:
    """Benign finance prompts that models commonly over-refuse."""
    rows: List[dict] = _load(path)["benign"]
    items = [FinancePrompt(r["prompt"], r["category"], benign=True) for r in rows]
    return tuple(items[:limit] if limit else items)


def load_toxic(
    limit: Optional[int] = None, path: Optional[Path] = None
) -> Tuple[FinancePrompt, ...]:
    """Genuinely unsafe finance prompts, used to measure leakage."""
    rows: List[dict] = _load(path)["toxic"]
    items = [FinancePrompt(r["prompt"], str(r["category"]), benign=False) for r in rows]
    return tuple(items[:limit] if limit else items)
