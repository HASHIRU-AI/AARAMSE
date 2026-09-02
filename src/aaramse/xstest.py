"""Loader for XSTest, a validated over-refusal benchmark.

XSTest (Roettger et al., NAACL 2024) pairs 250 safe prompts that aligned models
commonly over-refuse with 200 unsafe prompts sharing their surface form. Types
`X` and `contrast_X` line up index-for-index and are cross-checked here on the
`focus` column, which gives real contrastive pairs rather than hand-written ones.

The seed corpus in `corpus.py` was invented; this is measured data and should be
preferred for every claim.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .certification import ContrastivePair

__all__ = ["DEFAULT_XSTEST_PATH", "SAFE_TYPES", "load_xstest", "stratified_pairs", "xstest_pairs"]

logger = logging.getLogger(__name__)

DEFAULT_XSTEST_PATH = Path(__file__).resolve().parents[2] / "data" / "xstest_prompts.csv"

# Only these six have a 1:1 `contrast_X` partner. The privacy and discrimination
# types share a single contrast set between two safe types each, so pairing them
# would be arbitrary; they are excluded rather than guessed at.
SAFE_TYPES: Tuple[str, ...] = (
    "definitions",
    "figurative_language",
    "historical_events",
    "homonyms",
    "safe_contexts",
    "safe_targets",
)


def load_xstest(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Read the XSTest CSV into a list of row dictionaries."""
    source = Path(path or DEFAULT_XSTEST_PATH)
    if not source.exists():
        raise FileNotFoundError(
            f"XSTest not found at {source}. Fetch xstest_prompts.csv from "
            "github.com/paul-rottger/exaggerated-safety into data/."
        )
    with source.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def xstest_pairs(path: Optional[Path] = None) -> Tuple[ContrastivePair, ...]:
    """Build contrastive pairs from XSTest, verifying the focus term matches.

    Returns:
        Every (safe, unsafe) pair whose `focus` agrees, so a mismatch in the
        upstream file surfaces as a dropped pair rather than a bad pair.
    """
    rows = load_xstest(path)
    by_type: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    pairs: List[ContrastivePair] = []
    for safe_type in SAFE_TYPES:
        safe_rows = by_type.get(safe_type, [])
        unsafe_rows = by_type.get(f"contrast_{safe_type}", [])
        # strict=False is deliberate: a type whose safe and contrast sets differ
        # in length should yield the pairs it can, not raise.
        for safe, unsafe in zip(safe_rows, unsafe_rows, strict=False):
            if safe["focus"] != unsafe["focus"]:
                logger.warning(
                    "focus mismatch in %s: %r vs %r", safe_type, safe["focus"], unsafe["focus"]
                )
                continue
            pairs.append(
                ContrastivePair(
                    benign=safe["prompt"],
                    prohibited_twin=unsafe["prompt"],
                    note=f"{safe_type}/{safe['focus']}",
                )
            )
    return tuple(pairs)


def stratified_pairs(
    per_type: int = 2, path: Optional[Path] = None
) -> Tuple[ContrastivePair, ...]:
    """Take an even sample across XSTest types to keep evaluation tractable.

    Args:
        per_type: Pairs to draw from each safe type.
        path: Optional override for the CSV location.

    Returns:
        Up to `per_type * len(SAFE_TYPES)` pairs, in a deterministic order.
    """
    grouped: Dict[str, List[ContrastivePair]] = defaultdict(list)
    for pair in xstest_pairs(path):
        grouped[pair.note.split("/", 1)[0]].append(pair)
    sample: List[ContrastivePair] = []
    for safe_type in SAFE_TYPES:
        sample.extend(grouped[safe_type][:per_type])
    return tuple(sample)


def safe_prompts(pairs: Sequence[ContrastivePair]) -> Tuple[str, ...]:
    """Extract the benign half of a pair set, for over-refusal probing."""
    return tuple(pair.benign for pair in pairs)
