"""Loader for the vendored FinQA control slice.

This is a **negative** control, and that distinction is the whole point.
FalseReject and OR-Bench measure the two things the layer is supposed to do:
repair benign refusals, and not leak prohibited ones. Neither measures the
third thing it must do, which is *nothing at all* to a query the model was
always going to answer.

FinQA supplies that: numerical-reasoning questions over public filings, which
no safety-aligned model has a reason to refuse. Every one of them should come
back `PASSTHROUGH`, byte-identical. Any other decision is a false intervention
-- the layer rewriting a query nobody objected to -- and it costs both latency
and the passthrough guarantee the design rests on.

Regenerate with `scripts/build_finqa_control.py`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["DEFAULT_CONTROL_PATH", "ControlPrompt", "load_control", "source_note"]

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_PATH = Path(__file__).resolve().parents[2] / "data" / "finqa_control.json"


@dataclass(frozen=True)
class ControlPrompt:
    """One benign financial-analysis prompt.

    Attributes:
        prompt: Self-contained prompt, question plus its rendered table.
        question: The bare FinQA question, for reporting.
        answer: The dataset's executed answer. Not used for scoring refusal;
            kept so a reviewer can see these are answerable questions.
        source_id: FinQA item id, so any prompt can be traced to its filing.
    """

    prompt: str
    question: str
    answer: str
    source_id: str


def _load(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the vendored control file."""
    source = Path(path or DEFAULT_CONTROL_PATH)
    if not source.exists():
        raise FileNotFoundError(
            f"FinQA control slice not found at {source}; rebuild it with "
            "scripts/build_finqa_control.py from FinQA's dataset/test.json."
        )
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_control(
    limit: Optional[int] = None, path: Optional[Path] = None
) -> Tuple[ControlPrompt, ...]:
    """Benign financial-analysis prompts that must never be refused.

    Args:
        limit: Cap on how many to return, in file order.
        path: Override the vendored corpus location.

    Returns:
        The control prompts.
    """
    rows: List[Dict[str, str]] = _load(path)["items"]
    items = [
        ControlPrompt(
            prompt=row["prompt"],
            question=row["question"],
            answer=row.get("answer", ""),
            source_id=row.get("source_id", ""),
        )
        for row in rows
    ]
    return tuple(items[:limit] if limit else items)


def source_note(path: Optional[Path] = None) -> str:
    """Return the provenance line for this corpus, for a results header."""
    payload = _load(path)
    return f"{payload.get('source', 'unknown')} <{payload.get('url', '')}>"
