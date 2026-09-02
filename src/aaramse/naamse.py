"""Loader for NAAMSE evaluation reports.

NAAMSE evolves prompts against a target model and records what came back. Those
recordings are the thing AARAMSE cannot otherwise afford: a refusal label costs
a model call, and on a real model that is 20-30 seconds. A NAAMSE report hands
over thousands of *already generated* replies, so the baseline is free and
model time is spent only on prompts already known to be refused.

Two report shapes exist in the wild, and this loader accepts both:

* a standalone `final_report_{benign,adversarial}_*.json`, whose top level is
  the report itself;
* a combined `assessment_report_*.json` / `final_report_*.json` carrying both
  under `benign_report` and `adversarial_report`.

**Prompt sets are seed-scoped.** Two models evaluated under the same NAAMSE
seed share prompts and can be compared directly; two models under different
seeds share none. Cross-model claims are therefore only meaningful inside a
seed cohort, and `seed` is parsed off the filename so a caller can group by it
rather than assume.

**What these reports cannot do.** They carry no generation settings, so a reply
recorded here was produced under NAAMSE's conditions, not yours. Comparing a
recorded baseline against a live re-probe measures the gap between two
harnesses as much as the effect of a repair. Use recorded replies for baseline
labelling and analysis; re-probe under one consistent condition before claiming
a repair rate.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "GENERATION_ERROR_PATTERNS",
    "NaamseRecord",
    "NaamseReport",
    "load_directory",
    "load_report",
    "shared_prompts",
]

logger = logging.getLogger(__name__)

_SEED_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)

# Harness failures, not model behaviour. 341 records in the published corpus are
# the identical string "I apologize, but an error occurred while generating a
# response.", and another 87 are empty -- 13% of 3240 records. Scoring them is
# worse than dropping them: the three-way judge called one a full refusal and
# NAAMSE scored it 80, so an infrastructure timeout enters the results as a
# model that refused a benign question.
#
# Anchored to the start of the reply on purpose. An unanchored "error" matches
# prose that merely discusses one.
GENERATION_ERROR_PATTERNS: Tuple[str, ...] = (
    r"^\s*i\s+apologize,?\s+but\s+an\s+error\s+occurred",
    r"^\s*(?:an\s+)?error\s+occurred",
    r"^\s*\[?error\]?\s*[:\-]",
    r"^\s*(?:the\s+)?request\s+(?:timed\s+out|failed)",
    r"^\s*failed\s+to\s+generate",
)

_GENERATION_ERROR_RE = re.compile("|".join(GENERATION_ERROR_PATTERNS), re.IGNORECASE)


def is_generation_error(reply: str) -> bool:
    """Return True when a reply is a harness failure rather than model output.

    An empty reply counts: the model produced nothing, so there is nothing to
    label.
    """
    return not reply.strip() or bool(_GENERATION_ERROR_RE.search(reply[:200]))
POLARITIES: Tuple[str, ...] = ("benign", "adversarial")


@dataclass(frozen=True)
class NaamseRecord:
    """One prompt, its recorded reply, and NAAMSE's own score.

    Attributes:
        prompt: The prompt put to the target model.
        reply: The target model's final assistant turn, verbatim.
        score: NAAMSE's score. On a benign report a high score tracks refusal;
            on an adversarial report it tracks a successful attack. It is a
            different instrument from AARAMSE's judge, which is what makes it
            useful for agreement measurement rather than as ground truth.
        polarity: "benign" or "adversarial", from the report it came from.
        model: Target model, from the directory name.
        cluster: NAAMSE's semantic cluster label, when present.
        mutation_type: How the prompt was produced, when present.
    """

    prompt: str
    reply: str
    score: float
    polarity: str
    model: str
    cluster: str = ""
    mutation_type: str = ""

    @property
    def usable(self) -> bool:
        """True when the reply is model behaviour and can be scored."""
        return not is_generation_error(self.reply)

    @property
    def benign(self) -> bool:
        """True when the source report labelled this prompt benign.

        NAAMSE's label, not an adjudicated one. At least one jailbreak template
        appears in a benign set, so treat it as a starting point for
        adjudication rather than a finding.
        """
        return self.polarity == "benign"


@dataclass(frozen=True)
class NaamseReport:
    """Every record from one report file.

    Attributes:
        model: Target model name.
        seed: NAAMSE seed parsed from the filename, or "" when absent. Prompt
            sets are seed-scoped, so this is what makes two models comparable.
        source: Path the report was read from.
        records: The records, in file order.
    """

    model: str
    seed: str
    source: Path
    records: Tuple[NaamseRecord, ...]

    def polarity(self, polarity: str, usable_only: bool = True) -> Tuple[NaamseRecord, ...]:
        """Return the records of one polarity.

        Args:
            polarity: "benign" or "adversarial".
            usable_only: Drop harness failures. On by default, because scoring
                a timeout as a refusal is a silent corruption of every rate
                computed downstream.
        """
        return tuple(
            r for r in self.records
            if r.polarity == polarity and (r.usable or not usable_only)
        )

    def prompts(self, polarity: Optional[str] = None) -> Tuple[str, ...]:
        """Return unique prompts, optionally restricted to one polarity.

        Reports contain duplicates -- one file had 63 unique prompts in 80
        records -- so a denominator taken from the record count is inflated.
        """
        seen: Dict[str, None] = {}
        for record in self.records:
            if not record.usable:
                continue
            if polarity is None or record.polarity == polarity:
                seen.setdefault(record.prompt, None)
        return tuple(seen)


def _final_reply(record: Dict[str, Any]) -> str:
    """Return the last assistant turn from a conversation history."""
    history = record.get("conversation_history") or {}
    messages = history.get("messages") or []
    replies = [
        m.get("content", "")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    return replies[-1] if replies else ""


def _prompt_text(record: Dict[str, Any]) -> str:
    """Return the prompt, joining the list form NAAMSE sometimes emits."""
    prompt = record.get("prompt", "")
    if isinstance(prompt, list):
        return " ".join(str(part) for part in prompt).strip()
    return str(prompt).strip()


def _records(section: Dict[str, Any], polarity: str, model: str) -> List[NaamseRecord]:
    """Convert one report section into records."""
    out: List[NaamseRecord] = []
    for row in section.get("all_prompts_with_scores_and_history", []):
        prompt = _prompt_text(row)
        if not prompt:
            continue
        meta = row.get("metadata") or {}
        cluster_info = meta.get("cluster_info") or {}
        out.append(NaamseRecord(
            prompt=prompt,
            reply=_final_reply(row),
            score=float(row.get("score") or 0.0),
            polarity=polarity,
            model=model,
            cluster=str(cluster_info.get("label", "")),
            mutation_type=str(meta.get("mutation_type", "")),
        ))
    return out


def load_report(path: Path, model: Optional[str] = None) -> NaamseReport:
    """Load one report file, whichever of the two shapes it has.

    Args:
        path: The JSON report.
        model: Target model name. Defaults to the parent directory name.

    Returns:
        Every record the file holds, both polarities.

    Raises:
        ValueError: When the file carries no recognisable report section.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    name = model or path.parent.name
    seed_match = _SEED_RE.search(path.name)
    seed = seed_match.group(1) if seed_match else ""

    records: List[NaamseRecord] = []
    if any(f"{p}_report" in payload for p in POLARITIES):
        for polarity in POLARITIES:
            section = payload.get(f"{polarity}_report")
            if isinstance(section, dict):
                records.extend(_records(section, polarity, name))
    elif "all_prompts_with_scores_and_history" in payload:
        # Standalone file; polarity is only recoverable from the filename.
        polarity = "adversarial" if "adversarial" in path.name.lower() else "benign"
        records.extend(_records(payload, polarity, name))
    else:
        raise ValueError(f"{path} carries no NAAMSE report section")

    logger.info("loaded %d records from %s (model=%s seed=%s)", len(records), path, name, seed)
    return NaamseReport(model=name, seed=seed, source=path, records=tuple(records))


def load_directory(root: Path, pattern: str = "*.json") -> Tuple[NaamseReport, ...]:
    """Load every report under a directory tree.

    Args:
        root: Directory to walk.
        pattern: Glob applied to file names.

    Returns:
        One report per readable file. Unreadable files are logged and skipped,
        so one malformed export does not lose a whole corpus.
    """
    reports: List[NaamseReport] = []
    for path in sorted(Path(root).rglob(pattern)):
        try:
            reports.append(load_report(path))
        except (ValueError, json.JSONDecodeError, OSError) as error:
            logger.warning("skipping %s: %s", path, error)
    return tuple(reports)


def shared_prompts(
    reports: Sequence[NaamseReport], polarity: str = "benign"
) -> Tuple[str, ...]:
    """Return prompts every report has in common.

    Empty unless the reports share a NAAMSE seed. That is the intended
    behaviour: two models evaluated on different prompt sets have nothing to
    compare, and returning a union would invite exactly that mistake.
    """
    if not reports:
        return ()
    common = set(reports[0].prompts(polarity))
    for report in reports[1:]:
        common &= set(report.prompts(polarity))
    return tuple(sorted(common))


def group_by_seed(reports: Iterable[NaamseReport]) -> Dict[str, Tuple[NaamseReport, ...]]:
    """Group reports into comparable cohorts by NAAMSE seed."""
    groups: Dict[str, List[NaamseReport]] = {}
    for report in reports:
        groups.setdefault(report.seed, []).append(report)
    return {seed: tuple(items) for seed, items in sorted(groups.items())}
