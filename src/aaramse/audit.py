"""Tamper-evident audit log for every rewrite decision.

Records are chained by SHA-256 over the previous record's hash, so any edit to
history invalidates every record after it. What is logged is the operator
*program*, not just a pair of opaque strings: a supervisor reads
"DEPERSONALIZE o ENTITY_GENERALIZE, margin 2" and can replay it exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .certification import Certificate
from .types import RepairResult

__all__ = ["GENESIS_HASH", "AuditLog"]

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64


def _diagnostics(result: RepairResult) -> Optional[Dict[str, Any]]:
    """Serialize the search failure counters, when the result carries them."""
    diag = result.diagnostics
    if diag is None:
        return None
    return {
        "candidates_generated": diag.candidates_generated,
        "probed_refused": diag.probed_refused,
        "blocked_candidates": list(diag.blocked_candidates),
    }


def _localizations(result: RepairResult) -> List[Dict[str, Any]]:
    """Extract the delta-debugging record from each step that has one."""
    out: List[Dict[str, Any]] = []
    for step in result.program.steps:
        loc = step.localization
        if loc is None:
            continue
        out.append({
            "operator": step.operator,
            "mrtf": loc.text,
            "granularity": loc.granularity,
            "probes": loc.tests,
            "reduction_ratio": round(loc.reduction_ratio, 3),
        })
    return out


def _digest(previous: str, payload: Mapping[str, Any]) -> str:
    """Return the chained hash for a payload following `previous`."""
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((previous + body).encode("utf-8")).hexdigest()


@dataclass
class AuditLog:
    """Append-only, hash-chained JSONL log of rewrite interventions.

    Attributes:
        path: Destination JSONL file. Created on first append.
        certificates: Operator certificates in force, stamped onto each record
            so a reviewer can tell which certified algebra produced a decision.
    """

    path: Path
    certificates: Dict[str, Certificate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        """Return the hash of the final record, or the genesis hash."""
        records = list(self.read())
        return records[-1]["hash"] if records else GENESIS_HASH

    def append(self, result: RepairResult) -> str:
        """Write one intervention record and return its hash.

        Args:
            result: The repair outcome to record.

        Returns:
            The chained SHA-256 hash of the new record.
        """
        previous = self._last_hash()
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "decision": result.decision.value,
            "query": result.query,
            "rewritten": result.rewritten,
            "program": list(result.program.names),
            "program_render": result.program.render(),
            "refusal_margin": result.refusal_margin,
            "actionability_before": result.actionability_before.score,
            "actionability_after": result.actionability_after.score,
            "features_before": result.actionability_before.as_dict(),
            "generalizations": [list(g) for g in result.program.generalizations()],
            "localizations": _localizations(result),
            "dropped_clauses": list(result.program.dropped()),
            "oracle_calls": result.oracle_calls,
            "search_space": result.search_space,
            "reason": result.reason,
            "diagnostics": _diagnostics(result),
            "certificates": {
                name: cert.to_dict() for name, cert in sorted(self.certificates.items())
            },
            "prev_hash": previous,
        }
        record_hash = _digest(previous, payload)
        record = {**payload, "hash": record_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record_hash

    def read(self) -> Iterator[Dict[str, Any]]:
        """Yield every record in write order."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def verify(self) -> Optional[int]:
        """Verify the hash chain.

        Returns:
            None when the chain is intact, otherwise the zero-based index of
            the first record that fails verification.
        """
        previous = GENESIS_HASH
        for index, record in enumerate(self.read()):
            stored = record.get("hash")
            payload = {k: v for k, v in record.items() if k != "hash"}
            if payload.get("prev_hash") != previous:
                return index
            if _digest(previous, payload) != stored:
                return index
            previous = stored
        return None

    def summary(self) -> Dict[str, Any]:
        """Aggregate the log into the figures a supervisor asks for first."""
        records: List[Dict[str, Any]] = list(self.read())
        margins = [r["refusal_margin"] for r in records if r["decision"] == "repaired"]
        by_decision: Dict[str, int] = {}
        by_program: Dict[str, int] = {}
        for record in records:
            by_decision[record["decision"]] = by_decision.get(record["decision"], 0) + 1
            if record["decision"] == "repaired":
                key = record["program_render"]
                by_program[key] = by_program.get(key, 0) + 1
        total = len(records)
        return {
            "total": total,
            "by_decision": by_decision,
            "escalation_rate": by_decision.get("escalated", 0) / total if total else 0.0,
            "repair_rate": by_decision.get("repaired", 0) / total if total else 0.0,
            "mean_refusal_margin": sum(margins) / len(margins) if margins else 0.0,
            "programs_used": dict(sorted(by_program.items(), key=lambda kv: -kv[1])),
            "chain_intact": self.verify() is None,
        }
