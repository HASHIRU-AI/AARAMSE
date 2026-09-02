"""Renders the audit log into the report a supervisor actually reads.

`AuditLog.summary()` returns a dict, which is the right shape for a program and
the wrong shape for a person. Phase 3 of the concept note promises an
intervention report; this is it -- Markdown, deterministic, and derived only
from what the hash chain already holds, so nothing in the report is assertable
that the log cannot prove.

Two things get prominence because they are the two a regulator asks about
first: whether the chain verifies, and which queries the layer could *not*
repair. Escalations are failures the deployer has to answer for, so they are
listed individually rather than counted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .audit import AuditLog

__all__ = ["InterventionReport", "build_report"]

logger = logging.getLogger(__name__)

# Queries are user text. The report is a document that leaves the building, so
# it carries enough of each query to identify it and no more.
DEFAULT_QUERY_CHARS = 160


def _clip(text: str, limit: int) -> str:
    """Shorten text for display, marking that it was shortened."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Render a Markdown table, or nothing when there are no rows."""
    if not rows:
        return []
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(cell for cell in row) + " |" for row in rows)
    return lines


@dataclass(frozen=True)
class InterventionReport:
    """A rendered account of everything one audit log holds.

    Attributes:
        model: Model spec the gateway fronted.
        generated_at: UTC timestamp of rendering.
        summary: The aggregate figures from `AuditLog.summary()`.
        chain_break: None when the chain verifies, else the index of the first
            record that fails.
        escalations: Records the layer could not repair.
        repairs: Records the layer repaired.
        certificates: Certificates in force on the most recent record.
    """

    model: str
    generated_at: str
    summary: Dict[str, Any]
    chain_break: Optional[int]
    escalations: List[Dict[str, Any]] = field(default_factory=list)
    repairs: List[Dict[str, Any]] = field(default_factory=list)
    certificates: Dict[str, Any] = field(default_factory=dict)

    @property
    def chain_intact(self) -> bool:
        """True when every record verifies against its predecessor."""
        return self.chain_break is None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view, for machine consumers."""
        return {
            "model": self.model,
            "generated_at": self.generated_at,
            "chain_intact": self.chain_intact,
            "chain_break": self.chain_break,
            "summary": self.summary,
            "escalations": self.escalations,
            "repairs": self.repairs,
            "certificates": self.certificates,
        }

    def render_markdown(self, query_chars: int = DEFAULT_QUERY_CHARS) -> str:
        """Render the report as Markdown."""
        lines: List[str] = [
            "# AARAMSE intervention report",
            "",
            f"- **Model fronted:** `{self.model}`",
            f"- **Generated:** {self.generated_at}",
            f"- **Interventions logged:** {self.summary.get('total', 0)}",
        ]
        lines.extend(self._integrity_section())
        lines.extend(self._volume_section())
        lines.extend(self._certificate_section())
        lines.extend(self._escalation_section(query_chars))
        lines.extend(self._repair_section(query_chars))
        return "\n".join(lines).rstrip() + "\n"

    def _integrity_section(self) -> List[str]:
        """State plainly whether the record can be trusted at all."""
        if self.chain_intact:
            verdict = "Hash chain verifies. No record has been altered since it was written."
        else:
            verdict = (
                f"**Hash chain BROKEN at record {self.chain_break}.** "
                "Every record from that point on is unverifiable; treat the whole "
                "log as compromised until the break is explained."
            )
        return ["", "## Integrity", "", verdict]

    def _volume_section(self) -> List[str]:
        """What the layer did, in aggregate."""
        by_decision = self.summary.get("by_decision", {})
        rows = [[decision, str(count)] for decision, count in sorted(by_decision.items())]
        lines = ["", "## What the layer did", ""]
        lines.extend(_table(("Decision", "Count"), rows) or ["No interventions recorded."])
        if by_decision:
            lines.extend([
                "",
                f"- Repair rate: {self.summary.get('repair_rate', 0.0):.1%}",
                f"- Escalation rate: {self.summary.get('escalation_rate', 0.0):.1%}",
                f"- Mean refusal margin: {self.summary.get('mean_refusal_margin', 0.0):.2f}"
                " operators per repair",
            ])
        programs = self.summary.get("programs_used", {})
        if programs:
            lines.extend(["", "### Programs used", ""])
            lines.extend(_table(
                ("Program", "Times applied"),
                [[f"`{name}`", str(count)] for name, count in programs.items()],
            ))
        return lines

    def _certificate_section(self) -> List[str]:
        """Which operators were admitted, and on what evidence."""
        if not self.certificates:
            return [
                "", "## Certificates in force", "",
                "**None recorded.** No operator in this log carried a certificate, "
                "so nothing here was admitted on evidence.",
            ]
        rows = []
        for name, cert in sorted(self.certificates.items()):
            rows.append([
                f"`{name}`",
                "pass" if cert.get("passed") else "**FAIL**",
                str(cert.get("trials", 0)),
                str(cert.get("flips", 0)),
                str(cert.get("issued_at", "")),
                f"`{cert.get('corpus_digest', '')}`",
            ])
        lines = ["", "## Certificates in force", ""]
        lines.extend(_table(
            ("Operator", "Verdict", "Trials", "Flips", "Issued", "Corpus"), rows
        ))
        lines.extend([
            "",
            "A certificate binds an operator to *this* model and *this* corpus. "
            "A different model voids it.",
        ])
        return lines

    def _escalation_section(self, query_chars: int) -> List[str]:
        """The queries a human still owes an answer to."""
        lines = ["", "## Escalations", ""]
        if not self.escalations:
            return [*lines, "None. Every refused query was repaired by a certified program."]
        lines.append(
            f"{len(self.escalations)} refused queries were not repaired and were "
            "passed to a human. Each is a user who did not get an answer."
        )
        lines.append("")
        lines.extend(_table(
            ("Timestamp", "Query", "Reason"),
            [[
                str(r.get("timestamp", "")),
                _clip(str(r.get("query", "")), query_chars),
                str(r.get("reason", "")),
            ] for r in self.escalations],
        ))
        return lines

    def _repair_section(self, query_chars: int) -> List[str]:
        """Every edit, so a reviewer can replay the decision."""
        lines = ["", "## Repairs", ""]
        if not self.repairs:
            return [*lines, "None recorded."]
        rows = []
        for record in self.repairs:
            mrtfs = [str(loc.get("mrtf", "")) for loc in record.get("localizations", [])]
            rows.append([
                str(record.get("timestamp", "")),
                _clip(str(record.get("query", "")), query_chars),
                _clip(str(record.get("rewritten", "")), query_chars),
                f"`{record.get('program_render', '')}`",
                str(record.get("refusal_margin", "")),
                ", ".join(f"`{m}`" for m in mrtfs) if mrtfs else "--",
            ])
        lines.extend(_table(
            ("Timestamp", "Query", "Sent to agent", "Program", "Margin", "Localized fragment"),
            rows,
        ))
        return lines


def build_report(
    log: AuditLog, model: str = "unknown", limit: Optional[int] = None
) -> InterventionReport:
    """Build a report from an audit log.

    Args:
        log: The log to read. Its chain is verified as part of building.
        model: Model spec the gateway fronted, for the header.
        limit: Cap on how many repairs and escalations to list individually.
            The aggregate figures always cover every record.

    Returns:
        A rendered-ready report.
    """
    records = list(log.read())
    escalations = [r for r in records if r.get("decision") == "escalated"]
    repairs = [r for r in records if r.get("decision") == "repaired"]
    if limit is not None:
        escalations = escalations[:limit]
        repairs = repairs[:limit]
    certificates = records[-1].get("certificates", {}) if records else {}
    return InterventionReport(
        model=model,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        summary=log.summary(),
        chain_break=log.verify(),
        escalations=escalations,
        repairs=repairs,
        certificates=certificates,
    )
