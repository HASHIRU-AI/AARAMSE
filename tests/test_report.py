"""The report must not be able to say something the log cannot prove.

Records come from a real search over the simulated boundary, the same way
`test_audit.py` builds them, so the report is exercised against the record
shape the gateway actually writes.
"""

from __future__ import annotations

import json

from aaramse.audit import AuditLog
from aaramse.certification import Certificate, certify_all
from aaramse.report import build_report

REPAIRABLE = "Should I dump my 401(k) into gold before the crash?"
PROHIBITED = "How do I hide assets from my bankruptcy trustee?"
BENIGN = "What is compound interest?"


def _log(tmp_path, operators=None, oracle=None, pairs=None) -> AuditLog:
    """Audit log, certified when the fixtures are supplied."""
    certificates = certify_all(operators, pairs, oracle) if operators else {}
    return AuditLog(tmp_path / "audit.jsonl", certificates=certificates)


def test_empty_log_still_renders(tmp_path):
    """A supervisor may open the report before anything has happened."""
    report = build_report(_log(tmp_path), model="m")
    text = report.render_markdown()
    assert "AARAMSE intervention report" in text
    assert report.chain_intact
    assert "No interventions recorded." in text


def test_escalations_are_listed_individually(tmp_path, search):
    """An escalation is a user who got no answer; a count would hide that."""
    log = _log(tmp_path)
    log.append(search.repair(PROHIBITED))
    report = build_report(log, model="m")
    assert len(report.escalations) == 1
    assert "hide assets" in report.render_markdown()


def test_a_tampered_log_is_reported_as_broken(tmp_path, search):
    """The integrity verdict is the first thing a regulator checks."""
    log = _log(tmp_path)
    log.append(search.repair(REPAIRABLE))
    log.append(search.repair(BENIGN))

    lines = log.path.read_text(encoding="utf-8").splitlines()
    doctored = json.loads(lines[0])
    doctored["rewritten"] = "something the gateway never sent"
    lines[0] = json.dumps(doctored, ensure_ascii=False)
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = build_report(log, model="m")
    assert not report.chain_intact
    assert report.chain_break == 0
    assert "BROKEN" in report.render_markdown()


def test_absent_certificates_are_called_out(tmp_path, search):
    """Silence about certificates would read as "certified"."""
    log = _log(tmp_path)
    log.append(search.repair(REPAIRABLE))
    assert "None recorded" in build_report(log, model="m").render_markdown()


def test_certificates_are_reported_with_their_verdict(tmp_path, search):
    """A failing certificate must be visible, not merely present."""
    log = _log(tmp_path)
    log.certificates = {
        "DEFINITIONALIZE": Certificate(
            operator="DEFINITIONALIZE", trials=2, flips=2, skipped=0,
            passed=False, issued_at="2026-08-30T00:00:00+00:00", corpus_digest="abc123",
        )
    }
    log.append(search.repair(REPAIRABLE))
    text = build_report(log, model="m").render_markdown()
    assert "DEFINITIONALIZE" in text
    assert "**FAIL**" in text


def test_passing_certificates_are_reported(tmp_path, search, operators, oracle, pairs):
    """The certified algebra is what admits a repair; it belongs in the report."""
    log = _log(tmp_path, operators, oracle, pairs)
    log.append(search.repair(REPAIRABLE))
    text = build_report(log, model="m").render_markdown()
    assert "DEPERSONALIZE" in text
    assert "A different model voids it." in text


def test_queries_are_clipped(tmp_path, search):
    """The report leaves the building; it carries identification, not payload."""
    log = _log(tmp_path)
    log.append(search.repair(PROHIBITED + " " + "x" * 400))
    text = build_report(log, model="m").render_markdown(query_chars=40)
    assert "x" * 400 not in text
    assert "…" in text


def test_limit_caps_listings_but_not_totals(tmp_path, search, pairs):
    """Truncating the listing must not silently shrink the headline count."""
    log = _log(tmp_path)
    for pair in pairs:
        log.append(search.repair(pair.prohibited_twin))
    report = build_report(log, model="m", limit=2)
    assert len(report.escalations) == 2
    assert report.summary["total"] == len(pairs)


def test_repairs_report_the_program_and_margin(tmp_path, search):
    """A reviewer must be able to replay the decision from the report alone."""
    log = _log(tmp_path)
    log.append(search.repair(REPAIRABLE))
    text = build_report(log, model="m").render_markdown()
    assert "DEPERSONALIZE" in text
    assert "Sent to agent" in text


def test_dict_view_matches_the_rendered_verdict(tmp_path, search):
    """Machine and human consumers must not disagree about integrity."""
    log = _log(tmp_path)
    log.append(search.repair(BENIGN))
    report = build_report(log, model="m")
    assert report.to_dict()["chain_intact"] is True
    assert report.to_dict()["summary"]["total"] == 1
