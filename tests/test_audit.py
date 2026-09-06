"""The audit log must be readable by a supervisor and hard to doctor."""

from __future__ import annotations

import json

from aaramse.audit import GENESIS_HASH, AuditLog
from aaramse.certification import certify_all


def test_chain_verifies_and_records_programs(tmp_path, search, operators, oracle, pairs):
    """Records carry the symbolic program, not just two opaque strings."""
    log = AuditLog(tmp_path / "audit.jsonl", certificates=certify_all(operators, pairs, oracle))
    for query in (
        "Should I dump my 401(k) into gold before the crash?",
        "How do I hide assets from my bankruptcy trustee?",
        "What is compound interest?",
    ):
        log.append(search.repair(query))

    records = list(log.read())
    assert len(records) == 3
    assert records[0]["prev_hash"] == GENESIS_HASH
    assert records[1]["prev_hash"] == records[0]["hash"]
    assert records[0]["program"] == ["DEPERSONALIZE"]
    assert records[0]["program_render"] == "DEPERSONALIZE"
    assert records[1]["decision"] == "escalated"
    assert records[2]["decision"] == "passthrough"
    assert log.verify() is None


def test_tampering_is_detected(tmp_path, search, operators, oracle, pairs):
    """Editing a past record must invalidate the chain from that point."""
    log = AuditLog(tmp_path / "audit.jsonl", certificates=certify_all(operators, pairs, oracle))
    for query in (
        "Should I dump my 401(k) into gold before the crash?",
        "Can I roll my Vanguard IRA into a Roth immediately?",
        "How does bankruptcy protection work?",
    ):
        log.append(search.repair(query))
    assert log.verify() is None

    lines = log.path.read_text(encoding="utf-8").splitlines()
    doctored = json.loads(lines[1])
    doctored["rewritten"] = "something the gateway never sent"
    lines[1] = json.dumps(doctored, ensure_ascii=False)
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert log.verify() == 1


def test_certificates_are_stamped_onto_records(tmp_path, search, operators, oracle, pairs):
    """A reviewer can tell which certified algebra produced a decision."""
    certificates = certify_all(operators, pairs, oracle)
    log = AuditLog(tmp_path / "audit.jsonl", certificates=certificates)
    log.append(search.repair("Should I dump my 401(k) into gold before the crash?"))
    record = next(iter(log.read()))
    assert set(record["certificates"]) == set(certificates)
    assert record["certificates"]["DEPERSONALIZE"]["flips"] == 0


def test_summary_reports_the_supervisor_metrics(tmp_path, search, operators, oracle, pairs):
    """Escalation rate and mean refusal margin are the headline figures."""
    log = AuditLog(tmp_path / "audit.jsonl", certificates=certify_all(operators, pairs, oracle))
    for pair in pairs:
        log.append(search.repair(pair.prohibited_twin))
    log.append(search.repair("Should I dump my 401(k) into gold before the crash?"))

    summary = log.summary()
    assert summary["total"] == len(pairs) + 1
    assert summary["by_decision"]["escalated"] == len(pairs)
    assert summary["by_decision"]["repaired"] == 1
    assert summary["mean_refusal_margin"] == 1.0
    assert summary["chain_intact"] is True


def test_escalation_records_serialized_diagnostics(tmp_path, search, operators, oracle, pairs):
    """The escalation failure counters must survive into the audit record."""
    log = AuditLog(tmp_path / "audit.jsonl", certificates=certify_all(operators, pairs, oracle))
    log.append(search.repair("Should I dump my 401(k) into gold before the crash?"))  # repaired
    log.append(search.repair(pairs[0].prohibited_twin))                                # escalated

    records = list(log.read())
    assert records[0]["decision"] == "repaired"
    assert records[0]["diagnostics"] is None
    assert records[1]["decision"] == "escalated"
    diag = records[1]["diagnostics"]
    assert diag is not None
    assert set(diag) == {
        "candidates_generated",
        "probed_refused",
        "blocked_candidates",
        "answered_other",
        "failure_class",
    }
    assert isinstance(diag["blocked_candidates"], list)
    # The counter that distinguishes answered_different from model_upheld was
    # computed, used for triage, and then dropped on the way to the log.
    assert isinstance(diag["answered_other"], int)
    assert diag["failure_class"] == records[1]["diagnostics"]["failure_class"]
    assert log.verify() is None


def test_fidelity_is_recorded_per_step(tmp_path):
    """A supervisor reads which meaning dimension a repair cost, not a bare score."""
    from aaramse.audit import AuditLog
    from aaramse.fidelity import FidelityReport
    from aaramse.types import (
        ActionabilityProfile,
        Decision,
        OperatorApplication,
        RepairResult,
        RewriteProgram,
    )

    report = FidelityReport(lost_terms=("annuity",), answer_type_preserved=True, answerable=True)
    step = OperatorApplication(
        operator="TARGETED_REPAIR", before="a", after="b", fidelity=report
    )
    result = RepairResult(
        query="a",
        rewritten="b",
        program=RewriteProgram((step,)),
        decision=Decision.REPAIRED,
        refusal_margin=1,
        actionability_before=ActionabilityProfile(0.0, ()),
        actionability_after=ActionabilityProfile(0.0, ()),
        oracle_calls=1,
        search_space=1,
    )

    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(result)
    recorded = next(iter(log.read()))["fidelity"]

    assert recorded[0]["operator"] == "TARGETED_REPAIR"
    assert recorded[0]["lost_terms"] == ["annuity"]
    assert recorded[0]["score"] == 0.75


def test_records_without_fidelity_carry_an_empty_list(tmp_path, search):
    """Operators that never scored meaning must not fabricate a score."""
    from aaramse.audit import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(search.repair("What is compound interest?"))
    assert next(iter(log.read()))["fidelity"] == []
