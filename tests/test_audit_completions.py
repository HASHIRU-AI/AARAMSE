"""Raw completions: retained for diagnosis, hashed into the chain, kept out of git.

The audit record dropped what the model actually said, which is why the two
falsely-escalated FinQA controls could not be diagnosed -- there was no way to
tell a misfiring refusal classifier from a model that genuinely produced a
non-answer on a dense table.

Completions are therefore retained, but split. The committed record carries only
a SHA-256; the text goes to a gitignored sidecar. Because the hash sits inside
the chained payload, a completion cannot be altered and still verify, which is
what makes a local sidecar trustworthy evidence rather than an unattested file.
"""

from __future__ import annotations

import hashlib
import json

from aaramse.audit import GENESIS_HASH, AuditLog


def _result(search):
    """One repaired result to hang completions off."""
    return search.repair("Should I dump my 401(k) into gold before the crash?")


def test_completion_hash_is_recorded_in_the_chained_payload(tmp_path, search):
    """The hash is payload, not metadata, so the chain covers it."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(_result(search), completions={"query": "I cannot help with that."})

    record = next(iter(log.read()))
    expected = hashlib.sha256("I cannot help with that.".encode("utf-8")).hexdigest()
    assert record["completions"]["query"]["sha256"] == expected
    assert record["completions"]["query"]["chars"] == len("I cannot help with that.")
    assert log.verify() is None


def test_completion_text_goes_to_the_sidecar_not_the_record(tmp_path, search):
    """The committed log must not contain the model's words."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(_result(search), completions={"query": "SENSITIVE PROCEDURE TEXT"})

    assert "SENSITIVE PROCEDURE TEXT" not in (tmp_path / "audit.jsonl").read_text()
    sidecar = tmp_path / "audit.completions.jsonl"
    assert sidecar.exists()
    entry = json.loads(sidecar.read_text().strip())
    assert entry["text"] == "SENSITIVE PROCEDURE TEXT"
    assert entry["role"] == "query"
    assert entry["record_hash"] == next(iter(log.read()))["hash"]


def test_both_roles_are_retained(tmp_path, search):
    """The rewritten query's answer matters too: it is what a user would see."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(
        _result(search),
        completions={"query": "refused", "rewritten": "here is the answer"},
    )

    record = next(iter(log.read()))
    assert set(record["completions"]) == {"query", "rewritten"}
    roles = {
        json.loads(line)["role"]
        for line in (tmp_path / "audit.completions.jsonl").read_text().splitlines()
    }
    assert roles == {"query", "rewritten"}


def test_verify_completions_accepts_an_untampered_sidecar(tmp_path, search):
    """The happy path: sidecar text still hashes to what the chain committed."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(_result(search), completions={"query": "the original answer"})
    assert log.verify_completions() is None


def test_verify_completions_detects_edited_text(tmp_path, search):
    """Editing the sidecar must be caught; otherwise retention proves nothing."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(_result(search), completions={"query": "the original answer"})

    sidecar = tmp_path / "audit.completions.jsonl"
    entry = json.loads(sidecar.read_text().strip())
    entry["text"] = "a doctored answer"
    sidecar.write_text(json.dumps(entry) + "\n")

    assert log.verify_completions() == 0


def test_records_without_completions_still_chain(tmp_path, search):
    """Completions are optional; omitting them writes no sidecar and no field."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(_result(search))

    record = next(iter(log.read()))
    assert record["completions"] is None
    assert not (tmp_path / "audit.completions.jsonl").exists()
    assert log.verify() is None
    assert log.verify_completions() is None


def test_existing_committed_evidence_still_verifies(tmp_path):
    """Cited audit artifacts predate this field and must not break.

    `audit/e2e_heldout_live.jsonl` and its siblings are referenced by
    `plan/harm-gate.md` as the evidence behind the held-out run. A schema change
    that silently invalidated them would destroy the record it was added to
    improve.
    """
    from pathlib import Path

    committed = Path(__file__).resolve().parents[1] / "audit" / "e2e_heldout_live.jsonl"
    if not committed.exists():  # pragma: no cover - artifact not vendored
        return
    log = AuditLog(committed)
    assert log.verify() is None
    assert log.verify_completions() is None


def test_sidecar_path_is_derived_from_the_log_path(tmp_path):
    """A predictable name is what lets .gitignore exclude every sidecar."""
    log = AuditLog(tmp_path / "nested" / "run.jsonl")
    assert log.completions_path == tmp_path / "nested" / "run.completions.jsonl"
