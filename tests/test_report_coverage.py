"""Operator exclusions must reach the report body, not only stderr.

The offline suite printed PASSED while `TARGETED_REPAIR` -- the sole admitted
operator against a live model, and the operator that produced both induced
leaks -- was never exercised. The exclusion reached a `logger.error` line and
nowhere else, so a reader of the report had no way to learn that half the
shipped algebra was dark.

Untestable and failed are different findings and must read differently. A
failed operator weakened the boundary; an untestable one says the corpus is
wrong for it. Collapsing them hides which of the two you are looking at.
"""

from __future__ import annotations

from aaramse.report import InterventionReport


def _report(certificates):
    return InterventionReport(
        model="sim:test",
        generated_at="2026-09-04T00:00:00+00:00",
        summary={"total": 3},
        chain_break=None,
        certificates=certificates,
    )


PASSING = {"passed": True, "trials": 8, "flips": 0, "skipped": 0}
UNTESTABLE = {"passed": False, "trials": 0, "flips": 0, "skipped": 8}
FAILED = {"passed": False, "trials": 8, "flips": 3, "skipped": 0}


def test_untestable_operator_is_named_in_the_body():
    """Zero trials means the corpus never exercised it; say so in words."""
    report = _report({"FRAME_ASSERT": PASSING, "TARGETED_REPAIR": UNTESTABLE})
    body = report.render_markdown()

    assert "## Coverage" in body
    assert "TARGETED_REPAIR" in body.split("## Coverage", 1)[1]
    assert "never exercised" in body.lower()


def test_untestable_is_distinguished_from_failed():
    """The two exclusions have different causes and different remedies."""
    untestable = _report({"TARGETED_REPAIR": UNTESTABLE}).render_markdown().lower()
    failed = _report({"TARGETED_REPAIR": FAILED}).render_markdown().lower()

    assert "never exercised" in untestable
    assert "never exercised" not in failed
    assert "weakened the boundary" in failed


def test_full_coverage_does_not_raise_a_false_alarm():
    """A clean algebra must not be reported as if something were wrong."""
    body = report = _report({"FRAME_ASSERT": PASSING, "TARGETED_REPAIR": PASSING})
    rendered = report.render_markdown()

    assert "## Coverage" in rendered
    section = rendered.split("## Coverage", 1)[1]
    assert "never exercised" not in section.lower()
    assert "weakened" not in section.lower()


def test_exclusions_are_exposed_to_machine_consumers():
    """`e2e_smoke` reads the report programmatically; it needs the same finding."""
    report = _report({"FRAME_ASSERT": PASSING, "TARGETED_REPAIR": UNTESTABLE})
    payload = report.to_dict()

    assert payload["exclusions"] == [
        {"operator": "TARGETED_REPAIR", "cause": "untestable", "trials": 0, "flips": 0}
    ]


def test_failed_exclusion_reports_its_flips():
    """A reader needs the magnitude, not just the verdict."""
    report = _report({"TARGETED_REPAIR": FAILED})
    assert report.to_dict()["exclusions"] == [
        {"operator": "TARGETED_REPAIR", "cause": "failed", "trials": 8, "flips": 3}
    ]


def test_no_certificates_reports_no_exclusions():
    """Absent certificates is already its own loud finding; do not double-count."""
    assert _report({}).to_dict()["exclusions"] == []
