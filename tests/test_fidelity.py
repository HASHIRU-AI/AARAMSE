"""Meaning-fidelity scoring: what a rewrite cost the question."""

from __future__ import annotations

import pytest

from aaramse.fidelity import (
    FidelityReport,
    MeaningFidelity,
    extract_negations,
    extract_quantities,
)
from aaramse.operators import FrameAssert, FrameConfig


def _judge(answer_type: str = "YES", answerable: str = "YES"):
    """Stub judge returning a fixed two-line verdict, counting its calls."""
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return f"ANSWER_TYPE: {answer_type}\nANSWERABLE: {answerable}"

    generate.calls = calls  # type: ignore[attr-defined]
    return generate


class TestQuantityExtraction:
    def test_extracts_monetary_amounts(self):
        assert "50000" in extract_quantities("Can I withdraw $50,000 from my 401k?")

    def test_extracts_percentages(self):
        assert "7%" in extract_quantities("Is a 7% withdrawal rate sustainable?")

    def test_extracts_bare_numbers(self):
        assert "59" in extract_quantities("What happens before age 59?")

    def test_ignores_currency_symbol_and_separators(self):
        assert extract_quantities("$1,200") == extract_quantities("1200 dollars")

    def test_empty_for_text_without_quantities(self):
        assert extract_quantities("What is compound interest?") == frozenset()


class TestNegationExtraction:
    def test_extracts_without(self):
        assert "without" in extract_negations("Can I withdraw without penalty?")

    def test_extracts_exclusion_markers(self):
        assert "excluding" in extract_negations("Total excluding fees?")

    def test_empty_for_plain_text(self):
        assert extract_negations("What is a Roth IRA?") == frozenset()


class TestDeterministicDimensions:
    def test_identical_text_preserves_everything(self):
        report = MeaningFidelity().assess("What is a Roth IRA?", "What is a Roth IRA?")
        assert report.blocking_loss is None
        assert report.score == 1.0

    def test_dropped_quantity_blocks(self):
        report = MeaningFidelity().assess(
            "Can I withdraw $50,000 from my 401k?",
            "Can an investor withdraw from a 401k?",
        )
        assert report.lost_quantities == ("50000",)
        assert report.blocking_loss is not None

    def test_dropped_negation_blocks(self):
        report = MeaningFidelity().assess(
            "Can I withdraw from a 401k without penalty?",
            "Can an investor withdraw from a 401k?",
        )
        assert report.lost_negations == ("without",)
        assert report.blocking_loss is not None

    def test_dropped_domain_term_scores_down_but_does_not_block(self):
        # IntentGuard owns blocking on topic drift; fidelity only prices it.
        report = MeaningFidelity().assess(
            "How does a Roth IRA differ from an annuity?",
            "How does a Roth IRA work?",
        )
        assert "annuity" in report.lost_terms
        assert report.blocking_loss is None
        assert report.score < 1.0

    def test_deployer_frame_costs_no_fidelity(self):
        # FRAME_ASSERT prepends config text and leaves the payload byte-identical,
        # so the frame's own authorisation reference must not read as a quantity.
        query = "Can I withdraw $50,000 from my 401k without penalty?"
        framed = FrameAssert(FrameConfig(authorisation_ref="FRN-123456")).apply(query)
        report = MeaningFidelity().assess(query, framed.after)
        assert report.blocking_loss is None
        assert report.score == 1.0


class TestJudgedDimensions:
    def test_judge_not_called_without_generate(self):
        report = MeaningFidelity().assess("What is an ETF?", "What is a fund?")
        assert report.answer_type_preserved is None
        assert report.answerable is None

    def test_judge_verdicts_are_recorded(self):
        judge = _judge(answer_type="NO", answerable="YES")
        report = MeaningFidelity(generate=judge).assess(
            "How do I rebalance a portfolio?", "What is portfolio rebalancing?"
        )
        assert report.answer_type_preserved is False
        assert report.answerable is True

    def test_judged_failure_lowers_score_below_deterministic_only(self):
        good = MeaningFidelity(generate=_judge()).assess("What is an ETF?", "What is an ETF?")
        bad = MeaningFidelity(generate=_judge(answer_type="NO", answerable="NO")).assess(
            "What is an ETF?", "What is an ETF?"
        )
        assert bad.score < good.score

    def test_unparseable_verdict_fails_closed_without_blocking(self):
        report = MeaningFidelity(generate=lambda _: "maybe?").assess(
            "What is an ETF?", "What is an ETF?"
        )
        assert report.answer_type_preserved is False
        assert report.blocking_loss is None

    def test_judge_is_cached_across_repeat_assessments(self):
        judge = _judge()
        fidelity = MeaningFidelity(generate=judge)
        fidelity.assess("What is an ETF?", "What is a fund?")
        fidelity.assess("What is an ETF?", "What is a fund?")
        assert len(judge.calls) == 1

    def test_judge_asked_once_for_both_dimensions(self):
        judge = _judge()
        MeaningFidelity(generate=judge).assess("What is an ETF?", "What is a fund?")
        assert len(judge.calls) == 1

    def test_judge_is_not_called_when_a_deterministic_loss_already_blocks(self):
        judge = _judge()
        MeaningFidelity(generate=judge).assess(
            "Can I withdraw $50,000?", "Can an investor withdraw?"
        )
        assert judge.calls == []


class TestReporting:
    def test_report_serialises_for_the_audit_log(self):
        report = MeaningFidelity(generate=_judge()).assess(
            "Can I withdraw from a 401k?", "Can an investor withdraw from a 401k?"
        )
        data = report.to_dict()
        assert data["score"] == pytest.approx(report.score)
        assert set(data) >= {
            "score", "lost_terms", "lost_quantities", "lost_negations",
            "answer_type_preserved", "answerable",
        }

    def test_score_is_a_fraction_of_assessed_dimensions(self):
        report = FidelityReport(
            lost_terms=("annuity",),
            lost_quantities=(),
            lost_negations=(),
            answer_type_preserved=True,
            answerable=True,
        )
        # subject lost, constraints kept, both judged dimensions kept -> 3 of 4.
        assert report.score == pytest.approx(0.75)
