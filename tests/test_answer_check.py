"""A repair that gets an answer to a different question is not a repair.

The re-probe historically asked only "did the refusal boilerplate go away?".
That is the criterion a jailbreak loop optimises, and it is also the criterion a
meaning-destroying rewrite passes trivially: ask something easier, get an
answer, score a success.
"""

from __future__ import annotations

from typing import List

from aaramse.fidelity import AnswerCheck
from aaramse.refusal import ResponseVerdict
from aaramse.search import RepairSearch, SearchConfig
from aaramse.types import Decision, SearchDiagnostics

QUERY = "Should I dump my 401(k) into gold before the crash?"


class _Probe:
    """Refuses the original, answers every rewrite."""

    def __init__(self) -> None:
        self.n = 0

    def probe(self, prompt: str) -> ResponseVerdict:
        self.n += 1
        if self.n == 1:
            return ResponseVerdict(refused=True, delivers_procedure=False, text="I can't help")
        return ResponseVerdict(
            refused=False, delivers_procedure=False, text="Gold is a commodity."
        )


def _check(verdict: str, seen: List[str] | None = None) -> AnswerCheck:
    """An answer check whose judge always returns the same verdict."""

    def generate(prompt: str) -> str:
        if seen is not None:
            seen.append(prompt)
        return verdict

    return AnswerCheck(generate=generate)


class TestAnswerCheck:
    def test_yes_means_the_original_was_answered(self):
        assert _check("YES").answers("What is an ETF?", "An ETF is a fund.") is True

    def test_no_means_it_was_not(self):
        assert _check("NO").answers("What is an ETF?", "Gold is a commodity.") is False

    def test_unparseable_verdict_fails_closed(self):
        # Escalating sends a human a query the model had already refused; the
        # cost of the conservative choice here is attention, not a wrong answer.
        assert _check("perhaps").answers("What is an ETF?", "...") is False

    def test_verdicts_are_cached(self):
        seen: List[str] = []
        check = _check("YES", seen)
        check.answers("What is an ETF?", "An ETF is a fund.")
        check.answers("What is an ETF?", "An ETF is a fund.")
        assert len(seen) == 1

    def test_the_judge_sees_both_the_original_and_the_reply(self):
        seen: List[str] = []
        _check("YES", seen).answers("What is an ETF?", "An ETF is a fund.")
        assert "What is an ETF?" in seen[0]
        assert "An ETF is a fund." in seen[0]


class TestSearchIntegration:
    def test_search_has_no_answer_check_by_default(self, operators):
        """Existing deployments must not change behaviour by upgrading."""
        search = RepairSearch(operators, oracle=None, probe=_Probe())
        assert search.answer_check is None

    def test_repair_stands_when_the_answer_addresses_the_original(self, operators):
        search = RepairSearch(
            operators, oracle=None, config=SearchConfig(max_depth=3),
            probe=_Probe(), answer_check=_check("YES"),
        )
        assert search.repair(QUERY).decision is Decision.REPAIRED

    def test_repair_is_rejected_when_the_answer_is_to_another_question(self, operators):
        search = RepairSearch(
            operators, oracle=None, config=SearchConfig(max_depth=3),
            probe=_Probe(), answer_check=_check("NO"),
        )
        result = search.repair(QUERY)
        assert result.decision is Decision.ESCALATED
        assert "answered a different question" in result.reason

    def test_the_original_query_is_returned_untouched_on_that_escalation(self, operators):
        search = RepairSearch(
            operators, oracle=None, config=SearchConfig(max_depth=3),
            probe=_Probe(), answer_check=_check("NO"),
        )
        assert search.repair(QUERY).rewritten == QUERY

    def test_diagnostics_count_the_off_target_answers(self, operators):
        search = RepairSearch(
            operators, oracle=None, config=SearchConfig(max_depth=3),
            probe=_Probe(), answer_check=_check("NO"),
        )
        diagnostics = search.repair(QUERY).diagnostics
        assert diagnostics is not None
        assert diagnostics.answered_other > 0

    def test_the_judge_is_asked_about_the_original_not_the_rewrite(self, operators):
        seen: List[str] = []
        search = RepairSearch(
            operators, oracle=None, config=SearchConfig(max_depth=3),
            probe=_Probe(), answer_check=_check("YES", seen),
        )
        search.repair(QUERY)
        assert seen and QUERY in seen[0]


class TestFailureClass:
    def test_off_target_answers_get_their_own_class(self):
        diagnostics = SearchDiagnostics(
            candidates_generated=2, probed_refused=0, answered_other=2
        )
        assert diagnostics.failure_class == "answered_different"

    def test_existing_classes_are_unchanged(self):
        assert SearchDiagnostics(0, 0).failure_class == "no_candidate"
        assert (
            SearchDiagnostics(2, 0, blocked_candidates=("x: y",)).failure_class
            == "guard_blocked"
        )
        assert SearchDiagnostics(2, 2).failure_class == "model_upheld"
