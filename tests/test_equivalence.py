"""The intent-equivalence judge is the one thing standing behind a rewrite.

`TargetedRepair` refuses to act without it, so its failure modes -- an
unparseable verdict, a repeated call -- are the failure modes of the whole
repair path.
"""

from __future__ import annotations

from aaramse.equivalence import EQUIVALENCE_PROMPT, SemanticEquivalence


def test_equivalence_judge_fails_closed_on_unparseable_output():
    """An ambiguous verdict must not be read as approval."""
    checker = SemanticEquivalence(generate=lambda _: "hmm, hard to say")
    assert not checker.equivalent("a", "b")
    assert SemanticEquivalence(generate=lambda _: "YES").equivalent("a", "b")


def test_equivalence_judge_can_be_told_to_fail_open():
    """Fail-closed is the default, but it is a setting, not a law."""
    checker = SemanticEquivalence(generate=lambda _: "unclear", fail_closed=False)
    assert checker.equivalent("a", "b")


def test_verdict_parsing_is_substring_based_and_leans_negative():
    """A documented sharp edge, asserted so it cannot change silently.

    The verdict is parsed by substring, so any hedge containing "no" -- "no
    idea", "cannot tell" -- reads as NO rather than as unparseable. That lands
    on the safe side (a rewrite is rejected, never admitted) and "yes" is
    checked first, so it costs recall, not safety. Worth knowing before reading
    a judge-rejection rate.
    """
    for hedge in ("no idea", "cannot tell", "not sure"):
        assert not SemanticEquivalence(generate=lambda _, h=hedge: h).equivalent("a", "b")


def test_a_negative_verdict_is_read_as_no():
    """The judge's whole job is being able to say no."""
    checker = SemanticEquivalence(generate=lambda _: "NO")
    assert not checker.equivalent("a", "b")


def test_equivalence_verdicts_are_cached():
    """Judging is a model call; it must not repeat for the same pair."""
    calls = {"n": 0}

    def gen(_: str) -> str:
        calls["n"] += 1
        return "YES"

    checker = SemanticEquivalence(generate=gen)
    checker.equivalent("a", "b")
    checker.equivalent("a", "b")
    assert calls["n"] == 1


def test_cache_is_keyed_on_both_sides_of_the_pair():
    """A cache keyed only on the original would answer for the wrong rewrite."""
    seen = []

    def gen(prompt: str) -> str:
        seen.append(prompt)
        return "YES" if "gold" in prompt else "NO"

    checker = SemanticEquivalence(generate=gen)
    assert checker.equivalent("original", "a question about gold")
    assert not checker.equivalent("original", "a question about something else")
    assert len(seen) == 2


def test_prompt_carries_both_questions():
    """The judge cannot compare what it was not shown."""
    captured = {}

    def gen(prompt: str) -> str:
        captured["prompt"] = prompt
        return "YES"

    SemanticEquivalence(generate=gen).equivalent("ORIGINAL_TEXT", "REWRITTEN_TEXT")
    assert "ORIGINAL_TEXT" in captured["prompt"]
    assert "REWRITTEN_TEXT" in captured["prompt"]
    assert EQUIVALENCE_PROMPT.split("\n")[0] in captured["prompt"]
