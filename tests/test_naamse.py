"""NAAMSE reports are the only corpus we can grow without model access.

The loader's job is to be right about two things a careless reader gets wrong:
prompt sets are seed-scoped, so cross-seed comparison is meaningless; and
reports contain duplicates, so a record count is not a denominator.
"""

from __future__ import annotations

import json

import pytest

from aaramse.naamse import (
    group_by_seed,
    load_directory,
    load_report,
    shared_prompts,
)


def _record(prompt, reply, score=0.0, cluster="C", mutation="explore"):
    """One report record in NAAMSE's shape."""
    return {
        "prompt": prompt,
        "score": score,
        "metadata": {"mutation_type": mutation, "cluster_info": {"label": cluster}},
        "conversation_history": {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]},
    }


def _section(records):
    """A report section."""
    return {"summary": {"total_prompts_tested": len(records)},
            "all_prompts_with_scores_and_history": records}


def _write(tmp_path, model, filename, payload):
    """Write a report under a model directory, as the repo lays them out."""
    directory = tmp_path / model
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_the_combined_shape(tmp_path):
    """`assessment_report_*.json` carries both polarities under one roof."""
    path = _write(tmp_path, "modelA", "assessment_report_a_seed_75687.json", {
        "benign_report": _section([_record("q1", "Here is an answer.")]),
        "adversarial_report": _section([_record("bad", "I can't help with that.")]),
    })
    report = load_report(path)
    assert report.model == "modelA"
    assert report.seed == "75687"
    assert len(report.polarity("benign")) == 1
    assert len(report.polarity("adversarial")) == 1


def test_loads_the_standalone_shape(tmp_path):
    """A standalone file's polarity is only recoverable from its name."""
    benign = _write(tmp_path, "modelB", "final_report_benign_b.json",
                    _section([_record("q", "answer")]))
    adversarial = _write(tmp_path, "modelB", "final_report_adversarial_b.json",
                         _section([_record("bad", "no")]))
    assert load_report(benign).records[0].polarity == "benign"
    assert load_report(adversarial).records[0].polarity == "adversarial"


def test_the_final_assistant_turn_is_the_reply(tmp_path):
    """A multi-turn history must not yield the user's own text."""
    row = _record("q", "first")
    row["conversation_history"]["messages"] += [
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "final"},
    ]
    path = _write(tmp_path, "m", "final_report_benign_m.json", _section([row]))
    assert load_report(path).records[0].reply == "final"


def test_list_prompts_are_joined(tmp_path):
    """NAAMSE sometimes emits the prompt as a list of parts."""
    path = _write(tmp_path, "m", "final_report_benign_m.json",
                  _section([_record(["part one", "part two"], "answer")]))
    assert load_report(path).records[0].prompt == "part one part two"


def test_duplicate_prompts_are_not_a_denominator(tmp_path):
    """One real file held 63 unique prompts in 80 records."""
    path = _write(tmp_path, "m", "final_report_benign_m.json",
                  _section([_record("same", "a"), _record("same", "a"), _record("other", "b")]))
    report = load_report(path)
    assert len(report.records) == 3
    assert len(report.prompts("benign")) == 2


def test_seed_is_parsed_or_empty(tmp_path):
    """Seed drives comparability, so an absent one must not be invented."""
    with_seed = _write(tmp_path, "m", "final_report_benign_m_seed_4547331533.json",
                       _section([_record("q", "a")]))
    without = _write(tmp_path, "n", "final_report_benign_n.json", _section([_record("q", "a")]))
    assert load_report(with_seed).seed == "4547331533"
    assert load_report(without).seed == ""


def test_shared_prompts_is_an_intersection_not_a_union(tmp_path):
    """Two models on different prompt sets have nothing to compare.

    Returning a union here would silently produce a cross-model table whose
    rows were answered different questions.
    """
    a = load_report(_write(tmp_path, "a", "final_report_benign_a_seed_1.json",
                           _section([_record("shared", "x"), _record("only-a", "x")])))
    b = load_report(_write(tmp_path, "b", "final_report_benign_b_seed_1.json",
                           _section([_record("shared", "y"), _record("only-b", "y")])))
    assert shared_prompts([a, b]) == ("shared",)


def test_disjoint_reports_share_nothing(tmp_path):
    """Different seeds must yield an empty intersection, not a fallback."""
    a = load_report(_write(tmp_path, "a", "final_report_benign_a_seed_1.json",
                           _section([_record("p1", "x")])))
    b = load_report(_write(tmp_path, "b", "final_report_benign_b_seed_2.json",
                           _section([_record("p2", "y")])))
    assert shared_prompts([a, b]) == ()


def test_grouping_by_seed_separates_cohorts(tmp_path):
    """Comparability is per seed; grouping is how a caller respects that."""
    for model, seed in (("a", "1"), ("b", "1"), ("c", "2")):
        _write(tmp_path, model, f"final_report_benign_{model}_seed_{seed}.json",
               _section([_record("q", "a")]))
    groups = group_by_seed(load_directory(tmp_path))
    assert {s: len(g) for s, g in groups.items()} == {"1": 2, "2": 1}


def test_a_malformed_file_does_not_lose_the_corpus(tmp_path):
    """One bad export must not take out nineteen good ones."""
    _write(tmp_path, "good", "final_report_benign_good.json", _section([_record("q", "a")]))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "final_report_benign_bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "final_report_benign_empty.json").write_text("{}", encoding="utf-8")
    assert len(load_directory(tmp_path)) == 1


def test_unrecognised_payload_is_rejected(tmp_path):
    """Silently returning zero records would read as "this model never refused"."""
    path = _write(tmp_path, "m", "final_report_benign_m.json", {"something": "else"})
    with pytest.raises(ValueError, match="no NAAMSE report section"):
        load_report(path)


def test_missing_reply_does_not_crash(tmp_path):
    """A truncated record must load with an empty reply, not raise."""
    row = _record("q", "a")
    row["conversation_history"] = {"messages": [{"role": "user", "content": "q"}]}
    path = _write(tmp_path, "m", "final_report_benign_m.json", _section([row]))
    assert load_report(path).records[0].reply == ""


def test_harness_failures_are_not_model_behaviour(tmp_path):
    """The published corpus has 341 identical error strings and 87 empty replies.

    A model whose every reply is an error once scored 0% over-refusal and read
    as the most permissive in a league table. Dropping them is not tidiness.
    """
    from aaramse.naamse import is_generation_error

    assert is_generation_error("I apologize, but an error occurred while generating a response.")
    assert is_generation_error("")
    assert is_generation_error("   ")
    assert is_generation_error("Error: upstream timeout")


def test_prose_about_errors_is_kept(tmp_path):
    """An unanchored match would delete replies that merely discuss an error."""
    from aaramse.naamse import is_generation_error

    assert not is_generation_error("Here is why an error occurred in your calculation: ...")
    assert not is_generation_error("As an author, trial and error is part of the craft.")


def test_unusable_records_are_excluded_by_default(tmp_path):
    """A rate computed over harness failures is not a measurement."""
    path = _write(tmp_path, "m", "final_report_benign_m.json", _section([
        _record("good", "Here is an answer."),
        _record("bad", "I apologize, but an error occurred while generating a response."),
        _record("empty", ""),
    ]))
    report = load_report(path)
    assert len(report.records) == 3
    assert len(report.polarity("benign")) == 1
    assert len(report.polarity("benign", usable_only=False)) == 3
    assert report.prompts("benign") == ("good",)


def test_a_wholly_failed_model_yields_nothing(tmp_path):
    """llama-4-maverick returned 80/80 errors; it must not look like a model that never refuses."""
    path = _write(tmp_path, "dead", "final_report_benign_dead_seed_1.json", _section([
        _record(f"q{i}", "I apologize, but an error occurred while generating a response.")
        for i in range(5)
    ]))
    report = load_report(path)
    assert report.polarity("benign") == ()
    assert report.prompts("benign") == ()
