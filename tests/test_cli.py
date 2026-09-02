"""The CLI is the container's entry point, so its defaults are deployment policy."""

from __future__ import annotations

import json

import pytest

from aaramse.__main__ import build_parser, main


def test_certificates_are_required_by_default():
    """Fail-closed must be the default, not an opt-in flag."""
    args = build_parser().parse_args(["repair", "q"])
    assert args.allow_uncertified is False


def test_uncertified_operation_must_be_asked_for():
    """Running an uncertified algebra is a decision someone has to make."""
    args = build_parser().parse_args(["repair", "--allow-uncertified", "q"])
    assert args.allow_uncertified is True


def test_environment_configures_the_container(monkeypatch):
    """A container is configured by environment, not by argv."""
    monkeypatch.setenv("AARAMSE_MODEL", "anthropic:claude-opus-5")
    monkeypatch.setenv("AARAMSE_PORT", "9000")
    monkeypatch.setenv("AARAMSE_ALLOW_UNCERTIFIED", "1")
    args = build_parser().parse_args(["serve"])
    assert args.model == "anthropic:claude-opus-5"
    assert args.port == 9000
    assert args.allow_uncertified is True


def test_flags_beat_the_environment(monkeypatch):
    """An operator overriding on the command line must win."""
    monkeypatch.setenv("AARAMSE_MODEL", "from-env")
    args = build_parser().parse_args(["serve", "--model", "from-flag"])
    assert args.model == "from-flag"


def test_report_on_a_missing_log_fails_loudly(tmp_path, capsys):
    """A silent empty report would read as "nothing happened"."""
    code = main(["report", "--audit", str(tmp_path / "absent.jsonl")])
    assert code == 1
    assert "no audit log" in capsys.readouterr().err


def test_report_renders_markdown_and_json(tmp_path, capsys, search):
    """Both shapes come from the same log."""
    from aaramse.audit import AuditLog

    path = tmp_path / "audit.jsonl"
    AuditLog(path).append(search.repair("What is compound interest?"))

    assert main(["report", "--audit", str(path)]) == 0
    assert "AARAMSE intervention report" in capsys.readouterr().out

    assert main(["report", "--audit", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["total"] == 1
    assert payload["chain_intact"] is True


def test_a_subcommand_is_required():
    """Bare `python -m aaramse` must not start something by accident."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize("argv", [
    ["--log-level", "DEBUG", "repair", "q"],
    ["repair", "--log-level", "DEBUG", "q"],
])
def test_log_level_is_accepted_on_either_side_of_the_subcommand(argv):
    """`aaramse repair --log-level DEBUG q` is the order a person types.

    A shared `parents=` action plus a subparser's fresh namespace made the
    pre-subcommand form silently revert to INFO.
    """
    assert build_parser().parse_args(argv).log_level == "DEBUG"


def test_log_level_is_absent_when_not_given():
    """SUPPRESS leaves it unset; `main` resolves the default."""
    assert not hasattr(build_parser().parse_args(["repair", "q"]), "log_level")


def test_main_runs_without_an_explicit_log_level(tmp_path, capsys):
    """The resolution path must work, not just the parser."""
    assert main(["report", "--audit", str(tmp_path / "absent.jsonl")]) == 1
