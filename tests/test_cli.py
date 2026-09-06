"""The CLI is the container's entry point, so its defaults are deployment policy."""

from __future__ import annotations

import json
import os
import signal
import threading

import pytest

from aaramse.__main__ import (
    _shutdown_event,
    _wait_for_shutdown,
    build_parser,
    main,
)


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


def test_shutdown_wait_does_not_return_until_requested():
    """signal.pause() returns on any interruption, so waiting must be a loop.

    LiteLLM's lazy first-call import interrupts pause() within ~2s, which shut
    the sidecar down on the first turn with exit code 0 and no log line.
    """
    stopping = threading.Event()
    returned = threading.Event()

    def wait() -> None:
        _wait_for_shutdown(stopping, poll=0.01)
        returned.set()

    threading.Thread(target=wait, daemon=True).start()
    assert not returned.wait(0.3), "the wait returned without a shutdown request"


def test_shutdown_wait_returns_once_requested():
    """A real SIGINT/SIGTERM must still stop the server promptly."""
    stopping = threading.Event()
    returned = threading.Event()

    def wait() -> None:
        _wait_for_shutdown(stopping, poll=0.01)
        returned.set()

    threading.Thread(target=wait, daemon=True).start()
    stopping.set()
    assert returned.wait(2.0), "the wait did not return after shutdown was requested"


def test_sigterm_requests_shutdown():
    """The handler must record the request, not shut the server down inline."""
    previous = signal.getsignal(signal.SIGTERM)
    try:
        stopping = _shutdown_event()
        os.kill(os.getpid(), signal.SIGTERM)
        assert stopping.wait(2.0), "SIGTERM did not request shutdown"
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_localization_budget_defaults_small_for_the_console():
    """Localization probes dominate a repair's call burst.

    The library default is 32, which is a measurement setting. A hosted free
    tier throttles inside one repair at that size, so the deployment entry
    point ships the value examples/serve_demo.py already demo-tuned to.
    """
    args = build_parser().parse_args(["serve"])
    assert args.localization_budget == 8


def test_localization_budget_can_be_raised():
    """A smaller burst buys latency at the cost of a coarser fragment."""
    args = build_parser().parse_args(["serve", "--localization-budget", "32"])
    assert args.localization_budget == 32


def test_localization_budget_reads_the_environment(monkeypatch):
    """A container is configured by environment, not by argv."""
    monkeypatch.setenv("AARAMSE_LOCALIZATION_BUDGET", "16")
    args = build_parser().parse_args(["repair", "q"])
    assert args.localization_budget == 16


def test_answer_verification_is_on_for_deployment():
    """A repair the user cannot read is not a repair.

    FRAME_ASSERT cleared a prohibited query on one judge sample and the turn
    was logged `repaired`, while the reply the user received was still a
    refusal. AnswerCheck reads what the user would actually get, and returned
    False on that exact pair.
    """
    assert build_parser().parse_args(["serve"]).verify_answers is True


def test_answer_verification_can_be_turned_off():
    """It moves the recovery rate, so an evaluation must be able to opt out."""
    args = build_parser().parse_args(["serve", "--no-verify-answers"])
    assert args.verify_answers is False


def test_answer_verification_reads_the_environment(monkeypatch):
    """A container is configured by environment, not by argv."""
    monkeypatch.setenv("AARAMSE_VERIFY_ANSWERS", "0")
    assert build_parser().parse_args(["repair", "q"]).verify_answers is False


def test_default_model_is_the_preloaded_console_model():
    """The console ships pointed at a model, not at whatever is on localhost."""
    assert build_parser().parse_args(["serve"]).model == "meta/muse-spark-1.2"
