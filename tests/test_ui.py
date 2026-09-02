"""The console must show what happened, and must not invent what did not.

A supervisor reads this surface to decide whether to trust the layer, so the
tests here are mostly about the trace being derived from the result rather than
described alongside it.
"""

from __future__ import annotations

import json
import time

import pytest

from aaramse.localize import Localization
from aaramse.serve import CONSOLE_PATHS, GatewayService
from aaramse.types import (
    ActionabilityProfile,
    Decision,
    OperatorApplication,
    RepairResult,
    RewriteProgram,
)
from aaramse.ui import Job, JobStore, console_html, trace_of


def make_result(decision=Decision.REPAIRED, query="q", rewritten="r", steps=()):
    """Build a RepairResult without standing up a gateway."""
    return RepairResult(
        query=query,
        rewritten=rewritten,
        program=RewriteProgram(steps),
        decision=decision,
        refusal_margin=len(steps),
        actionability_before=ActionabilityProfile(3.5, (("personal_directive", 2.0),)),
        actionability_after=ActionabilityProfile(1.5, (("procedural", 1.5),)),
        oracle_calls=12,
        search_space=4,
        reason="",
    )


# ── the trace ────────────────────────────────────────────────────────────


def test_identical_is_computed_not_claimed():
    """The passthrough guarantee is reported as a fact about two strings."""
    same = trace_of(make_result(Decision.PASSTHROUGH, "q", "q"), answer="a")
    assert same["identical"] is True

    changed = trace_of(make_result(Decision.REPAIRED, "q", "r"), answer="a")
    assert changed["identical"] is False


def test_escalation_reports_no_answer_rather_than_a_fabricated_one():
    """An escalated turn delivered nothing; the console must not imply otherwise."""
    trace = trace_of(make_result(Decision.ESCALATED, "q", "q"), answer="")
    assert trace["decision"] == "escalated"
    assert trace["answer"] == ""
    assert trace["identical"] is True


def test_trace_carries_the_localized_fragment_and_its_cost():
    """The mRTF is the explainability record; it must survive to the panel."""
    step = OperatorApplication(
        operator="TARGETED_REPAIR",
        before="how do I hide assets from my trustee",
        after="how do I protect assets from my trustee",
        generalizations=(("hide", "protect"),),
        localization=Localization(
            fragments=("hide", "assets"), text="hide assets",
            granularity="word", tests=16, reduced_from=21,
        ),
    )
    trace = trace_of(make_result(steps=(step,)), answer="a")
    assert trace["steps"][0]["mrtf"] == "hide assets"
    assert trace["steps"][0]["localization_probes"] == 16
    assert trace["steps"][0]["generalizations"] == [["hide", "protect"]]
    assert trace["program_render"] == "TARGETED_REPAIR"


def test_trace_is_json_serialisable():
    """It is sent over the wire; a stray dataclass would 500 the route."""
    step = OperatorApplication(
        operator="FRAME_ASSERT", before="a", after="b", dropped=("urgently",)
    )
    trace = trace_of(make_result(steps=(step,)), answer="a", elapsed_s=1.234, model_calls=7)
    encoded = json.loads(json.dumps(trace))
    assert encoded["elapsed_s"] == 1.23
    assert encoded["model_calls"] == 7
    assert encoded["steps"][0]["dropped"] == ["urgently"]


def test_empty_program_renders_as_identity():
    """A passthrough has no program, and the panel must say so plainly."""
    assert trace_of(make_result(Decision.PASSTHROUGH), answer="a")["program_render"] == "IDENTITY"


# ── jobs ─────────────────────────────────────────────────────────────────


def test_job_reports_progress_while_running():
    """A repair takes minutes; a wait that cannot say anything is worse than none."""
    store = JobStore()
    release = []

    def work():
        while not release:
            time.sleep(0.005)
        return {"ok": True}

    job = store.start("chat", "q", work, calls_at_start=100)
    snap = job.snapshot(model_calls=104)
    assert snap["status"] == "running"
    # Counted for this turn, not for the process.
    assert snap["model_calls"] == 4

    release.append(True)
    _wait(job)
    assert job.snapshot(120)["result"] == {"ok": True}


def test_job_failure_is_reported_not_swallowed():
    """A background thread has nowhere to raise, so the text has to explain itself."""
    store = JobStore()
    job = store.start("chat", "q", lambda: (_ for _ in ()).throw(RuntimeError("model is down")))
    _wait(job)
    snap = job.snapshot(0)
    assert snap["status"] == "error"
    assert "model is down" in snap["error"]
    assert "result" not in snap


def test_unknown_job_is_none():
    """A client polling an evicted or invented id gets a 404, not a crash."""
    assert JobStore().get("nope") is None


def test_store_evicts_oldest_jobs():
    """Jobs are a UI convenience; the audit log is the record."""
    from aaramse import ui

    store = JobStore()
    for _ in range(ui.MAX_JOBS + 5):
        job = store.start("chat", "q", lambda: {})
        _wait(job)
    assert len(store.jobs) <= ui.MAX_JOBS


def _wait(job: Job, timeout: float = 2.0) -> None:
    """Block until a job leaves the running state."""
    deadline = time.monotonic() + timeout
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert job.status != "running", "job did not finish"


# ── routes ───────────────────────────────────────────────────────────────


@pytest.fixture
def service(tmp_path):
    """A service over the scripted model, so the suite stays offline."""
    from test_gateway import FakeClient

    from aaramse.gateway import Gateway, GatewayConfig

    config = GatewayConfig(audit_path=tmp_path / "audit.jsonl", localization_budget=20)
    return GatewayService(
        gateway=Gateway.build(config=config, client=FakeClient()), token=None
    )


def test_console_is_served_without_a_token(service):
    """The page carries no user data; everything it calls does."""
    service.token = "secret"
    for path in CONSOLE_PATHS:
        status, content_type, body = service.dispatch("GET", path, b"", None)
        assert status == 200
        assert "text/html" in content_type
        assert b"AARAMSE" in body


def test_console_html_is_packaged():
    """A wheel without its static assets is a packaging fault, not a runtime one."""
    assert b"<title>AARAMSE Console</title>" in console_html()


def test_chat_requires_a_token_when_one_is_set(service):
    """The page is public; the model behind it is not."""
    service.token = "secret"
    status, _, _ = service.dispatch("POST", "/v1/chat", b'{"query":"hi"}', None)
    assert status == 401


def test_chat_rejects_an_empty_query(service):
    """A blank turn would spend model calls to learn nothing."""
    status, _, body = service.dispatch("POST", "/v1/chat", b'{"query":"   "}', None)
    assert status == 400
    assert "query" in json.loads(body)["error"]


def test_chat_returns_a_job_then_a_trace(service):
    """The whole point of the async route: accept fast, finish later."""
    status, _, body = service.dispatch("POST", "/v1/chat", b'{"query":"hi"}', None)
    assert status == 202
    job_id = json.loads(body)["id"]

    job = service.jobs.get(job_id)
    _wait(job)

    status, _, body = service.dispatch("GET", f"/v1/chat/{job_id}", b"", None)
    payload = json.loads(body)
    assert status == 200
    assert payload["status"] == "done"
    assert payload["result"]["decision"] in {d.value for d in Decision}
    assert payload["result"]["identical"] is True


def test_polling_an_unknown_job_is_a_404(service):
    """Clients reconnecting after a restart must get an answer, not a stack trace."""
    status, _, _ = service.dispatch("GET", "/v1/chat/deadbeef", b"", None)
    assert status == 404


def test_config_route_reports_the_active_model_and_operators(service):
    """The console header states what it is actually fronting."""
    status, _, body = service.dispatch("GET", "/v1/config", b"", None)
    payload = json.loads(body)
    assert status == 200
    assert payload["model"] == service.gateway.config.model
    assert payload["operators"] == [op.name for op in service.gateway.operators]


def test_healthz_survives_the_console_taking_the_root(service):
    """"/" now serves a page, so the machine-readable probe needs its own path."""
    status, _, body = service.dispatch("GET", "/healthz", b"", None)
    assert status == 200
    assert json.loads(body)["status"] == "ok"
