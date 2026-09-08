"""Console backend: transforms repair decisions into inspectable compliance traces.

Provides the backend logic for the interactive web console. While automated agents
only need a simple rewritten string, compliance supervisors need full transparency
into the evidence behind every decision.

### Key capabilities:

* **Asynchronous job store:** Because evaluating multiple repair candidates against a
  live model requires several calls and can take up to a few minutes, chat interactions
  are executed as asynchronous background jobs. The console returns immediately and
  polls progress, exposing elapsed runtime and live model-call counts.
* **Inspectable decision traces:** Formats the underlying evidence for every turn —
  including the operator program, actionability profiles, baseline vs. repaired queries,
  and cryptographic hash checks — into a human-readable interface.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .types import RepairResult

__all__ = [
    "MAX_JOBS",
    "Job",
    "JobStore",
    "attempts_of",
    "console_html",
    "trace_of",
]

logger = logging.getLogger(__name__)

# Jobs are a UI convenience, not a record; the audit log is the record. Keeping
# the newest few bounds memory on a long-lived server.
MAX_JOBS = 64

_STATIC = Path(__file__).parent / "static"


def console_html() -> bytes:
    """Return the console page.

    Raises:
        FileNotFoundError: When the package was installed without its static
            assets, which is a packaging fault and should say so.
    """
    return (_STATIC / "index.html").read_bytes()


def attempts_of(operators: Any) -> List[Dict[str, Any]]:
    """Report what each operator tried this turn, including when it lost.

    The search returns the winning program and nothing else, so an operator
    that localized a fragment, proposed a substitution and was then beaten by a
    cheaper rewrite leaves no trace at all. On a live model FRAME_ASSERT wins
    most turns, which means a reader watching the console sees a prefix being
    prepended and never learns that the confined operator ran -- and the
    confined operator is the interesting half of the method.

    Args:
        operators: The gateway's operator set, after the turn.

    Returns:
        One entry per operator that has something to report, oldest first.
        Operators that recorded nothing are omitted, because a panel of empty
        sections teaches a reader to stop looking at it.
    """
    attempts: List[Dict[str, Any]] = []
    for operator in operators:
        localization = getattr(operator, "last_localization", None)
        rejected = list(getattr(operator, "rejected", ()) or ())
        proposal = getattr(operator, "last_proposal", None)
        if localization is None and not rejected and proposal is None:
            continue
        attempts.append({
            "operator": getattr(operator, "name", type(operator).__name__),
            "mrtf": getattr(localization, "text", None),
            "localization_probes": getattr(localization, "tests", None),
            "rejected": [[text, reason] for text, reason in rejected],
            "proposed": {
                "after": proposal.after,
                "substitutions": [list(pair) for pair in proposal.generalizations],
            } if proposal is not None else None,
        })
    return attempts


def trace_of(
    result: RepairResult,
    answer: str,
    baseline: Optional[Dict[str, Any]] = None,
    audit: Optional[Dict[str, Any]] = None,
    elapsed_s: float = 0.0,
    model_calls: int = 0,
    attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble everything a reader needs to judge one decision.

    Args:
        result: What the search decided.
        answer: The reply the user actually receives.
        baseline: How the model answered the untouched query, when known.
        audit: The hash-chain record this decision produced.
        elapsed_s: Wall-clock seconds the turn took.
        model_calls: Model calls the turn consumed.
        attempts: What each operator tried, from `attempts_of`. Included so a
            losing operator is still visible; the winning program alone hides
            the confined rewrite behind whichever operator was cheaper.

    Returns:
        A JSON-serialisable trace. `identical` is the passthrough guarantee
        stated as a fact about these two strings, not as a claim about the code.
    """
    steps: List[Dict[str, Any]] = []
    for step in result.program.steps:
        entry: Dict[str, Any] = {
            "operator": step.operator,
            "before": step.before,
            "after": step.after,
            "generalizations": [list(g) for g in step.generalizations],
            "dropped": list(step.dropped),
        }
        if step.localization is not None:
            entry["mrtf"] = step.localization.text
            entry["localization_probes"] = step.localization.tests
        steps.append(entry)

    return {
        "decision": result.decision.value,
        "query": result.query,
        "rewritten": result.rewritten,
        # The whole passthrough claim, checked rather than asserted.
        "identical": result.rewritten == result.query,
        "answer": answer,
        "baseline": baseline or {},
        "program": list(result.program.names),
        "program_render": result.program.render(),
        "steps": steps,
        "refusal_margin": result.refusal_margin,
        "oracle_calls": result.oracle_calls,
        "search_space": result.search_space,
        "reason": result.reason,
        "actionability_before": {
            "score": result.actionability_before.score,
            "features": [list(f) for f in result.actionability_before.features],
        },
        "actionability_after": {
            "score": result.actionability_after.score,
            "features": [list(f) for f in result.actionability_after.features],
        },
        "audit": audit or {},
        "elapsed_s": round(elapsed_s, 2),
        "model_calls": model_calls,
        # What each operator tried, including operators that lost. Without this
        # a reader only ever sees the winner.
        "attempts": attempts or [],
    }


@dataclass
class Job:
    """One unit of slow work, observable while it runs.

    Attributes:
        id: Opaque handle the client polls on.
        kind: What is running, so the console can label the wait.
        query: The text this job is about, for display.
        status: "running", "done", or "error".
        started: Monotonic start time.
        finished: Monotonic end time, once set.
        result: The payload, on success.
        error: A human-readable failure, on error.
        calls_at_start: Model-call counter when the job began, so progress is
            reported for this turn rather than for the process.
    """

    id: str
    kind: str
    query: str
    status: str = "running"
    started: float = field(default_factory=time.monotonic)
    finished: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    calls_at_start: int = 0

    def snapshot(self, model_calls: int) -> Dict[str, Any]:
        """Return the job's current state.

        Args:
            model_calls: The client's current cumulative call count.

        Returns:
            A JSON-serialisable view, including live elapsed and call counts
            while the job is still running.
        """
        end = self.finished if self.finished is not None else time.monotonic()
        payload: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "query": self.query,
            "status": self.status,
            "elapsed_s": round(end - self.started, 1),
            "model_calls": max(0, model_calls - self.calls_at_start),
        }
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass
class JobStore:
    """Runs slow work on threads and keeps the newest results.

    Attributes:
        jobs: Live and recently finished jobs, oldest first.
        lock: Guards `jobs`. The work itself serialises on the service lock,
            not this one.
    """

    jobs: "Dict[str, Job]" = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def start(
        self,
        kind: str,
        query: str,
        work: Callable[[], Dict[str, Any]],
        calls_at_start: int = 0,
    ) -> Job:
        """Begin a job on a daemon thread and return it immediately.

        Args:
            kind: Label for the work, e.g. "chat" or "model".
            query: Text the job is about, for display.
            work: The callable to run. Its return value becomes the result.
            calls_at_start: Model-call counter at submission time.

        Returns:
            The job, already running.
        """
        job = Job(id=uuid.uuid4().hex[:16], kind=kind, query=query,
                  calls_at_start=calls_at_start)
        with self.lock:
            self.jobs[job.id] = job
            self._evict()

        def run() -> None:
            try:
                job.result = work()
                job.status = "done"
            except Exception as error:  # - reported, not swallowed
                # A background thread has nowhere to raise. The console shows
                # this text, so it has to be the thing that explains itself.
                logger.exception("job %s (%s) failed", job.id, kind)
                job.error = f"{type(error).__name__}: {error}"
                job.status = "error"
            finally:
                job.finished = time.monotonic()

        threading.Thread(target=run, name=f"aaramse-{kind}-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Return a job by id, or None when it is unknown or evicted."""
        with self.lock:
            return self.jobs.get(job_id)

    def _evict(self) -> None:
        """Drop the oldest finished jobs once the store is over its cap."""
        while len(self.jobs) > MAX_JOBS:
            oldest = min(self.jobs.values(), key=lambda j: j.started)
            del self.jobs[oldest.id]


def baseline_of(probe: Any, query: str) -> Dict[str, Any]:
    """Report how the model answered the untouched query, if the probe saw it.

    The console's first question is "was this refused at all", and the search
    has already asked it. Reading the probe's cache answers it for free rather
    than spending another model call to ask again.

    Args:
        probe: A `JudgedProbe`, or anything with `seen` and `answers` caches.
        query: The original query.

    Returns:
        The label and reply for the untouched query, or an empty mapping when
        the probe never saw it.
    """
    label = getattr(probe, "seen", {}).get(query)
    if label is None:
        return {}
    return {
        "label": getattr(label, "value", str(label)),
        "refused": getattr(label, "is_over_refusal", label.value == "full_refusal"),
        "text": getattr(probe, "answers", {}).get(query, ""),
    }
