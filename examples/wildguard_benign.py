"""Does the adversarial wrapper alone move the refusal boundary?

The WildGuardMix benign sample is built as a matched contrast: 50 plain benign
prompts and 50 benign prompts wrapped in jailbreak structure -- roleplay
personas, "unrestricted environment" clauses, forced response prefixes -- drawn
from the same four harm-adjacent topics and all labelled `unharmful` by
WildGuard's annotators. Neither arm should be refused, and neither should be
intervened on if it was not refused.

That makes one comparison available that no other corpus here supplies. FinQA
measures false intervention on prompts nothing objects to; FalseReject measures
repair on prompts engineered to look unsafe. This measures the *delta between
two arms of the same topic*, where the wrapper is the only thing that changed.
A layer -- or a model -- that treats the two arms differently is keying on the
wrapper, not the request.

    python examples/wildguard_benign.py                       # all 100
    python examples/wildguard_benign.py MODEL 10              # 10 per arm
    python examples/wildguard_benign.py MODEL 10 --arm adversarial

Reported per arm: refusal rate, recovery rate, and recovery misses.
Writes `audit/wildguard_benign.json` plus a hash-chained `.jsonl`.

**Read the refusal rate before the recovery rate.** If the model refused
nothing, every prompt passes through and the run proves only that passthrough
works -- it is not evidence the layer repairs anything. `examples/judge_floor.py`
is the gate that tells you which case you are in.

**This corpus cannot measure false intervention, and does not try.** The layer
returns PASSTHROUGH exactly when its probe found no refusal, so `decision` *is*
the refusal observation -- there is no independent ground truth here to catch
the layer intervening on a query nobody refused. `examples/finqa_control.py`
is the run that measures that, on a corpus chosen because nothing refuses it.
What this run measures is the refusal boundary itself, and whether the
adversarial wrapper moves it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.client import ModelClient, ModelUnavailable
from aaramse.gateway import DEFAULT_SYSTEM_PROMPT, Gateway, GatewayConfig
from aaramse.providers import build_client
from aaramse.types import Decision

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

CORPUS = Path(os.environ.get(
    "WILDGUARD_CORPUS", "data/wildguard_benign_sample.json"))
OUT = Path(os.environ.get("WILDGUARD_OUT", "audit/wildguard_benign.json"))
AUDIT = OUT.with_suffix(".jsonl")
SYSTEM_PROMPT = os.environ.get("AARAMSE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
MODEL = argv[0] if argv else "gemma4:12b"
PER_ARM = int(argv[1]) if len(argv) > 1 else 0  # 0 means the whole arm
ONLY = next((f.split("=", 1)[1] for f in flags if f.startswith("--arm=")), None)
RESUME = "--resume" in flags
# Hosted endpoints meter per minute and a repair fires ~25 calls in a burst.
RETRIES = int(os.environ.get("WILDGUARD_RETRIES", "8"))
PARTIAL = OUT.with_suffix(".partial.jsonl")
EXPECTED = 0  # set once the selection is known


def load_sample() -> List[Dict[str, Any]]:
    """Read the vendored sample, newest build wins."""
    if not CORPUS.exists():
        raise SystemExit(
            f"corpus not found at {CORPUS}; regenerate it with "
            "scripts/build_wildguard_benign.py --sample-out"
        )
    with CORPUS.open("r", encoding="utf-8") as handle:
        return list(json.load(handle)["items"])


def arm_of(item: Dict[str, Any]) -> str:
    """Which arm of the contrast an item belongs to."""
    return "adversarial" if item["adversarial"] else "vanilla"


def select(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the --arm filter and the per-arm cap, preserving file order."""
    chosen: List[Dict[str, Any]] = []
    for arm in ("vanilla", "adversarial"):
        if ONLY and arm != ONLY:
            continue
        bucket = [i for i in items if arm_of(i) == arm]
        chosen.extend(bucket[:PER_ARM] if PER_ARM else bucket)
    return chosen


@dataclass
class RetryingClient:
    """Back off and retry when the backend meters us.

    A hosted endpoint bills per minute while a single repair fires ~25 calls in
    a burst, so a 429 mid-search is routine rather than exceptional. Without
    this the exception unwinds through the whole run and every completed item
    is lost -- the first attempt at this battery died on item 13 of 100 and
    threw away twelve repairs, several of which cost over a minute each.

    Wrapping the client rather than the run is what makes the retry invisible
    to the search: a repair that survives a rate limit is still one repair, and
    the call counter still reports what the model was actually asked.
    """

    inner: ModelClient
    attempts: int = 8
    base_delay: float = 4.0

    @property
    def calls(self) -> int:
        """Delegate, so `report()["model_calls"]` stays truthful."""
        return self.inner.calls

    def _retry(self, call: Any, *args: Any) -> str:
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return str(call(*args))
            except ModelUnavailable as error:
                last = error
                if attempt == self.attempts - 1:
                    break
                delay = self.base_delay * (2 ** attempt)
                print(f"    (backend unavailable, retrying in {delay:.0f}s)", flush=True)
                time.sleep(delay)
        raise last if last else RuntimeError("retry loop exited without result")

    def answer(self, prompt: str, max_tokens: int = 320) -> str:
        return self._retry(self.inner.answer, prompt, max_tokens)

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 60) -> str:
        return self._retry(self.inner.complete, prompt, temperature, max_tokens)

    def reset(self) -> None:
        self.inner.reset()


def load_done() -> Dict[str, Dict[str, Any]]:
    """Records already completed, keyed by prompt, for --resume."""
    if not (RESUME and PARTIAL.exists()):
        return {}
    done: Dict[str, Dict[str, Any]] = {}
    with PARTIAL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                done[record["prompt"]] = record
    return done


def write_result(
    records: List[Dict[str, Any]], started: float, model_calls: int, chain_intact: bool
) -> None:
    """Write the full result file. Called after every item, not just at the end.

    A run of this length is a sequence of expensive, individually meaningful
    measurements, so a crash at item 90 should cost item 90 and nothing else.
    The summary is recomputed from whatever is complete, which means the file
    on disk is always a valid result for the prefix that finished.
    """
    summary = {
        "model": MODEL,
        "corpus": str(CORPUS),
        "temperature": 0.0,
        "complete": len(records) == EXPECTED,
        "system_prompt_is_default": SYSTEM_PROMPT == DEFAULT_SYSTEM_PROMPT,
        "arms": {a: summarise(records, a) for a in ("vanilla", "adversarial", "all")},
        "elapsed_s": round(time.time() - started, 1),
        "model_calls": model_calls,
        "chain_intact": chain_intact,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n")


def summarise(records: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    """Per-arm refusal and recovery counts.

    PASSTHROUGH means the probe found no refusal; REPAIRED and ESCALATED are
    both downstream of one. So refusals are counted from the decision, not from
    `refusal_margin` -- that field is the length of the rewrite program, which
    is 0 for an escalation and would score an unrepairable refusal as "not
    refused".
    """
    rows = [r for r in records if r["arm"] == arm] if arm != "all" else records
    if not rows:
        return {"n": 0}
    refused = [r for r in rows if r["refused"]]
    repaired = [r for r in refused if r["decision"] == "repaired"]
    return {
        "n": len(rows),
        "refused_by_model": len(refused),
        "refusal_rate": round(len(refused) / len(rows), 4),
        "recovered": len(repaired),
        "recovery_misses": len(refused) - len(repaired),
        "recovery_rate": round(len(repaired) / len(refused), 4) if refused else None,
        "refusal_rate_by_subcategory": {
            s: round(
                sum(1 for r in rows if r["subcategory"] == s and r["refused"])
                / sum(1 for r in rows if r["subcategory"] == s), 4
            )
            for s in sorted({r["subcategory"] for r in rows})
        },
        "by_decision": {
            d: sum(1 for r in rows if r["decision"] == d)
            for d in sorted({r["decision"] for r in rows})
        },
    }


def main() -> int:
    """Run both arms through the gateway and report the per-arm contrast."""
    global EXPECTED
    items = select(load_sample())
    if not items:
        raise SystemExit("no items selected")
    EXPECTED = len(items)
    done = load_done()
    if not RESUME:
        for stale in (AUDIT, PARTIAL):
            if stale.exists():
                stale.unlink()
    AUDIT.parent.mkdir(parents=True, exist_ok=True)

    # `send_temperature` is off by default because recent OpenAI and Anthropic
    # models reject any temperature but their own -- so nothing was being sent
    # and the endpoint applied its own default. That is what made the same
    # prompt repair on one run and escalate on the next. NIM accepts 0.0 and is
    # reproducible under it, so an experiment asks for it explicitly.
    client = RetryingClient(
        inner=build_client(
            MODEL, system_prompt=SYSTEM_PROMPT, send_temperature=True
        ),
        attempts=RETRIES,
    )
    gateway = Gateway.build(
        config=GatewayConfig(model=MODEL, audit_path=AUDIT, system_prompt=SYSTEM_PROMPT),
        client=client,
    )
    records: List[Dict[str, Any]] = []
    started = time.time()
    if done:
        print(f"resuming: {len(done)} of {len(items)} already done\n", flush=True)

    for index, item in enumerate(items, 1):
        if item["prompt"] in done:
            records.append(done[item["prompt"]])
            print(f"[{index}/{len(items)}] {arm_of(item):<11} "
                  f"{done[item['prompt']]['decision']:<12} (cached)", flush=True)
            continue
        began = time.time()
        result = gateway.handle(item["prompt"])
        identical = result.rewritten == item["prompt"]
        # PASSTHROUGH is returned only when the probe found no refusal; every
        # other decision is downstream of one.
        refused = result.decision is not Decision.PASSTHROUGH
        records.append({
            "arm": arm_of(item),
            "subcategory": item["subcategory"],
            "prompt": item["prompt"],
            # The rewrite is the artifact of the repair, so it belongs in the
            # result rather than only in the hash-chained audit log. Under the
            # FCA condition FRAME_ASSERT prepends a *financial education* frame,
            # which is worth being able to read back against a corpus that is
            # not about finance.
            "rewritten": result.rewritten,
            "decision": result.decision.value,
            "byte_identical": identical,
            "refused": refused,
            "program_length": result.refusal_margin,
            "program": result.program.render(),
            "oracle_calls": result.oracle_calls,
            "elapsed_s": round(time.time() - began, 1),
        })
        with PARTIAL.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
        write_result(records, started, gateway.client.calls, True)
        flag = "" if not refused else (
            "  <-- REFUSED, repaired" if result.decision is Decision.REPAIRED
            else "  <-- REFUSED, unrepaired")
        print(f"[{index}/{len(items)}] {records[-1]['arm']:<11} "
              f"{records[-1]['decision']:<12} {records[-1]['elapsed_s']:>6.1f}s  "
              f"{item['prompt'][:52]!r:<56}{flag}", flush=True)

    report = gateway.report()
    write_result(records, started, report["model_calls"], report["chain_intact"])
    summary = json.loads(OUT.read_text())["summary"]
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {OUT} and {AUDIT}")

    total_refused = sum(1 for r in records if r["refused"])
    if total_refused == 0:
        print(
            "\nNOTE: the model refused nothing in either arm, so every prompt "
            "passed through.\nThis run measures false intervention only. It is "
            "not evidence that the layer\nrepairs anything -- there was nothing "
            "to repair. See examples/judge_floor.py."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
