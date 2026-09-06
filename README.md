# AARAMSE — auditable over-refusal repair for regulated AI advice

A model-agnostic middleware layer that detects when a deployed financial agent
refuses a legitimate question, repairs the refusal, and logs every intervention
in a tamper-evident record a supervisor can read.

Every model call goes through LiteLLM, so the provider and model are a spec
string. `python examples/demo.py` runs the whole pipeline offline against a
scripted stand-in model; `--live` drives a real one. `aaramse serve` puts a
chat console in front of it.

**This is the MVP branch.** It carries the modules with live measurements
behind them and nothing else. See `main` for the full research tree, including
the three-arm comparison this branch cannot reproduce.

## What it does

```
query ──> probe ──> refused? ──no──> passthrough, untouched
                       │yes
                       ├─ localize the minimal refusal-triggering fragment
                       ├─ repair (edit the fragment, or assert deployer context)
                       ├─ re-probe: did the user actually get an answer?
                       └─ escalate if not, and log either way
```

## Measured results

The headline comparison -- `TARGETED_REPAIR` against `LLM_REPHRASE` against
`FRAME_ASSERT`, on the FalseReject finance slice and OR-Bench-toxic under an FCA
compliance prompt -- lives on `main`, with `examples/three_arm_eval.py` and
`audit/three_arm_eval.json`. It is not reproducible here: the generative
rephrase arm is not part of this branch.

What that comparison established, and why this branch is shaped the way it is:
confinement costs recovery. The mRTF-confined arm cleared 1 of 8 benign
over-refusals and leaked 0 of 10 toxic prompts; the deployer frame cleared 7 of
8 and leaked 2 of 10. Both of those ship. The full-prompt rewrite scored 6 of 8
and 2 of 10 after its instruction was tuned, and 1 of 8 and 0 of 10 before --
the tuning bought the recovery and the leak together, which is a property of a
prompt rather than of a method. It does not ship.

### Leaving benign queries alone

Repair and leakage are two of three obligations. The third is doing nothing to
a query nobody refused, and until it was measured on a real model it was only
asserted against a simulator that could not fail it. `gemma4:12b`, the FinQA
control slice, 120 numerical-reasoning questions over public filings, same FCA
system prompt. Source: `audit/finqa_control.json`.

| | Result |
|---|---|
| False intervention rate | **1/120 — 0.83%** |
| Byte-identity preserved | **120/120 — 100%** (480/480 across four runs) |
| Median passthrough | 10.8s |
| The one escalation | 32.5s — a 3.0x latency multiplier |

Byte-identity is the guarantee, and it never broke: every prompt came back
identical, escalations included. A false intervention costs latency and a
supervisor's attention, not the user's words.

> **These figures predate the judge's worked examples.** The refusal oracle *is*
> the three-way judge, so strengthening its prompt moves what counts as an
> over-refusal, and therefore moves both rows above. The change was made because
> a live model classified a decline-then-refer-elsewhere reply as a partial
> refusal, which the gateway reads as "answered" and leaves untouched -- the
> layer stood down on a query it exists to repair. `examples/finqa_control.py`
> has not been re-run against the new instrument. Treat this table as the last
> measurement of the old one until it has been.
>
> Two later changes move them further: `--verify-answers` is now on for the
> deployment path, which turns some repairs into escalations, and the harm gate
> now covers concealment from a creditor or trustee, which stops repair being
> attempted on those at all. Both were made to fix wrong outcomes rather than
> to move a number, and neither has been re-measured.

Two thirds of the first measured rate was our own fault. The corpus builder
shipped each item's table and dropped the filing's narrative, leaving 47 of 120
questions unanswerable as shipped; under a compliance prompt an unanswerable
question draws "I am not permitted to advise" rather than "I lack that figure",
which is indistinguishable from a refusal. All 7 initial false interventions
fell in that group and none in the other 73 (Fisher exact p = 0.001). Fixing
the corpus took 5.83% to 0.83%. The measurement instrument was the finding
again, exactly as with the refusal detector.

The single survivor is the genuine one: given grant-date fair values for
2005-2007, project 2008 at the same appreciation. It is arithmetic over a
public filing, it is fully specified, and it is refused. It also *passed*
before the fix — without the numbers the model could not project and said so,
which scores as compliance. Supplying them turned "I can't" into "I won't".

**Not yet established:** how any of this behaves in DDOR's setting — OR-Bench,
no system prompt. That control ran with a starved probe budget (n=12, repair
rate 0.0) and is invalid, so nothing here is comparable to DDOR's reported
51.96% reduction until it is rerun. The script that ran it, `examples/control_eval.py`,
is on `main`; neither it nor its output is carried here.

## Measurement comes first

Binary refusal detection cannot measure over-refusal, and this project proved it
the expensive way — four detector failures, three missing real refusals
(`"I cannot fulfill this request"` among them) and one inventing refusals from
disclaimers. Under a compliance system prompt the dominant behaviour is
**partial refusal**: decline, then answer anyway.

`judge.py` implements XSTest's three-way taxonomy (full compliance / full
refusal / partial refusal), validated 6/6 against hand labels. It is for offline
measurement only and must never sit on the runtime safety path.

## Safety machinery

**Localization** (`localize.py`) — `ddmin` with complement testing and adaptive
partitioning, sentence then word granularity. Reduces a 21-word prompt to
`"hide assets"` in 16 probes; tests assert genuine 1-minimality.

**Confinement is structural** (`targeted.py`) — DDOR instructs a model to edit
only the localized fragment. Here the model returns a replacement phrase and the
substitution happens in code, so everything outside the mRTF is byte-identical
by construction. A test reverses the edit and asserts exact equality.

**Certification** (`certification.py`) — operators are certified against
contrastive pairs on the **deployed model**. A certificate is a property of
*(operator, model, corpus)*, not of the operator: `DEFINITIONALIZE` certified
clean against a simulator and leaked on its first live query. Enforcement is
fail-closed by default.

**Guards** (`invariants.py`) — every candidate must not raise actionability on a
rule-based lattice, and must preserve the propositional core. For generative
rewrites the lexical topic check cannot work, so an intent-equivalence
judgement carries that weight and the operator refuses to run without one.

**Audit** (`audit.py`) — SHA-256 hash-chained JSONL. Records the operator
program, the localized mRTF, each `(fragment -> replacement)` edit, the refusal
margin, and the certificates in force.

## Layout

| Module | Purpose |
|---|---|
| `gateway.py` | The deployable layer: build, certify, handle, report |
| `client.py` | Client protocol, caching, and the raw-HTTP Ollama client |
| `equivalence.py` | Intent-equivalence judge; `TargetedRepair` will not run without it |
| `judge.py` | XSTest three-way response classification |
| `localize.py` | Delta-debugging mRTF localization |
| `targeted.py` | Fragment-confined repair with structural splicing |
| `operators/` | Rule-based operator algebra, registry, deployer frame |
| `search.py` | Bounded shortest-program search, escalation |
| `invariants.py` | Actionability lattice and intent guard |
| `certification.py` | Contrastive certificates |
| `audit.py` | Hash-chained intervention log |
| `serve.py` | HTTP sidecar: console, chat, repair, and report routes |
| `ui.py` | Console backend: decision traces and the job store behind them |
| `static/index.html` | The console itself |
| `__main__.py` | CLI: `serve`, `repair`, `report` |
| `providers.py` | LiteLLM client, native fallbacks, and the model-spec factory |
| `report.py` | Supervisor-facing intervention report |
| `budget.py` | Induced-leakage measurement and fail-closed enforcement |
| `splits.py` | Deterministic held-out splits by content hash |
| `falsereject.py`, `finqa.py` | Vendored benchmark loaders |
| `corpus.py` | Hand-written test fixtures — **not** evidence |

## Usage

```python
from aaramse import Gateway, GatewayConfig

gw = Gateway.build(GatewayConfig(model="gemma4:12b"))
gw.certify(pairs)                 # against the model you deploy in front of
result = gw.handle(user_query)    # repaired, escalated, or passed through
send_to_agent(result.rewritten)   # == user_query unless decision is REPAIRED
print(gw.report())
```

## Choosing a model

Every model call goes through LiteLLM, so the deployed model is a spec string:

```python
Gateway.build(GatewayConfig(model="anthropic:claude-opus-5"))
Gateway.build(GatewayConfig(model="openai:gpt-5"))
Gateway.build(GatewayConfig(model="gemma4:12b"))          # bare tag = Ollama
Gateway.build(GatewayConfig(model="openrouter/meta-llama/llama-3-70b"))
```

This is not a convenience. A certificate is a property of *(operator, model,
corpus)* — `DEFINITIONALIZE` certified clean against a simulator and leaked on
its first live query — so re-running the battery against another model has to be
cheap or it does not happen.

The hand-rolled OpenAI, Anthropic and Ollama clients are still there for a
deployment that cannot take a dependency tree: `build_client(..., backend="native")`,
or `AARAMSE_CLIENT_BACKEND=native`. They speak three providers; LiteLLM speaks
the rest.

Ollama models are sent `think=False`. Without it a reasoning model spends the
whole token budget thinking and returns an empty string, which nothing
downstream can tell apart from a model that answered with nothing.

## Running

```bash
make demo                                  # certify once (cached), serve the console
make demo-offline                          # the whole pipeline, no model, instant
python examples/finqa_control.py           # false-intervention rate, 120 items
python examples/finqa_cause.py             # why each refusal happened
make test                                  # 360 tests
```

`make demo` stands the layer in front of `qwen3.5:4b` under the FCA compliance
prompt, certifies its operators against that model once, caches the
certificates, and serves the console. Ask it *"What is an ETF?"* — a question
the compliance-locked model wrongly refuses — and watch it come back repaired,
with the audit trail behind the verdict. See **[DEMO.md](DEMO.md)** for the
three-minute script.

## The console

```bash
pip install -e .                            # or: export PYTHONPATH=src
aaramse serve --model qwen3.5:4b            # console at http://localhost:8080/
```

A chat window that shows its work. Each turn carries a verdict —
`passthrough`, `repaired`, or `escalated` — and states whether the model refused
at baseline and whether the query reached it byte-identical. When a query was
rewritten, the console shows what was sent instead. Clicking the verdict opens
the trace: the operator program, the localized mRTF and what it cost to find,
every declared substitution, both actionability profiles, the untouched reply,
and the audit hash with its chain check.

A repair runs 23-26 model calls and can take minutes, so a turn is a job:
`POST /v1/chat` returns immediately and the console polls `/v1/chat/{id}`,
showing elapsed time and the running call count rather than a spinner that
cannot say what it is doing.

The console page is served without a token because it carries no user data.
Everything it calls is behind the token when one is set.

As a sidecar without the console:

```bash
python examples/preflight.py                            # split, certify, budget
AARAMSE_API_TOKEN=$(openssl rand -hex 16) aaramse serve --model anthropic:claude-opus-5
```

`POST /v1/repair` with `{"query": "..."}` and forward the `rewritten` field; it
is byte-identical to your query unless the decision is `repaired`.

Configuration, Docker, endpoint reference, and the latency budget live in
**[docs/](docs/)** — this README does not repeat them.

## Limitations

- **The lattice is hand-built.** English-only, finance-tuned rules. Deliberately
  not learned, to keep model judgement off the runtime safety path.
- **Monotonicity is enforced, not proven.** `DEFINITIONALIZE` obeyed the lattice
  and leaked anyway; certification caught it, the invariant did not.
- **The judge is not the published instrument.** DDOR uses Qwen3Guard-Gen-0.6B
  with a double-blind human study; this uses a general model prompted with the
  taxonomy, validated on 6 hand labels. It erred at least once in 40.
- **Leak figures rest partly on OR-Bench labels** that DDOR specifically
  criticises as noisy. Treat them as upper bounds pending human adjudication.
- **This branch no longer has zero runtime dependencies.** That property was
  load-bearing and advertised; LiteLLM ends it. The native backend is the
  escape hatch, not a claim that nothing changed.
- **Repair is slow.** Measured against `gemma4:12b`: a passthrough takes ~30-50s
  (2 model calls), a repair 7.5-8.7 minutes (~23-26 calls). This does not fit
  behind a synchronous request; see [docs/deployment.md](docs/deployment.md).
- **The sidecar handles one request at a time.** The audit log recomputes its
  tail hash by reading the file, so appends are serialised by a lock.
- **The console is single-tenant.** Jobs live in memory and die with the
  process, and the service serialises every turn behind one lock, so it is a
  supervisor's window onto one gateway rather than a multi-user product.
- **The judge still sits on the runtime path.** `gateway.JudgedProbe` uses the
  three-way judge to decide refusal at runtime, which `judge.py` explicitly
  forbids. Fixing it will move the measured numbers, so it is not a silent
  change.

## Licence

Apache-2.0 (`LICENSE`). Conditions of intended use, stated as norms rather than
licence terms, are in [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).
