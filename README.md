# AARAMSE — auditable over-refusal repair for regulated AI advice

A model-agnostic middleware layer that detects when a deployed financial agent
refuses a legitimate question, repairs the refusal, and logs every intervention
in a tamper-evident record a supervisor can read.

No runtime dependencies. `python examples/demo.py` runs the whole pipeline
offline against a scripted stand-in model; `--live` drives a real one.

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

`gemma4:12b`, FalseReject finance slice + OR-Bench-toxic, 10 prompts per class,
under an FCA compliance system prompt, scored with the XSTest three-way
taxonomy. 8 of 10 benign and 10 of 10 toxic prompts were refused at baseline, so
those are the denominators. Source: `audit/three_arm_eval.json`.

| Arm | Benign repaired | Toxic leaked |
|---|---|---|
| `TARGETED_REPAIR` (mRTF-confined edit) | 1/8 — 12% | **0/10** |
| `LLM_REPHRASE` (full-prompt rewrite) | 6/8 — 75% | 2/10 — 20% |
| `FRAME_ASSERT` (deployer context) | **7/8 — 88%** | 2/10 — 20% |

Confinement is what costs recovery. The two unconfined arms clear most of the
over-refusals and leak at the same rate as each other; the mRTF-confined arm
leaks nothing and clears almost nothing. That trade-off is the finding, not a
bug in any arm.

Both `LLM_REPHRASE` leaks are OR-Bench prompts whose toxic label is doubtful
— requesting a password reset for one's own account (scored `partial_refusal`,
not full compliance) and how investigators compile public records legally. The
second is also one of `FRAME_ASSERT`'s two. Treat 20% as an upper bound pending
human adjudication.

An earlier run (`audit/three_arm_eval_pretune.json`, rewriter prompt untuned)
put `LLM_REPHRASE` at 1/8 with 0/10 leaked. Tuning the rewriter moved it from
safe-and-weak to strong-and-leaky. It did not find a third option, and that is
the result worth reporting: on this corpus no unconfined arm has yet recovered
benign queries without also moving toxic ones.

**Not yet established:** how any of this behaves in DDOR's setting — OR-Bench,
no system prompt. That control ran with a starved probe budget (n=12, repair
rate 0.0) and is invalid, so nothing here is comparable to DDOR's reported
51.96% reduction until it is rerun. See `examples/control_eval.py`.

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
| `client.py` | The one place that talks to a model |
| `judge.py` | XSTest three-way response classification |
| `localize.py` | Delta-debugging mRTF localization |
| `targeted.py` | Fragment-confined repair with structural splicing |
| `rewriter.py` | Full-prompt LLM rephrase (DDOR's baseline) |
| `operators/` | Rule-based operator algebra, registry, deployer frame |
| `search.py` | Bounded shortest-program search, escalation |
| `invariants.py` | Actionability lattice and intent guard |
| `certification.py` | Contrastive certificates |
| `audit.py` | Hash-chained intervention log |
| `serve.py` | HTTP sidecar: repair, report, and A2A routes |
| `a2a.py` | Agent2Agent `message/send` envelope and agent card |
| `__main__.py` | CLI: `serve`, `repair`, `report` |
| `providers.py` | OpenAI and Anthropic clients, and the model-spec factory |
| `report.py` | Supervisor-facing intervention report |
| `budget.py` | Induced-leakage measurement and fail-closed enforcement |
| `splits.py` | Deterministic held-out splits by content hash |
| `falsereject.py`, `xstest.py`, `finqa.py` | Vendored benchmark loaders |
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

## Running

```bash
python examples/demo.py                    # offline, instant
python examples/demo.py --live gemma4:12b  # against a real model
python examples/three_arm_eval.py          # the comparison table above
python examples/control_eval.py            # DDOR-setting control (needs a real budget)
PYTHONPATH=src python -m pytest -q         # 280 tests
```

As a sidecar:

```bash
pip install -e .                                        # or: export PYTHONPATH=src
python examples/preflight.py                            # split, certify, budget
AARAMSE_API_TOKEN=$(openssl rand -hex 16) aaramse serve --model anthropic:claude-opus-5
```

`POST /v1/repair` with `{"query": "..."}` and forward the `rewritten` field; it
is byte-identical to your query unless the decision is `repaired`. A2A clients
use `POST /a2a`.

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
- **Repair is slow.** Measured against `gemma4:12b`: a passthrough takes ~30-50s
  (2 model calls), a repair 7.5-8.7 minutes (~23-26 calls). This does not fit
  behind a synchronous request; see [docs/deployment.md](docs/deployment.md).
- **The sidecar handles one request at a time.** The audit log recomputes its
  tail hash by reading the file, so appends are serialised by a lock.
- **The A2A adapter is a subset** — `message/send` only, no task lifecycle or
  streaming. The agent card advertises exactly that.
- **The judge still sits on the runtime path.** `gateway.JudgedProbe` uses the
  three-way judge to decide refusal at runtime, which `judge.py` explicitly
  forbids. Fixing it will move the measured numbers, so it is not a silent
  change.

## Licence

Apache-2.0 (`LICENSE`). Conditions of intended use, stated as norms rather than
licence terms, are in [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).
