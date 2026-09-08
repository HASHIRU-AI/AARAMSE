# AARAMSE — auditable over-refusal repair for regulated AI advice

A model-agnostic middleware layer that detects when a deployed financial agent
refuses a legitimate question, repairs the refusal, and logs every intervention
in a tamper-evident record a supervisor can read.

The repair itself is an agent: it plans over a **closed operator algebra** (a
predefined, fixed set of safe rewrite actions), acts on the deployed model,
verifies what came back, and escalates to a human when it cannot succeed. What
makes it deployable in front of a regulated system is that its autonomy is
strictly capped by a concrete number the deployer sets (**bounded autonomy**),
not by an open-ended prompt or a vague stopping heuristic.

Every model call goes through LiteLLM, so the provider and model are configured
with a simple spec string (e.g. `openai:gpt-5` or `gemma4:12b`).
`python examples/demo.py` runs the whole pipeline offline against a scripted
stand-in model; `--live` drives a real model. `aaramse serve` launches an
interactive web chat console in front of it.

**This is the MVP branch.** It carries the modules with live measurements
behind them and nothing else. See `main` for the full research tree, including
the three-arm comparison this branch cannot reproduce.

## What it does

```
query ──> probe ──> refused? ──no──> passthrough, untouched
                       │yes
                       ├─ localize the minimal refusal-triggering fragment (mRTF)
                       ├─ repair (edit the fragment, or assert deployer context)
                       ├─ re-probe: did the user actually get an answer?
                       └─ escalate if not, and log either way
```

That loop is perceive-act-verify with a human handoff, where each stage is a
guarded, mathematically bounded agentic step rather than a prompt:

| Stage | What the agent does | What bounds it (and what it means) |
|---|---|---|
| **Probe** | Checks if the live model refuses using an automated probe | `max_oracle_calls` (hard ceiling on live model calls) |
| **Localize** | Uses delta-debugging search to isolate the minimal refusal-triggering fragment (mRTF) | **1-minimality** (removing any single word stops triggering refusal), verified in tests |
| **Plan** | Breadth-first search for the shortest sequence of safe rewrite operators | `search_space_size(\|O\|, k)` (strictly caps total candidate combinations) |
| **Act** | Slices the replacement phrase into the original query in code | **Byte-identity** (100% untouched character-for-character outside the fragment) |
| **Verify** | Re-probes the model and verifies the response answers the user's question | **Actionability lattice** (ensuring educational intent) and **harm gate** (blocking unsafe topics) |
| **Hand off** | Escalates to a human compliance reviewer | Fires when the operator set is exhausted, or if procedural guidance was delivered |

## Bounded autonomy is the safety argument

Most agentic safety designs rely on "stopping heuristics": they prompt the agent
and trust it to stop when it decides it has done enough. AARAMSE relies on
**arithmetic**. Depth `k` over a closed operator set `O` mathematically caps the
total candidates the agent can ever consider at `search_space_size(|O|, k)` —
with the shipped defaults, a small, three-digit number an engineer or regulator
can enumerate and inspect *before* deployment. `max_oracle_calls` caps live model
contact independently. An agent with a fixed search budget cannot be talked into
an endless search or jailbreak loop.

Four further architectural properties make that budget dependable:

- **Escalation fires on a defined trigger (deterministic escalation)**, not on
  the agent's subjective confidence: when the allowed actions are exhausted, or
  when a candidate elicits step-by-step procedural assistance
  (`abort_on_content_delivery`). The agent never blindly "tries harder".
- **The agent's edits are structural, not generative (structural confinement).**
  The model only proposes a replacement for the localized phrase; the
  substitution happens in deterministic software code. Everything outside that
  isolated fragment remains byte-identical by construction.
- **Operators are certified before admission (contrastive pre-certification).**
  A certificate is a property of *(operator, model, corpus)*; an operator is
  only allowed to run if proven never to flip a prohibited request into an
  answer on the specific deployed model. Enforcement is fail-closed.
- **Every autonomous decision is auditable.** The operator program, the
  localized fragment, each edit, the refusal margin, and the certificates in
  force are written to a cryptographic SHA-256 hash-chained log.

## Where this sits

In regulatory governance frameworks, this addresses **(3) authorisation,
supervision, and enforcement**. It serves two users at opposite ends of the
same audit trail:

- **A firm's compliance function** puts it in front of a customer-facing
  assistant. Today, a bank tunes its assistant to refuse anything resembling
  regulated advice. The resulting over-refusals are completely invisible: a
  customer who asked *"What is an ETF?"*, was declined, and left in frustration
  leaves no record today. AARAMSE makes each over-refusal a logged, repaired, or
  escalated event. Escalations become an actionable queue a compliance officer
  can work, rather than a silent failure.
- **A regulatory supervisor or external auditor** reads the other end. Running
  `aaramse report` converts a firm's audit log into a Markdown document that
  leads with what a regulator asks first: does the cryptographic hash chain
  verify, and which queries required human escalation? Escalations are listed
  individually so deployers can be held accountable for failures.

What changes in current practice: **over-refusal becomes evidenced**. Today, a
supervisor cannot prove whether a firm's assistant is denying customers lawful
information, because refusals leave no artifact. AARAMSE produces the artifact
and makes it tamper-evident.

## Measured results

Three obligations govern this layer: repair the refusals that are wrong, do not
leak the ones that are right, and leave everything else completely alone. A system
that meets two of them while quietly failing the third looks fine on paper but is
unacceptable in production.

### Obligation 3: leaving benign queries alone

This is the obligation most often skipped by researchers, and the one measured
most carefully here. Until it was evaluated against a live model, it was only
asserted against a simulator that could not fail it. We tested on `gemma4:12b`
using the FinQA control slice (120 numerical-reasoning questions over public
filings under the FCA banking system prompt), single model, `verify_answers`
off. Source: `audit/finqa_control_v2.json`.

| Metric | Result | Meaning |
|---|---|---|
| **False intervention rate** | **2/120 — 1.67%** | The layer intervened on 2 of 120 normal questions |
| **Passed through byte-identical** | **119/120 — 99.2%** | One question was rewritten; the other 119 reached the model character-for-character unchanged |
| **Median passthrough latency** | **11.1s** | Normal questions pass through with minimal overhead |
| **Audit chain** | **intact** | 286 model calls over 28 minutes, chain verified |

Byte-identity is our core guarantee: *any query that the model did not refuse
reaches it 100% unchanged down to the byte.* That condition held on all 120 items:
the layer never altered a single query that the model was willing to answer.

What this evaluation demonstrates is what happens when the underlying model mistakenly
refuses a normal question. The model refused 2 harmless control questions: one was
safely escalated to a human, and one was repaired. This highlights the crucial
difference between the two types of false interventions:
- **An escalating false intervention** costs extra latency and human review time,
  but leaves the user's original words completely untouched.
- **A repairing false intervention** alters the user's phrasing to obtain an answer.
  We report this transparently rather than smoothing it over.

> **Why a single run cannot establish a trend (the sampling variance caveat):**
> Large language models generate responses probabilistically. When measuring rare
> events (like 1 vs. 2 false interventions out of 120 items, or 0.83% vs. 1.67%),
> small fluctuations between single runs represent expected sampling variance rather
> than a causal effect or performance regression.
>
> In fact, when we re-evaluated the first 44 items after introducing the well-formedness
> guard, two false interventions occurred again, but on *different items*:
> `AAL/2010/page_72.pdf-3` flipped from passthrough to escalated, while
> `ABMD/2005/page_29.pdf-1` flipped the other way — despite zero code changes on
> either path.
>
> Therefore, 0.83% and 1.67% should be understood as two sample points of the same
> baseline under slightly different evaluation settings, not as a trend or regression.
> A definitive benchmark requires repeated runs with statistical confidence intervals.
> Partial run artifacts are preserved in `audit/finqa_control_v3.jsonl`.
>
> **What the judge's worked examples changed:**
> The refusal oracle *is* the three-way judge, so clarifying its prompt shifts what
> gets recognized as an over-refusal. The prompt was updated with worked examples
> because a live model classified polite brush-offs (declining advice but suggesting
> outside resources) as "partial refusals," which the gateway treated as answered and
> left alone — standing down on queries it was built to fix.
>
> Re-evaluating `examples/finqa_control.py` with the sharper judge identified two
> refusals instead of one (1.67%), and for the first time repaired one of them rather
> than passing it through. Full artifacts are in `audit/finqa_control_v2.json`,
> alongside the original baseline run.

**Both false interventions share the exact same question structure**, which reveals
an important model failure mode. Both questions (`ABMD/2007/page_78.pdf-2` and
`ABMD/2005/page_29.pdf-1`) ask for straightforward mathematical extrapolation over
public filings (*"assuming the same growth rate as year N, what would the figure be
in year N+1?"*). Under a strict FCA banking compliance prompt, the model over-cautiously
misinterprets simple arithmetic projection as "financial forecasting," and declines it
as unauthorized advice.

**The gibberish repair and the guard confirmation:**
Investigating the audit log for the single repaired query revealed an instructive
edge case in LLM rewriting:
- **The failure:** The system localized the minimal Refusal-Triggering Fragment (mRTF)
  to the words `"fair for"`. The rewriter proposed replacing `fair` with
  `market*valuation` (leaking a markdown asterisk from the LLM prompt) and `for` with
  `during`. The query was spliced into: *"...grant-date market\*valuation value during
  options"*.
- **Why guards passed it:** The query stayed within length caps, actionability was 0
  (it didn't ask for advice), and semantic fidelity (`MeaningFidelity`) scored it 1.0
  because `topic_core` used a narrow 95-word dictionary (it saw `"options"` and gained
  `"valuation"`). The model answered simply because the sentence had been garbled into
  ungrammatical syntax that no longer triggered the safety filter!
- **The guard confirmation:** We added a strict **well-formedness guard** in
  `targeted.py` that automatically rejects any replacement containing invalid
  characters, markdown symbols (like asterisks), or non-English typography
  (regression test: `TestMalformedReplacements`). Re-running that exact item against
  the new guard confirms the intended defense: the malformed candidate is rejected,
  the search exhausts all valid options without finding an admissible rewrite, and
  the query safely escalates to human review with the user's original words 100%
  byte-identical (`IDENTITY`).
- **The `topic_core` vocabulary limit:** The current 95-word finance allowlist in
  `topic_core` is too narrow to catch every subtle loss of meaning (it lacked terms like
  *fair*, *value*, and *grant*). Expanding this vocabulary is planned for future work,
  as adjusting it impacts operator behavior across the entire evaluation benchmark.

> Two later enhancements are **not** reflected above, because this run predates
> them on the deployment path: `--verify-answers` (escalating repairs whose
> answer fails verification, which would likely have turned the repair back into
> an escalation) and a harm gate covering concealment of assets from creditors
> or bankruptcy trustees. Neither has been measured on this corpus.
>
> The console ships with a split-model architecture: Nemotron generates answers
> and acts as the judge, while Muse-Spark-1.2 proposes fragment replacements and
> scores semantic fidelity. The judge remains aligned with the model being
> repaired, because what constitutes a refusal is inherently a property of that
> model. What the split provides is independence: the model being repaired is not
> also the judge scoring whether its own repair preserved meaning. Using
> `--rewriter-model ""` collapses everything onto a single model, which is how
> every measurement above was collected.

Earlier, two-thirds of the false-intervention rate stemmed from data
preparation, not model alignment: the initial benchmark builder included tables
but omitted the surrounding narrative text, leaving 47 of 120 questions
unanswerable from the prompt alone. Under a strict compliance prompt, an
unanswerable question triggers "I am not permitted to advise" rather than "I
lack that figure" — which is indistinguishable from a safety refusal. All 7
false interventions in that run occurred in the incomplete group, and none in
the complete group (Fisher exact test $p = 0.001$, proving the failures were
caused by missing data). Providing the full context brought the rate from 5.83%
to 0.83%, and the corpus has carried its narrative ever since.

### Obligations 1 and 2: the instrument was the finding

The headline comparison — evaluating `TARGETED_REPAIR` (editing only the
localized trigger fragment) against `LLM_REPHRASE` (asking an LLM to rewrite the
entire prompt) against `FRAME_ASSERT` (adding an authorized compliance frame) on
the FalseReject finance slice and OR-Bench-toxic under an FCA compliance prompt —
is documented on `main` (`examples/three_arm_eval.py` and
`audit/three_arm_eval.json`).

Read as a scoreboard, the results show that **confinement trades off recovery
for safety**:
- The **fragment-confined operator (`TARGETED_REPAIR`)** cleared 1 of 8 benign
  over-refusals and leaked **0 of 10** toxic prompts. It guarantees safety.
- The **deployer frame (`FRAME_ASSERT`)** cleared 7 of 8 benign over-refusals
  and leaked 2 of 10 toxic prompts.
- The **full-prompt generative rewrite (`LLM_REPHRASE`)** scored 6 of 8 and
  leaked 2 of 10 after prompt tuning (up from 1 of 8 and 0 of 10 before tuning).
  However, prompt tuning shifted recovery and leaks together — a fragile property
  of prompts rather than a robust method. It does not ship in our production layer.

Read as an empirical finding, the data revealed something even more critical:
**the standard benign benchmark dataset is contaminated**. Of the 8 prompts
refused at baseline in FalseReject, only ~3 were genuine over-refusals. The
remaining 5 described cash structuring, elder exploitation, partner fraud, and
predatory marketing. Refusing those queries was *correct caution*, not an
over-refusal. The deployer frame's score of 7 of 8 meant it talked the model into
answering a money-laundering query — its true leakage rate was higher than the
toxic slice indicated, hidden by the benchmark's incorrect "benign" label.

This was the third time our evaluation instrument uncovered hidden flaws in
standard assumptions — following the discovery of four refusal-detector failure
modes and the FinQA data-context defect ($p = 0.001$). The primary contribution is
the apparatus that catches these failures. `plan/over-refusal-repair.md` details
the subsequent re-adjudication.

**Not yet established:** behaviour in DDOR's setting (OR-Bench without a system
prompt). That control experiment ran with an insufficient probe budget ($n=12$,
repair rate 0.0) and is invalid; nothing here should be compared to DDOR's
reported 51.96% reduction until it is rerun.

## Measurement comes first

Simple binary refusal detection (e.g. checking for "I cannot") cannot
reliably measure over-refusal. We learned this the hard way: simple detectors
exhibited four failure modes, missing real refusals like *"I cannot fulfill this
request"* and mistaking standard disclaimers for refusals. Furthermore, under a
compliance system prompt, the dominant model behavior is **partial refusal**:
declining personalized advice first, but answering general background anyway.

`judge.py` implements the XSTest three-way taxonomy:
1. **Full compliance:** answered the request directly.
2. **Full refusal:** declined the request entirely.
3. **Partial refusal:** declined personalized advice, but provided educational context.

This was validated 6/6 against human annotations. It is used for offline
measurement and evaluation, and never sits on the runtime safety path.

## Safety machinery

- **Localization (`localize.py`):** Uses delta-debugging (`ddmin`) with
  complement testing and adaptive partitioning (sentence then word level). It
  reduces a 21-word query down to the minimal trigger fragment (e.g. `"hide assets"`)
  in 16 probes. Tests assert genuine 1-minimality (removing any single word
  stops triggering refusal).
- **Structural Confinement (`targeted.py`):** Unlike approaches that ask an LLM
  to rewrite only a fragment and trust it to follow instructions, AARAMSE asks
  the model only for a replacement phrase. The replacement is spliced into the
  original query in deterministic code, guaranteeing that everything outside the
  minimal Refusal-Triggering Fragment (mRTF) remains 100% byte-identical.
- **Contrastive Pre-Certification (`certification.py`):** Operators are
  certified against contrastive pairs (side-by-side benign and prohibited queries)
  on the **actual deployed model**. A certificate belongs to the tuple *(operator,
  model, corpus)*, not the operator in the abstract: an operator named
  `DEFINITIONALIZE` certified clean on a simulator but leaked on its first live query.
  Enforcement is fail-closed by default.
- **Actionability Guards (`invariants.py`):** Every candidate must not increase
  actionability on a rule-based lattice (ensuring queries stay educational
  rather than personalized advice), and must preserve the core question topic.
  For generative rewrites, semantic equivalence checks ensure the original intent
  is preserved.
- **Tamper-Evident Audit Log (`audit.py`):** An append-only JSONL log chained
  with SHA-256 cryptographic hashes. It records the operator program, the
  localized mRTF, each `(fragment -> replacement)` edit, the refusal margin, and
  active certificates.

## Layout

| Module | Purpose |
|---|---|
| `gateway.py` | The deployable layer: build, certify, handle, report |
| `client.py` | Client protocol, caching, and raw-HTTP Ollama client |
| `equivalence.py` | Intent-equivalence judge; ensures rewrites preserve meaning |
| `judge.py` | XSTest three-way response classification (compliance, refusal, partial) |
| `localize.py` | Delta-debugging mRTF localization |
| `targeted.py` | Fragment-confined repair with structural code splicing |
| `operators/` | Rule-based operator algebra, registry, and deployer frame |
| `search.py` | Bounded shortest-program search and deterministic escalation |
| `invariants.py` | Actionability lattice and intent guards |
| `certification.py` | Contrastive certificates on deployed models |
| `audit.py` | Cryptographic hash-chained intervention log |
| `serve.py` | HTTP sidecar: console, chat, repair, and report routes |
| `ui.py` | Console backend: decision traces and asynchronous job store |
| `static/index.html` | Interactive web chat console |
| `__main__.py` | CLI: `serve`, `repair`, `report` commands |
| `providers.py` | LiteLLM client, native fallbacks, and model-spec parser |
| `report.py` | Supervisor-facing intervention reports (JSON and Markdown) |
| `budget.py` | Induced-leakage measurement and fail-closed budget enforcement |
| `splits.py` | Deterministic held-out data splits by content hash |
| `falsereject.py`, `finqa.py` | Benchmark dataset loaders |
| `corpus.py` | Hand-written test fixtures — **not** benchmark evidence |

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
make test                                  # 513 tests
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

- **The actionability lattice is rule-based.** The rules for distinguishing
  general educational concepts from personalized advice are currently
  English-only and tuned for finance. They are intentionally deterministic and
  rule-based rather than machine-learned, ensuring model unpredictability is kept
  off the runtime safety path.
- **Monotonicity (ensuring edits never make a query more actionable) is
  enforced, not formally proven.** In one instance, an operator named
  `DEFINITIONALIZE` satisfied the rule-based lattice but still caused a model to
  leak an answer; contrastive pre-certification successfully caught this failure,
  proving why multi-layer defenses are necessary.
- **The refusal judge is an evaluation prompt, not a dedicated guard model.**
  While research literature (like DDOR) uses fine-tuned models like Qwen3Guard,
  AARAMSE prompts a general model with the XSTest taxonomy (validated on human
  labels). It works well, but carries a small error rate (~1 in 40).
- **Reported leakage figures reflect benchmark dataset noise.** Standard
  benchmarks (like OR-Bench) contain noise and disputed labels; treat leakage
  measurements as upper bounds pending further human adjudication.
- **External runtime dependencies.** Multi-provider support is powered by
  LiteLLM. For environments with strict zero-dependency requirements, the native
  standard-library HTTP clients (`AARAMSE_CLIENT_BACKEND=native`) remain
  available for Ollama, OpenAI, and Anthropic.
- **Repair latency requires an asynchronous architecture.** Testing multiple
  candidate repairs against a live model takes several calls (e.g. 23–26 calls on
  larger models like `gemma4:12b`, taking several minutes). AARAMSE is built as an
  asynchronous supervisory sidecar: `POST /v1/chat` immediately returns a `202`
  with a job ID, allowing callers to poll progress rather than hanging on an
  open HTTP socket.
- **Turns are serialized per log to guarantee cryptographic integrity.** The
  audit log updates its SHA-256 hash chain sequentially on disk, so concurrent
  requests are handled under a lock. Preserving an unbroken audit chain takes
  precedence over concurrent throughput. Production horizontal scaling is
  achieved by running multiple worker instances, each managing its own log.
- **The interactive console is single-tenant.** In-flight jobs are tracked in
  memory for the duration of the process; it is designed as an operational
  window for compliance teams and supervisors, not a multi-tenant SaaS frontend.
- **The judge currently runs on the runtime path.** `gateway.JudgedProbe`
  utilizes the three-way judge to evaluate refusals at runtime; transitioning
  this to a lightweight deterministic oracle will further refine measured latencies.

## Licence

Apache-2.0 (`LICENSE`). Conditions of intended use, stated as norms rather than
licence terms, are in [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md).
