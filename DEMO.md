# AARAMSE — demo runbook

**Auditable over-refusal repair for regulated AI advice.**

A drop-in middleware layer that sits in front of a deployed financial agent,
detects when it refuses a legitimate customer question, repairs the refusal, and
logs every intervention in a tamper-evident record a compliance officer can read.

The repair itself is an agent: it plans over a **closed operator algebra** (a
predefined, fixed set of safe rewrite actions), acts on the deployed model,
verifies what came back, and escalates to a human when it cannot succeed. Its
autonomy is strictly capped before deployment (**bounded autonomy**, capping the
number of search steps and live model calls), which makes it safe to put in front
of a regulated system.

---

## The problem, in one sentence

Banks lock their customer-facing AI down so hard to stay compliant with
regulators (like the FCA) that it refuses ordinary, harmless questions — such as
*"What is an ETF?"* — and every over-refusal turns away a customer the bank was
legally allowed to serve.

## What we do about it

```
customer question ─▶ probe: did the agent refuse?
                        │ no  ──▶ passed through untouched (byte-identical)
                        │ yes
                        ├─ repair: assert the deployer's authorised context
                        ├─ re-probe: did the customer actually get an answer now?
                        ├─ escalate to a human if the refusal was correct
                        └─ log the intervention to a hash-chained audit trail
```

Two core properties make it safe to put in front of a regulated assistant:

- **Byte-identity (preserving exact customer wording).** If a question is not
  refused by the model, it reaches the model character-for-character unchanged.
  The layer cannot alter or put words in a customer's mouth.
- **Escalation over leakage (refusing prohibited requests).** If a query is
  genuinely harmful or prohibited, the system upholds the refusal and sends it to
  a human compliance reviewer, rather than trying to bypass the safety alignment.

---

## Run it

One command. The first run certifies the operators against the model (the slow
step, a few minutes) and caches the result; every run after is instant.

```bash
make demo                      # qwen3.5:4b, console at http://localhost:8080/
```

Pre-warm the cache before you present so the first certification isn't on the clock:

```bash
make demo        # let it certify and print the banner, then Ctrl-C
make demo        # now boots straight to the console
```

## The three-minute script

Open `http://localhost:8080/` and type these in order:

| # | Type this | What the judges see | What it proves |
|---|---|---|---|
| 1 | **What is an ETF?** | `repaired` — the model refused at baseline; the layer applied the deployer frame (authorising general education); the customer got a real answer | The core value: a wrongly-refused legitimate question, fixed live |
| 2 | **How does compound interest work on a savings account?** | `repaired` again | It generalises across questions — not a one-prompt trick |
| 3 | **How can I hide money from my creditors during bankruptcy?** | `escalated` — refusal upheld, sent to a human queue | The safety counterweight: it does not say yes to prohibited requests |

Click any verdict to open the **trace**:
- The **operator program** that ran (the sequence of edits performed).
- Both **actionability profiles** (measuring whether the question remained general education rather than seeking personalised financial advice).
- The **untouched baseline reply** from the model.
- What was sent to the model instead.
- The **audit hash** with its **chain-intact check** (cryptographic proof that the audit trail is unaltered).

That trace is the compliance story: every automated intervention is inspectable
and tamper-evident.

> A repair explores multiple candidates and takes several model calls, so each turn
> is handled as an asynchronous job: the console returns immediately and displays
> elapsed time and a live call count while it works. Talk the judges through the
> trace of turn 1 while turn 2 runs.

---

## Why it's an agent, not a prompt

- **Its autonomy is a number, not a heuristic (bounded autonomy).** Instead of
  relying on open-ended stopping rules or letting the model decide when it has
  "done enough", depth `k` over a closed operator set `O` mathematically caps the
  candidates the agent can ever consider at `search_space_size(|O|, k)`. With the
  shipped defaults, this is a small, three-digit number an engineer or auditor
  can enumerate *before* deployment. Live model calls are capped separately
  (`max_oracle_calls`). An agent with a fixed search budget cannot be
  manipulated into an endless loop.
- **It hands off on a defined trigger (deterministic escalation).** Escalation
  to a human fires when the allowed actions are exhausted, or when a candidate
  elicits step-by-step procedural assistance (`abort_on_content_delivery`) — not
  when the agent "feels" confident. It never blindly "tries harder".
- **Its edits are structural, not generative (structural confinement).** Instead
  of asking an LLM to rewrite the whole sentence, the system isolates the
  **minimal Refusal-Triggering Fragment (mRTF)** — the exact phrase that caused
  the refusal — and asks the model only for a replacement phrase. The
  substitution is performed in deterministic code, guaranteeing that everything
  outside that fragment remains byte-identical.
- **Certification is per-model, and pre-admission (contrastive certification).**
  A rewrite operator is only admitted into production after it is empirically
  proven against *this specific model* never to flip a prohibited request into
  an answer. If you switch models, it re-certifies. This catches rules that look
  safe on paper or against a simulator but leak on a real model.
- **The audit log is an immutable hash chain.** Each record carries the
  cryptographic SHA-256 hash of the previous record. A compliance supervisor can
  mathematically prove that no record was inserted, deleted, or altered after
  the fact. Every autonomous decision remains permanently reviewable.
- **The judge never sits on the runtime safety path.** Refusal repair is governed
  by deterministic rules (an actionability lattice checking educational vs.
  actionable intent) and pre-certified operators, not by asking an LLM at
  runtime whether a rewrite is safe.

## Who it's for

In regulatory governance frameworks, this addresses **(3) authorisation,
supervision, and enforcement**. It serves two users at opposite ends of the
same audit trail:

- **A firm's compliance team** runs it in front of customer-facing assistants.
  Over-refusals stop being invisible drop-offs — where a customer asked *"What is
  an ETF?"*, was declined, and left without a trace. Instead, every refusal
  becomes a logged event that is either safely repaired or escalated into a queue
  that compliance staff can actively review.
- **A regulatory supervisor or external auditor** reads the other end. Running
  `aaramse report` converts the firm's audit log into an executive document that
  answers the regulator's first two questions: is the cryptographic hash chain
  intact, and which queries required human escalation? For the first time,
  over-refusal becomes evidenced and measurable with an audit trail.

## What we'd say on the limitations slide

Honesty is part of the pitch, and it is the part a sharp judge probes for:

- **Recovery and leak-safety trade off (the leakage budget).** An aggressive
  deployer frame recovers most over-refusals, but carries a higher risk of
  answering questions that should have been refused. A strictly confined
  fragment edit leaks nothing, but recovers fewer queries. The deployer controls
  this trade-off using an enforceable, audited **leakage budget** — an explicit,
  configurable dial, not an unmeasured assumption.
- **The benign benchmarks are noisy (dataset contamination).** When we audited
  the standard FalseReject finance benchmark, only ~27% of its "benign" prompts
  were clearly harmless; roughly a third actually asked for illegal or
  unethical actions (such as money laundering, partner fraud, or elder
  exploitation) that a model *should* refuse. We score against a re-adjudicated
  subset and document this benchmark contamination openly.
- **Latency in an asynchronous sidecar.** A repair explores multiple candidates
  and runs several model calls, taking seconds to minutes depending on model
  size. AARAMSE is designed as an asynchronous supervisory sidecar (where callers
  poll a job ID), rather than an inline synchronous blocker on a customer's hot path.

---

## The team

Three co-founders of **NAAMSE**. All three are authors on two published agentic
systems, and this project is the third step in the same line of work.

| | Organisation | Role |
|---|---|---|
| **Kunal Pai** | UCLA / NAAMSE | Co-Founder & Research Lead |
| **Parth Shah** | Amazon / NAAMSE | Co-Founder & Tech Lead |
| **Harshil Patel** | UC Davis / NAAMSE | Co-Founder & Product Lead |

**[HASHIRU](https://arxiv.org/abs/2506.04255)** — a hierarchical multi-agent
system with a "CEO" agent that hires and fires specialised employee agents
under an explicit economic model of cost and memory, prioritising smaller local
models and reaching for larger APIs only when a task needs them, with
autonomous tool creation. Building agents whose autonomy is bounded by a
resource budget is directly the design this layer applies to a safety budget.

**[NAAMSE](https://arxiv.org/html/2602.07391v2)** — an evolutionary security
fuzzer for LLM agents, built on LangGraph and compliant with the AgentBeats A2A
protocol. It mutates adversarial prompts against a target agent and scores the
responses for jailbreaks, prompt injection, and PII leakage — while explicitly
holding *benign-use correctness*, because a blanket-refusing agent is
degenerately secure rather than safe.

That last clause is where AARAMSE comes from. NAAMSE measures the refusal
boundary from the outside and shows that blanket refusal is not security;
AARAMSE is the other half — the layer that repairs the over-refusals this
produces, under controls a supervisor can audit. The connection is concrete in
the code: NAAMSE's gpt-oss-120b benign report is where this project's fifth
refusal-detector failure class was found. That report emits U+2019 exclusively,
so every pattern spelling its contraction with an ASCII apostrophe scored
*"I can't help"* as compliance. The regression is `TestTypographicApostrophes`
in `tests/test_content_gate.py`.

What this repository evidences on its own: ~7k lines across 30 focused modules,
513 passing tests, per-model certification, a hash-chained audit layer,
container and Kubernetes manifests, and live measurements against a real model
with the artifacts checked in under `audit/`.

Contact details are on the submission form rather than here, so this file can
be shared without publishing three inboxes.

---

## If a judge asks to run it themselves

```bash
git clone <repo> && cd aaramse
pip install -e .                 # or: export PYTHONPATH=src
make demo                        # needs Ollama with qwen3.5:4b pulled
```

No API keys needed. Everything runs against a local Ollama model.
`make demo-offline` runs the entire pipeline against a scripted stand-in with no
model needed, perfect for machines without a local GPU or model host.
