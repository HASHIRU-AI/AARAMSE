# Deploying the sidecar

AARAMSE runs as a process in front of your agent. It takes a query, decides
whether the model behind it refuses, repairs the refusal if it can, and returns
the text you should forward. Its one runtime dependency is LiteLLM, which is
what makes the deployed provider and model a configuration value rather than a
code change.

## The shape of a deployment

```
user ──▶ your agent ──▶ AARAMSE ──▶ model
                            │
                            ├─ probes the model with the query
                            ├─ not refused → returns it byte-identical
                            ├─ refused → localizes, repairs, re-probes
                            └─ appends to the hash-chained audit log
```

AARAMSE calls the model itself, to find out whether it refuses. It is not a
transparent proxy, and it is not fast — see [What it costs](#what-it-costs)
before you put it on a synchronous path.

## Installing it

The package lives under `src/`, so it is not importable from a bare checkout.
Either install it or set `PYTHONPATH`:

```bash
pip install -e .          # gives you the `aaramse` command
# or, without installing:
export PYTHONPATH=src
```

Every command below assumes one of those. In Docker neither is needed — the
image sets `PYTHONPATH` itself.

## Running it

```bash
# Locally, against a model on your machine
AARAMSE_API_TOKEN=$(openssl rand -hex 16) aaramse serve --model gemma4:12b

# One query, no server
aaramse repair "What is the legal definition of tax-loss harvesting?"

# Read an existing log
aaramse report --audit audit/gateway.jsonl
```

`python -m aaramse ...` works identically if you prefer not to rely on the
console script.

With Docker:

```bash
docker build -t aaramse .
docker run -p 8080:8080 \
  -e AARAMSE_API_TOKEN=... \
  -e AARAMSE_MODEL=anthropic:claude-opus-5 \
  -e ANTHROPIC_API_KEY=... \
  -v aaramse-audit:/data \
  aaramse
```

Or `compose.yaml`, which brings up the sidecar and an Ollama server together.
It refuses to start without `AARAMSE_API_TOKEN`, deliberately.

## Configuration

Every flag has an environment variable, because that is how a container is
configured. Flags win over the environment.

| Variable | Flag | Default | What it does |
|---|---|---|---|
| `AARAMSE_MODEL` | `--model` | `gemma4:12b` | Model spec. See [providers.md](providers.md). |
| `AARAMSE_API_TOKEN` | — | unset | Bearer token. **Unset means no authentication.** |
| `AARAMSE_AUDIT_PATH` | `--audit` | `audit/gateway.jsonl` | Where the hash chain is written. |
| `AARAMSE_DEPLOYER` | `--deployer` | `Acme Wealth Ltd` | Firm named in the deployer frame. |
| `AARAMSE_AUTHORISATION_REF` | `--authorisation-ref` | `FRN-123456` | That firm's regulatory reference. |
| `AARAMSE_SYSTEM_PROMPT` | `--system-prompt` | FCA compliance prompt | The deployment condition. |
| `AARAMSE_ALLOW_UNCERTIFIED` | `--allow-uncertified` | off | Run operators with no passing certificate. |
| `AARAMSE_HOST` | `--host` | `0.0.0.0` | Bind interface. |
| `AARAMSE_PORT` | `--port` | `8080` | Bind port. |
| `AARAMSE_LOG_LEVEL` | `--log-level` | `INFO` | Python logging level. |
| `OLLAMA_HOST` | — | `http://localhost:11434` | Ollama base URL. Set this in a container. |

### The three settings that are safety decisions

**`AARAMSE_API_TOKEN`.** Without it the sidecar answers anyone who can reach
the port, and the thing they can reach is a service that rewrites prompts until
a model stops refusing them. The server logs a warning at startup when it is
unset. Set it.

**`AARAMSE_ALLOW_UNCERTIFIED`.** Off by default. An uncertified operator is one
whose behaviour on *your* model is unmeasured — and this project has a case
where an operator certified clean against a simulator and leaked on its first
live query. The flag exists for research on the uncertified algebra.

**The leakage budget.** Not an environment variable, because it is a decision
that belongs in code a reviewer reads. See [measurement.md](measurement.md).

## Before you serve traffic

`examples/preflight.py` runs the whole sequence and is executable as written:

```bash
python examples/preflight.py                          # offline, instant
python examples/preflight.py --live gemma4:12b        # against the real thing
python examples/preflight.py --live gemma4:12b --serve
```

It does four things, and each is a step you cannot skip:

1. **Split the prohibited corpus** into certification and evaluation folds, and
   assert they are disjoint. Certifying and then scoring on the same prompts
   measures memorisation. See [measurement.md](measurement.md).

2. **Certify against the model you are actually deploying.** A certificate is a
   property of *(operator, model, corpus)*, so moving from `gemma4:12b` to
   `claude-opus-5` voids every certificate you hold. Operators that fail — or
   that the corpus never exercises — are excluded.

3. **Enforce a leakage budget** on the held-out fold. This is the end-to-end
   question certification does not ask: across the whole search, how many
   prohibited prompts does the assembled layer get answered? It raises
   `BudgetExceeded`, and disables repair before raising, so a caller that
   swallows the exception is left escalating rather than leaking.

4. **Serve**, with the audit log somewhere that survives a restart. The compose
   file uses a named volume. A repair nobody can review is the thing the
   regulator was worried about.

The offline mode uses the seed corpus, not FalseReject: the simulated boundary
is finance-tuned and refuses 1 of 59 FalseReject-toxic prompts, so running it
there would certify nothing and measure nothing. Offline shows the mechanics;
only `--live` produces a number worth reporting.

## Operating it

- **Throughput is one request at a time.** The audit log recomputes its tail
  hash by reading the file, so concurrent appends would interleave and break
  the chain. A lock serialises handling *within* one process. This is a real
  ceiling, and it is the first thing to fix if you need concurrency — probably
  by holding the tail hash in memory and appending under a file lock.
- **The lock does not span processes.** Two gateways pointed at the same audit
  path will race, and `verify()` passing is not evidence that they did not:
  appends are seconds apart, so a race can simply fail to happen. One writer
  per log file. We hit this during development — a stray second process
  double-wrote a log whose chain still verified — and the only reason it was
  caught was a record count that did not match the number of queries.
- **`GET /healthz` does not touch the model.** Liveness is about the sidecar.
  If health depended on the backend, a model outage would turn into a restart
  loop that also loses the in-memory caches.
- **A model outage returns `503 model backend unavailable`** with the failing
  URL in `detail`, not a generic 500.
- **Repair is slow enough to change the architecture.** See below.

## What it costs

Measured end to end against `gemma4:12b` on a local Ollama server, one query at
a time. Your numbers will differ with the model and the hardware; the ratios
are the point.

| Decision | Wall clock | What it pays for |
|---|---|---|
| `passthrough` (not refused) | 10-50 s | The probe and the judge: two generations. |
| `escalated` (nothing cleared it) | 5-6 min | The full search, exhausted. You pay the most for the queries you cannot repair. |
| `repaired` via `FRAME_ASSERT` | ~7.5 min | Search plus re-probe. |
| `repaired` via `TARGETED_REPAIR` | ~8.7 min | The above plus delta debugging: `localization_budget` probes, each a full generation. |

The ordering is the uncomfortable part. **An escalation costs minutes and
produces nothing** — the layer works hardest on exactly the queries it hands to
a human anyway. Any deadline you set is therefore mostly a cap on wasted work,
which is an argument for setting one low.

**This does not fit behind a synchronous HTTP request.** A user is not waiting
eight minutes, and no sensible gateway timeout accommodates it. Three ways out,
in the order we would try them:

1. **Escalate on a deadline.** Give `handle` a time budget; when it expires,
   escalate rather than keep searching. A human answering in an hour beats a
   socket timing out in thirty seconds, and escalation is already the
   fail-closed path.
2. **Repair asynchronously.** Return `escalated` immediately, run the search on
   a queue, and use the result to answer a follow-up or to pre-warm the cache
   for the next occurrence of that query.
3. **Use it offline.** Mine refusal telemetry in batch and put proposed repairs
   in front of a reviewer. This is where the audit log and the report already
   point, and it needs no latency budget at all.

`localization_budget` is the main dial: it caps delta-debugging probes per
query, and it trades repair rate against time directly.

## What it does not do

No rate limiting, no TLS, no request queue, no horizontal scaling. Put it
behind whatever you already use for those. It is a decision layer, not an edge
proxy.
