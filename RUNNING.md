# Running and testing AARAMSE

Everything below assumes you are at the repository root.

---

## 0. What you need

| | |
|---|---|
| Python | 3.11 or newer (verified on 3.13.12) |
| Packages | none, for the tests and the offline demo — the package is pure standard library |
| Ollama | only for the live demo and the benchmark scripts |
| API keys | none |

The test suite and `make demo-offline` need **nothing installed**. Dependencies
only matter once you point the layer at a real model.

---

## 1. Install

Two ways. Pick one.

**A. No install at all** — put `src` on the path. This is what the `Makefile`
does, and what every command below uses.

```bash
export PYTHONPATH=src
```

**B. Editable install**, which also gives you the `aaramse` console script:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements-dev.txt
```

`requirements.txt` holds the runtime dependency (`litellm`);
`requirements-dev.txt` adds `pytest`, `mypy` and `ruff`. Both mirror
`pyproject.toml`, which stays the source of truth.

> **On this machine:** the repo lives under a directory with a `:` in its name
> (`prototypes/c:dir/`), which `uv` cannot parse. Use `pip` and the
> `PYTHONPATH=src` form here, not `uv run`.

---

## 2. Test

```bash
PYTHONPATH=src python3 -m pytest -q        # or: make test
```

Expected: **360 passed** in well under a second. No network, no model, no
fixtures to download.

Narrow it down while working:

```bash
PYTHONPATH=src python3 -m pytest tests/test_gateway.py -q
PYTHONPATH=src python3 -m pytest -q -k certif
PYTHONPATH=src python3 -m pytest -q -x --ff      # stop at first failure, failures first
```

Lint and type-check the same way CI would:

```bash
make lint
# = .venv/bin/ruff check src tests examples
#   .venv/bin/mypy src/aaramse
```

---

## 3. Run it — no model needed

The whole pipeline against a scripted stand-in. Instant, deterministic, and the
right thing to run on a machine that cannot host a model.

```bash
make demo-offline                # = PYTHONPATH=src python3 examples/demo.py
```

You get the three stages (certification, live turns, supervisor report) and an
audit log at `audit/gateway_demo.jsonl`. Last known-good run: 1 passthrough,
2 repaired, 1 escalated, `chain_intact True`.

The deployer's full pre-flight — fold split, certification, leakage budget —
also runs offline:

```bash
PYTHONPATH=src python3 examples/preflight.py
```

---

## 4. Run it — against a real model

Needs [Ollama](https://ollama.com) running with the model pulled:

```bash
ollama pull qwen3.5:4b
```

Then:

```bash
make verify                      # certify + check the curated prompts, warms the cache
make demo                        # certify once (cached), console at http://localhost:8080/
```

The first `make demo` certifies the operators against the model — minutes — and
caches the result in `audit/cert_cache.json`. Every run after boots straight to
the console. Pre-warm before you present. `make clean-cache` forces a
re-certify; swapping the model forces one too.

Demo script and what each prompt proves: **`DEMO.md`**.

Override the model anywhere:

```bash
make demo MODEL=gemma4:12b
PYTHONPATH=src python3 examples/serve_demo.py gemma4:12b
```

---

## 4b. End-to-end smoke test

`make test` is offline and proves the code is internally consistent. It says
nothing about whether the layer works in front of a model. These two scripts do,
and they run in that order.

### Step 1 — is the model usable at all?

```bash
PYTHONPATH=src python3 examples/judge_floor.py              # gemma4:12b
PYTHONPATH=src python3 examples/judge_floor.py qwen3.5:4b
```

Two ways a live run can pass while proving nothing, and this catches both.

If the model **doesn't over-refuse**, every query passes through and a green run
means only that passthrough works — there has to be something broken before a
repair layer can be shown to fix it. If the **judges are a yes-machine**, every
model-judged gate rubber-stamps: `gemma3:270m` accepted "write me a poem about
cats" as equivalent to a bond-yield question.

Exits non-zero and says which check failed. `gemma4:12b` is the only model known
to pass it — `qwen3.5:4b`, the `make demo` default, has never been checked.

### Step 2 — drive the whole thing over HTTP

```bash
PYTHONPATH=src python3 examples/e2e_smoke.py --offline        # no model, seconds
PYTHONPATH=src python3 examples/e2e_smoke.py --per-bucket 1   # gemma4:12b, minutes
PYTHONPATH=src python3 examples/e2e_smoke.py gemma4:12b --per-bucket 3
```

Stands the gateway up, certifies it, **serves it on a real socket**, and drives
it the way a deployer's agent would — which nothing else here does; the HTTP
tests use a scripted client. Then it audits the record a supervisor would read.

Three buckets in one run:

| Bucket | Corpus | Must |
|---|---|---|
| control | FinQA filing arithmetic | pass through, byte-identical |
| benign | curated over-refusals | ideally repair |
| prohibited | OR-Bench-toxic finance | not be repaired |

**Hard failures are bugs. Soft failures are the model.** It exits non-zero only
for: a control query that came back altered, a prohibited query the layer talked
the model into answering, a broken hash chain, a 5xx, or auth that didn't fail
closed. A benign query that escalated instead of repairing is a *recovery miss* —
the measured rate is roughly 1 in 3 — and is reported, not failed. Wiring that
the other way gives you a test that flaps on model variance until nobody trusts
it.

`--offline` uses the simulated boundary. It proves the harness and the plumbing
and is worth running before you spend model time; it is **not** evidence about
any model.

**Budget the time.** A repair is 23–26 model calls, and against a local
`gemma4:12b` that measured 7.5–8.7 minutes each. Start with `--per-bucket 1`.
The first run also pays for certification; it is cached after that.

---

## 5. The CLI

With `pip install -e .` the command is `aaramse`; otherwise
`PYTHONPATH=src python3 -m aaramse`.

```bash
python3 -m aaramse repair "What is an ETF?"      # one query, result as JSON
python3 -m aaramse serve                         # the HTTP sidecar
python3 -m aaramse report --audit audit/gateway.jsonl
python3 -m aaramse --help
```

Sidecar routes: `GET /` (console), `GET /healthz`, `POST /v1/repair`,
`POST /v1/chat` + `GET /v1/chat/<id>` (a repair is a job — poll it),
`GET /v1/config`, `GET /v1/report` and `/v1/report.md`.

```bash
curl -s localhost:8080/healthz
curl -s localhost:8080/v1/repair -d '{"query":"What is an ETF?"}'
```

Set `AARAMSE_API_TOKEN` to require `Authorization: Bearer …` on every route but
the console. Without it the sidecar answers unauthenticated requests and says so
in the log.

---

## 6. Benchmarks (live model, slow)

```bash
PYTHONPATH=src python3 examples/finqa_control.py             # false-intervention rate, 120 items
PYTHONPATH=src python3 examples/finqa_control.py gemma4:12b 10   # 10-item smoke test
PYTHONPATH=src python3 examples/finqa_cause.py               # why each refusal happened
PYTHONPATH=src python3 examples/adjudicated_repair.py --scripted   # offline self-check
PYTHONPATH=src python3 examples/adjudicated_repair.py gemma4:12b   # live
```

Results land in `audit/`. The measured artifacts the docs cite —
`audit/finqa_*.{json,jsonl}`, `audit/benign_adjudication.json` — are tracked in
git on purpose; regenerated run output is gitignored. Don't delete the tracked
ones.

---

## 7. Configuration

Every flag has an environment variable; flags win.

| Variable | Default | What it does |
|---|---|---|
| `AARAMSE_MODEL` | `gemma4:12b` | Model spec. Bare tag = Ollama; also `openai:gpt-5`, `anthropic:claude-opus-5` |
| `AARAMSE_CLIENT_BACKEND` | `litellm` | `native` uses the zero-dependency urllib clients |
| `AARAMSE_AUDIT_PATH` | `audit/gateway.jsonl` | Hash-chained audit log |
| `AARAMSE_CERT_CACHE` | `audit/cert_cache.json` | Certificate cache the demo reuses |
| `AARAMSE_DEPLOYER` | `Acme Wealth Ltd` | Authorised firm operating the gateway |
| `AARAMSE_AUTHORISATION_REF` | `FRN-123456` | That firm's regulatory reference |
| `AARAMSE_SYSTEM_PROMPT` | FCA compliance prompt | System prompt the agent runs under |
| `AARAMSE_HOST` / `AARAMSE_PORT` | `0.0.0.0` / `8080` | Where the sidecar listens |
| `AARAMSE_API_TOKEN` | unset | Bearer token required on non-console routes |
| `AARAMSE_ALLOW_UNCERTIFIED` | unset | Set to `1` to skip certification. Debugging only |
| `AARAMSE_LOG_LEVEL` | `INFO` | Python logging level |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | unset | Only for hosted models |

---

## 8. Docker

```bash
docker compose up            # see compose.yaml
```

---

## If something breaks

- **`ModuleNotFoundError: aaramse`** — `export PYTHONPATH=src`, or `pip install -e .`.
- **`uv` errors on the path** — see the note in §1; use `pip`.
- **`litellm is not installed`** — `pip install -r requirements.txt`, or set
  `AARAMSE_CLIENT_BACKEND=native`.
- **The demo hangs on boot** — it is certifying against the model. First run
  only; `make verify` gets it over with beforehand.
- **Connection refused on a live run** — Ollama isn't up, or the model isn't
  pulled: `ollama list`.
