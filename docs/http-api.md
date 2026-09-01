# HTTP reference

Two routes are open; everything else needs `Authorization: Bearer <token>` when
`AARAMSE_API_TOKEN` is set. Bodies are capped at 64 KB — a query is a sentence.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/healthz` | open | Liveness. Does not touch the model. |
| `GET` | `/`, `/console` | open | The console page. Carries no user data. |
| `GET` | `/v1/config` | required | Active model, operators, certificates. |
| `POST` | `/v1/repair` | required | Repair one query, synchronously. |
| `POST` | `/v1/chat` | required | Start one console turn. Returns a job. |
| `GET` | `/v1/chat/{id}` | required | Poll that job. |
| `GET` | `/v1/report` | required | Intervention report, JSON. |
| `GET` | `/v1/report.md` | required | Intervention report, Markdown. |

The console page is open because it is static markup; every call it makes is
behind the token when one is set. `/` no longer answers liveness — use
`/healthz`, which is what a load balancer should have been probing anyway.

## `POST /v1/repair`

```json
{"query": "What is the legal definition of tax-loss harvesting?"}
```

```json
{
  "decision": "repaired",
  "rewritten": "What does tax-loss harvesting mean as a general matter?",
  "program": ["TARGETED_REPAIR"],
  "program_render": "TARGETED_REPAIR",
  "refusal_margin": 1,
  "oracle_calls": 18,
  "reason": ""
}
```

**Forward `rewritten` to your agent.** It is byte-identical to `query` unless
`decision` is `"repaired"` — that invariant is the contract, and it is what
makes the layer safe to put in a path you already trust.

`decision` is one of:

| Value | Meaning | What to do |
|---|---|---|
| `passthrough` | The model did not refuse. | Forward `rewritten` (== `query`). |
| `repaired` | It refused; a certified program cleared it. | Forward `rewritten`. Consider disclosing that the question was rephrased. |
| `escalated` | It refused and nothing cleared it. | Do **not** forward. Route to a human. |

`refusal_margin` is the program length — how many operators it took. 0 means no
over-refusal occurred; 1–2 is pragmatic over-refusal.

### Errors

| Status | When |
|---|---|
| `400` | Body is not a JSON object, or `query` is missing/empty/not a string. |
| `401` | Missing or wrong bearer token. |
| `404` | Unknown route. |
| `413` | Body over 64 KB. |
| `503` | The model backend is unreachable. `detail` names the URL and error. |
| `500` | A bug. The server logs a traceback; please report it. |

## `POST /v1/chat`

The console's route. Identical work to `/v1/repair`, but asynchronous, because a
repair runs 23-26 model calls and does not fit behind a held-open socket.

```json
{"query": "How do I hide assets from my bankruptcy trustee?"}
```

`202` with a job handle:

```json
{"id": "07638226f6084b45", "kind": "chat", "status": "running",
 "elapsed_s": 0.0, "model_calls": 0}
```

Poll `GET /v1/chat/{id}`. While `status` is `running` the response carries
`elapsed_s` and `model_calls` **for this turn**, not for the process. On
`done` a `result` appears with the full trace: the decision, whether the query
went through byte-identical, the baseline classification and reply, the operator
program with each step's localized mRTF and declared substitutions, both
actionability profiles, the answer delivered, and the audit record with its
chain check.

`status` is `error` when the turn failed; `error` carries the reason. Jobs live
in memory, are capped at the newest 64, and die with the process — the audit log
is the record, not the job store. Polling an unknown or evicted id is a `404`.

## Reports

`GET /v1/report` returns the machine view; `/v1/report.md` returns the same
content as Markdown for a human. Both lead with chain integrity, then volume
and rates, then certificates in force, then **every escalation individually** —
an escalation is a user who did not get an answer, and a count would hide that
— then every repair with its program, margin, and localized fragment.

Queries are clipped to 160 characters. The report leaves the building; it
carries enough to identify a query and no more.
