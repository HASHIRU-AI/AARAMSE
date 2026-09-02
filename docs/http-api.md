# HTTP and A2A reference

Two routes are open; everything else needs `Authorization: Bearer <token>` when
`AARAMSE_API_TOKEN` is set. Bodies are capped at 64 KB — a query is a sentence.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/healthz`, `/` | open | Liveness. Does not touch the model. |
| `GET` | `/.well-known/agent-card.json` | open | A2A discovery. |
| `POST` | `/v1/repair` | required | Repair one query. |
| `POST` | `/a2a` | required | The same, in A2A's JSON-RPC envelope. |
| `GET` | `/v1/report` | required | Intervention report, JSON. |
| `GET` | `/v1/report.md` | required | Intervention report, Markdown. |

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

## A2A

The agent card at `/.well-known/agent-card.json` advertises exactly one method.

```json
{"capabilities": {"streaming": false, "pushNotifications": false},
 "x-aaramse": {"model": "gemma4:12b", "supportedMethods": ["message/send"]}}
```

**This is a subset, deliberately.** There is no task lifecycle, no
`message/stream`, no push notifications, no artifact store. Those are what a
*destination* agent needs; a middleware that rewrites one message and hands it
back needs the envelope and nothing else. Anything unsupported returns JSON-RPC
`-32601` naming what is supported, so a client discovers the limit rather than
hitting it silently.

### `message/send`

```json
{"jsonrpc": "2.0", "id": 1, "method": "message/send",
 "params": {"message": {"kind": "message", "role": "user", "messageId": "m1",
                        "parts": [{"kind": "text", "text": "..."}]}}}
```

```json
{"jsonrpc": "2.0", "id": 1,
 "result": {"kind": "message", "role": "agent", "messageId": "…",
            "parts": [{"kind": "text", "text": "the text to forward"}],
            "metadata": {"aaramse/decision": "repaired",
                         "aaramse/program": ["TARGETED_REPAIR"],
                         "aaramse/refusalMargin": 1,
                         "aaramse/reason": ""}}}
```

Every text part is concatenated in order and treated as one query. Non-text
parts are ignored — a file attachment is not a question. `contextId` and
`taskId` are echoed when you send them and **never invented when you do not**,
because emitting a `taskId` would imply a task lifecycle this adapter does not
have.

Protocol problems come back as JSON-RPC errors with HTTP 200 (`-32700` parse,
`-32600` invalid request, `-32601` method not found, `-32602` invalid params),
which is what an A2A client parses. Only transport problems are HTTP errors.

## Reports

`GET /v1/report` returns the machine view; `/v1/report.md` returns the same
content as Markdown for a human. Both lead with chain integrity, then volume
and rates, then certificates in force, then **every escalation individually** —
an escalation is a user who did not get an answer, and a count would hide that
— then every repair with its program, margin, and localized fragment.

Queries are clipped to 160 characters. The report leaves the building; it
carries enough to identify a query and no more.
