# Hosting the console persistently

A plan, not an implementation. The question is what it takes to leave the
console reachable at a stable URL rather than behind a quick tunnel that dies
with the terminal it was started from.

## What the code decides for us

Four properties of the layer rule out most of the obvious answers before any
platform is compared.

**It is single-instance by construction.** `GatewayService` serialises every
request on one lock because `RepairSearch` and `AuditLog` share mutable state,
and `AuditLog._last_hash()` recomputes the tail by reading the file. Two
replicas appending to one chain would interleave and break it. There is no
horizontal scaling story here and there should not be one: an unbroken chain is
the point, throughput is not.

**The audit log has to outlive the container.** It is the artifact a supervisor
reads and the thing every claim in the README is checked against. Any platform
with an ephemeral filesystem discards the evidence on each deploy, which is
worse than not logging at all -- it looks like a record and is not one. The
existing `Dockerfile` already declares `VOLUME ["/data"]` for exactly this.

**Work continues after the response is sent.** A chat turn returns a job id
immediately and the repair runs on a thread for 90-320 seconds. Anything that
suspends the process between requests -- scale-to-zero, CPU throttling outside a
request -- kills repairs in flight, and the console polls a job that will never
finish.

**Turns are long but requests are short.** The poll design means no connection
is held open for minutes, so proxy idle timeouts are a non-issue. This is the
one constraint that makes hosting easy rather than hard.

## What has to change before it faces the public

These are independent of platform. The first two are blocking; the rest are the
difference between a demo and something that can be left up.

1. **`POST /v1/model` is a credential sink.** It writes a submitted API key into
   the process environment, globally. That was designed for a local tool and
   `serve.py` says so in as many words. Publicly hosted, one visitor's key
   silently pays for the next visitor's turns. It needs an env flag that refuses
   the route, or per-session isolation, which is a much larger change.

2. **There is no usable authentication.** `AARAMSE_API_TOKEN` protects the API
   but the console page sends no `Authorization` header, so setting a token
   serves a page that cannot call anything. Either the token moves into the page
   (read from the URL fragment, kept in `sessionStorage`) or authentication
   happens at the edge and the app stays open behind it.

3. **Two unbounded growths.** Each client's `_cache` never evicts, and
   `_last_hash()` is O(n) per append, so a long session is O(n²) in the number
   of records. Neither matters for an evaluation run of 120 prompts. Both matter
   for a server left up for a month.

4. **Every turn spends the host's quota.** A repair is 18-66 model calls. Either
   the edge rate-limits per visitor, or visitors bring their own key -- which
   conflicts with (1) unless keys are per-session.

5. **Visitor queries are written to disk**, hash-chained, in whatever the
   deployer's audit path is. That is the intended behaviour and it needs to be
   stated on the page, or query text needs to stay out of the record.

## The recommendation

**One small always-on VM with a persistent volume, reached through a named
Cloudflare Tunnel, with Cloudflare Access in front of it.**

The tunnel is the same technology as the quick tunnel already used, in its
durable form: a named tunnel keeps a stable hostname across restarts and needs
no inbound port, so the machine has no public attack surface at all. Access
handles authentication at the edge -- an email allowlist or one-time PIN --
which resolves the console's missing `Authorization` header without touching the
page, because the app never sees an unauthenticated request.

Concretely: a shared-CPU instance with a gigabyte of memory and a small volume
mounted at `/data`, running the existing image with `restart: unless-stopped`,
scale-to-zero disabled, and one instance maximum. Fly.io, Hetzner, and a plain
DigitalOcean droplet all satisfy this; the choice between them is billing, not
capability. Expect a few dollars a month, with the tunnel and Access on free
tiers.

`compose.yaml` needs one edit: it currently depends on a local `ollama` service,
which a hosted deployment does not want. Drop that service and set
`AARAMSE_MODEL` and `AARAMSE_REWRITER_MODEL` to the hosted specs.

## What was considered and rejected

**Serverless request-scoped platforms** (Vercel, Netlify, Lambda behind API
Gateway). Work continues after the response returns, so the repair dies with the
request. Disqualified by the job model, not by cold starts.

**Cloud Run.** Now supports volume mounts, so it is no longer disqualified
outright, but it needs concurrency pinned to 1, minimum instances at 1, and CPU
always allocated -- at which point it is a more expensive VM with a GCS FUSE
mount underneath an append-heavy hash chain, which is the worst possible
storage shape for this workload.

**Free tiers that sleep** (Render free, HF Spaces). Ephemeral disk resets the
chain on every wake. The console would appear to work while quietly discarding
the evidence, which is the failure mode this project exists to avoid.

**Kubernetes.** The single-instance constraint means an orchestrator manages one
pod that must never be evicted or replaced concurrently. All of the cost, none
of the benefit.

## Sequence

1. Add the flag that disables `/v1/model`, and decide whether hosted visitors
   bring keys or spend the deployer's.
2. Stand up the VM, volume, and named tunnel; put Access in front.
3. Deploy the existing image with the compose file's `ollama` service removed.
4. Bound the client cache and memoise the audit tail hash.
5. Decide what the page says about queries being recorded.

Steps 1 and 2 are what separate "reachable" from "safe to leave reachable".
Steps 4 and 5 can follow the first deployment; nothing breaks in the first week
without them.
