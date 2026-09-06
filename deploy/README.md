# Hosting the console for a reviewer

A stable HTTPS URL, one login, no domain purchase, no bill. Roughly twenty
minutes, most of which is waiting for Oracle to provision.

Everything here assumes one instance. That is not a simplification: the sidecar
serialises requests on a lock because the audit log recomputes its tail hash by
reading the file, so two containers appending to one chain would break it. Do
not scale this.

## 1. An always-on instance

Oracle Cloud's Always Free tier, an Ampere A1 instance -- it is genuinely free
rather than trial credit, and does not sleep. Ubuntu 22.04, 1 OCPU and 6 GB is
ample; the process is a Python server and its memory goes on LiteLLM.

Two things to get right while creating it:

- **Assign a reserved public IP.** An ephemeral one changes on stop/start and
  the URL below is derived from it.
- **Open 80 and 443** in the subnet's security list. Oracle blocks inbound by
  default, and its images also carry local iptables rules, so:

```bash
sudo iptables -I INPUT -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 2. Docker

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker "$USER" && newgrp docker
```

## 3. The code and its configuration

```bash
git clone https://github.com/HASHIRU-AI/AARAMSE.git
cd AARAMSE
git checkout fix/serve-shutdown-and-refusal-oracle   # until this is merged
cd deploy
```

The checkout matters: everything in `deploy/`, the model-swap kill switch, and
the console's four seeds are on that branch. `main` has none of it, and a clone
without the checkout would build an image whose console cannot be locked down.

Hash the reviewer's password. It is never stored in plaintext and never
committed:

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'THE-PASSWORD'
```

Write `deploy/.env` with the output. **This file holds credentials -- confirm it
is gitignored before you save it.**

```ini
SITE_ADDRESS=203.0.113.10.sslip.io      # your reserved IP, with .sslip.io appended
JUDGE_USER=judge
JUDGE_PASSWORD_HASH=$2a$14$...          # the hash printed above, not the password
NVIDIA_NIM_API_KEY=...
META_API_KEY=...
```

`sslip.io` resolves a hostname containing an IP back to that IP, which is what
lets Let's Encrypt issue a certificate without owning a domain.

## 4. Start it

```bash
docker compose -f compose.hosted.yaml up -d --build
docker compose -f compose.hosted.yaml logs -f caddy   # watch the certificate issue
```

The console is at `https://<SITE_ADDRESS>/`, and the browser asks for the
login before it will render.

## 5. Check it before sending the link

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://<SITE_ADDRESS>/          # 401
curl -s -u judge:THE-PASSWORD https://<SITE_ADDRESS>/healthz              # {"status": "ok", ...}
curl -s -u judge:THE-PASSWORD https://<SITE_ADDRESS>/v1/config | grep swap  # "model_swap_allowed": false
```

The first must be 401. If it is 200, basic auth is not applied and the link is
open to anyone who finds it.

Then open it and take one turn. The first seed is a passthrough and costs two
model calls, so it is a cheap way to confirm the provider credentials work
without spending a repair.

## What the reviewer can and cannot do

They can take turns and read every trace. They cannot change the model or
supply a credential: `AARAMSE_ALLOW_MODEL_SWAP=0` makes `/v1/model` return 403,
and the page hides the form rather than offering one the server will refuse.

That restriction is the point. There is one process environment and one
gateway, so a key pasted by one visitor would sit in the environment and serve
the next visitor's turns. Anyone who wants to point the layer at their own
model runs it locally, where the bench is safe because the process is theirs.

## What this costs and what it risks

The hosting is free. Every turn spends **your** NVIDIA NIM and Meta quota -- a
repair is 18-66 model calls -- and NIM throttles inside a single repair, which
the backoff rides out by waiting. Two reviewers working at once will queue
behind the lock rather than interleave.

Their queries are written to a hash-chained log on the `aaramse-audit` volume.
That is deliberate and it is the artifact the project is about, but it means
someone else's questions are recorded on your instance. Say so, or run with an
audit path you are willing to keep.

## Operating it

```bash
docker compose -f compose.hosted.yaml logs -f aaramse            # what it is doing
docker compose -f compose.hosted.yaml exec aaramse \
  python -m aaramse report --audit /data/gateway.jsonl           # the supervisor report
docker compose -f compose.hosted.yaml down                       # stop; the volumes persist
```

Two known limits, neither urgent and both real if this stays up for weeks. The
client's response cache never evicts, so memory grows with distinct queries.
And the audit log recomputes its tail hash by reading the whole file on every
append, so appends slow as the chain grows. Restarting clears the first; the
second wants a fix before the log reaches tens of thousands of records.
