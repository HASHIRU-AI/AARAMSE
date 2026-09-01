# Model backends

One string selects the backend. Everything else in the layer is unchanged,
which is the point: a measurement that differs between two models should differ
because the models differ, not because two evaluation scripts grew two clients.

```python
from aaramse import build_client

build_client("gemma4:12b")                  # Ollama (bare tags stay Ollama)
build_client("ollama:gemma4:12b")           # the same, explicit
build_client("openai:gpt-5")                # OpenAI
build_client("anthropic:claude-opus-5")     # Anthropic
```

The spec splits on the **first** colon only, so `ollama:gemma4:12b` keeps its
tag intact. A prefix with no model (`openai:`) is an error rather than a
silently-mangled Ollama tag.

| Provider | Credential | Endpoint |
|---|---|---|
| `ollama` | none | `OLLAMA_HOST`, default `http://localhost:11434` |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |

A missing key raises `MissingCredential` naming the variable to export. Keys
are read at request time, never at construction, so building a gateway never
requires credentials — which is what lets the test suite run offline.

## The interface

Every client implements two methods, and the split is load-bearing:

- **`answer(prompt)`** runs under the deployment system prompt. This is the
  thing being measured — the model as the user meets it.
- **`complete(prompt)`** runs with no system prompt. This is the instrument:
  rewriting, judging, equivalence checks.

Letting the compliance prompt leak into the instrument would mean scoring a
refusal with a judge that is itself under orders to refuse. Tests assert the
separation for every provider.

Both memoise on their arguments. The cache exists because six evaluation
scripts once each grew their own client, and the answer settings drifted
between the search's accept decision and the scorer's judgement — two calls
generating different answers for the same repair.

## LiteLLM by default, raw HTTP as the fallback

`LiteLLMClient` is what `build_client` returns unless told otherwise, and every
model call in the package goes through it. The reason is not convenience: a
certificate is a property of *(operator, model, corpus)*, and `DEFINITIONALIZE`
certified clean against a simulator and leaked on its first live query. Re-running
the battery against a different model has to be one string, or nobody does it.

Specs translate to LiteLLM routes — `openai:gpt-5` becomes `openai/gpt-5`, a
bare `gemma4:12b` becomes `ollama/gemma4:12b`, and anything already containing a
slash passes through so an unmodelled route still works.

**Ollama models are sent `think=False`.** A reasoning model otherwise spends the
whole token budget thinking and returns an empty `content`, which nothing
downstream can distinguish from a model that answered with nothing — the judge
would score silence as a refusal on every prompt. `qwen3.5:4b` did exactly this
until the default was added.

The hand-rolled clients remain for a deployment that cannot take a dependency
tree: `build_client(..., backend="native")`, or `AARAMSE_CLIENT_BACKEND=native`.
They speak OpenAI, Anthropic and Ollama; asking them for anything else is an
error rather than a silent fallback to the wrong provider. Every request goes
through `client.post_json`, the single seam a test replaces.

The cost of hand-rolling is that provider quirks are ours to track. Two are
encoded in the native clients:

**Anthropic never receives `temperature`.** Current Claude models removed the
sampling parameters and reject `temperature` with a 400. This package asks for
0 on every call, so passing it through would fail every request. The argument
is accepted for interface parity and used only in the cache key.

**OpenAI omits `temperature` by default** (`send_temperature=False`) for the
same reason — recent models reject anything but their default — and sends
`max_completion_tokens`, not `max_tokens`. Set `send_temperature=True` for an
older model or a compatible gateway that wants it.

Both are tested. If a provider changes shape, the test names the assumption.

## Adding one

Subclass `CachingClient`, add `calls` and `_cache` fields, implement `_post`,
`answer`, and `complete`, then extend `_PROVIDERS` and `build_client`. Keep
`_post` as the only method that touches the network — that is what makes the
client testable without a key.

## What is not handled

No retries, no backoff, no rate-limit handling, no streaming. A transport
failure becomes `ModelUnavailable`, which the HTTP layer maps to a `503` naming
the URL. For an evaluation run that is the right behaviour: a retried call is a
second measurement, and silently averaging the two is how a result stops
meaning anything.
