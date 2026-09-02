"""Minimal model client. One place that talks to a model.

Six evaluation scripts each grew their own copy of this, which is how the
answer-generation settings drifted between the search's accept decision and the
scorer's judgement -- two calls with different truncation, disagreeing about the
same repair. Everything now goes through one cached client.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

__all__ = [
    "CachingClient",
    "ModelClient",
    "ModelUnavailable",
    "OllamaClient",
    "post_json",
]

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """The model backend could not be reached.

    Distinguished from every other failure because it is the one an operator
    can act on: the model is down, misconfigured, or unreachable from here.
    A generic 500 sends them reading this code instead of their deployment.
    """


@runtime_checkable
class ModelClient(Protocol):
    """Anything the package can put a prompt to.

    Two methods, because the package makes exactly two kinds of call: `answer`
    runs under the deployment system prompt and is the thing being measured;
    `complete` runs without it and is used for rewriting, judging and
    equivalence checks. Keeping them separate is what stops the deployment
    condition leaking into the instruments that score it.
    """

    calls: int

    def answer(self, prompt: str, max_tokens: int = 320) -> str:
        """Answer as the deployed agent, under the system prompt."""
        ...

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 60) -> str:
        """Plain completion with no system prompt."""
        ...

    def reset(self) -> None:
        """Clear the cache and the call counter."""
        ...


def post_json(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Dict[str, Any]:
    """POST JSON and return the parsed response.

    The only outbound network call in the package. Every provider client routes
    through it so a test can substitute one seam instead of one per provider.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # The backend answered, so this is a request problem: a bad model id, a
        # rejected parameter, a missing credential. Surface its body -- that is
        # where the provider says which.
        detail = error.read()[:500].decode("utf-8", "replace")
        raise ModelUnavailable(f"{url} returned {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ModelUnavailable(f"{url} is unreachable: {error}") from error


class CachingClient:
    """Memoisation and call counting, shared by every provider client.

    Six evaluation scripts once grew six copies of this. Providers now inherit
    it, so a cache-key change cannot apply to one provider and not another.
    Subclasses supply `calls` and `_cache` as dataclass fields; this class only
    supplies the behaviour, so it does not disturb their field order.
    """

    calls: int
    _cache: Dict[Tuple[str, ...], str]

    def cached(self, key: Tuple[str, ...], produce: Callable[[], str]) -> str:
        """Return a memoised result, producing it on first request."""
        if key in self._cache:
            return self._cache[key]
        value = produce()
        self._cache[key] = value
        return value

    def reset(self) -> None:
        """Clear the cache and the call counter."""
        self._cache.clear()
        self.calls = 0


@dataclass
class OllamaClient(CachingClient):
    """Calls a local Ollama server, memoising identical requests.

    Attributes:
        model: Model tag to run.
        endpoint: Base URL of the Ollama server. Defaults to `OLLAMA_HOST`
            when set, so a container can point at a sibling service.
        system_prompt: Applied to `answer` calls only. This is the deployment
            condition; rewriting and judging run without it.
        timeout: Per-request timeout in seconds.
    """

    model: str = "gemma4:12b"
    endpoint: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    system_prompt: Optional[str] = None
    timeout: int = 300
    calls: int = 0
    _cache: Dict[Tuple[str, ...], str] = field(default_factory=dict, repr=False)

    def _post(self, path: str, payload: Dict[str, Any], key: str) -> str:
        """Issue one request and pull the text field out of the response."""
        self.calls += 1
        body = post_json(f"{self.endpoint}{path}", payload, {}, self.timeout)
        return body["message"]["content"] if key == "message" else body[key]

    def answer(self, prompt: str, max_tokens: int = 320) -> str:
        """Answer as the deployed agent, under the system prompt."""
        key = ("answer", prompt, str(max_tokens))
        if key in self._cache:
            return self._cache[key]
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        reply = self._post("/api/chat", {
            "model": self.model, "stream": False, "think": False, "messages": messages,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }, "message")
        self._cache[key] = reply
        return reply

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 60) -> str:
        """Plain completion with no system prompt, for rewriting and judging."""
        key = ("complete", prompt, str(temperature), str(max_tokens))
        if key in self._cache:
            return self._cache[key]
        reply = self._post("/api/generate", {
            "model": self.model, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, "response")
        self._cache[key] = reply
        return reply
