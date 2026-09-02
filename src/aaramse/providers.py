"""Hosted-provider clients, so the layer can front a model it does not run.

The concept note promises results on frontier models; every measurement so far
came from a local Ollama server, because that was the only client. These add
OpenAI and Anthropic behind the same two-method interface, so an evaluation
script changes one string and nothing else.

Raw HTTP rather than the vendors' SDKs: this package declares no runtime
dependencies, and that is load-bearing -- the whole thing is meant to drop into
a deployment without pulling a dependency tree in behind it. Every request goes
through `client.post_json`, which is the one seam a test replaces.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .client import CachingClient, ModelClient, OllamaClient, post_json

__all__ = [
    "ANTHROPIC_VERSION",
    "AnthropicClient",
    "MissingCredential",
    "OpenAIClient",
    "build_client",
    "parse_spec",
]

logger = logging.getLogger(__name__)

# Pinned deliberately. The header is required on every Messages API request and
# is the contract the response shape is guaranteed against.
ANTHROPIC_VERSION = "2023-06-01"


class MissingCredential(RuntimeError):
    """Raised when a provider client is used without its API key in the environment."""


def _require_key(env_var: str) -> str:
    """Read an API key from the environment or fail with an actionable message."""
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise MissingCredential(
            f"{env_var} is not set; export it before running against this provider."
        )
    return key


@dataclass
class OpenAIClient(CachingClient):
    """Chat-completions client for OpenAI-compatible endpoints.

    Attributes:
        model: Model id to call.
        endpoint: API base URL, overridable for compatible gateways.
        system_prompt: Applied to `answer` calls only, never to `complete`.
        api_key_env: Environment variable holding the credential.
        timeout: Per-request timeout in seconds.
        send_temperature: Off by default. Recent OpenAI models reject any
            temperature other than their default, and this package asks for 0
            everywhere, so sending it turns every call into a 400.
    """

    model: str
    endpoint: str = "https://api.openai.com/v1"
    system_prompt: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: int = 300
    send_temperature: bool = False
    calls: int = 0
    _cache: Dict[Tuple[str, ...], str] = field(default_factory=dict, repr=False)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Issue one chat-completions request. Overridden in tests."""
        self.calls += 1
        return post_json(
            f"{self.endpoint}/chat/completions",
            payload,
            {"Authorization": f"Bearer {_require_key(self.api_key_env)}"},
            self.timeout,
        )

    def _chat(
        self, messages: List[Dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        """Send messages and pull the assistant text out of the response."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if self.send_temperature:
            payload["temperature"] = temperature
        body = self._post(payload)
        choices = body.get("choices") or []
        if not choices:
            logger.warning("openai returned no choices for model %s", self.model)
            return ""
        return choices[0].get("message", {}).get("content") or ""

    def answer(self, prompt: str, max_tokens: int = 320) -> str:
        """Answer as the deployed agent, under the system prompt."""
        def produce() -> str:
            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            return self._chat(messages, 0.0, max_tokens)

        return self.cached(("answer", prompt, str(max_tokens)), produce)

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 60) -> str:
        """Plain completion with no system prompt, for rewriting and judging."""
        key = ("complete", prompt, str(temperature), str(max_tokens))
        return self.cached(
            key,
            lambda: self._chat([{"role": "user", "content": prompt}], temperature, max_tokens),
        )


@dataclass
class AnthropicClient(CachingClient):
    """Messages API client for Claude models.

    Attributes:
        model: Model id to call.
        endpoint: API base URL.
        system_prompt: Sent as the top-level `system` parameter on `answer`
            calls only. The Messages API keeps it out of the message list.
        api_key_env: Environment variable holding the credential.
        timeout: Per-request timeout in seconds.
        anthropic_version: Value of the required `anthropic-version` header.

    Temperature is never sent. Current Claude models removed the sampling
    parameters and reject `temperature` with a 400, and this package asks for
    0 on every call, so passing it through would fail every request. Requests
    are deterministic enough for our purposes without it; the `temperature`
    argument on `complete` is accepted for interface parity and used only as
    part of the cache key.
    """

    model: str
    endpoint: str = "https://api.anthropic.com/v1"
    system_prompt: Optional[str] = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout: int = 300
    anthropic_version: str = ANTHROPIC_VERSION
    calls: int = 0
    _cache: Dict[Tuple[str, ...], str] = field(default_factory=dict, repr=False)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Issue one Messages request. Overridden in tests."""
        self.calls += 1
        return post_json(
            f"{self.endpoint}/messages",
            payload,
            {
                "x-api-key": _require_key(self.api_key_env),
                "anthropic-version": self.anthropic_version,
            },
            self.timeout,
        )

    def _message(self, prompt: str, system: Optional[str], max_tokens: int) -> str:
        """Send one user turn and concatenate the text blocks that come back."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        body = self._post(payload)
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text:
            logger.warning(
                "anthropic returned no text blocks (stop_reason=%s)", body.get("stop_reason")
            )
        return text

    def answer(self, prompt: str, max_tokens: int = 320) -> str:
        """Answer as the deployed agent, under the system prompt."""
        return self.cached(
            ("answer", prompt, str(max_tokens)),
            lambda: self._message(prompt, self.system_prompt, max_tokens),
        )

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 60) -> str:
        """Plain completion with no system prompt, for rewriting and judging."""
        key = ("complete", prompt, str(temperature), str(max_tokens))
        return self.cached(key, lambda: self._message(prompt, None, max_tokens))


# Provider prefixes. Anything unprefixed is an Ollama tag, because those
# contain colons themselves ("gemma4:12b") and predate this factory.
_PROVIDERS = ("openai", "anthropic", "ollama")


def parse_spec(spec: str) -> Tuple[str, str]:
    """Split a model spec into (provider, model).

    Args:
        spec: Either "provider:model" or a bare Ollama tag.

    Returns:
        The provider name and the model id. Splitting happens on the first
        colon only, so "ollama:gemma4:12b" keeps its tag intact.

    Raises:
        ValueError: When the spec names a provider prefix but no model.
    """
    provider, separator, model = spec.partition(":")
    if separator and provider in _PROVIDERS:
        if not model:
            raise ValueError(f"model spec {spec!r} names a provider but no model")
        return provider, model
    return "ollama", spec


def build_client(
    spec: str, system_prompt: Optional[str] = None, **kwargs: Any
) -> ModelClient:
    """Build the client a model spec names.

    Args:
        spec: "openai:gpt-5", "anthropic:claude-opus-5", "ollama:gemma4:12b",
            or a bare Ollama tag such as "gemma4:12b".
        system_prompt: Deployment condition, applied to `answer` calls only.
        **kwargs: Passed to the client constructor (endpoint, timeout, ...).

    Returns:
        A client satisfying the ModelClient protocol.

    Raises:
        ValueError: When the spec is empty or names an unknown provider.
    """
    if not spec.strip():
        raise ValueError("model spec is empty")
    provider, model = parse_spec(spec)
    if provider == "openai":
        return OpenAIClient(model=model, system_prompt=system_prompt, **kwargs)
    if provider == "anthropic":
        return AnthropicClient(model=model, system_prompt=system_prompt, **kwargs)
    return OllamaClient(model=model, system_prompt=system_prompt, **kwargs)
