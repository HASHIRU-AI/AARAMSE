"""Hosted-provider clients, so the layer can front a model it does not run.

The concept note promises results on frontier models; every measurement so far
came from a local Ollama server, because that was the only client. These add
OpenAI and Anthropic behind the same two-method interface, so an evaluation
script changes one string and nothing else.

`LiteLLMClient` is the default and every model call in the package goes through
it, so which provider and model the layer fronts is a spec string rather than a
code change. That control is the point: a certificate is a property of
*(operator, model, corpus)*, so re-running the battery against another model has
to be cheap or it does not happen.

The hand-rolled OpenAI, Anthropic and Ollama clients below are kept as a
zero-dependency fallback -- `build_client(..., backend="native")`, or
`AARAMSE_CLIENT_BACKEND=native`. They speak raw HTTP through `client.post_json`,
which is the one seam a test replaces, and they are what to reach for when the
layer must drop into a deployment without pulling a dependency tree behind it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .client import (
    CachingClient,
    ModelClient,
    ModelUnavailable,
    OllamaClient,
    post_json,
)

__all__ = [
    "ANTHROPIC_VERSION",
    "DEFAULT_BACKEND",
    "AnthropicClient",
    "LiteLLMClient",
    "MissingCredential",
    "OpenAIClient",
    "build_client",
    "litellm_spec",
    "parse_spec",
]

# Which client `build_client` returns when the caller does not say. LiteLLM by
# default: provider control is worth one dependency.
DEFAULT_BACKEND = "litellm"

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


@dataclass
class LiteLLMClient(CachingClient):
    """The default client: every model call in the package goes through LiteLLM.

    LiteLLM normalises ~100 providers onto one chat-completions shape, so
    changing which model the layer fronts is a spec string and nothing else.
    That is the point: certificates, budgets and audit records are properties of
    *(operator, model, corpus)*, so being able to re-run the same battery
    against a different model without touching code is what makes those numbers
    comparable.

    The two-method split is preserved exactly. `answer` runs under the
    deployment system prompt and is the thing being measured; `complete` runs
    without it and is used for judging and equivalence. Keeping them separate is
    what stops the deployment condition leaking into the instruments that score
    it.

    Attributes:
        model: LiteLLM model string, e.g. "openai/gpt-5", "anthropic/claude-opus-5",
            "ollama/gemma4:12b".
        system_prompt: Applied to `answer` calls only, never to `complete`.
        api_base: Provider base URL override. Required for Ollama, optional
            elsewhere; `None` lets LiteLLM use the provider default.
        timeout: Per-request timeout in seconds.
        send_temperature: Off by default. Recent OpenAI and Anthropic models
            reject any temperature but their own default, and this package asks
            for 0 nearly everywhere, so sending it turns every call into a 400.
        extra: Additional keyword arguments forwarded to `litellm.completion`
            verbatim, for provider-specific parameters this class does not model.
    """

    model: str
    system_prompt: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 300
    send_temperature: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    calls: int = 0
    _cache: Dict[Tuple[str, ...], str] = field(default_factory=dict, repr=False)

    def _completion(self, **kwargs: Any) -> Any:
        """Call LiteLLM. Overridden in tests, which never import litellm.

        Imported lazily so the package still imports without litellm installed;
        the native clients below remain usable in that case.
        """
        try:
            import litellm
        except ImportError as error:  # pragma: no cover - exercised by hand
            raise ModelUnavailable(
                "litellm is not installed; `pip install litellm` or build a native "
                "client with build_client(..., backend='native')"
            ) from error
        return litellm.completion(**kwargs)

    def _chat(
        self, messages: List[Dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        """Send messages and pull the assistant text out of the response.

        Every LiteLLM exception becomes `ModelUnavailable`, because that is the
        one failure a deployer can act on and the sidecar turns it into a 503
        rather than a stack trace.
        """
        self.calls += 1
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
            **self.extra,
        }
        if self.api_base:
            payload["api_base"] = self.api_base
        if self.send_temperature:
            payload["temperature"] = temperature
        try:
            response = self._completion(**payload)
        except ModelUnavailable:
            raise
        except Exception as error:
            raise ModelUnavailable(f"{self.model} call failed: {error}") from error

        choices = getattr(response, "choices", None) or []
        if not choices:
            logger.warning("litellm returned no choices for model %s", self.model)
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        return content or ""

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
        """Plain completion with no system prompt, for judging and equivalence."""
        key = ("complete", prompt, str(temperature), str(max_tokens))
        return self.cached(
            key,
            lambda: self._chat([{"role": "user", "content": prompt}], temperature, max_tokens),
        )


# Provider prefixes recognised in a spec. Anything unprefixed is an Ollama tag,
# because those contain colons themselves ("gemma4:12b") and predate this factory.
_PROVIDERS = (
    "anthropic",
    "azure",
    "bedrock",
    "cohere",
    "deepseek",
    "gemini",
    "groq",
    "mistral",
    "ollama",
    "openai",
    "openrouter",
    "together_ai",
    "vertex_ai",
    "xai",
)

# Providers the native (zero-dependency) backend can speak. Everything else is
# LiteLLM-only, and asking for it natively is an error rather than a silent
# fallback to the wrong provider.
_NATIVE_PROVIDERS = ("anthropic", "ollama", "openai")

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


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


def litellm_spec(spec: str) -> str:
    """Translate a model spec into the "provider/model" form LiteLLM expects.

    A spec that already contains a slash is passed through untouched, so a
    caller can hand over a LiteLLM route this factory does not model, such as
    "openrouter/anthropic/claude-opus-5".

    Args:
        spec: "openai:gpt-5", "gemma4:12b", "openrouter/meta-llama/llama-3-70b".

    Returns:
        A LiteLLM model string.
    """
    if "/" in spec:
        return spec
    provider, model = parse_spec(spec)
    return f"{provider}/{model}"


def _resolve_backend(backend: Optional[str]) -> str:
    """Pick the client backend from the argument, the environment, or the default."""
    chosen = (backend or os.environ.get("AARAMSE_CLIENT_BACKEND") or DEFAULT_BACKEND).strip()
    if chosen not in ("litellm", "native"):
        raise ValueError(f"unknown client backend {chosen!r}; expected 'litellm' or 'native'")
    return chosen


def build_client(
    spec: str,
    system_prompt: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs: Any,
) -> ModelClient:
    """Build the client a model spec names.

    Args:
        spec: "openai:gpt-5", "anthropic:claude-opus-5", "ollama:gemma4:12b",
            a bare Ollama tag such as "gemma4:12b", or any LiteLLM route
            containing a slash.
        system_prompt: Deployment condition, applied to `answer` calls only.
        backend: "litellm" (default) or "native" for the zero-dependency
            clients. Falls back to `AARAMSE_CLIENT_BACKEND`, then to
            `DEFAULT_BACKEND`.
        **kwargs: Passed to the client constructor (api_base, timeout, ...).

    Returns:
        A client satisfying the ModelClient protocol.

    Raises:
        ValueError: When the spec is empty, names an unknown provider, or names
            a provider the requested backend cannot speak.
    """
    if not spec.strip():
        raise ValueError("model spec is empty")

    if _resolve_backend(backend) == "litellm":
        model = litellm_spec(spec)
        if model.startswith("ollama/"):
            # Ollama has no default base URL to fall back on, so supply one.
            # Every other provider gets its own default from LiteLLM.
            kwargs.setdefault("api_base", os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
            # Reasoning models spend the whole token budget thinking and return
            # an empty `content`, which this package cannot tell apart from a
            # model that answered with nothing. `qwen3.5:4b` burned all 60
            # tokens and returned "" until this was set. The native client sent
            # think=False on every request for the same reason.
            extra = dict(kwargs.get("extra") or {})
            extra.setdefault("think", False)
            kwargs["extra"] = extra
        return LiteLLMClient(model=model, system_prompt=system_prompt, **kwargs)

    provider, model = parse_spec(spec)
    if provider not in _NATIVE_PROVIDERS:
        raise ValueError(
            f"the native backend cannot speak {provider!r}; "
            f"use the litellm backend or one of {_NATIVE_PROVIDERS}"
        )
    if provider == "openai":
        return OpenAIClient(model=model, system_prompt=system_prompt, **kwargs)
    if provider == "anthropic":
        return AnthropicClient(model=model, system_prompt=system_prompt, **kwargs)
    return OllamaClient(model=model, system_prompt=system_prompt, **kwargs)
