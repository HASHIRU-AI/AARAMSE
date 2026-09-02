"""Hosted providers must behave identically to the local one, or results drift.

Every test substitutes `_post`, so the suite still runs offline with no keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from aaramse.client import ModelClient, ModelUnavailable, OllamaClient
from aaramse.providers import (
    ANTHROPIC_VERSION,
    AnthropicClient,
    LiteLLMClient,
    MissingCredential,
    OpenAIClient,
    build_client,
    litellm_spec,
    parse_spec,
)


@dataclass
class RecordingOpenAI(OpenAIClient):
    """Captures payloads instead of issuing requests."""

    sent: List[Dict[str, Any]] = field(default_factory=list)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        self.calls += 1
        self.sent.append(payload)
        return {"choices": [{"message": {"content": "reply"}}]}


@dataclass
class RecordingAnthropic(AnthropicClient):
    """Captures payloads instead of issuing requests."""

    sent: List[Dict[str, Any]] = field(default_factory=list)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        self.calls += 1
        self.sent.append(payload)
        return {"content": [{"type": "text", "text": "reply"}]}


def test_litellm_is_the_default_backend():
    """Every model call goes through LiteLLM unless a caller opts out."""
    for spec in ("openai:gpt-5", "anthropic:claude-opus-5", "gemma4:12b"):
        assert isinstance(build_client(spec), LiteLLMClient)


def test_native_backend_routes_to_the_right_provider():
    """The zero-dependency clients stay reachable and stay correct."""
    assert isinstance(build_client("openai:gpt-5", backend="native"), OpenAIClient)
    assert isinstance(build_client("anthropic:claude-opus-5", backend="native"), AnthropicClient)
    assert isinstance(build_client("ollama:gemma4:12b", backend="native"), OllamaClient)


def test_backend_can_be_selected_by_environment(monkeypatch):
    """A deployer switches backend without touching a call site."""
    monkeypatch.setenv("AARAMSE_CLIENT_BACKEND", "native")
    assert isinstance(build_client("gemma4:12b"), OllamaClient)


def test_unknown_backend_is_rejected():
    """A typo must not silently pick a backend."""
    with pytest.raises(ValueError):
        build_client("gemma4:12b", backend="litelm")


def test_native_backend_refuses_providers_it_cannot_speak():
    """Falling back to the wrong provider would be worse than failing."""
    with pytest.raises(ValueError):
        build_client("groq:llama-3", backend="native")


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("openai:gpt-5", "openai/gpt-5"),
        ("anthropic:claude-opus-5", "anthropic/claude-opus-5"),
        ("gemma4:12b", "ollama/gemma4:12b"),
        ("ollama:gemma4:12b", "ollama/gemma4:12b"),
        ("gemini:gemini-2.5-pro", "gemini/gemini-2.5-pro"),
        ("openrouter/meta-llama/llama-3-70b", "openrouter/meta-llama/llama-3-70b"),
    ],
)
def test_spec_translation_to_litellm(spec, expected):
    """A spec with a slash is already a LiteLLM route and passes through."""
    assert litellm_spec(spec) == expected


def test_ollama_gets_a_base_url_and_others_do_not():
    """Ollama has no default base URL to fall back on; hosted providers do."""
    assert build_client("gemma4:12b").api_base == "http://localhost:11434"
    assert build_client("openai:gpt-5").api_base is None


def test_ollama_base_url_follows_the_environment(monkeypatch):
    """A container points at a sibling service without a code change."""
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    assert build_client("gemma4:12b").api_base == "http://ollama:11434"


def test_bare_tag_stays_ollama():
    """Existing configs say "gemma4:12b" and must keep working untouched."""
    assert parse_spec("gemma4:12b") == ("ollama", "gemma4:12b")
    assert build_client("gemma4:12b", backend="native").model == "gemma4:12b"


def test_prefixed_ollama_tag_keeps_its_colon():
    """Splitting on every colon would truncate the tag to "gemma4"."""
    assert parse_spec("ollama:gemma4:12b") == ("ollama", "gemma4:12b")


def test_provider_without_model_is_rejected():
    """A truncated spec must fail loudly, not silently become an Ollama tag."""
    with pytest.raises(ValueError):
        parse_spec("openai:")
    with pytest.raises(ValueError):
        build_client("   ")


def test_clients_satisfy_the_protocol():
    """Anything the gateway accepts must present the same two methods."""
    for client in (OpenAIClient(model="m"), AnthropicClient(model="m"), OllamaClient()):
        assert isinstance(client, ModelClient)


def test_anthropic_never_sends_temperature():
    """Current Claude models reject sampling params with a 400.

    The package asks for temperature 0 everywhere, so passing it through would
    fail every single request.
    """
    client = RecordingAnthropic(model="claude-opus-5")
    client.complete("p", temperature=0.0)
    assert "temperature" not in client.sent[0]


def test_anthropic_sends_required_fields():
    """max_tokens is mandatory on the Messages API; the version header is too."""
    client = RecordingAnthropic(model="claude-opus-5")
    client.answer("q", max_tokens=128)
    assert client.sent[0]["max_tokens"] == 128
    assert client.anthropic_version == ANTHROPIC_VERSION


def test_openai_omits_temperature_by_default():
    """Recent OpenAI models reject a non-default temperature."""
    client = RecordingOpenAI(model="gpt-5")
    client.complete("p", temperature=0.0)
    assert "temperature" not in client.sent[0]
    assert client.sent[0]["max_completion_tokens"] == 60


def test_system_prompt_is_only_applied_to_answers():
    """The deployment condition must not leak into the scoring instruments."""
    openai = RecordingOpenAI(model="gpt-5", system_prompt="COMPLIANCE")
    openai.answer("q")
    openai.complete("p")
    assert openai.sent[0]["messages"][0]["role"] == "system"
    assert all(m["role"] != "system" for m in openai.sent[1]["messages"])

    claude = RecordingAnthropic(model="claude-opus-5", system_prompt="COMPLIANCE")
    claude.answer("q")
    claude.complete("p")
    assert claude.sent[0]["system"] == "COMPLIANCE"
    assert "system" not in claude.sent[1]


def test_responses_are_cached_per_provider():
    """Caching is inherited, so it cannot hold for one provider and not another."""
    for client in (RecordingOpenAI(model="m"), RecordingAnthropic(model="m")):
        client.answer("q")
        client.answer("q")
        assert client.calls == 1
        client.reset()
        assert client.calls == 0
        client.answer("q")
        assert client.calls == 1


def test_empty_response_does_not_crash():
    """A refusal-shaped or truncated reply must return text, not raise."""

    class Empty(AnthropicClient):
        def _post(self, payload):  # type: ignore[override]
            return {"content": [], "stop_reason": "refusal"}

    assert Empty(model="m").answer("q") == ""


def test_missing_key_names_the_variable(monkeypatch):
    """The failure has to say which variable to export."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingCredential, match="ANTHROPIC_API_KEY"):
        AnthropicClient(model="m")._post({})


@dataclass
class RecordingLiteLLM(LiteLLMClient):
    """Captures kwargs instead of calling LiteLLM, so the suite stays offline."""

    sent: List[Dict[str, Any]] = field(default_factory=list)
    reply: str = "reply"

    def _completion(self, **kwargs: Any) -> Any:  # type: ignore[override]
        self.sent.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_litellm_answer_carries_the_system_prompt():
    """`answer` is the deployment condition and must include it."""
    client = RecordingLiteLLM(model="openai/gpt-5", system_prompt="COMPLIANCE")
    assert client.answer("q") == "reply"
    roles = [m["role"] for m in client.sent[0]["messages"]]
    assert roles == ["system", "user"]
    assert client.sent[0]["messages"][0]["content"] == "COMPLIANCE"


def test_litellm_complete_never_carries_the_system_prompt():
    """Judging and equivalence must not run under the deployment condition.

    This is the separation the whole measurement rests on: if the compliance
    prompt leaked into the judge, the instrument would be scoring itself.
    """
    client = RecordingLiteLLM(model="openai/gpt-5", system_prompt="COMPLIANCE")
    client.complete("q")
    roles = [m["role"] for m in client.sent[0]["messages"]]
    assert roles == ["user"]


def test_litellm_omits_temperature_by_default():
    """Recent OpenAI and Anthropic models 400 on any temperature but their own."""
    client = RecordingLiteLLM(model="openai/gpt-5")
    client.complete("q", temperature=0.7)
    assert "temperature" not in client.sent[0]

    opted_in = RecordingLiteLLM(model="ollama/gemma4:12b", send_temperature=True)
    opted_in.complete("q", temperature=0.7)
    assert opted_in.sent[0]["temperature"] == 0.7


def test_litellm_passes_api_base_and_extra_through():
    """Provider-specific parameters reach LiteLLM verbatim."""
    client = RecordingLiteLLM(
        model="ollama/gemma4:12b",
        api_base="http://ollama:11434",
        extra={"num_ctx": 8192},
    )
    client.complete("q")
    assert client.sent[0]["api_base"] == "http://ollama:11434"
    assert client.sent[0]["num_ctx"] == 8192


def test_litellm_results_are_cached_and_counted():
    """A repeated prompt is one call; the counter is what reports cost."""
    client = RecordingLiteLLM(model="openai/gpt-5")
    client.answer("q")
    client.answer("q")
    assert len(client.sent) == 1
    assert client.calls == 1

    client.reset()
    assert client.calls == 0
    client.answer("q")
    assert len(client.sent) == 2


def test_litellm_answer_and_complete_do_not_share_a_cache_entry():
    """They are different conditions; collapsing them would corrupt both."""
    client = RecordingLiteLLM(model="openai/gpt-5", system_prompt="COMPLIANCE")
    client.answer("q")
    client.complete("q")
    assert len(client.sent) == 2


def test_litellm_failures_become_model_unavailable():
    """The one failure a deployer can act on; the sidecar turns it into a 503."""

    class Exploding(RecordingLiteLLM):
        def _completion(self, **kwargs: Any) -> Any:
            raise RuntimeError("connection reset by peer")

    with pytest.raises(ModelUnavailable, match="connection reset"):
        Exploding(model="openai/gpt-5").answer("q")


def test_litellm_empty_response_is_survivable():
    """A filtered or truncated response must degrade, not raise."""

    class Empty(RecordingLiteLLM):
        def _completion(self, **kwargs: Any) -> Any:
            return SimpleNamespace(choices=[])

    assert Empty(model="openai/gpt-5").answer("q") == ""


def test_litellm_client_satisfies_the_protocol():
    """The gateway must not be able to tell which backend it was handed."""
    assert isinstance(LiteLLMClient(model="openai/gpt-5"), ModelClient)


def test_ollama_models_disable_thinking():
    """A reasoning model spends the whole budget thinking and returns "".

    `qwen3.5:4b` returned an empty string for every call until `think=False`
    was sent, which is indistinguishable from a model that declined to answer
    -- the judge would have scored silence as a refusal on every prompt.
    """
    assert build_client("qwen3.5:4b").extra["think"] is False
    assert build_client("openai:gpt-5").extra == {}


def test_thinking_can_be_re_enabled_explicitly():
    """A default, not a lock: a caller who wants reasoning traces can have them."""
    client = build_client("qwen3.5:4b", extra={"think": True})
    assert client.extra["think"] is True
