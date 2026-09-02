"""Hosted providers must behave identically to the local one, or results drift.

Every test substitutes `_post`, so the suite still runs offline with no keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from aaramse.client import ModelClient, OllamaClient
from aaramse.providers import (
    ANTHROPIC_VERSION,
    AnthropicClient,
    MissingCredential,
    OpenAIClient,
    build_client,
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


def test_specs_route_to_the_right_provider():
    """One string is the only thing an evaluation changes between models."""
    assert isinstance(build_client("openai:gpt-5"), OpenAIClient)
    assert isinstance(build_client("anthropic:claude-opus-5"), AnthropicClient)
    assert isinstance(build_client("ollama:gemma4:12b"), OllamaClient)


def test_bare_tag_stays_ollama():
    """Existing configs say "gemma4:12b" and must keep working untouched."""
    assert parse_spec("gemma4:12b") == ("ollama", "gemma4:12b")
    assert build_client("gemma4:12b").model == "gemma4:12b"


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
