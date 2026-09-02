"""One client, so answer settings cannot drift between decision and score."""

from __future__ import annotations

from aaramse.client import OllamaClient


class CountingClient(OllamaClient):
    """Counts underlying requests without issuing any."""

    def _post(self, path, payload, key):  # type: ignore[override]
        self.calls += 1
        return "reply"


def test_answers_are_cached():
    """The drift bug came from generating the same answer twice."""
    client = CountingClient(model="fake")
    client.answer("q")
    client.answer("q")
    assert client.calls == 1


def test_completions_are_cached_per_temperature():
    """Sampling at a different temperature is a different request."""
    client = CountingClient(model="fake")
    client.complete("p", 0.0)
    client.complete("p", 0.0)
    client.complete("p", 0.7)
    assert client.calls == 2


def test_reset_clears_cache_and_counter():
    """Runs must be independently measurable."""
    client = CountingClient(model="fake")
    client.answer("q")
    client.reset()
    assert client.calls == 0
    client.answer("q")
    assert client.calls == 1


def test_system_prompt_is_only_applied_to_answers():
    """Rewriting and judging must run outside the deployment condition."""
    seen = {}

    class Recording(OllamaClient):
        def _post(self, path, payload, key):  # type: ignore[override]
            seen[path] = payload
            return "reply"

    client = Recording(model="fake", system_prompt="COMPLIANCE")
    client.answer("q")
    client.complete("p")
    roles = [m["role"] for m in seen["/api/chat"]["messages"]]
    assert "system" in roles
    assert "prompt" in seen["/api/generate"]
    assert "COMPLIANCE" not in seen["/api/generate"]["prompt"]


def test_endpoint_follows_ollama_host(monkeypatch):
    """A container must be able to reach a sibling service, not its own localhost."""
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    assert OllamaClient().endpoint == "http://ollama:11434"


def test_endpoint_defaults_to_localhost(monkeypatch):
    """Unset means the developer's own machine, as before."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaClient().endpoint == "http://localhost:11434"
