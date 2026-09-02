"""Certificate caching: certify once, reuse only under the same identity."""

from __future__ import annotations

from aaramse.certcache import cache_key, load_certificates, save_certificates
from aaramse.certification import certify_all


def _certs(operators, pairs, oracle):
    return certify_all(operators, pairs, oracle)


def test_roundtrip_preserves_certificates(tmp_path, operators, pairs, oracle):
    """A cached set reloads identical, so a reload admits the same operators."""
    certs = _certs(operators, pairs, oracle)
    path = tmp_path / "certs.json"
    save_certificates(path, "sim-model", "sys prompt", pairs, certs)

    loaded = load_certificates(path, "sim-model", "sys prompt", pairs)
    assert loaded is not None
    assert set(loaded) == set(certs)
    for name, cert in certs.items():
        assert loaded[name] == cert  # frozen dataclass equality


def test_missing_file_is_a_clean_miss(tmp_path, pairs):
    """No cache file means certify from scratch, not an error."""
    assert load_certificates(tmp_path / "absent.json", "m", "s", pairs) is None


def test_a_different_model_misses(tmp_path, operators, pairs, oracle):
    """A certificate is a property of the model; another model must not reuse it."""
    certs = _certs(operators, pairs, oracle)
    path = tmp_path / "certs.json"
    save_certificates(path, "model-a", "sys", pairs, certs)
    assert load_certificates(path, "model-b", "sys", pairs) is None


def test_a_different_prompt_misses(tmp_path, operators, pairs, oracle):
    """The system prompt is part of the certification condition."""
    certs = _certs(operators, pairs, oracle)
    path = tmp_path / "certs.json"
    save_certificates(path, "m", "prompt one", pairs, certs)
    assert load_certificates(path, "m", "prompt two", pairs) is None


def test_a_corrupt_cache_is_ignored_not_raised(tmp_path, pairs):
    """A garbled cache file degrades to a miss, never a crash mid-demo."""
    path = tmp_path / "certs.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_certificates(path, "m", "s", pairs) is None


def test_cache_key_moves_with_each_input(pairs):
    """The key separates model, prompt and corpus so none can be confused."""
    base = cache_key("m", "s", pairs)
    assert cache_key("m2", "s", pairs) != base
    assert cache_key("m", "s2", pairs) != base


def test_cached_certs_admit_the_same_operators(tmp_path, operators, pairs, oracle, guard):
    """The whole point: a cache-loaded gateway repairs like a freshly-certified one."""
    from aaramse.gateway import Gateway, GatewayConfig

    cfg = GatewayConfig(model="sim", audit_path=tmp_path / "a.jsonl")
    fresh = Gateway.build(config=cfg, client=_ScriptedRefuser())
    fresh_certs = fresh.certify(pairs)
    fresh_admitted = {op.name for op in fresh.operators}

    path = tmp_path / "certs.json"
    save_certificates(path, "sim", cfg.system_prompt, pairs, fresh_certs)
    loaded = load_certificates(path, "sim", cfg.system_prompt, pairs)

    cached = Gateway.build(
        config=GatewayConfig(model="sim", audit_path=tmp_path / "b.jsonl"),
        client=_ScriptedRefuser(),
    )
    cached.apply_certificates(loaded)
    assert {op.name for op in cached.operators} == fresh_admitted


class _ScriptedRefuser:
    """Minimal client: refuses the seed prohibited twins, answers the rest."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, prompt: str, max_tokens: int = 320) -> str:
        self.calls += 1
        low = prompt.lower()
        framed = "authorisation reference" in low
        if not framed and any(w in low for w in ("hide", "conceal", "evade", "launder")):
            return "I cannot help with that request."
        return "Here is general, factual information."

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 60) -> str:
        self.calls += 1
        if "1_full_compliance" in prompt:
            return "2_full_refusal" if "cannot" in prompt.split("RESPONSE:", 1)[-1].lower() \
                else "1_full_compliance"
        if "request the same information" in prompt:
            return "YES"
        return prompt.split('Blocking fragment: "', 1)[-1].split('"', 1)[0]
