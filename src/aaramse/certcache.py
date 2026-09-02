"""On-disk cache for live-earned certificates, so a demo certifies once.

Certification probes the deployed model with every contrastive pair and is the
slow part of standing the gateway up -- minutes against a local model. But a
certificate is a property of *(operator, model, corpus)* and of the system
prompt the model runs under, so it is safe to reuse only while all four are
unchanged. This cache keys on exactly those, refusing a hit when any of them
moves, so `DEFINITIONALIZE certified against a simulator` can never be silently
reloaded against a different model.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

from .certification import Certificate, ContrastivePair, corpus_digest

__all__ = ["cache_key", "load_certificates", "save_certificates"]

logger = logging.getLogger(__name__)


def cache_key(model: str, system_prompt: str, pairs: Sequence[ContrastivePair]) -> str:
    """Identity a cached certificate set is valid for.

    Combines the model spec, a digest of the system prompt the model was
    certified under, and the corpus digest. A change in any of the three is a
    different certification question and must miss.
    """
    prompt_digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    return f"{model}|{prompt_digest}|{corpus_digest(pairs)}"


def save_certificates(
    path: Path,
    model: str,
    system_prompt: str,
    pairs: Sequence[ContrastivePair],
    certificates: Dict[str, Certificate],
) -> None:
    """Write certificates to ``path`` under the key they are valid for."""
    payload = {
        "key": cache_key(model, system_prompt, pairs),
        "model": model,
        "certificates": {name: cert.to_dict() for name, cert in certificates.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info("cached %d certificate(s) to %s", len(certificates), path)


def load_certificates(
    path: Path,
    model: str,
    system_prompt: str,
    pairs: Sequence[ContrastivePair],
) -> Optional[Dict[str, Certificate]]:
    """Load certificates from ``path`` iff they match the current identity.

    Returns None -- a cache miss the caller must certify past -- when the file is
    absent, unreadable, or was written for a different model, prompt, or corpus.
    A stale cache is never returned in place of re-certifying.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("certificate cache at %s unreadable (%s); ignoring", path, exc)
        return None
    want = cache_key(model, system_prompt, pairs)
    if payload.get("key") != want:
        logger.info("certificate cache at %s is for a different identity; ignoring", path)
        return None
    certificates = {
        name: Certificate.from_dict(data)
        for name, data in payload.get("certificates", {}).items()
    }
    logger.info("loaded %d cached certificate(s) from %s", len(certificates), path)
    return certificates
