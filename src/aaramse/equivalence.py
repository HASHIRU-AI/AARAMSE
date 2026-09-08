"""Intent-equivalence judging for candidate rewrites.

Verifies that a candidate rewrite still fulfills the user's original information need.
`TargetedRepair` requires this check before admitting any proposed rewrite.

Why intent equivalence replaces cosine embedding similarity:
Embedding cosine distance often registers high similarity scores between any two
sentences discussing the same general topic (e.g., scoring high between "How do I evade taxes?"
and "What is the tax code?"), failing to detect material shifts in intent. Intent-equivalence
directly verifies whether answering the rewritten query provides the information the user
originally sought.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

__all__ = ["EQUIVALENCE_PROMPT", "SemanticEquivalence"]

logger = logging.getLogger(__name__)

EQUIVALENCE_PROMPT = """You are checking whether a rephrased question still serves the \
original information need.

Original:  {a}
Rephrased: {b}

Different wording, framing, or level of generality is fine and expected. Answer NO only if the \
rephrased question asks about a materially different subject, or if answering it would leave \
the original question unanswered.

Answer with exactly one word: YES or NO."""


@dataclass
class SemanticEquivalence:
    """Judges whether a rewrite still asks for the same information.

    Attributes:
        generate: Completion callable used to ask the model for a verdict.
        fail_closed: When the verdict is unparseable, treat it as NO.
    """

    generate: Callable[[str], str]
    fail_closed: bool = True

    def __post_init__(self) -> None:
        self._cache: Dict[Tuple[str, str], bool] = {}

    def equivalent(self, original: str, rewrite: str) -> bool:
        """Return True when the rewrite requests the same information.

        Args:
            original: The user's query as received.
            rewrite: The candidate replacement.

        Returns:
            True when the two ask for the same information.
        """
        key = (original, rewrite)
        if key in self._cache:
            return self._cache[key]
        raw = self.generate(EQUIVALENCE_PROMPT.format(a=original, b=rewrite))
        head = raw.strip().lower()[:40]
        if "yes" in head:
            verdict = True
        elif "no" in head:
            verdict = False
        else:
            logger.warning("unparseable equivalence verdict %r", raw[:60])
            verdict = not self.fail_closed
        self._cache[key] = verdict
        return verdict
