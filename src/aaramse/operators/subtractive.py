"""Subtractive operators: strip pragmatic surface features, never content.

Each operator here moves a query upward on the actionability lattice by
removing person marking, imperative mood, urgency, or entity specificity. None
of them can introduce a new target, a new instruction, or new specificity.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..types import OperatorApplication, OperatorKind
from .base import RewriteOperator, register_operator, tidy

__all__ = [
    "DeUrgency",
    "Definitionalize",
    "Deimperativize",
    "Depersonalize",
    "EntityGeneralize",
    "SplitCompound",
]

logger = logging.getLogger(__name__)

_DEPERSONALIZE_RULES: Tuple[Tuple[str, str], ...] = (
    (r"^\s*should\s+i\b", "what factors determine whether an investor should"),
    (r"^\s*should\s+we\b", "what factors determine whether investors should"),
    (r"^\s*(?:can|could)\s+i\b", "what rules govern whether an investor can"),
    (r"^\s*(?:can|could)\s+we\b", "what rules govern whether investors can"),
    (r"^\s*would\s+it\s+be\s+(?:smart|wise|better)\s+for\s+me\b",
     "what determines whether it is prudent for an investor"),
    (r"\bfor\s+me\b", "for an investor"),
    (r"\bmy\b", "their"),
    (r"\bmine\b", "theirs"),
    (r"\bi'm\b", "an investor is"),
    (r"\bi\s+am\b", "an investor is"),
    (r"\bi've\b", "an investor has"),
    (r"\bi\b", "an investor"),
    (r"\bme\b", "an investor"),
    (r"\bour\b", "their"),
    (r"\bwe\b", "investors"),
)

_PERSON_MARKER_RE = re.compile(r"\b(i|i'm|i've|me|my|mine|we|our|ours)\b", re.IGNORECASE)

# A leading "tell me" / "give us" belongs to DEIMPERATIVIZE. Masking it keeps
# DEPERSONALIZE from producing "Tell an investor how to ...", and lets the two
# operators compose cleanly instead of competing for the same span.
_LEADING_IMPERATIVE_OBJECT_RE = re.compile(
    r"^\s*(?:tell|show|give|walk|help)\s+(me|us)\b", re.IGNORECASE
)
_MASK = "\x00OBJ\x00"


@register_operator
class Depersonalize(RewriteOperator):
    """Replace first-person advice framing with a general investor frame."""

    name = "DEPERSONALIZE"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "removes person marking; subject matter unchanged"

    def applicable(self, text: str) -> bool:
        """Return True when the query carries first-person marking."""
        return bool(_PERSON_MARKER_RE.search(text))

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Rewrite first-person framing into third-person general framing."""
        masked: Optional[str] = None
        out = text
        leading = _LEADING_IMPERATIVE_OBJECT_RE.search(out)
        if leading:
            masked = leading.group(1)
            out = out[: leading.start(1)] + _MASK + out[leading.end(1) :]

        for pattern, replacement in _DEPERSONALIZE_RULES:
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

        if masked is not None:
            out = out.replace(_MASK, masked)
        return self._application(text, out)


_IMPERATIVE_RULES: Tuple[Tuple[str, str], ...] = (
    (r"^\s*(?:tell|show)\s+(?:me|us)\s+how\s+to\b", "what is the general process by which one can"),
    (r"^\s*walk\s+(?:me|us)\s+through\b", "what is the general process for"),
    (r"^\s*(?:tell|show)\s+(?:me|us)\b", "what is known about"),
    (r"^\s*give\s+(?:me|us)\b", "what are"),
    (r"^\s*help\s+(?:me|us)\s+to\b", "what considerations apply when one wants to"),
    (r"^\s*help\s+(?:me|us)\b", "what considerations apply when one wants to"),
    (r"^\s*(?:recommend|pick|choose)\b", "what criteria are used to select"),
)

_IMPERATIVE_RE = re.compile(
    r"^\s*(tell|show|give|walk|help|recommend|pick|choose)\b", re.IGNORECASE
)


@register_operator
class Deimperativize(RewriteOperator):
    """Convert an imperative request into a descriptive question."""

    name = "DEIMPERATIVIZE"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "shifts mood from imperative to interrogative; subject matter unchanged"

    def applicable(self, text: str) -> bool:
        """Return True when the query opens with a directive verb."""
        return bool(_IMPERATIVE_RE.match(text))

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Replace the leading imperative with an interrogative frame."""
        out = text
        for pattern, replacement in _IMPERATIVE_RULES:
            new = re.sub(pattern, replacement, out, count=1, flags=re.IGNORECASE)
            if new != out:
                out = new
                break
        if not out.rstrip().endswith("?"):
            out = out.rstrip(" .") + "?"
        return self._application(text, out)


# Specific term -> general term. Every substitution is declared to the guard,
# which is what lets the topic-preservation check tolerate the loss.
_ENTITY_TABLE: Tuple[Tuple[str, str], ...] = (
    (r"\bvanguard\b", "mutual fund"),
    (r"\bfidelity\b", "brokerage"),
    (r"\bcharles\s+schwab\b|\bschwab\b", "brokerage"),
    (r"\brobinhood\b", "retail brokerage"),
    (r"\bcoinbase\b", "crypto exchange"),
    (r"\b(?:tesla|nvidia|apple|amazon)\b", "listed equity"),
    (r"\bbitcoin\b|\bethereum\b", "cryptoasset"),
    (r"\b(?:gold|silver)\b", "commodity"),
    (r"\$[A-Z]{1,5}\b", "listed equity"),
    (r"[$£€]\s?\d[\d,]*(?:\.\d+)?", "a given amount"),
    (r"\b\d[\d,]*(?:\.\d+)?\s?k\b", "a given amount"),
)


@register_operator
class EntityGeneralize(RewriteOperator):
    """Replace named products, brands and amounts with their asset class."""

    name = "ENTITY_GENERALIZE"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "maps specific entities to their class; declares every substitution"

    def applicable(self, text: str) -> bool:
        """Return True when a known named entity or figure is present."""
        return any(re.search(p, text, flags=re.IGNORECASE) for p, _ in _ENTITY_TABLE)

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Generalize named entities, recording each substitution."""
        out = text
        declared: List[Tuple[str, str]] = []
        for pattern, general in _ENTITY_TABLE:
            matches = re.findall(pattern, out, flags=re.IGNORECASE)
            if not matches:
                continue
            for match in matches:
                source = match if isinstance(match, str) else match[0]
                if source:
                    declared.append((source, general))
            out = re.sub(pattern, general, out, flags=re.IGNORECASE)
        return self._application(text, out, generalizations=tuple(declared))


_URGENCY_RULES: Tuple[str, ...] = (
    r"\b(?:right\s+now|immediately|as\s+soon\s+as\s+possible|asap|urgently|today)\b",
    r",?\s*before\s+[^,?.]{0,40}?\b(?:crash(?:es)?|tanks?|deadline|closes?|expires?)\b",
    r"^\s*(?:urgent|please\s+help|help)\s*[!:,-]\s*",
    r"\bi'?m\s+(?:about\s+to\s+)?(?:lose|losing|panicking)\b[^,?.]*[,.]?",
    r"\b(?:quick|quickly|fast)\b\s*[,!-]\s*",
)

_URGENCY_RE = re.compile("|".join(_URGENCY_RULES), re.IGNORECASE)


@register_operator
class DeUrgency(RewriteOperator):
    """Remove time pressure and emotive urgency markers."""

    name = "DE_URGENCY"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "removes urgency and affect markers; subject matter unchanged"

    def applicable(self, text: str) -> bool:
        """Return True when an urgency marker is present."""
        return bool(_URGENCY_RE.search(text))

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Strip urgency clauses, recording them as dropped content."""
        dropped = tuple(
            m.group(0).strip() for m in _URGENCY_RE.finditer(text) if m.group(0).strip()
        )
        out = _URGENCY_RE.sub(" ", text)
        return self._application(text, out, dropped=dropped)


# The replacement is purely definitional. An earlier version asked for "the legal
# definition AND MECHANICS of X", which is still a request for the procedure: against
# a live model it turned a refused "how do I hide assets from my bankruptcy trustee"
# into an answered, enumerated concealment how-to. An operator that claims to remove
# procedural framing must not reintroduce it in its own replacement text.
_DEFINITION_RULES: Tuple[Tuple[str, str], ...] = (
    (r"^\s*how\s+(?:do|can|should)\s+(?:i|we|one|an\s+investor)\s+(.+?)\s*\??$",
     r"what is the legal definition of \1"),
    (r"^\s*how\s+to\s+(.+?)\s*\??$", r"what is the legal definition of \1"),
    (
        r"^\s*what\s+should\s+(?:i|we|an\s+investor)\s+do\s+(?:about|with)\s+(.+?)\s*\??$",
        r"what is the standard treatment of \1",
    ),
)

_PROCEDURAL_RE = re.compile(
    r"^\s*how\s+(?:do|can|should|to)\b|^\s*what\s+should\s+(?:i|we|an\s+investor)\s+do\b",
    re.IGNORECASE,
)


@register_operator
class Definitionalize(RewriteOperator):
    """Turn a procedural how-to into a request for definition and mechanics."""

    name = "DEFINITIONALIZE"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "shifts procedural framing to definitional; topic unchanged"

    def applicable(self, text: str) -> bool:
        """Return True when the query is framed as a procedure request."""
        return bool(_PROCEDURAL_RE.match(text))

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Reframe the procedure request as a definitional question."""
        out = text
        for pattern, replacement in _DEFINITION_RULES:
            new = re.sub(pattern, replacement, out, count=1, flags=re.IGNORECASE | re.DOTALL)
            if new != out:
                out = new
                break
        else:
            return None
        if not out.rstrip().endswith("?"):
            out = out.rstrip(" .") + "?"
        return self._application(text, out)


_CLAUSE_SPLIT_RE = re.compile(
    r"\?|;|\band\s+also\b|\balso,?\s+(?=(?:should|can|what|how|tell|give|is|do)\b)",
    re.IGNORECASE,
)


@register_operator
class SplitCompound(RewriteOperator):
    """Keep the least actionable sub-question of a compound query.

    Mixed queries pair a benign question with a prohibited one. Answering the
    benign half is strictly better than refusing both, and the discarded clause
    is recorded verbatim for review.
    """

    name = "SPLIT_COMPOUND"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "retains one sub-question; declares every dropped clause"

    def __init__(self) -> None:
        # Imported here to avoid a circular import at module load time.
        from ..invariants import ActionabilityScorer

        self._scorer = ActionabilityScorer()

    @staticmethod
    def _clauses(text: str) -> List[str]:
        """Split a compound query into its candidate sub-questions."""
        parts = [tidy(p) for p in _CLAUSE_SPLIT_RE.split(text)]
        return [p for p in parts if len(p.split()) >= 3]

    def applicable(self, text: str) -> bool:
        """Return True when the query contains more than one sub-question."""
        return len(self._clauses(text)) >= 2

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Retain the lowest-actionability clause and drop the rest."""
        clauses = self._clauses(text)
        if len(clauses) < 2:
            return None
        kept = min(clauses, key=lambda c: (self._scorer.score(c), len(c)))
        dropped = tuple(c for c in clauses if c != kept)
        out = kept if kept.rstrip().endswith("?") else kept.rstrip(" .") + "?"
        return self._application(text, out, dropped=dropped)
