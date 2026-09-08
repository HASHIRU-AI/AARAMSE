"""The actionability lattice and the invariants every rewrite must satisfy.

Defines the safety rules that distinguish safe refusal repair from adversarial
jailbreaking. Every rewrite candidate must satisfy two key invariants:

1. **Monotone generalization on an actionability lattice:**
   Actionability measures how directive, specific, or personalized a query is (e.g.
   "Should I sell my TSLA shares now?" is highly actionable, whereas "How are stock
   options valued?" is general education). The **actionability lattice** is a
   rule-based hierarchy that enforces that every rewrite is *no more actionable*
   than the query it replaces. A rewrite can only move toward general education,
   never toward personalized advice or procedural execution.
2. **Topic preservation:**
   The core subject matter of the query is preserved, except for specific words
   the operator explicitly declared it generalized or removed. This prevents the
   repair from drifting into answering an entirely different question.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import FrozenSet, Tuple

from .types import ActionabilityProfile, RewriteProgram

__all__ = [
    "FRAME_SEPARATOR",
    "ActionabilityScorer",
    "GuardVerdict",
    "IntentGuard",
    "strip_frame",
    "topic_core",
]

logger = logging.getLogger(__name__)

# Additive operators prepend a deployer frame terminated by this separator.
# Scoring always runs on the payload so that framing cannot deflate the score.
FRAME_SEPARATOR = "\n\nQuestion: "

# Feature name, pattern, weight. A feature contributes its weight at most once,
# so scores stay stable under harmless repetition.
_FEATURE_PATTERNS: Tuple[Tuple[str, str, float], ...] = (
    ("personal_stake", r"\b(my|mine|our|ours|i'm|i am|i've|i have)\b", 2.0),
    ("personal_directive", r"\b(should|shall|ought|must)\s+(i|we)\b", 2.0),
    (
        "imperative_request",
        r"^\s*(tell|show|give|walk|help)\s+(me|us)\b|^\s*(recommend|pick|choose|buy|sell)\b",
        1.5,
    ),
    (
        # Any request for a procedure, however it is dressed. "mechanics of X" and
        # "the process by which one can X" are procedural requests; scoring only the
        # surface string "how do i" let DEFINITIONALIZE launder a how-to into a
        # "definitional" query that the model then answered as a how-to.
        "procedural",
        r"\bhow\s+(?:do|can|should|would|might)\s+(?:i|we|one|an\s+investor)\b"
        r"|\bhow\s+to\b|\bhow\s+it\s+works\b|\bsteps?\s+(?:to|for)\b"
        r"|\bmechanics\b|\bprocedure\b|\bprocess\s+(?:by|for|of)\b"
        r"|\btechniques?\s+(?:to|for)\b|\bmethods?\s+(?:to|for|of)\b"
        r"|\bwalk\s+(?:me|us)\s+through\b",
        1.5,
    ),
    (
        "named_entity",
        r"\$[A-Za-z]{1,5}\b|\b(vanguard|fidelity|schwab|robinhood|coinbase|tesla|nvidia|bitcoin)\b",
        1.0,
    ),
    (
        "specific_amount",
        r"[$£€]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|pounds|euros|k)\b",
        1.0,
    ),
    # "right away" was missing, and the omission was load-bearing: asked to
    # rewrite the fragment "immediately?", muse-spark-1.2 returned "right away".
    # The request is exactly as urgent, but the score fell, so the guard read a
    # lateral swap as a generalization and admitted a rewrite that generalized
    # nothing. Synonyms of an entry already here must score as that entry.
    (
        "urgency",
        r"\b(right now|right away|straight away|at once|immediately|asap"
        r"|as soon as possible|urgent(?:ly)?|today)\b",
        1.0,
    ),
)

# Domain vocabulary used to check that the propositional core survived.
_DOMAIN_TERMS: FrozenSet[str] = frozenset(
    (
        "401k", "account", "accounts", "adviser", "advisor", "allocation", "amortization",
        "annuitant", "annuity", "apr", "assets", "audit", "bankruptcy", "beneficiary", "bitcoin",
        "bond", "bonus", "broker", "brokerage", "capital", "chapter", "claim", "collateral",
        "commodities", "commodity", "compliance", "credit", "creditor", "crypto", "cryptoasset",
        "custody", "debt", "deductible", "deductible", "deduction", "disclosure",
        "diversification", "dividend", "earnings", "equity", "escrow", "etf", "fca", "fiduciary",
        "filing", "fund", "futures", "gains", "gold", "harvesting", "hedge", "hmrc", "income",
        "index", "insurance", "interest", "invoice", "invoices", "ira", "irs", "leverage",
        "liquidity", "loan", "long", "losses", "margin", "mortgage", "mutual", "option",
        "options", "pension", "policy", "portfolio", "premium", "prospectus", "rebalance",
        "refinance", "retirement", "roth", "sale", "savings", "sec", "shares", "short", "silver",
        "stock", "stocks", "suitability", "tax", "taxable", "taxes", "trustee", "underwriting",
        "valuation", "wash", "yield",
    )
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_NORMALIZE_RE = re.compile(r"401\s*\(\s*k\s*\)", re.IGNORECASE)


def strip_frame(text: str) -> str:
    """Return the question payload, discarding any additive deployer frame."""
    if FRAME_SEPARATOR in text:
        return text.split(FRAME_SEPARATOR, 1)[1]
    return text


def topic_core(text: str) -> FrozenSet[str]:
    """Extract the domain terms carrying the query's propositional content."""
    normalized = _NORMALIZE_RE.sub("401k", strip_frame(text).lower())
    tokens = set(_WORD_RE.findall(normalized))
    return frozenset(tokens & _DOMAIN_TERMS)


@dataclass(frozen=True)
class ActionabilityScorer:
    """Scores where a query sits on the actionability lattice.

    The scorer is deterministic and rule-based on purpose. A learned scorer
    would put a model's judgement back on the runtime safety path, which is the
    variance this design exists to remove.
    """

    def profile(self, text: str) -> ActionabilityProfile:
        """Return the actionability profile of the query payload."""
        payload = strip_frame(text).lower()
        matched: list[Tuple[str, float]] = []
        for name, pattern, weight in _FEATURE_PATTERNS:
            if re.search(pattern, payload, flags=re.IGNORECASE | re.MULTILINE):
                matched.append((name, weight))
        return ActionabilityProfile(
            score=sum(weight for _, weight in matched),
            features=tuple(matched),
        )

    def score(self, text: str) -> float:
        """Return only the scalar actionability score."""
        return self.profile(text).score


@dataclass(frozen=True)
class GuardVerdict:
    """Result of checking a candidate rewrite against the invariants."""

    admitted: bool
    reason: str = ""


@dataclass(frozen=True)
class IntentGuard:
    """Enforces monotone generalization and topic preservation.

    Attributes:
        scorer: Lattice scorer used for the monotonicity check.
        tolerance: Slack allowed on the monotonicity comparison.
    """

    scorer: ActionabilityScorer = ActionabilityScorer()
    tolerance: float = 1e-9

    def check(self, query: str, candidate: str, program: RewriteProgram) -> GuardVerdict:
        """Admit a candidate rewrite only if both invariants hold.

        Args:
            query: The original user query.
            candidate: The proposed rewrite.
            program: The operator program that produced the candidate.

        Returns:
            A verdict carrying the reason for any rejection.
        """
        before = self.scorer.score(query)
        after = self.scorer.score(candidate)
        if after > before + self.tolerance:
            return GuardVerdict(
                admitted=False,
                reason=f"monotonicity violated: actionability {before:.1f} -> {after:.1f}",
            )

        lost = topic_core(query) - topic_core(candidate)
        if lost:
            declared = self._declared_losses(program)
            undeclared = {term for term in lost if term not in declared}
            if undeclared:
                return GuardVerdict(
                    admitted=False,
                    reason=f"topic drift: undeclared loss of {sorted(undeclared)}",
                )

        return GuardVerdict(admitted=True)

    @staticmethod
    def _declared_losses(program: RewriteProgram) -> FrozenSet[str]:
        """Collect terms the program openly declared it generalized or dropped."""
        declared: set[str] = set()
        for source, _target in program.generalizations():
            declared.update(_WORD_RE.findall(source.lower()))
        for clause in program.dropped():
            declared.update(_WORD_RE.findall(clause.lower()))
        return frozenset(declared)
