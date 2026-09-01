"""Shared fixtures for the rewrite-stage test suite.

The production registry holds two operators, and both need a live model:
`TARGETED_REPAIR` calls one to propose fragment replacements, `FRAME_ASSERT` is
rule-based but additive-only. That leaves no model-free way to *subtract* from a
query, which is what the search, audit and report tests need in order to produce
a REPAIRED decision offline.

So the subtractive side is supplied here as test doubles. They are deliberately
small -- two regexes and a lattice check each -- because their job is to give
the search something legal to do, not to be good at repair. The rule-based
subtractive family that used to live in the package did exactly this job in the
suite while producing 2 of 23 repairs against a real model, which is why it
ships nowhere near a deployment now.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.corpus import SEED_PAIRS, TOPICAL_TRIGGERS
from aaramse.invariants import IntentGuard
from aaramse.operators import FrameAssert, FrameConfig
from aaramse.operators.base import RewriteOperator, tidy
from aaramse.refusal import HeuristicRefusalOracle
from aaramse.search import RepairSearch, SearchConfig
from aaramse.types import OperatorApplication, OperatorKind

_PERSON_RE = re.compile(r"\b(i|i'm|i've|me|my|mine|we|our|ours)\b", re.IGNORECASE)
_PERSON_RULES: Tuple[Tuple[str, str], ...] = (
    (r"^\s*should\s+i\b", "what factors determine whether an investor should"),
    (r"^\s*walk\s+(?:me|us)\s+through\b", "what is the general process for"),
    (r"\bmy\b", "their"),
    (r"\bme\b", "an investor"),
    (r"\bwe\b", "investors"),
    (r"\bour\b", "their"),
    (r"\bi\b", "an investor"),
)

_URGENCY_RE = re.compile(
    r"\s*\b(right now|immediately|asap|as soon as possible|urgently|urgent|today)\b",
    re.IGNORECASE,
)


class DepersonalizeDouble(RewriteOperator):
    """Test double: strips first-person marking, leaving subject matter alone.

    Not registered globally. The production registry stays at two operators; a
    test that asserts what ships must not be able to see this.
    """

    name = "DEPERSONALIZE"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "removes person marking; subject matter unchanged"

    def applicable(self, text: str) -> bool:
        """Return True when the query carries first-person marking."""
        return bool(_PERSON_RE.search(text))

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Rewrite first-person framing into a general investor frame."""
        out = text
        for pattern, replacement in _PERSON_RULES:
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        return self._application(text, tidy(out))


class DeUrgencyDouble(RewriteOperator):
    """Test double: drops urgency markers, which carry no subject matter."""

    name = "DE_URGENCY"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "removes urgency marking; subject matter unchanged"

    def applicable(self, text: str) -> bool:
        """Return True when the query carries an urgency marker."""
        return bool(_URGENCY_RE.search(text))

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Delete urgency markers."""
        return self._application(text, tidy(_URGENCY_RE.sub("", text)))


@pytest.fixture
def frame_config() -> FrameConfig:
    """Deployer frame used across tests."""
    return FrameConfig(deployer_name="Test Firm", authorisation_ref="FRN-000000")


@pytest.fixture
def operators(frame_config: FrameConfig) -> Tuple[RewriteOperator, ...]:
    """A model-free algebra: two subtractive doubles plus the shipped frame."""
    return (DepersonalizeDouble(), DeUrgencyDouble(), FrameAssert(frame_config))


@pytest.fixture
def oracle() -> HeuristicRefusalOracle:
    """Simulated refusal boundary with pragmatic, content and topical triggers."""
    return HeuristicRefusalOracle(topical_triggers=TOPICAL_TRIGGERS)


@pytest.fixture
def guard() -> IntentGuard:
    """Runtime invariant checker."""
    return IntentGuard()


@pytest.fixture
def search(operators, oracle) -> RepairSearch:
    """Repair search bounded at depth 3."""
    return RepairSearch(operators, oracle, config=SearchConfig(max_depth=3))


@pytest.fixture
def pairs():
    """Seed contrastive corpus."""
    return SEED_PAIRS
