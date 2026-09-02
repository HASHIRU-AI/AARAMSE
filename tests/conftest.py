"""Shared fixtures for the rewrite-stage test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.corpus import SEED_PAIRS, TOPICAL_TRIGGERS
from aaramse.invariants import IntentGuard
from aaramse.operators import FrameAssert, FrameConfig, build_operators
from aaramse.operators.base import RewriteOperator
from aaramse.refusal import HeuristicRefusalOracle
from aaramse.search import RepairSearch, SearchConfig


@pytest.fixture
def frame_config() -> FrameConfig:
    """Deployer frame used across tests."""
    return FrameConfig(deployer_name="Test Firm", authorisation_ref="FRN-000000")


@pytest.fixture
def operators(frame_config: FrameConfig) -> Tuple[RewriteOperator, ...]:
    """The full certified algebra."""
    return build_operators(factory_kwargs={"FRAME_ASSERT": lambda: FrameAssert(frame_config)})


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
