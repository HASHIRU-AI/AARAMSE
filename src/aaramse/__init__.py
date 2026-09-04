"""AARAMSE rewrite stage: a monotone pragmatic rewrite calculus.

Repairs over-refusal without becoming a jailbreak primitive. Rewrites are
programs over a closed, certified operator algebra that can only move a query
upward on an actionability lattice, so the worst case is that a user receives
general financial education instead of personalised advice.
"""

from .audit import AuditLog
from .budget import BudgetExceeded, BudgetVerdict, LeakageBudget, measure_leakage
from .certification import (
    Certificate,
    ContrastivePair,
    admit_certified,
    certify_all,
    certify_operator,
)
from .client import CachingClient, ModelClient, OllamaClient
from .equivalence import SemanticEquivalence
from .fidelity import AnswerCheck, FidelityReport, MeaningFidelity
from .gateway import Gateway, GatewayConfig, JudgedProbe
from .invariants import ActionabilityScorer, IntentGuard, topic_core
from .operators import FrameConfig, build_operators
from .providers import AnthropicClient, OpenAIClient, build_client
from .refusal import HeuristicRefusalOracle, ModelRefusalOracle, RefusalOracle
from .report import InterventionReport, build_report
from .search import RepairSearch, SearchConfig, search_space_size
from .splits import Split, assert_disjoint, split_items
from .types import Decision, OperatorKind, RepairResult, RewriteProgram

__version__ = "0.1.0"

__all__ = [
    "ActionabilityScorer",
    "AnswerCheck",
    "AnthropicClient",
    "AuditLog",
    "BudgetExceeded",
    "BudgetVerdict",
    "CachingClient",
    "Certificate",
    "ContrastivePair",
    "Decision",
    "FidelityReport",
    "FrameConfig",
    "Gateway",
    "GatewayConfig",
    "HeuristicRefusalOracle",
    "IntentGuard",
    "InterventionReport",
    "JudgedProbe",
    "LeakageBudget",
    "MeaningFidelity",
    "ModelClient",
    "ModelRefusalOracle",
    "OllamaClient",
    "OpenAIClient",
    "OperatorKind",
    "RefusalOracle",
    "RepairResult",
    "RepairSearch",
    "RewriteProgram",
    "SearchConfig",
    "SemanticEquivalence",
    "Split",
    "admit_certified",
    "assert_disjoint",
    "build_client",
    "build_operators",
    "build_report",
    "certify_all",
    "certify_operator",
    "measure_leakage",
    "search_space_size",
    "split_items",
    "topic_core",
]
