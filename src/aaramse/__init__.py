"""AARAMSE rewrite stage: a monotone pragmatic rewrite calculus.

Repairs over-refusal without becoming a jailbreak primitive. Rewrites are
programs over a closed, certified operator algebra that can only move a query
upward on an actionability lattice, so the worst case is that a user receives
general financial education instead of personalised advice.
"""

from .agreement import Agreement, agree, best_threshold
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
from .gateway import Gateway, GatewayConfig, JudgedProbe
from .invariants import ActionabilityScorer, IntentGuard, topic_core
from .naamse import NaamseRecord, NaamseReport, load_report, shared_prompts
from .operators import FrameConfig, build_operators
from .providers import AnthropicClient, OpenAIClient, build_client
from .refusal import HeuristicRefusalOracle, ModelRefusalOracle, RefusalOracle
from .report import InterventionReport, build_report
from .rewriter import (
    LLMRephrase,
    RewriterConfig,
    SemanticEquivalence,
)
from .search import RepairSearch, SearchConfig, search_space_size
from .splits import Split, assert_disjoint, split_items
from .types import Decision, OperatorKind, RepairResult, RewriteProgram

__version__ = "0.1.0"

__all__ = [
    "ActionabilityScorer",
    "Agreement",
    "AnthropicClient",
    "AuditLog",
    "BudgetExceeded",
    "BudgetVerdict",
    "CachingClient",
    "Certificate",
    "ContrastivePair",
    "Decision",
    "FrameConfig",
    "Gateway",
    "GatewayConfig",
    "HeuristicRefusalOracle",
    "IntentGuard",
    "InterventionReport",
    "JudgedProbe",
    "LLMRephrase",
    "LeakageBudget",
    "ModelClient",
    "ModelRefusalOracle",
    "NaamseRecord",
    "NaamseReport",
    "OllamaClient",
    "OpenAIClient",
    "OperatorKind",
    "RefusalOracle",
    "RepairResult",
    "RepairSearch",
    "RewriteProgram",
    "RewriterConfig",
    "SearchConfig",
    "SemanticEquivalence",
    "Split",
    "admit_certified",
    "agree",
    "assert_disjoint",
    "best_threshold",
    "build_client",
    "build_operators",
    "build_report",
    "certify_all",
    "certify_operator",
    "load_report",
    "measure_leakage",
    "search_space_size",
    "shared_prompts",
    "split_items",
    "topic_core",
]
