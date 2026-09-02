"""Closed operator algebra for the AARAMSE rewrite stage.

Importing this package registers every operator. The registry is deliberately
small and enumerable: with |O| operators and search depth k, the layer's total
optimization power is at most |O|^k candidates, all of them inspectable.
"""

from ..rewriter import LLMRephrase, RewriterConfig, SemanticEquivalence
from ..targeted import TargetedConfig, TargetedRepair
from .additive import FrameAssert, FrameConfig
from .base import (
    OPERATOR_REGISTRY,
    RewriteOperator,
    build_operators,
    operator_factory,
    register_operator,
    tidy,
)
from .subtractive import (
    Definitionalize,
    Deimperativize,
    Depersonalize,
    DeUrgency,
    EntityGeneralize,
    SplitCompound,
)

__all__ = [
    "OPERATOR_REGISTRY",
    "DeUrgency",
    "Definitionalize",
    "Deimperativize",
    "Depersonalize",
    "EntityGeneralize",
    "FrameAssert",
    "FrameConfig",
    "LLMRephrase",
    "RewriteOperator",
    "RewriterConfig",
    "SemanticEquivalence",
    "SplitCompound",
    "TargetedConfig",
    "TargetedRepair",
    "build_operators",
    "operator_factory",
    "register_operator",
    "tidy",
]
