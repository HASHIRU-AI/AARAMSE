"""Closed operator algebra for the AARAMSE rewrite stage.

Importing this package registers every operator. The registry is deliberately
small and enumerable: with |O| operators and search depth k, the layer's total
optimization power is at most |O|^k candidates, all of them inspectable.

The MVP registry holds exactly two operators. The rule-based subtractive family
produced 2 of 23 repairs in the judged evaluation and the generative rephrase
operator was never wired into the gateway, so neither ships here.
"""

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

__all__ = [
    "OPERATOR_REGISTRY",
    "FrameAssert",
    "FrameConfig",
    "RewriteOperator",
    "TargetedConfig",
    "TargetedRepair",
    "build_operators",
    "operator_factory",
    "register_operator",
    "tidy",
]
