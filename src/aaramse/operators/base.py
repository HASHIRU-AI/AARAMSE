"""Operator abstraction and registry for the rewrite calculus.

The operator set is closed and enumerable on purpose. Its size and the search
depth together bound the layer's total optimization power to a number a
supervisor can read off a compliance document, unlike a free-form paraphraser
whose search space is unbounded and undocumentable.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type

from ..types import OperatorApplication, OperatorKind

__all__ = [
    "OPERATOR_REGISTRY",
    "RewriteOperator",
    "build_operators",
    "operator_factory",
    "register_operator",
    "tidy",
]

logger = logging.getLogger(__name__)

OPERATOR_REGISTRY: Dict[str, Type["RewriteOperator"]] = {}

_WS_RE = re.compile(r"\s+")
_SPACE_PUNCT_RE = re.compile(r"\s+([?.,;!])")
_DUP_PUNCT_RE = re.compile(r"([?.,;!])\1+")
_LEADING_PUNCT_RE = re.compile(r"^[\s,;]+")


def tidy(text: str) -> str:
    """Normalise whitespace and punctuation left behind by a substitution."""
    out = _WS_RE.sub(" ", text).strip()
    out = _LEADING_PUNCT_RE.sub("", out)
    out = _SPACE_PUNCT_RE.sub(r"\1", out)
    out = _DUP_PUNCT_RE.sub(r"\1", out)
    out = re.sub(r",\s*\?", "?", out)
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


class RewriteOperator(ABC):
    """A single typed transformation on the pragmatic layer of a query.

    Subclasses must not add specificity, actionability, or a new target to a
    query. Any operator that violates this is rejected at runtime by the
    IntentGuard and, before deployment, by contrastive certification.
    """

    name: str = ""
    kind: OperatorKind = OperatorKind.SUBTRACTIVE
    invariant: str = ""
    idempotent: bool = True

    @abstractmethod
    def applicable(self, text: str) -> bool:
        """Return True when this operator has something to act on."""

    @abstractmethod
    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Transform the text, or return None when the result is a no-op."""

    def reset(self) -> None:  # noqa: B027 - a deliberate no-op default, not an abstract hook
        """Forget any per-turn state. Concrete: most operators keep none.

        An operator instance outlives the turn it runs in, so anything it
        recorded about one query has to be cleared before the next, or a
        console showing what the operator tried would attribute one reader's
        attempt to another's question.
        """

    def _application(
        self,
        before: str,
        after: str,
        generalizations: Tuple[Tuple[str, str], ...] = (),
        dropped: Tuple[str, ...] = (),
    ) -> Optional[OperatorApplication]:
        """Build an application record, collapsing no-ops to None."""
        after = tidy(after)
        if after == tidy(before) or not after:
            return None
        return OperatorApplication(
            operator=self.name,
            before=before,
            after=after,
            generalizations=generalizations,
            dropped=dropped,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.name} kind={self.kind.value}>"


def register_operator(cls: Type[RewriteOperator]) -> Type[RewriteOperator]:
    """Register an operator class under its declared name."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must declare a name")
    if cls.name in OPERATOR_REGISTRY:
        raise ValueError(f"operator {cls.name} is already registered")
    OPERATOR_REGISTRY[cls.name] = cls
    return cls


def operator_factory(name: str) -> Type[RewriteOperator]:
    """Look up a registered operator class by name."""
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown operator {name!r}; registered: {sorted(OPERATOR_REGISTRY)}"
        ) from exc


def build_operators(
    names: Optional[List[str]] = None,
    factory_kwargs: Optional[Dict[str, Callable[[], RewriteOperator]]] = None,
) -> Tuple[RewriteOperator, ...]:
    """Instantiate operators by name, defaulting to the whole registry.

    Args:
        names: Operator names to build. Defaults to every registered operator.
        factory_kwargs: Per-name constructors for operators needing configuration.

    Returns:
        Instantiated operators in a stable, sorted order so that search is
        reproducible across runs.
    """
    factory_kwargs = factory_kwargs or {}
    selected = sorted(names if names is not None else OPERATOR_REGISTRY.keys())
    built: List[RewriteOperator] = []
    for name in selected:
        if name in factory_kwargs:
            built.append(factory_kwargs[name]())
        else:
            built.append(operator_factory(name)())
    return tuple(built)
