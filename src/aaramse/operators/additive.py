"""Additive operators: supply deployer context without touching the question.

Some over-refusals are triggered by the topic itself rather than by how the
question is phrased. "What is the legal definition of tax-loss harvesting?"
carries no pragmatic feature left to strip, so subtraction cannot help it.

The honest repair is to supply the regulatory context the guardrail lacked.
Frame text is drawn only from deployer configuration and never from the query,
which is what stops this operator class from becoming a prompt-injection
surface: a user cannot influence a single token of what gets prepended.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..invariants import FRAME_SEPARATOR
from ..types import OperatorApplication, OperatorKind
from .base import RewriteOperator, register_operator

__all__ = ["FrameAssert", "FrameConfig"]

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = (
    "The following question is answered as general financial education by "
    "{deployer_name} (authorisation reference {authorisation_ref}). "
    "Provide factual, general information about the subject matter. "
    "Do not provide personalised investment, tax or legal advice, and state "
    "plainly where a licensed professional is required."
)


@dataclass(frozen=True)
class FrameConfig:
    """Deployer-supplied regulatory context.

    Attributes:
        deployer_name: Name of the authorised firm operating the gateway.
        authorisation_ref: Regulatory authorisation reference for that firm.
        template: Frame body. Must not interpolate anything user-controlled.
    """

    deployer_name: str = "the deploying firm"
    authorisation_ref: str = "UNSET"
    template: str = _DEFAULT_TEMPLATE

    def render(self) -> str:
        """Render the frame prefix, terminated by the payload separator."""
        body = self.template.format(
            deployer_name=self.deployer_name,
            authorisation_ref=self.authorisation_ref,
        )
        return body + FRAME_SEPARATOR


@register_operator
class FrameAssert(RewriteOperator):
    """Prepend a fixed, config-supplied regulatory frame to the query."""

    name = "FRAME_ASSERT"
    kind = OperatorKind.ADDITIVE
    invariant = "prepends deployer-configured context; the question is byte-identical"

    def __init__(self, config: Optional[FrameConfig] = None) -> None:
        self._config = config or FrameConfig()

    @property
    def config(self) -> FrameConfig:
        """Return the frame configuration in force."""
        return self._config

    def applicable(self, text: str) -> bool:
        """Return True when the query is not already framed."""
        return FRAME_SEPARATOR not in text

    def apply(self, text: str) -> Optional[OperatorApplication]:
        """Attach the deployer frame ahead of the unmodified question."""
        if FRAME_SEPARATOR in text:
            return None
        framed = self._config.render() + text
        # Bypasses _application: tidy() would mangle the frame's punctuation,
        # and the payload must stay byte-identical for the invariant to hold.
        return OperatorApplication(operator=self.name, before=text, after=framed)
