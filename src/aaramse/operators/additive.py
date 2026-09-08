"""Additive operators: supply authorized compliance context without altering the query.

Some over-refusals are triggered by financial keywords themselves rather than by
how the user phrased the question. For example, "What is the legal definition of
tax-loss harvesting?" contains no unnecessary words to remove.

The appropriate fix is to provide the model with the authorized regulatory context
it was missing. The **deployer frame** prepends an explicit statement establishing
that the response is permitted general financial education under the firm's regulatory
authorization.

Security note: Frame text is generated exclusively from deployer configuration
(`deployer_name` and `authorisation_ref`), never from the user's prompt. This prevents
the operator from becoming a prompt-injection attack vector.
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
