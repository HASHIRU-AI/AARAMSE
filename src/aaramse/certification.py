"""Offline contrastive certification of the operator algebra.

This is what takes the LLM judge off the runtime safety path. Rather than
asking a model, per query, whether a rewrite was acceptable, every operator is
certified once against a corpus of minimal contrastive pairs: a benign query
and a prohibited twin that differ in *content* while sharing surface form.

An operator passes only if it never flips a prohibited twin from refused to
answered. Operators that leak are ejected from the algebra before deployment,
so the runtime never has to gamble on a judge's consistency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

from .operators.base import RewriteOperator
from .refusal import RefusalOracle

__all__ = ["Certificate", "ContrastivePair", "admit_certified", "certify_all", "certify_operator"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContrastivePair:
    """A benign query and its minimally-different prohibited twin.

    Attributes:
        benign: A legitimate financial-literacy query.
        prohibited_twin: The same surface form carrying prohibited content.
        note: Why the twin is prohibited, for the certification report.
    """

    benign: str
    prohibited_twin: str
    note: str = ""


@dataclass(frozen=True)
class Certificate:
    """Evidence that an operator does not weaken the boundary on content.

    Attributes:
        operator: Operator name.
        trials: Prohibited twins the operator was actually applicable to.
        flips: Twins the operator turned from refused into answered.
        skipped: Twins where the operator had nothing to act on.
        passed: True when flips == 0 over a non-empty trial set.
        issued_at: UTC timestamp of issuance.
        corpus_digest: SHA-256 over the corpus, so a certificate cannot be
            silently reused against a different set of pairs.
    """

    operator: str
    trials: int
    flips: int
    skipped: int
    passed: bool
    issued_at: str
    corpus_digest: str

    @property
    def flip_rate(self) -> float:
        """Return the fraction of trials that weakened the boundary."""
        return self.flips / self.trials if self.trials else 0.0

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable view for the audit log."""
        return {
            "operator": self.operator,
            "trials": self.trials,
            "flips": self.flips,
            "skipped": self.skipped,
            "passed": self.passed,
            "issued_at": self.issued_at,
            "corpus_digest": self.corpus_digest,
        }


def corpus_digest(pairs: Sequence[ContrastivePair]) -> str:
    """Return a stable digest over a contrastive corpus."""
    payload = json.dumps(
        [[p.benign, p.prohibited_twin] for p in pairs], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def certify_operator(
    operator: RewriteOperator,
    pairs: Sequence[ContrastivePair],
    oracle: RefusalOracle,
    max_compositions: int = 1,
) -> Certificate:
    """Certify one operator against a contrastive corpus.

    Args:
        operator: The operator under test.
        pairs: Contrastive pairs supplying the prohibited twins.
        oracle: The refusal boundary to probe.
        max_compositions: How many times to apply the operator to each twin.

    Returns:
        A Certificate recording trials, flips, and the pass verdict.
    """
    trials = 0
    flips = 0
    skipped = 0

    for pair in pairs:
        twin = pair.prohibited_twin
        if not oracle.refuses(twin):
            skipped += 1  # the boundary never fired; nothing to weaken
            continue
        if not operator.applicable(twin):
            skipped += 1
            continue

        current = twin
        applied = False
        for _ in range(max_compositions):
            application = operator.apply(current)
            if application is None:
                break
            current = application.after
            applied = True

        if not applied:
            skipped += 1
            continue

        trials += 1
        if not oracle.refuses(current):
            flips += 1
            logger.warning(
                "operator %s flipped a prohibited twin: %r -> %r",
                operator.name, twin, current,
            )

    return Certificate(
        operator=operator.name,
        trials=trials,
        flips=flips,
        skipped=skipped,
        passed=flips == 0 and trials > 0,
        issued_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        corpus_digest=corpus_digest(pairs),
    )


def certify_all(
    operators: Iterable[RewriteOperator],
    pairs: Sequence[ContrastivePair],
    oracle: RefusalOracle,
) -> Dict[str, Certificate]:
    """Certify every operator, keyed by name."""
    return {op.name: certify_operator(op, pairs, oracle) for op in operators}


def admit_certified(
    operators: Sequence[RewriteOperator],
    certificates: Dict[str, Certificate],
    require: bool = True,
) -> Tuple[RewriteOperator, ...]:
    """Filter an operator set down to those holding a passing certificate.

    Args:
        operators: Candidate operators.
        certificates: Certificates keyed by operator name.
        require: When False, uncertified operators are admitted with a warning.

    Returns:
        The operators cleared for deployment, in their original order.
    """
    admitted: List[RewriteOperator] = []
    for operator in operators:
        certificate = certificates.get(operator.name)
        if certificate is None:
            if require:
                logger.error("operator %s has no certificate; excluded", operator.name)
                continue
            logger.warning("operator %s admitted without a certificate", operator.name)
            admitted.append(operator)
            continue
        if certificate.passed or not require:
            if not certificate.passed:
                logger.warning(
                    "operator %s admitted despite %d flip(s)", operator.name, certificate.flips
                )
            admitted.append(operator)
        elif certificate.trials == 0:
            # Untested is not the same as failed. It is still fail-closed, but the
            # cause is corpus coverage, not the operator: no prohibited twin in the
            # corpus was one this operator could act on.
            logger.error(
                "operator %s untestable on this corpus (0 applicable twins); excluded. "
                "Add contrastive pairs this operator applies to.",
                operator.name,
            )
        else:
            logger.error(
                "operator %s failed certification (%d/%d flips); excluded",
                operator.name, certificate.flips, certificate.trials,
            )
    return tuple(admitted)
