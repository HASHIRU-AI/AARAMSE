"""Offline certification keeps the LLM judge off the runtime safety path."""

from __future__ import annotations

from typing import Optional

from aaramse.certification import admit_certified, certify_all, certify_operator, corpus_digest
from aaramse.operators.base import RewriteOperator
from aaramse.types import OperatorApplication, OperatorKind


class ContentStripper(RewriteOperator):
    """A deliberately unsafe operator that deletes prohibited content words.

    It exists only to prove certification catches a leaky operator. It is not
    registered, so it can never reach a deployment.
    """

    name = "UNSAFE_CONTENT_STRIP"
    kind = OperatorKind.SUBTRACTIVE
    invariant = "claims to be pragmatic; actually removes content"

    _BANNED = ("hide", "evade", "insider", "falsify", "launder", "under-report", "front-run")

    def applicable(self, text: str) -> bool:
        return any(word in text.lower() for word in self._BANNED)

    def apply(self, text: str) -> Optional[OperatorApplication]:
        out = text
        for word in self._BANNED:
            out = out.replace(word, "handle").replace(word.capitalize(), "Handle")
        return OperatorApplication(operator=self.name, before=text, after=out)


def test_honest_operators_pass_certification(operators, oracle, pairs):
    """Every shipped operator must leave prohibited twins refused."""
    certificates = certify_all(operators, pairs, oracle)
    for name, certificate in certificates.items():
        assert certificate.flips == 0, f"{name} flipped {certificate.flips} twin(s)"


def test_leaky_operator_fails_and_is_excluded(oracle, pairs):
    """An operator that touches content must be caught and ejected."""
    bad = ContentStripper()
    certificate = certify_operator(bad, pairs, oracle)
    assert certificate.trials > 0
    assert certificate.flips > 0
    assert not certificate.passed
    assert certificate.flip_rate > 0

    admitted = admit_certified([bad], {bad.name: certificate}, require=True)
    assert admitted == ()


def test_uncertified_operator_is_excluded_when_required(operators, oracle, pairs):
    """Missing certificates fail closed."""
    certificates = certify_all(operators, pairs, oracle)
    dropped = next(iter(certificates))
    del certificates[dropped]
    admitted = admit_certified(operators, certificates, require=True)
    assert dropped not in {op.name for op in admitted}


def test_certificate_binds_to_its_corpus(operators, oracle, pairs):
    """A certificate cannot be silently reused against a different corpus."""
    certificates = certify_all(operators, pairs, oracle)
    digest = corpus_digest(pairs)
    assert all(c.corpus_digest == digest for c in certificates.values())
    assert corpus_digest(pairs[:-1]) != digest
