# Responsible use

AARAMSE is licensed under Apache-2.0 (see `LICENSE`). That licence places no
restriction on the field of use. This document states the project's conditions
of intended use. **They are norms, not additional licence terms**, and nothing
here is legal advice.

We state them separately, and say plainly that they are unenforced, because the
alternative — bolting use restrictions onto a permissive licence — produces a
document that is neither open source nor reliably enforceable, and reads as a
stronger guarantee than it is.

## What this tool is

A layer that detects when a deployed model refuses a legitimate question and
repairs the refusal. Mechanically, that is a system for getting a model to
answer something it declined to answer. The distance between that and a
jailbreak primitive is not the mechanism; it is the controls around it.

Those safety controls are:
- **Refusal-only entry:** Only queries the deployed model actually refused are
  eligible for repair; normal questions pass through untouched.
- **Pre-admission certification:** Only rewrite operators holding a verified
  passing certificate on that specific model may run.
- **Structural confinement:** Edits are strictly confined to the localized
  trigger fragment in deterministic code, keeping the rest of the question
  byte-identical.
- **Actionability limits:** Candidate rewrites that increase actionability
  (e.g., asking for specific, personalized advice rather than educational
  context) are rejected.
- **Deterministic escalation:** Any unresolved or potentially harmful query
  immediately escalates to a human reviewer.
- **Tamper-evident audit trail:** Every decision and edit is written to an
  append-only, cryptographically hash-chained log. Each record carries its
  predecessor's SHA-256, so a record cannot be altered after the fact and leave
  the rest of the chain intact.

## Intended use

- Restoring access to lawful financial, legal, and medical information that a
  safety-aligned model wrongly declines.
- Empirical research on over-refusal in LLMs, its measurement, and safe repair.
- Regulatory compliance and supervisory review of a deployed agent's refusal
  behaviour.

## Uses we ask you not to make of it

- **Bypassing legitimate safety refusals.** Repairing a refusal for a genuinely
  harmful or illegal request is the critical failure mode this project exists
  to prevent, not an application of it.
- **Disabling certification in production (`require_certificates=False`).**
  This flag exists solely for offline research on uncertified operators. In
  production, running uncertified operators means their safety on the deployed
  model has not been verified.
- **Raising the leakage budget without an audited decision.** The leakage
  budget is a deployer's explicit, measurable limit on accepted risk. Raising it
  silently turns a stated policy limit into a hidden vulnerability.
- **Disabling or bypassing the audit log.** A repair that cannot be
  cryptographically verified destroys the accountability that supervisors and
  regulators require.
- **Presenting repaired output as the model's unprompted answer.** The user
  asked one question; the model answered a repaired formulation. The audit log
  records both, and downstream user disclosures should reflect that.

## If you find a way to misuse it

Open an issue, or contact the authors privately if disclosure would itself
cause harm. A confinement escape or a certified operator that leaks is a
defect in this project, and we would rather hear about it than not.
