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

Those controls are: only queries the deployed model actually refuses are
eligible; only operators holding a passing certificate on that model may run;
edits are structurally confined to the localized fragment; candidates that
raise actionability are rejected; anything unresolved escalates to a human; and
every decision is written to a hash-chained log.

## Intended use

- Restoring access to lawful financial, legal, and medical information that a
  safety-aligned model wrongly declines.
- Research on over-refusal, its measurement, and its repair.
- Regulatory and compliance review of a deployed agent's refusal behaviour.

## Uses we ask you not to make of it

- **Removing a refusal that is correct.** Repairing a refusal of a genuinely
  prohibited request is the failure mode this project exists to avoid, not an
  application of it.
- **Running with `require_certificates=False` in production.** The flag exists
  for research on the uncertified algebra. An uncertified operator is one whose
  behaviour on the deployed model is unmeasured.
- **Raising the leakage budget without recording the decision.** The budget is
  a deployer's explicit statement of accepted risk. Raising it silently
  converts a stated risk into a hidden one.
- **Detaching the audit log.** A repair nobody can review is the thing the
  regulator was worried about in the first place.
- **Presenting repaired output as the model's unprompted answer.** The user
  asked one question; the model was asked another. The log records both, and
  downstream disclosures should reflect that.

## If you find a way to misuse it

Open an issue, or contact the authors privately if disclosure would itself
cause harm. A confinement escape or a certified operator that leaks is a
defect in this project, and we would rather hear about it than not.
