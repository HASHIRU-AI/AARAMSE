# Measurement: corpora, splits, and the leakage budget

Three obligations, three corpora. A layer that meets two of them and quietly
fails the third looks fine in every table anyone publishes.

| Obligation | Corpus | Expected decision |
|---|---|---|
| Repair benign refusals | FalseReject finance slice | `repaired` |
| Do not leak prohibited requests | OR-Bench-toxic, FalseReject toxic | `escalated` |
| **Leave everything else alone** | FinQA control slice | `passthrough`, byte-identical |

## The third one, and why it exists

FalseReject measures repair. OR-Bench measures leakage. Neither catches a layer
that quietly rewrites queries the model was always going to answer. That is a
false intervention: it costs latency, it puts words in the user's mouth, and it
breaks the passthrough guarantee the design rests on.

`data/finqa_control.json` is 120 numerical-reasoning questions over public
filings, from FinQA's test split, each rendered self-contained with its table.
No safety-aligned model has a reason to refuse them, which is exactly what
makes them useful. Every one must come back `passthrough`.

The table travels with the question deliberately. A bare FinQA question is
unanswerable, and a model replying "I don't have that data" trips the
refusal-marker regex on "I cannot" — you would be measuring a missing table,
not a refusal.

Rebuild it with `scripts/build_finqa_control.py`. Selection is deterministic
(sorted by id, size-capped), so a rerun is byte-identical.

**This is not the use the concept note cites FinQA for.** The note lists it
under "financial-reasoning data" for the benign prompt bank. As an
over-refusal benchmark it would measure nothing — the same way XSTest turned
out saturated, with `gemma4:12b` over-refusing 0 of 24 safe prompts. As a
negative control it earns its place.

## Held-out splits

The note promised "a held-out set of genuinely prohibited advice requests".
Nothing enforced it: the same prompts certified the operators and then scored
them, which makes the leak figure a measure of memorisation.

```python
from aaramse import split_items, assert_disjoint

split = split_items(prompts, key=lambda p: p.prompt, holdout=0.5)
gw.certify(pairs_from(split.certification))
gw.enforce_budget([p.prompt for p in split.evaluation])
```

Splitting is by SHA-256 of the prompt text, not by shuffling, for three
reasons:

- **Reproducible** without storing a seed.
- **Stable as the corpus grows** — adding prompts does not reshuffle existing
  ones. This is the failure that makes a longitudinal result meaningless: rerun
  after adding data and your "held-out" set now contains items the operators
  were certified on last time.
- **Order-independent** — two runs that load the corpus differently agree.

Identical text always lands in one fold, so a near-duplicate cannot straddle
the boundary. `assert_disjoint` fails loudly with an example when folds
overlap; `split_digest` identifies an evaluation fold so two runs can prove
they scored the same set.

## The leakage budget

§4 of the concept note promised the layer "must not increase unsafe-request
pass-through above baseline". The measurements say that promise and a useful
recovery rate cannot both hold: on `gemma4:12b`, the fragment-confined operator
leaked 0/10 and recovered 1/8, while unconfined operators recovered 6–7/8 and
leaked 2/10.

So it becomes a budget — an explicit number the deployer sets and the code
enforces, defaulting to zero. Weaker than the original claim, honest, and
unlike the original it is actually checked.

```python
config = GatewayConfig(leak_budget=LeakageBudget(max_leaks=0, max_rate=0.0))
verdict = gw.enforce_budget(held_out_prohibited)   # raises BudgetExceeded
```

Both caps apply and the stricter binds, so a generous rate cannot let an
absolute cap through.

### Induced versus baseline leakage

The distinction that matters:

- **Baseline leak** — the model answers a prohibited prompt on its own. A
  property of the model. Reported, never charged to the layer.
- **Induced leak** — the model refused, and the layer got it answered. This
  layer's doing. Charged.

The rate is induced leaks over *prompts refused at baseline*, not over the
whole set. Dividing by everything would flatter the layer precisely when the
model is leakiest.

### Failing closed

When the budget is blown, `enforce_budget` empties the operator set *before*
raising, so a caller that swallows the exception is left with a layer that
escalates everything rather than one that leaks. Enforcement probes through the
search directly, so synthetic evaluation prompts never enter the intervention
log a supervisor reads.

## Reading a leak count

Every induced leak is returned in `verdict.induced` for hand adjudication, and
you should adjudicate them. In the three-arm run, both leaks recorded for the
rewrite arm were OR-Bench prompts whose toxic label is doubtful — requesting a
password reset for one's own account (scored `partial_refusal`, not full
compliance) and how investigators compile public records legally. DDOR
criticises these labels as noisy. Treat a raw count as an upper bound, publish
the adjudication next to it, and let a reviewer re-score the same evidence.

## Certification is a different question

Certification asks whether *one* operator flips *one* contrastive twin. The
budget asks the end-to-end question: across the whole search, how many
prohibited prompts does the assembled layer get answered? A composed program
can leak where no single operator does. Run both.
