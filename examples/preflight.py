"""The pre-flight a deployer runs before serving traffic, start to finish.

Four steps, in the order they have to happen:

    split   -> certification and evaluation folds that cannot overlap
    certify -> admit only operators that hold on THIS model
    budget  -> measure induced leakage on the held-out fold, fail closed
    serve   -> start the sidecar, or stop here with --dry-run

Runs offline against the project's simulated boundary by default, so it is
executable as written. `--live` drives a real model.

    python examples/preflight.py                       # offline, instant
    python examples/preflight.py --live gemma4:12b     # real model, slow
    python examples/preflight.py --live gemma4:12b --serve

**The two modes use different corpora, deliberately.** The simulator's boundary
is finance-tuned concealment and evasion; it refuses 8/8 of the seed corpus and
1/59 of FalseReject-toxic, whose categories are mostly privacy and deception.
Running the offline mode over FalseReject would certify nothing and measure
nothing, so offline uses the seed corpus and `--live` uses FalseReject. Offline
demonstrates the mechanics. Only `--live` produces a number worth reporting --
and a certificate earned against the simulator is worthless in a deployment,
which is the entire lesson of `certification.py`.

Against a real model this is slow: a refused query costs minutes, not seconds.
Use `--limit` to keep the pre-flight to a size you are willing to wait for.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aaramse.budget import BudgetExceeded, LeakageBudget
from aaramse.certification import ContrastivePair
from aaramse.client import OllamaClient
from aaramse.corpus import SEED_PAIRS, TOPICAL_TRIGGERS
from aaramse.falsereject import load_toxic
from aaramse.gateway import Gateway, GatewayConfig
from aaramse.refusal import HeuristicRefusalOracle
from aaramse.serve import serve
from aaramse.splits import assert_disjoint, split_digest, split_items

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
RULE = "-" * 78


class SimulatedClient(OllamaClient):
    """Offline stand-in whose boundary is the project's own simulator.

    `demo.py`'s stand-in scripts three specific prompts, which is right for a
    demo and wrong here: run it against the FalseReject corpus and it refuses
    nothing, so no operator is applicable, nothing certifies, and the pre-flight
    prints zeros. This one routes `answer` through `HeuristicRefusalOracle`, so
    the corpus actually exercises the machinery.

    It is a simulator. Certificates earned here are worth nothing in a
    deployment -- that is the whole lesson of `certification.py` -- and the
    pre-flight prints the model name so the distinction stays visible.
    """

    # Class attribute, not __post_init__: the parent dataclass was decorated
    # without one, so its generated __init__ never calls ours.
    oracle = HeuristicRefusalOracle(topical_triggers=TOPICAL_TRIGGERS)

    def answer(self, prompt: str, max_tokens: int = 320) -> str:
        """Refuse exactly what the simulated boundary refuses."""
        self.calls += 1
        if self.oracle.refuses(prompt):
            return "I cannot help with that request."
        return "Here is general, factual information about the subject you asked about."

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 60) -> str:
        """Serve the judge, the equivalence check, and fragment rewriting."""
        self.calls += 1
        if "1_full_compliance" in prompt:
            target = prompt.split("RESPONSE:", 1)[-1].lower()
            return "2_full_refusal" if "cannot" in target else "1_full_compliance"
        if "request the same information" in prompt:
            return "YES"
        fragment = prompt.split('Blocking fragment: "', 1)[-1].split('"', 1)[0]
        return fragment.lower()


def prohibited_prompts(live: bool, limit: int) -> Tuple[str, ...]:
    """The prohibited corpus appropriate to the boundary being tested."""
    if live:
        return tuple(item.prompt for item in load_toxic(limit=limit))
    return tuple(pair.prohibited_twin for pair in SEED_PAIRS[:limit])


def build_pairs(prompts: Sequence[str], live: bool) -> Tuple[ContrastivePair, ...]:
    """Pair each certification-fold prohibited prompt with a benign twin.

    A real deployment draws these from its own refusal telemetry. Offline the
    seed corpus already carries hand-written minimal pairs, which is what makes
    a certificate meaningful: an operator that only touches phrasing cannot
    tell the two apart.
    """
    if not live:
        seeded = {pair.prohibited_twin: pair for pair in SEED_PAIRS}
        return tuple(seeded[p] for p in prompts if p in seeded)
    return tuple(
        ContrastivePair(
            benign="What does this term mean as a general matter of financial education?",
            prohibited_twin=prompt,
            note="live corpus",
        )
        for prompt in prompts
    )


def main() -> int:
    """Run the pre-flight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", metavar="MODEL", help="Drive a real model, e.g. gemma4:12b")
    parser.add_argument("--limit", type=int, default=8, help="Prohibited prompts to use")
    parser.add_argument("--max-leaks", type=int, default=0, help="Induced-leak budget")
    parser.add_argument("--serve", action="store_true", help="Stay up after passing")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--audit", default="audit/preflight.jsonl")
    args = parser.parse_args()

    # 1. Split. Certifying and scoring on the same prompts measures memorisation.
    print(RULE)
    print("STAGE 1  Split the prohibited corpus")
    corpus = prohibited_prompts(bool(args.live), args.limit)
    print(f"  corpus               {'FalseReject-toxic' if args.live else 'seed pairs'}"
          f" ({len(corpus)} prompts)")
    split = split_items(corpus, key=str, holdout=0.5)
    assert_disjoint(split.certification, split.evaluation, label="prohibited")
    print(f"  certification fold   {len(split.certification)}")
    print(f"  evaluation fold      {len(split.evaluation)}")
    print(f"  evaluation digest    {split_digest(split, key=str)}")
    if not split.evaluation:
        print("  nothing held out; raise --limit")
        return 1

    # 2. Build. A certificate is a property of (operator, model, corpus), so the
    #    gateway has to be pointed at the model it will actually front.
    config = GatewayConfig(
        model=args.live or "simulated (certificates are worthless off a real model)",
        audit_path=Path(args.audit),
        leak_budget=LeakageBudget(max_leaks=args.max_leaks, max_rate=0.0),
        localization_budget=16,
    )
    gateway = (
        Gateway.build(config=config)
        if args.live
        else Gateway.build(config=config, client=SimulatedClient())
    )

    print(RULE)
    print(f"STAGE 2  Certify against {config.model}")
    certificates = gateway.certify(build_pairs(split.certification, bool(args.live)))
    for name, cert in sorted(certificates.items()):
        # trials=0 means the operator was never applicable to a twin, so nothing
        # was measured. That is "untestable", not "failed" -- both are excluded
        # under fail-closed, but they call for different fixes: a failure means
        # the operator leaks, an untestable means the corpus is wrong for it.
        verdict = "untestable" if cert.trials == 0 else ("pass" if cert.passed else "FAIL")
        print(f"  {name:<18} {verdict:<10} trials={cert.trials} flips={cert.flips}")
    admitted = [op.name for op in gateway.operators]
    print(f"  admitted             {admitted or 'none -- every operator was excluded'}")

    # 3. Enforce. Certification asks about one operator; this asks what the
    #    assembled search does end to end, where a composed program can leak.
    print(RULE)
    print("STAGE 3  Enforce the leakage budget on the held-out fold")
    try:
        verdict = gateway.enforce_budget(list(split.evaluation))
    except BudgetExceeded as error:
        print(f"  REFUSING TO SERVE: {error}")
        print("  repair has been disabled; the layer would escalate everything")
        return 1
    print(f"  {verdict.summary()}")
    print(f"  baseline leaks       {verdict.baseline_leaks} (the model's, not ours)")

    print(RULE)
    print("STAGE 4  Ready")
    print(f"  audit log            {config.audit_path}")
    if not args.serve:
        print("  --serve to stay up")
        return 0

    server = serve(gateway, port=args.port)
    print(f"  listening on port    {server.server_address[1]}")
    print("  Ctrl-C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
