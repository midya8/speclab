"""SpecLab command-line interface.

Commands:
  evaluate  Run the speculative-decoding evaluation and report.
  gates     Evaluate and enforce policy gates; exit 1 when any gate fails.

examples/demo.sh exercises both sides of every gate.
"""

from __future__ import annotations

import argparse
import json
import sys

from .eval import build_pair, format_terminal, run_evaluation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="speclab",
        description="Speculative decoding with proven equivalence and measured savings.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--corpus-size", type=int, default=400)
        p.add_argument("--corpus-seed", type=int, default=42)
        p.add_argument("--draft-order", type=int, default=3)
        p.add_argument("--target-order", type=int, default=4)
        p.add_argument("--n-tokens", type=int, default=60)
        p.add_argument("--gamma", type=int, default=4)

    p_eval = sub.add_parser("evaluate", help="run the evaluation")
    common(p_eval)
    p_eval.add_argument("--format", choices=("terminal", "json", "markdown"),
                        default="terminal")

    p_gate = sub.add_parser("gates", help="evaluate and enforce policy gates")
    common(p_gate)
    p_gate.add_argument("--require-identical", action="store_true",
                        help="fail unless speculative output equals target-only on every prompt")
    p_gate.add_argument("--min-acceptance", type=float, default=None,
                        help="minimum mean draft acceptance rate")
    p_gate.add_argument("--min-eval-savings", type=float, default=None,
                        help="minimum fraction of target evaluations saved")

    args = parser.parse_args(argv)
    draft, target = build_pair(args.corpus_size, args.corpus_seed,
                               args.draft_order, args.target_order)
    results = run_evaluation(draft, target, n_tokens=args.n_tokens, gamma=args.gamma)

    if args.command == "evaluate":
        if args.format == "json":
            print(json.dumps(results, indent=2, sort_keys=True))
        elif args.format == "markdown":
            print("| prompt | identical | acceptance | target evals | naive |")
            print("|---|---|---|---|---|")
            for r in results["prompts"]:
                print(f"| {r['prompt']} | {r['identical']} | {r['acceptance']:.2f} "
                      f"| {r['spec_target_evals']} | {r['baseline_target_evals']} |")
            print(f"\nSavings: {results['target_eval_savings']*100:.1f}% "
                  f"of target evaluations; mean acceptance {results['mean_acceptance']:.3f}.")
        else:
            print(format_terminal(results))
        return 0

    failures = []
    if args.require_identical and not results["identical"]:
        failures.append("speculative output differs from target-only greedy on a prompt")
    if args.min_acceptance is not None and results["mean_acceptance"] < args.min_acceptance:
        failures.append(f"mean acceptance {results['mean_acceptance']:.3f} "
                        f"< required {args.min_acceptance:.3f}")
    if args.min_eval_savings is not None and results["target_eval_savings"] < args.min_eval_savings:
        failures.append(f"target-eval savings {results['target_eval_savings']:.3f} "
                        f"< required {args.min_eval_savings:.3f}")

    print(format_terminal(results))
    if failures:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
