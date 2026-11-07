"""Measured speculative-decoding evaluation.

The fixture pairs a cheaper order-3 draft with an order-4 target over the same
deterministic corpus, generates from fixed prompts, and reports what real
deployments care about: output equivalence (proven, not sampled), draft
acceptance rate, and target-evaluation savings versus one-evaluation-per-token
baseline decoding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import generate_corpus, vocabulary
from .model import CharNGram
from .spec import speculative_run, target_only

PROMPTS = ["the model", "the cache", "the verifier", "the parser",
           "the sampler", "the decoder", "the index", "the planner"]


def build_pair(corpus_size: int = 400, seed: int = 42,
               draft_order: int = 3, target_order: int = 4,
               draft_smoothing: float = 0.1,
               target_smoothing: float = 0.01) -> Tuple[CharNGram, CharNGram]:
    docs = generate_corpus(corpus_size, seed=seed)
    vocab = vocabulary(docs)
    draft = CharNGram(draft_order, vocab, smoothing=draft_smoothing)
    draft.train(docs)
    target = CharNGram(target_order, vocab, smoothing=target_smoothing)
    target.train(docs)
    return draft, target


def run_evaluation(draft: CharNGram, target: CharNGram,
                   prompts: Optional[Sequence[str]] = None,
                   n_tokens: int = 60, gamma: int = 4) -> Dict:
    prompts = list(prompts or PROMPTS)
    per_prompt: List[Dict] = []
    identical = True
    for p in prompts:
        base = target_only(target, p, n_tokens)
        spec = speculative_run(draft, target, p, n_tokens, gamma=gamma)
        same = base["text"] == spec["text"]
        identical = identical and same
        per_prompt.append({
            "prompt": p,
            "identical": same,
            "emitted": spec["emitted"],
            "acceptance": spec["acceptance"],
            "baseline_target_evals": base["target_evals"],
            "spec_target_evals": spec["target_evals"],
            "draft_evals": spec["draft_evals"],
        })
    n = len(per_prompt)
    base_evals = sum(r["baseline_target_evals"] for r in per_prompt)
    spec_evals = sum(r["spec_target_evals"] for r in per_prompt)
    return {
        "prompts": per_prompt,
        "identical": identical,
        "mean_acceptance": sum(r["acceptance"] for r in per_prompt) / n,
        "baseline_target_evals": base_evals,
        "spec_target_evals": spec_evals,
        "target_eval_savings": 1.0 - (spec_evals / base_evals if base_evals else 1.0),
        "total_draft_evals": sum(r["draft_evals"] for r in per_prompt),
        "n_tokens": n_tokens,
        "gamma": gamma,
    }


def format_terminal(results: Dict) -> str:
    lines = [
        f"SpecLab evaluation - {len(results['prompts'])} prompts x "
        f"{results['n_tokens']} tokens, gamma={results['gamma']}",
        "",
        f"{'prompt':<16} {'identical':>9} {'acceptance':>10} {'target evals':>22}",
        f"{'-'*16} {'-'*9} {'-'*10} {'-'*22}",
    ]
    for r in results["prompts"]:
        lines.append(f"{r['prompt']:<16} {str(r['identical']):>9} "
                     f"{r['acceptance']:>10.2f} "
                     f"{r['spec_target_evals']:>6} vs {r['baseline_target_evals']:<6} naive")
    lines += [
        "",
        f"Output identical to target-only greedy on every prompt: {results['identical']}",
        f"Mean draft acceptance: {results['mean_acceptance']:.3f}",
        f"Target evaluations: {results['spec_target_evals']} vs "
        f"{results['baseline_target_evals']} naive "
        f"(savings {results['target_eval_savings']*100:.1f}%)",
        f"Draft evaluations (cheap model): {results['total_draft_evals']}",
    ]
    return "\n".join(lines)
