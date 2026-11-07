"""Speculative decoding with exact verification and evaluation counting.

Greedy variant: the draft model proposes gamma tokens by its own greedy rule;
the target model then scores each drafted position in one batched verification
(counted as a single target evaluation). Drafted tokens are accepted while the
target's greedy choice agrees; on the first mismatch the target's token is
emitted instead and the block ends.

This yields output identical to target-only greedy decoding, by construction:
every emitted token is either a drafted token the target greedily confirms, or
the target's own greedy choice. The test suite proves it across configurations
rather than asserting it once.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .model import CharNGram


class EvalCounter:
    """Wraps a model and counts verification rounds and actual scoring calls.

    `rounds` models the batched verification passes a real deployment runs:
    one per drafted block, one per baseline token. `scoring_calls` counts the
    actual sequential scoring work this pure-Python implementation performs
    inside those rounds. Reporting both keeps the modeled count honest: the
    savings number is a verification-round count, not a measured reduction in
    arithmetic.
    """

    def __init__(self, model: CharNGram):
        self.model = model
        self.rounds = 0
        self.scoring_calls = 0

    @property
    def evaluations(self) -> int:
        return self.rounds

    def greedy_next(self, context: str) -> str:
        self.rounds += 1
        self.scoring_calls += 1
        return self.model.greedy_next(context)

    def verify_block(self, context: str, block: List[str]) -> Tuple[int, str]:
        """Score the drafted block as one modeled verification round.

        Returns (accepted_count, next_token): how many drafted tokens the
        target greedily confirms in order, plus the target's greedy token at
        the first mismatch (or after a fully accepted block).
        """
        self.rounds += 1  # one modeled batched pass per block
        accepted = 0
        ctx = context
        for tok in block:
            self.scoring_calls += 1
            choice = self.model.greedy_next(ctx)
            if choice != tok:
                return accepted, choice
            accepted += 1
            ctx = ctx + tok
        self.scoring_calls += 1
        return accepted, self.model.greedy_next(ctx)


def target_only(target: CharNGram, prompt: str, n_tokens: int,
                stop: str = "\n") -> Dict:
    """Baseline: one target evaluation per emitted token."""
    counter = EvalCounter(target)
    text = prompt
    emitted = 0
    for _ in range(n_tokens):
        ch = counter.greedy_next(text)
        if ch == stop:
            break
        text += ch
        emitted += 1
    return {"text": text, "emitted": emitted, "target_evals": counter.rounds,
            "target_scoring_calls": counter.scoring_calls}


def speculative_run(draft: CharNGram, target: CharNGram, prompt: str,
                    n_tokens: int, gamma: int = 4, stop: str = "\n") -> Dict:
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    t_counter = EvalCounter(target)
    d_evals = drafted = accepted_sum = 0
    text = prompt
    emitted = 0
    while emitted < n_tokens:
        block: List[str] = []
        d_ctx = text
        for _ in range(gamma):
            ch = draft.greedy_next(d_ctx)
            d_evals += 1
            if ch == stop:
                break
            block.append(ch)
            d_ctx += ch
        if not block:
            ch = t_counter.greedy_next(text)
            if ch == stop:
                break
            text += ch
            emitted += 1
            continue
        drafted += len(block)
        accepted, nxt = t_counter.verify_block(text, block)
        accepted_sum += accepted
        take = min(accepted, n_tokens - emitted)
        text += "".join(block[:take])
        emitted += take
        if emitted >= n_tokens:
            break
        if nxt == stop:
            break
        text += nxt
        emitted += 1
    return {
        "text": text,
        "emitted": emitted,
        "target_evals": t_counter.rounds,
        "target_scoring_calls": t_counter.scoring_calls,
        "draft_evals": d_evals,
        "drafted": drafted,
        "accepted": accepted_sum,
        "acceptance": (accepted_sum / drafted) if drafted else 0.0,
    }
