"""Deterministic corpus of short technical sentences."""

from __future__ import annotations

import random
from typing import List

_SUBJ = ["the model", "the cache", "the verifier", "the decoder", "the index",
         "the parser", "the planner", "the sampler"]
_VERB = ["checks", "scores", "prunes", "merges", "replays", "streams",
         "validates", "compiles"]
_OBJ = ["the draft tokens", "the batch", "the logits", "the grammar",
        "the rollout", "the candidate block", "the prefix", "the trace"]
_ADV = ["in one pass", "without recompute", "under a fixed seed",
        "with exact counts", "before commit", "at half the cost"]


def generate_corpus(n: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        s = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} {rng.choice(_ADV)}."
        docs.append(s)
    return docs


def vocabulary(docs: List[str]) -> List[str]:
    chars = set()
    for d in docs:
        chars.update(d)
    return sorted(chars)
