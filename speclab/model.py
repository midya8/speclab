"""Exact character-level n-gram model with add-k smoothing.

Probabilities come straight from counts, so the target model's greedy choice is
ground truth and draft/target agreement is a property of the models, not of any
approximation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Sequence


class CharNGram:
    def __init__(self, order: int, vocab: Sequence[str], smoothing: float = 0.1):
        if order < 1:
            raise ValueError("order must be >= 1")
        if not vocab:
            raise ValueError("vocab must be non-empty")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.order = order
        self.vocab = tuple(sorted(set(vocab)))
        self.smoothing = smoothing
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._totals: Dict[str, int] = defaultdict(int)

    def train(self, documents: Iterable[str]) -> None:
        for doc in documents:
            padded = "\x00" * (self.order - 1) + doc
            for i in range(self.order - 1, len(padded)):
                ctx = padded[i - (self.order - 1):i] if self.order > 1 else ""
                ch = padded[i]
                if ch not in self.vocab:
                    continue
                self._counts[ctx][ch] += 1
                self._totals[ctx] += 1

    def distribution(self, context: str) -> Dict[str, float]:
        ctx = context[-(self.order - 1):] if self.order > 1 else ""
        row = self._counts.get(ctx, {})
        total = self._totals.get(ctx, 0)
        k = self.smoothing
        v = len(self.vocab)
        denom = total + k * v
        return {ch: (row.get(ch, 0) + k) / denom for ch in self.vocab}

    def greedy_next(self, context: str) -> str:
        """Deterministic argmax: highest probability, ties broken by vocab order."""
        dist = self.distribution(context)
        best, best_p = None, -1.0
        for ch in self.vocab:
            if dist[ch] > best_p:
                best, best_p = ch, dist[ch]
        return best
