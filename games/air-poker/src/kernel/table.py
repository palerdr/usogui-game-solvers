"""The entry table: counts keyed by number prefix and public class prefix.

A kernel entry is ``(h, a, b)``: the numbers and result classes of the rounds
played so far and the next pair. The sampler produces counts per number
sequence and revealed class prefix; the same entry receives samples from every
sequence that extends its number prefix, and a hidden class (a fold) pools the
five revealed classes at that position. This module owns both aggregations.

Round ``r`` (0-based) has one dense array of shape
``(number prefixes of length r + 1, 6 ** r, 6)``: axis 1 is the class prefix in
base 6 with digits ``0..4`` for ``W0, W0c, W1, W1c, N`` and ``5`` for hidden;
axis 2 is the class count. Rounds 1 to 4 hold the 21,904,800 entries the
solver reads; round 0 is exact and kept only for parity.
"""

from __future__ import annotations

import itertools

import numpy as np

from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.kernel.confidence import boxes
from src.kernel.sampler import CLASSES, DIGIT, ROUNDS, ROW_OFFSET

HIDDEN = 5


class Sequences:
    """All 14,400 number sequences and the index of each number prefix."""

    def __init__(self) -> None:
        orders_0 = list(itertools.permutations(TEAM_0_METAL_CARDS))
        orders_1 = list(itertools.permutations(TEAM_1_METAL_CARDS))
        self.targets = np.array([[(a, b) for a, b in zip(order_0, order_1)]
                                 for order_0 in orders_0 for order_1 in orders_1], dtype=np.int64)
        self.prefix_index: list[dict[tuple, int]] = []
        self.prefix_of_sequence = np.empty((ROUNDS, len(self.targets)), dtype=np.int64)
        for r in range(ROUNDS):
            index: dict[tuple, int] = {}
            for s, sequence in enumerate(self.targets):
                key = tuple(map(tuple, sequence[:r + 1]))
                self.prefix_of_sequence[r, s] = index.setdefault(key, len(index))
            self.prefix_index.append(index)

    def __len__(self) -> int:
        return len(self.targets)


class KernelCounts:
    """Dense per-round count arrays with sequence aggregation and hidden pooling."""

    def __init__(self, sequences: Sequences) -> None:
        self.sequences = sequences
        self.revealed = [np.zeros((len(sequences.prefix_index[r]), 5 ** r, CLASSES), dtype=np.int64)
                         for r in range(ROUNDS)]
        self.entries = sum(len(sequences.prefix_index[r]) * 6 ** r for r in range(1, ROUNDS))

    def add(self, sequence: int, counts: np.ndarray) -> None:
        """Add one sequence's ``(781, 6)`` sampler output."""
        for r in range(ROUNDS):
            rows = counts[ROW_OFFSET[r]:ROW_OFFSET[r] + 5 ** r]
            self.revealed[r][self.sequences.prefix_of_sequence[r, sequence]] += rows

    def pooled(self, r: int) -> np.ndarray:
        """Return round ``r`` counts over base-6 class prefixes, hidden positions pooled."""
        revealed = self.revealed[r].reshape((-1,) + (5,) * r + (CLASSES,))
        pooled = np.zeros((revealed.shape[0],) + (6,) * r + (CLASSES,), dtype=np.int64)
        pooled[(slice(None),) + (slice(0, 5),) * r] = revealed
        for axis in range(1, r + 1):
            target: list[slice | int] = [slice(None)] * (r + 2)
            target[axis] = HIDDEN
            source: list[slice | int] = [slice(None)] * (r + 2)
            source[axis] = slice(0, 5)
            pooled[tuple(target)] = pooled[tuple(source)].sum(axis=axis)
        return pooled.reshape(revealed.shape[0], 6 ** r, CLASSES)

    def boxes(self, r: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
        """Confidence boxes of round ``r`` entries at overall level ``delta`` by union bound."""
        return boxes(self.pooled(r), delta / self.entries)


def class_code(prefix: tuple[int | None, ...]) -> int:
    """Base-6 code of a public class prefix; ``None`` is a hidden class."""
    code = 0
    for result in prefix:
        code = code * 6 + (HIDDEN if result is None else int(DIGIT[result]))
    return code
