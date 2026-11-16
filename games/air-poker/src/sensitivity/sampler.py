"""Exact uniform hand-pair sampling and result classification.

The sampler draws one hand uniformly from ``L(D, n)`` by filtering the
precomputed full-deck legal-hand list for subsets of the current deck. Every
legal hand of the depleted deck is a legal hand of the full deck, so the filter
is exact and no DP is needed. Classification mirrors ``_poker_winner`` and
``_settle_round`` in ``src.engine.game``; the parity test checks that claim
through the engine itself.
"""

from __future__ import annotations

import numpy as np

from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.engine.poker_math import FULL_DECK_MASK, evaluate_mask, generate_legal_hands

W0, W0C, W1, W1C, TIE, NOJ = range(6)
CLASS_NAMES = ("W0", "W0c", "W1", "W1c", "T", "N")
# N is a no-judgement round: both targets impossible, retry has no resolving
# mass, antes return and the deck is unchanged (MODEL.md M10). T is unreachable
# with distinct metal cards and is kept only so the classifier is total.
CANONICAL_TARGETS = tuple(sorted(TEAM_0_METAL_CARDS + TEAM_1_METAL_CARDS))


class HandTables:
    """Full-deck legal hands and their poker ranks for a set of targets."""

    def __init__(self, targets: tuple[int, ...] = CANONICAL_TARGETS) -> None:
        self.masks: dict[int, np.ndarray] = {}
        self.ranks: dict[int, np.ndarray] = {}
        for target in targets:
            hands = generate_legal_hands(target, FULL_DECK_MASK)
            self.masks[target] = np.array(hands, dtype=np.uint64)
            self.ranks[target] = np.array([evaluate_mask(h) for h in hands], dtype=np.int64)

    def legal_indices(self, target: int, deck: int) -> np.ndarray:
        """Indices into the full-deck list of the hands contained in ``deck``."""
        outside = np.uint64(FULL_DECK_MASK & ~deck)
        return np.flatnonzero((self.masks[target] & outside) == 0)

    def sample_hand(self, target: int, deck: int, rng: np.random.Generator):
        """Return ``(mask, rank)`` uniform over ``L(deck, target)``, or ``(None, None)``."""
        if deck == FULL_DECK_MASK:
            index = rng.integers(self.masks[target].size)
        else:
            legal = self.legal_indices(target, deck)
            if legal.size == 0:
                return None, None
            index = legal[rng.integers(legal.size)]
        return int(self.masks[target][index]), int(self.ranks[target][index])


def classify(hand_0, rank_0, hand_1, rank_1) -> int:
    """Return the lower-visible result class of a resolved hand pair."""
    if hand_0 is None:
        return W1
    if hand_1 is None:
        return W0
    if rank_0 == rank_1:
        return TIE
    overlap = bool(hand_0 & hand_1)
    if rank_0 > rank_1:
        return W0C if overlap else W0
    return W1C if overlap else W1


def sample_round(tables: HandTables, deck: int, target_0: int, target_1: int,
                 rng: np.random.Generator) -> tuple[int, int]:
    """Sample one resolved round; return ``(result_class, next_deck)``.

    A double no-rank is resampled, which is the retry kernel of ``MODEL.md``
    conditioned on eventual resolution. Two impossible targets settle with no
    judgement and leave the deck unchanged.
    """
    if (tables.legal_indices(target_0, deck).size == 0
            and tables.legal_indices(target_1, deck).size == 0):
        return NOJ, deck
    while True:
        hand_0, rank_0 = tables.sample_hand(target_0, deck, rng)
        hand_1, rank_1 = tables.sample_hand(target_1, deck, rng)
        if hand_0 is not None or hand_1 is not None:
            break
    consumed = (hand_0 or 0) | (hand_1 or 0)
    return classify(hand_0, rank_0, hand_1, rank_1), deck & ~consumed


def sample_trajectory(tables: HandTables, numbers: tuple[tuple[int, int], ...],
                      rng: np.random.Generator) -> tuple[int, ...]:
    """Sample the result-class sequence for a fixed number sequence."""
    deck = FULL_DECK_MASK
    classes = []
    for target_0, target_1 in numbers:
        result, deck = sample_round(tables, deck, target_0, target_1, rng)
        classes.append(result)
    return tuple(classes)
