"""Gates 1 and 2 of KERNEL.md: the DP counts and unranks exactly the legal hands."""

import unittest

import numpy as np

from src.engine.poker_math import (
    FULL_DECK_MASK,
    generate_legal_hands,
    make_card,
    rank_sum,
)
from src.kernel.dp import (
    build_table,
    count,
    hand_count,
    unrank,
    unrank_hand,
    value_counts,
)
from src.sensitivity.sampler import CANONICAL_TARGETS

REDUCED = sum(1 << make_card(r, 0) for r in range(13)) | sum(1 << make_card(r, 1) for r in range(6))


def seeded_decks(seed: int, n: int, low: int, high: int) -> list[int]:
    """``n`` decks with uniform sizes in ``[low, high]``, cards chosen without replacement."""
    rng = np.random.default_rng(seed)
    decks = []
    for _ in range(n):
        kept = rng.choice(52, size=int(rng.integers(low, high + 1)), replace=False)
        decks.append(int(sum(1 << int(c) for c in kept)))
    return decks


class CountGateTest(unittest.TestCase):
    def test_value_counts_of_the_full_deck_are_four_each(self):
        self.assertEqual(value_counts(FULL_DECK_MASK).tolist(), [0] + [4] * 13)
        self.assertEqual(value_counts(REDUCED).tolist(), [0, 1] + [2] * 6 + [1] * 6)

    def test_count_matches_enumeration_on_the_full_deck(self):
        for target in CANONICAL_TARGETS:
            self.assertEqual(hand_count(FULL_DECK_MASK, target),
                             len(generate_legal_hands(target, FULL_DECK_MASK)), target)

    def test_count_matches_enumeration_on_200_seeded_depleted_decks(self):
        for deck in seeded_decks(20261029, 200, 5, 42):  # sizes after round 1
            table = build_table(value_counts(deck))
            for target in CANONICAL_TARGETS:
                self.assertEqual(int(count(table, target)), len(generate_legal_hands(target, deck)),
                                 (deck, target))

    def test_count_is_zero_exactly_when_enumeration_is_empty(self):
        for deck in [REDUCED, *seeded_decks(7, 10, 4, 12)]:
            table = build_table(value_counts(deck))
            for target in range(70):
                enumerated = len(generate_legal_hands(target, deck))
                self.assertEqual(int(count(table, target)), enumerated, (deck, target))


class BijectionGateTest(unittest.TestCase):
    def test_unrank_enumerates_every_legal_hand_exactly_once_on_small_decks(self):
        for deck in [REDUCED, *seeded_decks(11, 20, 8, 14)]:
            a = value_counts(deck)
            table = build_table(a)
            for target in range(6, 65):
                total = int(count(table, target))
                hands = {int(unrank(table, a, deck, target, u)) for u in range(total)}
                self.assertEqual(len(hands), total, (deck, target))
                self.assertEqual(hands, set(generate_legal_hands(target, deck)), (deck, target))

    def test_unranked_full_deck_hands_are_legal_and_distinct(self):
        rng = np.random.default_rng(3)
        a = value_counts(FULL_DECK_MASK)
        table = build_table(a)
        for target in CANONICAL_TARGETS:
            total = int(count(table, target))
            picks = rng.choice(total, size=min(total, 2000), replace=False)
            hands = {int(unrank(table, a, FULL_DECK_MASK, target, int(u))) for u in picks}
            self.assertEqual(len(hands), picks.size)
            for hand in hands:
                self.assertEqual(hand.bit_count(), 5)
                self.assertEqual(hand & ~FULL_DECK_MASK, 0)
                self.assertEqual(rank_sum(hand), target)

    def test_out_of_range_rank_returns_minus_one(self):
        total = hand_count(REDUCED, 25)
        self.assertEqual(unrank_hand(REDUCED, 25, total), -1)
        self.assertEqual(unrank_hand(REDUCED, 8, 0), -1)


if __name__ == "__main__":
    unittest.main()
