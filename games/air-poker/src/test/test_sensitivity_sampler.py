"""Parity of the sensitivity sampler with the production engine."""

import unittest

import numpy as np

from src.engine import game
from src.engine.poker_math import FULL_DECK_MASK, generate_legal_hands
from src.sensitivity.sampler import (
    CLASS_NAMES,
    NOJ,
    TIE,
    W0,
    W0C,
    W1,
    W1C,
    HandTables,
    classify,
    sample_round,
    sample_trajectory,
)

TABLES = HandTables()
DEPLETED = FULL_DECK_MASK & ~((1 << 10) - 1) & ~(0b1111 << 30)  # 14 cards removed
CHECK = game.BettingAction(game.RoundAction.Check)


def engine_class(target_0: int, target_1: int, hands: game.HandPair) -> int:
    """Run one round through the engine on the full deck and read its record."""
    state = game.initial_state((target_0, 9, 10, 11, 12), (target_1, 13, 14, 16, 17))
    state = game.apply(game.apply(state, target_0), target_1)
    state = game.apply(state, hands)
    state = game.apply(game.apply(state, CHECK), CHECK)
    record = state.history[-1]
    if record.winner is None:
        return TIE
    if record.winner is game.Team.ZERO:
        return W0C if record.calamity else W0
    return W1C if record.calamity else W1


class LegalHandFilterTest(unittest.TestCase):
    def test_filter_equals_engine_enumeration_on_depleted_deck(self):
        for target in TABLES.masks:
            expected = set(generate_legal_hands(target, DEPLETED))
            got = {int(TABLES.masks[target][i]) for i in TABLES.legal_indices(target, DEPLETED)}
            self.assertEqual(got, expected, target)

    def test_sampled_hands_are_legal_subsets(self):
        rng = np.random.default_rng(1)
        for target in TABLES.masks:
            for _ in range(50):
                hand, _ = TABLES.sample_hand(target, DEPLETED, rng)
                if hand is None:
                    self.assertEqual(len(generate_legal_hands(target, DEPLETED)), 0)
                else:
                    self.assertEqual(hand & ~DEPLETED, 0)
                    self.assertIn(hand, generate_legal_hands(target, DEPLETED))


class ClassificationParityTest(unittest.TestCase):
    def test_classify_matches_engine_on_sampled_pairs(self):
        rng = np.random.default_rng(2)
        seen = set()
        for target_0 in game.TEAM_0_METAL_CARDS:
            for target_1 in game.TEAM_1_METAL_CARDS:
                for _ in range(8):
                    hand_0, rank_0 = TABLES.sample_hand(target_0, FULL_DECK_MASK, rng)
                    hand_1, rank_1 = TABLES.sample_hand(target_1, FULL_DECK_MASK, rng)
                    result = classify(hand_0, rank_0, hand_1, rank_1)
                    self.assertEqual(result, engine_class(target_0, target_1, (hand_0, hand_1)))
                    seen.add(result)
        self.assertIn(W0, seen)
        self.assertIn(W1, seen)

    def test_forced_overlap_and_tie_match_engine(self):
        # Two hands sharing the ace of clubs; ranks differ, so the winner takes calamity.
        hand_a = int(TABLES.masks[25][0])
        candidates = TABLES.masks[8][(TABLES.masks[8] & np.uint64(hand_a)) != 0]
        self.assertGreater(candidates.size, 0)
        hand_b = int(candidates[0])
        rank_a = int(TABLES.ranks[25][0])
        rank_b = int(TABLES.ranks[8][np.flatnonzero(TABLES.masks[8] == np.uint64(hand_b))[0]])
        self.assertEqual(classify(hand_a, rank_a, hand_b, rank_b),
                         engine_class(25, 8, (hand_a, hand_b)))
        self.assertIn(classify(hand_a, rank_a, hand_b, rank_b), (W0C, W1C))

    def test_distinct_targets_never_tie(self):
        # Equal HandRank implies an equal rank multiset, hence an equal sum, so two
        # distinct metal cards cannot produce a poker tie. TIE is unreachable here.
        targets = list(TABLES.ranks)
        for i, a in enumerate(targets):
            for b in targets[i + 1:]:
                self.assertFalse(set(TABLES.ranks[a].tolist()) & set(TABLES.ranks[b].tolist()),
                                 (a, b))

    def test_no_rank_classes(self):
        self.assertEqual(classify(None, None, 1, 5), W1)
        self.assertEqual(classify(1, 5, None, None), W0)


class RoundSamplingTest(unittest.TestCase):
    def test_impossible_target_always_loses(self):
        tiny = sum(1 << c for c in range(13))  # one suit: 63 impossible, 15 = A2345 possible
        rng = np.random.default_rng(3)
        self.assertEqual(len(generate_legal_hands(63, tiny)), 0)
        for _ in range(20):
            result, deck = sample_round(TABLES, tiny, 63, 15, rng)
            self.assertEqual(result, W1)
            self.assertEqual((tiny & ~deck).bit_count(), 5)

    def test_double_impossible_is_no_judgement_with_deck_unchanged(self):
        tiny = sum(1 << c for c in range(13))
        result, deck = sample_round(TABLES, tiny, 63, 63, np.random.default_rng(4))
        self.assertEqual(result, NOJ)
        self.assertEqual(deck, tiny)

    def test_trajectory_has_five_classes(self):
        numbers = tuple(zip(game.TEAM_0_METAL_CARDS, game.TEAM_1_METAL_CARDS))
        classes = sample_trajectory(TABLES, numbers, np.random.default_rng(5))
        self.assertEqual(len(classes), 5)
        self.assertTrue(all(0 <= c < len(CLASS_NAMES) for c in classes))


if __name__ == "__main__":
    unittest.main()
