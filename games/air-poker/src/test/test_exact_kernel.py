"""The exact kernel counter agrees with the engine's own chance enumeration."""

import unittest
from fractions import Fraction

import numpy as np

from src.engine import game
from src.engine.poker_math import FULL_DECK_MASK, generate_legal_hands, make_card
from src.kernel.exact import exact_kernel_counts
from src.sensitivity.sampler import CLASS_NAMES, NOJ, W0, W0C, W1, W1C, HandTables

TABLES = HandTables()
CHECK = game.BettingAction(game.RoundAction.Check)
# All clubs plus the diamonds 2..7: 19 cards. Supports run from 34 (15) to 533
# (26); 8 and 63 are impossible, so one-sided and two-sided no-rank are reachable.
REDUCED = sum(1 << make_card(r, 0) for r in range(13)) | sum(1 << make_card(r, 1) for r in range(6))


def engine_kernel(deck: int, target_0: int, target_1: int) -> dict[str, Fraction]:
    """Probability of each result class from the engine's enumeration on ``deck``."""
    base = game.apply(game.apply(game.initial_state((target_0, 9, 10, 11, 12),
                                                    (target_1, 13, 14, 16, 17)), target_0), target_1)
    state = game.GameState(base.phase, base.lower, deck, base.round)
    mass: dict[str, Fraction] = {name: Fraction(0) for name in CLASS_NAMES}

    def record_class(settled: game.GameState) -> str:
        record = settled.history[-1]
        if record.winner is None:
            return CLASS_NAMES[NOJ]
        if record.winner is game.Team.ZERO:
            return CLASS_NAMES[W0C if record.calamity else W0]
        return CLASS_NAMES[W1C if record.calamity else W1]

    for hands, probability in game.iter_chance_outcomes(state):
        successor = game.apply(state, hands)
        if successor.phase is game.Phase.UPPER_CHANCE:  # retry after double no-rank
            outcomes = tuple(game.iter_chance_outcomes(successor))
            self_check = sum(p for _, p in outcomes)
            assert abs(self_check - 1.0) < 1e-9
            for retry_hands, retry_probability in outcomes:
                settled = game.apply(successor, retry_hands)
                if settled.phase is game.Phase.BETTING:
                    settled = game.apply(game.apply(settled, CHECK), CHECK)
                mass[record_class(settled)] += Fraction(probability * retry_probability).limit_denominator(10**9)
            continue
        settled = game.apply(game.apply(successor, CHECK), CHECK)
        mass[record_class(settled)] += Fraction(probability).limit_denominator(10**9)
    return mass


class ExactKernelParityTest(unittest.TestCase):
    def assert_matches_engine(self, deck: int, target_0: int, target_1: int) -> None:
        counts = exact_kernel_counts(TABLES, deck, target_0, target_1)
        total = int(counts.sum())
        engine = engine_kernel(deck, target_0, target_1)
        for index, name in enumerate(CLASS_NAMES):
            exact = Fraction(int(counts[index]), total)
            self.assertAlmostEqual(float(exact), float(engine[name]), places=9, msg=(target_0, target_1, name))

    def test_support_product_and_engine_agreement_on_reduced_deck(self):
        for target_0, target_1 in ((25, 15), (36, 44), (39, 47), (45, 15)):
            counts = exact_kernel_counts(TABLES, REDUCED, target_0, target_1)
            support = (len(generate_legal_hands(target_0, REDUCED)),
                       len(generate_legal_hands(target_1, REDUCED)))
            self.assertGreater(min(support), 0)
            self.assertEqual(int(counts.sum()), support[0] * support[1])
            self.assert_matches_engine(REDUCED, target_0, target_1)

    def test_one_sided_no_rank_gives_the_other_side_a_plain_win(self):
        self.assertEqual(len(generate_legal_hands(8, REDUCED)), 0)
        counts = exact_kernel_counts(TABLES, REDUCED, 26, 8)
        self.assertEqual(int(counts[W0]), int(counts.sum()))
        self.assert_matches_engine(REDUCED, 26, 8)
        counts = exact_kernel_counts(TABLES, REDUCED, 25, 63)
        self.assertEqual(int(counts[W1]), 0)
        self.assert_matches_engine(REDUCED, 25, 63)

    def test_two_sided_no_rank_is_no_judgement(self):
        counts = exact_kernel_counts(TABLES, REDUCED, 36, 63)
        self.assertNotEqual(int(counts[NOJ]), 1)
        counts = exact_kernel_counts(TABLES, REDUCED & ~sum(1 << make_card(r, 0) for r in range(6, 13)), 36, 63)
        self.assertEqual(counts.tolist(), [0, 0, 0, 0, 0, 1])
        self.assert_matches_engine(REDUCED & ~sum(1 << make_card(r, 0) for r in range(6, 13)), 36, 63)

    def test_full_deck_pair_has_no_ties_or_no_judgement(self):
        counts = exact_kernel_counts(TABLES, FULL_DECK_MASK, 25, 63)
        self.assertEqual(int(counts.sum()), 61892 * 28)
        self.assertEqual(int(counts[4]), 0)
        self.assertEqual(int(counts[NOJ]), 0)
        self.assertTrue(np.all(counts >= 0))


if __name__ == "__main__":
    unittest.main()
