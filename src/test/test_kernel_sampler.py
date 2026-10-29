"""Gates 3, 4 and 5 of KERNEL.md at unit scale, plus the generator and the evaluator."""

import json
import unittest

import numpy as np

from src.engine import game
from src.engine.poker_math import FULL_DECK_MASK, evaluate_mask, make_card
from src.kernel.exact import exact_kernel_counts
from src.kernel.sampler import (
    CLASSES,
    ROWS,
    bounded,
    evaluate_many,
    mix64,
    row,
    sample_round_hands,
    sample_rounds,
    sample_sequence,
    stream,
)
from src.sensitivity.sampler import (
    CANONICAL_TARGETS,
    CLASS_NAMES,
    NOJ,
    TIE,
    W0,
    W0C,
    W1,
    W1C,
    HandTables,
)
from src.test.test_kernel_dp import seeded_decks

TABLES = HandTables()
CHECK = game.BettingAction(game.RoundAction.Check)
REDUCED = sum(1 << make_card(r, 0) for r in range(13)) | sum(1 << make_card(r, 1) for r in range(6))
PAIRS = [(a, b) for a in game.TEAM_0_METAL_CARDS for b in game.TEAM_1_METAL_CARDS]


def engine_replay(deck: int, target_0: int, target_1: int, hands) -> tuple[int, int]:
    """Apply ``hands`` through the engine on ``deck``; return ``(class, next_deck)``."""
    base = game.apply(game.apply(game.initial_state((target_0, 9, 10, 11, 12),
                                                    (target_1, 13, 14, 16, 17)), target_0), target_1)
    state = game.apply(game.GameState(base.phase, base.lower, deck, base.round), hands)
    if state.phase is game.Phase.UPPER_CHANCE:  # retry after double no-rank
        state = game.apply(state, hands)
    if state.phase is game.Phase.BETTING:
        state = game.apply(game.apply(state, CHECK), CHECK)
    record = state.history[-1]
    if record.winner is None:
        result = NOJ
    elif record.winner is game.Team.ZERO:
        result = W0C if record.calamity else W0
    else:
        result = W1C if record.calamity else W1
    return result, state.available_deck_mask


def assert_within(test, counts: np.ndarray, exact: np.ndarray, sigma: float, context) -> None:
    """Sampled ``counts`` agree with the kernel ``exact / exact.sum()`` within ``sigma`` errors."""
    n = int(counts.sum())
    p = exact / exact.sum()
    for index, name in enumerate(CLASS_NAMES):
        if p[index] in (0.0, 1.0):
            test.assertEqual(int(counts[index]), round(p[index] * n), (context, name))
        else:
            z = (counts[index] - n * p[index]) / np.sqrt(n * p[index] * (1 - p[index]))
            test.assertLess(abs(z), sigma, (context, name, z))


class GeneratorTest(unittest.TestCase):
    def test_mix64_is_a_bijection_on_a_sample(self):
        words = np.arange(0, 1 << 20, dtype=np.uint64)
        self.assertEqual(len({int(mix64(w)) for w in words[:50000]}), 50000)

    def test_bounded_is_uniform_and_exact_at_the_edges(self):
        for n in (1, 2, 3, 7, 100, 61892):
            draws = np.zeros(n, np.int64)
            trials = 20000 if n < 1000 else 200000
            for index in range(trials):
                u, _ = bounded(stream(5, 0, index), n)
                draws[u] += 1
            self.assertEqual(int(draws.sum()), trials)
            if n < 1000:
                expected = trials / n
                self.assertLess(np.abs(draws - expected).max(), 5 * np.sqrt(expected) + 1, n)
            else:
                self.assertLess(draws.max(), 20)
                self.assertGreater(np.count_nonzero(draws), 0.9 * n)

    def test_streams_differ_across_index_batch_and_seed(self):
        a = int(stream(1, 0, 0))
        self.assertNotEqual(a, int(stream(1, 0, 1)))
        self.assertNotEqual(a, int(stream(1, 1, 0)))
        self.assertNotEqual(a, int(stream(2, 0, 0)))


class EvaluatorTest(unittest.TestCase):
    def test_numba_evaluator_matches_poker_math_on_every_canonical_legal_hand(self):
        for target in CANONICAL_TARGETS:
            masks = TABLES.masks[target].astype(np.int64)
            np.testing.assert_array_equal(evaluate_many(masks), TABLES.ranks[target])

    def test_numba_evaluator_matches_poker_math_on_random_hands(self):
        rng = np.random.default_rng(9)
        masks = np.array([int(sum(1 << int(c) for c in rng.choice(52, 5, replace=False)))
                          for _ in range(20000)], dtype=np.int64)
        np.testing.assert_array_equal(evaluate_many(masks), [evaluate_mask(int(m)) for m in masks])


class Gate3RoundOneTest(unittest.TestCase):
    def test_single_rounds_match_the_exact_kernel_within_four_errors(self):
        with open("docs/results/kernel-round1.json") as handle:
            exact = {tuple(e["numbers"]): e["counts"] for e in json.load(handle)["entries"]}
        for batch, (a, b) in enumerate(PAIRS):
            counts = sample_rounds(FULL_DECK_MASK, a, b, 20261029, batch, 400_000)
            assert_within(self, counts, np.array([exact[(a, b)][k] for k in CLASS_NAMES], float), 4.0, (a, b))

    def test_round_one_row_of_a_trajectory_batch_matches_the_exact_kernel(self):
        with open("docs/results/kernel-round1.json") as handle:
            exact = {tuple(e["numbers"]): e["counts"] for e in json.load(handle)["entries"]}
        targets = [(36, 15), (25, 8), (45, 63), (26, 44), (39, 47)]
        counts, errors = sample_sequence(targets, 3, 0, 200_000)
        self.assertEqual(errors, 0)
        assert_within(self, counts[0], np.array([exact[(36, 15)][k] for k in CLASS_NAMES], float), 4.0, targets[0])


class Gate4DepletedDeckTest(unittest.TestCase):
    def test_single_rounds_on_50_seeded_depleted_decks_match_the_exact_counter(self):
        rng = np.random.default_rng(4)
        for deck in seeded_decks(20261030, 50, 14, 32):
            a, b = PAIRS[int(rng.integers(len(PAIRS)))]
            exact = exact_kernel_counts(TABLES, deck, a, b).astype(float)
            counts = sample_rounds(deck, a, b, 11, 0, 100_000)
            assert_within(self, counts, exact, 4.0, (deck, a, b))


class Gate5SettlementParityTest(unittest.TestCase):
    def test_sampled_rounds_replay_through_the_engine_to_the_same_class_and_deck(self):
        rng = np.random.default_rng(5)
        tiny = REDUCED & ~sum(1 << make_card(r, 0) for r in range(6, 13))  # 36 and 63 impossible
        cases = [(FULL_DECK_MASK, 25, 8), (REDUCED, 26, 8), (REDUCED, 25, 63), (tiny, 36, 63)]
        cases += [(deck, *PAIRS[int(rng.integers(len(PAIRS)))]) for deck in seeded_decks(6, 40, 10, 40) * 5]
        seen = set()
        for index, (deck, a, b) in enumerate(cases):
            result, next_deck, hand_0, hand_1 = sample_round_hands(deck, a, b, 77, index)
            self.assertEqual((result, next_deck), engine_replay(deck, a, b, (hand_0, hand_1)), (deck, a, b))
            seen.add(result)
        self.assertEqual(seen, {W0, W0C, W1, W1C, NOJ})


class DenseCountTest(unittest.TestCase):
    def test_rows_are_consistent_across_rounds_and_reproducible(self):
        targets = [(25, 15), (36, 8), (45, 44), (26, 47), (39, 63)]
        counts, errors = sample_sequence(targets, 42, 7, 50_000)
        self.assertEqual(errors, 0)
        self.assertEqual(counts.shape, (ROWS, CLASSES))
        self.assertEqual(int(counts[:, TIE].sum()), 0)
        for r in range(4):
            for prefix in np.ndindex(*(5,) * r):
                parent = counts[row(r, prefix)]
                for result in (W0, W0C, W1, W1C, NOJ):
                    child = counts[row(r + 1, (*prefix, result))]
                    self.assertEqual(int(child.sum()), int(parent[result]), (r, prefix, result))
        self.assertEqual(int(counts[0].sum()), 50_000)
        again, _ = sample_sequence(targets, 42, 7, 50_000)
        np.testing.assert_array_equal(again, counts)
        other, _ = sample_sequence(targets, 42, 8, 50_000)
        self.assertFalse(np.array_equal(other, counts))


if __name__ == "__main__":
    unittest.main()
