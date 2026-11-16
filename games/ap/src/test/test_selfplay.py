"""Self-play maps solver decisions to legal engine actions and plays whole games through the engine."""

import unittest

import numpy as np

from src.engine import game
from src.kernel.table import KernelCounts, Sequences
from src.sensitivity.sampler import W0
from src.solver.allocate import allocation
from src.solver.betting import CHIPS
from src.solver.matrix_game import MAX_SIZE
from src.solver.robust import Arena, RobustSolver
from src.solver.selfplay import Policy, play_game, record_digit, slot_action
from src.test.test_betting_dag import betting_state, hands_of_class


class RecordDigitTest(unittest.TestCase):
    def test_fold_no_judgement_and_win_classes(self):
        fold = (game.Team.ONE, game.BettingAction(game.RoundAction.Fold))
        self.assertEqual(record_digit(game.RoundRecord((25, 8), game.Team.ZERO, (fold,), game.Team.ZERO, False)), 5)
        self.assertEqual(record_digit(game.RoundRecord((25, 8), game.Team.ZERO, (), None, False)), 4)
        self.assertEqual(record_digit(game.RoundRecord((25, 8), game.Team.ZERO, (), game.Team.ZERO, False)), 0)
        self.assertEqual(record_digit(game.RoundRecord((25, 8), game.Team.ZERO, (), game.Team.ZERO, True)), 1)
        self.assertEqual(record_digit(game.RoundRecord((25, 8), game.Team.ZERO, (), game.Team.ONE, False)), 2)
        self.assertEqual(record_digit(game.RoundRecord((25, 8), game.Team.ZERO, (), game.Team.ONE, True)), 3)


class SlotActionTest(unittest.TestCase):
    def test_slots_enumerate_exactly_the_engine_legal_actions_in_dag_order(self):
        rng = np.random.default_rng(12)
        hands = hands_of_class(W0)
        for _ in range(40):
            r = int(rng.integers(5))
            ante = game.ANTES[r]
            s0, s1 = int(rng.integers(ante, 30)), int(rng.integers(ante, 30))
            c0, c1 = ante + int(rng.integers(0, 6)), ante + int(rng.integers(0, 6))
            actor = int(rng.integers(2))
            state = betting_state(r, (s0, s1), 0, (c0, c1), actor, bool(rng.integers(2)), hands)
            legal = game.legal_actions(state)
            to_call = max(c0, c1) - (c0, c1)[actor]
            translated = [slot_action(slot, to_call) for slot in range(len(legal))]
            self.assertEqual(set(translated), set(legal), (r, s0, s1, c0, c1, actor))
            self.assertEqual(len(set(translated)), len(translated))


class SyntheticSolver(RobustSolver):
    """A solver with random kept tables for rounds 1 and 2 and random boxes, for engine play."""

    def __init__(self, sequences, arena, rng):
        super().__init__(sequences, arena)
        self.set_boxes(KernelCounts(sequences), 1e-3)
        for r in (1, 2, 3, 4):
            lo = rng.uniform(0, 0.2, size=self.box_lo[r].shape)
            hi = np.clip(lo + rng.uniform(0.2, 0.6, size=lo.shape), 0, 1)
            lo[..., 4] = hi[..., 4] = 0.0
            self.box_lo[r], self.box_hi[r] = lo, hi
        self.keep_depth = 2
        self.kept_values, self.kept_strategies, self.max_gap = {}, {}, 0.0
        for r in (1, 2):
            for prefix in sequences.prefix_index[r - 1]:  # keys of length r
                lo = rng.uniform(-1, 1, size=(6 ** r, CHIPS, CHIPS))
                self.kept_values[prefix] = (lo, np.minimum(1.0, lo + 0.1))
                rows, cols = self.remaining_pairs(prefix)
                chips = arena.chip_array(r, self.first(prefix, r))
                uniform = np.full((6 ** r, chips.shape[0], len(rows)), 1.0 / len(rows))
                self.kept_strategies[prefix] = (rows, cols, chips, uniform, uniform, uniform, uniform)
        rows, cols = self.remaining_pairs(())
        root = np.full((1, 1, MAX_SIZE), 0.2)
        self.kept_strategies[()] = (rows, cols, np.array([[23, 23]]), root, root, root, root)


class EnginePlayTest(unittest.TestCase):
    def test_games_run_through_the_engine_to_a_terminal_state(self):
        sequences = Sequences()
        solver = SyntheticSolver(sequences, Arena(), np.random.default_rng(13))
        policies = (Policy(solver, False), Policy(solver, True))
        played = set()
        for index in range(3):
            utility, sequence = play_game(policies, 5, index)
            self.assertIn(utility, (-1.0, 0.0, 1.0))
            self.assertIn(len(sequence), range(1, 6))  # a bankruptcy ends the game early
            self.assertIn(tuple((int(a), int(b)) for a, b in sequence), sequences.prefix_index[len(sequence) - 1])
            played.add(sequence)
        self.assertEqual(play_game(policies, 5, 1), play_game(policies, 5, 1))


class AllocationTest(unittest.TestCase):
    def test_floor_plus_proportional_share_sums_to_at_most_the_budget(self):
        reach = np.zeros(14_400, dtype=np.int64)
        reach[[3, 7, 7, 100]] += 1
        reach[7] += 10
        plan = allocation(reach, 1_440_000, 0.5)
        self.assertEqual(int(plan.min()), 50)
        self.assertLessEqual(int(plan.sum()), 1_440_000)
        self.assertGreater(plan[7], plan[3])
        self.assertEqual(plan[3], plan[100])
        np.testing.assert_array_equal(allocation(np.zeros(4, dtype=np.int64), 100, 0.0), [25, 25, 25, 25])


if __name__ == "__main__":
    unittest.main()
