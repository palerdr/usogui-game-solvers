import unittest
from fractions import Fraction

import numpy as np

from src.solver.best_response import behavior_from_plan, best_response_value, certify
from src.solver.build_tree import build_tree
from src.solver.constraints import build_flow
from src.solver.payoff import build_payoff
from src.solver.primitives import Guess, Phase, Player
from src.solver.sequences import build_sequence_index
from src.solver.solve import solve


def hand_strategy(index, player, announce_eleven):
    """Uniform beads; guess uniformly over the hard support, or always announce 11."""
    behavior = {}
    for info in index.players[player].information_sets:
        state = info.information_state
        actions = [index.players[player].sequences[s].action for s in info.action_sequence_ids]
        if state.phase in (Phase.CHOOSE_P0, Phase.CHOOSE_P1):
            behavior[state] = {a: Fraction(1, 10) for a in actions}
            continue
        x = state.own_beads
        cands = set(range(1, 11))
        for miss in state.public_history:
            cands.discard(miss.total - x)
            if miss.player != state.player:
                cands = {c for c in cands if miss.total - 10 <= c <= miss.total - 1}
        if announce_eleven and (11 - x) in cands:
            cands = {11 - x}
        p = Fraction(1, len(cands))
        behavior[state] = {a: (p if a.total - x in cands else Fraction(0)) for a in actions}
    return behavior


class D1SolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = build_tree(1)
        cls.index = build_sequence_index(cls.tree)

    def test_payoff_holds_one_entry_per_hit_leaf(self) -> None:
        payoff = build_payoff(self.tree, self.index)
        self.assertEqual(payoff.shape, (111, 1_891))
        self.assertEqual(payoff.nnz, 1_000)
        self.assertEqual(int((payoff.data == 1).sum()), 100)
        self.assertEqual(int((payoff.data == -1).sum()), 900)

    def test_flow_matrices_balance_every_information_set(self) -> None:
        for player in Player:
            matrix, rhs = build_flow(self.index.players[player])
            self.assertEqual(matrix.shape[0], 1 + len(self.index.players[player].information_sets))
            self.assertEqual(rhs[0], 1.0)
            self.assertTrue(np.all(rhs[1:] == 0.0))
            self.assertTrue(np.all(np.asarray(matrix.sum(axis=1)).ravel()[1:] == 9.0))

    def test_d1_value_is_zero_and_certified(self) -> None:
        solution = solve(self.tree, self.index)
        self.assertAlmostEqual(solution.primal_value, 0.0, places=9)
        self.assertAlmostEqual(solution.dual_value, 0.0, places=9)
        p0 = behavior_from_plan(self.index, Player.P0, solution.p0_plan)
        p1 = behavior_from_plan(self.index, Player.P1, solution.p1_plan)
        lower, upper = certify(self.tree, p0, p1)
        self.assertLessEqual(upper - lower, Fraction(1, 10**9))

    def test_announcing_eleven_guarantees_zero_exactly(self) -> None:
        p0 = hand_strategy(self.index, Player.P0, announce_eleven=True)
        p1 = hand_strategy(self.index, Player.P1, announce_eleven=False)
        self.assertEqual(certify(self.tree, p0, p1), (Fraction(0), Fraction(0)))

    def test_uniform_guessing_leaks_for_p0(self) -> None:
        p0 = hand_strategy(self.index, Player.P0, announce_eleven=False)
        self.assertEqual(-best_response_value(self.tree, Player.P1, p0), Fraction(-9, 100))

    def test_behavior_sums_to_one_with_eleven_on_path(self) -> None:
        solution = solve(self.tree, self.index)
        p0 = behavior_from_plan(self.index, Player.P0, solution.p0_plan)
        for state, dist in p0.items():
            self.assertEqual(sum(dist.values()), 1)
            if state.phase is Phase.GUESS_P0:
                self.assertEqual(dist[Guess(11)], 1)


if __name__ == "__main__":
    unittest.main()
