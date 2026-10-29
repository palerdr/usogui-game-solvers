import sys
import unittest

import numpy as np

from src.solver.array_best_response import saddle_gap
from src.solver.array_lp import build_sequence_form, plan_behavior, solve_lp
from src.solver.build_tree import build_tree
from src.solver.primitives import Phase
from src.solver.public_abstraction import transcripts
from src.solver.public_tree import build_public_tree, sizes
from src.solver.sequences import build_sequence_index

sys.setrecursionlimit(10000)


def array_information_states(layers):
    """(player, own beads, transcript) of every guess information set in the arrays."""
    out = set()
    for d, layer in enumerate(layers):
        p = layer.actor
        paths = transcripts(layers)[d]
        for node, hand in zip(layer.slot_node[p], layer.slot_hand[p]):
            out.add((p, int(hand), tuple(int(g) for g in paths[node])))
    return out


class PublicTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trees = {k: build_tree(k) for k in (1, 2)}
        cls.indexes = {k: build_sequence_index(t) for k, t in cls.trees.items()}

    def test_sizes_match_the_dataclass_tree(self) -> None:
        for k, tree in self.trees.items():
            s = sizes(build_public_tree(k))
            index = self.indexes[k]
            self.assertEqual(s.nodes, len(tree.nodes))
            self.assertEqual(s.hit_leaves, sum(1 for n in tree.nodes if n.returns is not None and n.returns[0] != 0))
            self.assertEqual(s.draw_leaves, sum(1 for n in tree.nodes if n.returns == (0, 0)))
            self.assertEqual(s.information_sets, tuple(len(index.players[p].information_sets) for p in (0, 1)))
            self.assertEqual(s.sequences, tuple(len(index.players[p].sequences) for p in (0, 1)))

    def test_information_sets_match_the_dataclass_tree(self) -> None:
        for k, index in self.indexes.items():
            expected = {
                (int(info.information_state.player), info.information_state.own_beads,
                 tuple(m.total for m in info.information_state.public_history))
                for p in (0, 1)
                for info in index.players[p].information_sets
                if info.information_state.phase in (Phase.GUESS_P0, Phase.GUESS_P1)
            }
            self.assertEqual(array_information_states(build_public_tree(k)), expected)

    def test_d3_sizes(self) -> None:
        s = sizes(build_public_tree(3))
        self.assertEqual(s.nodes, 66_430_111)
        self.assertEqual(s.hit_leaves, 6_643_000)
        self.assertEqual(s.information_sets, (248_891, 2_926_157))
        self.assertEqual(s.sequences, (2_488_911, 29_261_571))


class ArrayLPTests(unittest.TestCase):
    def test_d1_lp_is_zero_and_certified(self) -> None:
        layers = build_public_tree(1)
        form = build_sequence_form(layers)
        self.assertEqual(form.payoff.nnz, 1_000)
        self.assertEqual(int((form.payoff.data == 1).sum()), 100)
        solution = solve_lp(form, layers)
        self.assertAlmostEqual(solution.value, 0.0, places=9)
        behavior, choose = [None] * len(layers), [None, None]
        plan_behavior(form, layers, 0, solution.p0_plan, behavior, choose)
        plan_behavior(form, layers, 1, solution.p1_plan, behavior, choose)
        lower, upper = saddle_gap(layers, behavior, tuple(choose))
        self.assertLess(upper - lower, 1e-9)
        for p in (0, 1):
            residual = form.flow[p] @ np.maximum(solution.p0_plan if p == 0 else solution.p1_plan, 0)
            self.assertAlmostEqual(residual[0], 1.0)
            self.assertLess(np.abs(residual[1:]).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
