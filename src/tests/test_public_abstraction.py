import unittest

import numpy as np

from src.solver.array_best_response import best_response_value
from src.solver.array_lp import build_sequence_form, plan_behavior, solve_lp
from src.solver.cfr_plus import CFRPlus
from src.solver.public_abstraction import (
    MergedCFRPlus,
    alternate,
    guarantee,
    last_layer,
    merged_best_response,
    slot_keys,
    transcripts,
)
from src.solver.public_tree import build_public_tree


def reference_key(name: str, p: int, own: int, totals: tuple[int, ...]):
    """The keys of order_blind.py, computed from the rules on one transcript."""
    alive = set()
    for opp in range(1, 11):
        hands = (own, opp) if p == 0 else (opp, own)
        legal = all(hands[j % 2] + 1 <= g <= hands[j % 2] + 10 for j, g in enumerate(totals))
        if legal and all(g != hands[0] + hands[1] for g in totals):
            alive.add(opp)
    mine = frozenset(totals[p::2])
    theirs = frozenset(totals[1 - p :: 2])
    return {
        "support": (own, frozenset(alive)),
        "hard": (own, mine, theirs),
        "opening+hard": (own, mine, theirs, totals[0]),
    }[name]


class SlotKeyTests(unittest.TestCase):
    def test_keys_partition_like_the_reference(self) -> None:
        layers = build_public_tree(2)
        for p in (0, 1):
            d = last_layer(layers, p)
            paths = transcripts(layers)[d]
            for name in ("support", "hard", "opening+hard"):
                key, n_keys = slot_keys(layers, p, name)
                ref = [
                    reference_key(name, p, int(h), tuple(int(g) for g in paths[node]))
                    for node, h in zip(layers[d].slot_node[p], layers[d].slot_hand[p])
                ]
                self.assertEqual(n_keys, len(set(ref)))
                self.assertEqual(len(set(zip(key.tolist(), ref))), n_keys)

    def test_d2_support_key_count(self) -> None:
        key, n_keys = slot_keys(build_public_tree(2), 0, "support")
        self.assertEqual((len(key), n_keys), (1858, 958))


class MergedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layers = build_public_tree(2)
        form = build_sequence_form(cls.layers)
        solution = solve_lp(form, cls.layers, method="highs-ipm")
        cls.value = solution.value
        behavior, choose = [None] * len(cls.layers), [None, None]
        plan_behavior(form, cls.layers, 0, solution.p0_plan, behavior, choose)
        plan_behavior(form, cls.layers, 1, solution.p1_plan, behavior, choose)
        cls.behavior, cls.choose = behavior, tuple(choose)

    def test_full_key_response_recovers_the_value(self) -> None:
        for p in (0, 1):
            key, n_keys = slot_keys(self.layers, p, "full")
            value, _ = merged_best_response(self.layers, p, self.behavior, self.choose, key, n_keys)
            self.assertAlmostEqual(value, self.value if p == 0 else -self.value, places=6)

    def test_merged_cfr_keeps_p0_value_on_support(self) -> None:
        key, n_keys = slot_keys(self.layers, 0, "support")
        cfr = MergedCFRPlus(self.layers, 0, self.behavior, self.choose, key, n_keys)
        for _ in range(1000):
            cfr.iterate()
        behavior, choose = cfr.average()
        guarantee = -best_response_value(self.layers, 1, behavior, choose)
        cap, _ = merged_best_response(self.layers, 0, behavior, choose, key, n_keys)
        self.assertLessEqual(guarantee, cap + 1e-12)
        self.assertGreater(guarantee, self.value - 1e-4)
        merged = cfr.merged_average()
        self.assertTrue(np.allclose(behavior[last_layer(self.layers, 0)], merged[key]))


if __name__ == "__main__":
    unittest.main()


def cfr_after(layers, iterations: int = 200) -> CFRPlus:
    cfr = CFRPlus(layers)
    for _ in range(iterations):
        cfr.iterate()
    return cfr


class AlternateTests(unittest.TestCase):
    def test_alternation_returns_a_key_measurable_strategy(self) -> None:
        layers = build_public_tree(2)
        cfr = cfr_after(layers)
        behavior, choose = cfr.average()
        key, n_keys = slot_keys(layers, 1, "support")
        steps, (b, c) = alternate(layers, 1, behavior, choose, key, n_keys, rounds=1, iterations=100, log=lambda s: None)
        self.assertEqual([s[0] for s in steps], ["merge1", "early1"])
        last = b[last_layer(layers, 1)]
        for k in range(n_keys):
            rows = last[key == k]
            self.assertTrue(np.allclose(rows, rows[0]))
        self.assertAlmostEqual(max(s[1] for s in steps), guarantee(layers, 1, b, c), places=12)
        self.assertLessEqual(steps[0][1], steps[0][2] + 1e-12)
