import unittest

from src.solver.array_best_response import profile_value, saddle_gap
from src.solver.cfr_plus import CFRPlus
from src.solver.public_tree import build_public_tree

D2_VALUE = -0.045519573417  # D_2 LP value, certified by the rational certificate


class CFRPlusTests(unittest.TestCase):
    def run_cfr(self, horizon: int, checkpoints: list[int]) -> tuple[list[tuple[float, float]], CFRPlus]:
        layers = build_public_tree(horizon)
        cfr = CFRPlus(layers)
        bounds = []
        for target in checkpoints:
            while cfr.iterations < target:
                cfr.iterate()
            bounds.append(saddle_gap(layers, *cfr.average()))
        return bounds, cfr

    def test_d1_gap_declines_toward_zero(self) -> None:
        bounds, _ = self.run_cfr(1, [10, 100, 2000])
        gaps = [upper - lower for lower, upper in bounds]
        self.assertGreater(gaps[0], gaps[1])
        self.assertGreater(gaps[1], gaps[2])
        self.assertLess(gaps[2], 1e-4)
        self.assertLessEqual(bounds[-1][0], 1e-12)
        self.assertGreaterEqual(bounds[-1][1], -1e-12)

    def test_d2_brackets_the_lp_value(self) -> None:
        bounds, cfr = self.run_cfr(2, [30, 300])
        (lo1, hi1), (lo2, hi2) = bounds
        self.assertLess(hi2 - lo2, hi1 - lo1)
        self.assertLess(hi2 - lo2, 1e-2)
        self.assertLessEqual(lo2, D2_VALUE)
        self.assertGreaterEqual(hi2, D2_VALUE)
        value = profile_value(cfr.layers, *cfr.average())
        self.assertLessEqual(lo2, value + 1e-12)
        self.assertGreaterEqual(hi2, value - 1e-12)

    def test_strategies_are_distributions(self) -> None:
        _, cfr = self.run_cfr(2, [5])
        for strategy in (cfr.current(), cfr.average()):
            behavior, choose = strategy
            for b in behavior:
                self.assertTrue(((b >= 0) & (abs(b.sum(axis=1) - 1) < 1e-12)[:, None]).all())
            for c in choose:
                self.assertAlmostEqual(c.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
