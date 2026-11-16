"""The robust round value agrees with an LP reference, collapses on point kernels, and widens monotonically."""

import unittest

import numpy as np
from scipy.optimize import linprog

from src.engine.game import ANTES
from src.kernel.sampler import DIGIT
from src.kernel.table import HIDDEN, KernelCounts, Sequences
from src.sensitivity.sampler import NOJ, W0, W0C, W1, W1C
from src.solver.betting import CHIPS, BettingDag, evaluate, settle
from src.solver.robust import Arena, RobustSolver, w_table

WIN = (W0, W0C, W1, W1C)
SEQUENCES = Sequences()
ARENA = Arena()


def lp_extreme(u, q_lo, q_hi, minimize: bool) -> float:
    """Extreme of ``q . u`` over the box intersected with the simplex, by linear program."""
    c = np.array(u) if minimize else -np.array(u)
    result = linprog(c, A_eq=np.ones((1, 4)), b_eq=[1.0], bounds=list(zip(q_lo, q_hi)), method="highs")
    assert result.success
    return float(result.fun if minimize else -result.fun)


def w_reference(r, first, s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi) -> tuple[float, float]:
    """Plain-Python ``W`` with LP extremes, the same DAG, settlement and continuation rules."""
    dag = BettingDag(s0, s1, ANTES[r], first)
    t_lo, t_hi = box_lo[NOJ], box_hi[NOJ]
    q_lo = [0.0 if t_lo >= 1 else box_lo[c] / (1 - t_lo) for c in WIN]
    q_hi = [1.0 if t_hi >= 1 else min(1.0, box_hi[c] / (1 - t_hi)) for c in WIN]

    def cont(c0, c1, result, child):
        terminal, utility, t0, t1 = settle(r, s0, s1, c0, c1, result)
        if terminal:
            return utility, utility
        return vnext_lo[child, t0, t1], vnext_hi[child, t0, t1]

    show_lo, show_hi = [], []
    for c0, c1 in dag.showdowns:
        u = [cont(c0, c1, c, code * 6 + int(DIGIT[c])) for c in WIN]
        show_lo.append(lp_extreme([x[0] for x in u], q_lo, q_hi, True))
        show_hi.append(lp_extreme([x[1] for x in u], q_lo, q_hi, False))
    fold = [cont(c0, c1, -1 - who, code * 6 + HIDDEN) for c0, c1, who in dag.folds]
    scratch = np.empty(len(dag))
    b_lo = evaluate(dag.actor, dag.action_start, dag.action_target, np.array(show_lo), np.array([f[0] for f in fold]), scratch)
    b_hi = evaluate(dag.actor, dag.action_start, dag.action_target, np.array(show_hi), np.array([f[1] for f in fold]), scratch)
    n_lo, n_hi = cont(ANTES[r], ANTES[r], NOJ, code * 6 + int(DIGIT[NOJ]))
    w_lo = min(t * n_lo + (1 - t) * b_lo for t in (t_lo, t_hi))
    w_hi = max(t * n_hi + (1 - t) * b_hi for t in (t_lo, t_hi))
    return w_lo, w_hi


def random_box(rng, point: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A box around a random kernel on the five reachable classes; ``point`` collapses it."""
    p = rng.dirichlet(np.ones(5) * rng.choice([0.2, 1.0, 5.0]))
    k = np.zeros(6)
    k[[W0, W0C, W1, W1C, NOJ]] = p
    if rng.random() < 0.5:
        k[NOJ] = 0.0
        k /= k.sum()
    width = rng.uniform(0, 0.3) * (0.0 if point else 1.0)  # same stream either way
    lo = np.clip(k - width * rng.random(6), 0, 1)
    hi = np.clip(k + width * rng.random(6), 0, 1)
    lo[4] = hi[4] = 0.0
    return lo, hi


class RoundValueTest(unittest.TestCase):
    def test_numba_round_value_matches_the_lp_reference(self):
        rng = np.random.default_rng(8)
        checked = 0
        for r in range(5):
            for _ in range(6):
                first = int(rng.integers(2))
                chips = ARENA.chips[r][first]
                s0, s1 = chips[int(rng.integers(len(chips)))]
                code = int(rng.integers(6 ** r))
                vnext_lo = rng.uniform(-1, 1, size=(6 ** (r + 1), CHIPS, CHIPS))
                vnext_hi = np.minimum(1.0, vnext_lo + rng.uniform(0, 0.5, size=vnext_lo.shape))
                box_lo, box_hi = random_box(rng)
                out_lo = np.full((6 ** r, CHIPS, CHIPS), np.nan)
                out_hi = np.full((6 ** r, CHIPS, CHIPS), np.nan)
                box_lo_t = np.zeros((6 ** r, 6))
                box_hi_t = np.zeros((6 ** r, 6))
                box_lo_t[code] = box_lo
                box_hi_t[code] = box_hi
                w_table(r, first, box_lo_t, box_hi_t, vnext_lo, vnext_hi, np.array([[s0, s1]], dtype=np.int64),
                        ARENA.max_nodes, ARENA.max_terminals, *ARENA.arrays(), out_lo, out_hi)
                expected = w_reference(r, first, s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi)
                self.assertAlmostEqual(out_lo[code, s0, s1], expected[0], places=9, msg=(r, first, s0, s1))
                self.assertAlmostEqual(out_hi[code, s0, s1], expected[1], places=9, msg=(r, first, s0, s1))
                self.assertLessEqual(out_lo[code, s0, s1], out_hi[code, s0, s1] + 1e-12)
                checked += 1
        self.assertEqual(checked, 30)


class SubtreeTest(unittest.TestCase):
    """A round-4 subtree (four round-5 tables and the 2x2 games) under synthetic boxes."""

    def subtree(self, rng, point: bool):
        solver = RobustSolver(SEQUENCES, ARENA)
        solver.set_boxes(KernelCounts(SEQUENCES), 1e-3)
        prefix = tuple((int(a), int(b)) for a, b in SEQUENCES.targets[7][:3])
        for r in (3, 4):  # boxes of the number prefixes under ``prefix`` only
            lo, hi = solver.box_lo[r], solver.box_hi[r]
            for key, index in SEQUENCES.prefix_index[r].items():
                if key[:3] == prefix:
                    for code in range(6 ** r):
                        lo[index, code], hi[index, code] = random_box(rng, point)
        solver.kept_values, solver.kept_strategies, solver.max_gap, solver.keep_depth = {}, {}, 0.0, -1
        lo, hi, strategies = solver._v(prefix, 3)
        return solver, lo, hi, strategies

    def test_point_kernels_collapse_the_interval_and_values_are_finite_on_reachable_chips(self):
        rng = np.random.default_rng(9)
        solver, lo, hi, strategies = self.subtree(rng, point=True)
        first = solver.first(tuple((int(a), int(b)) for a, b in SEQUENCES.targets[7][:3]), 3)
        reachable = np.zeros((CHIPS, CHIPS), bool)
        for s0, s1 in ARENA.chips[3][first]:
            reachable[s0, s1] = True
        self.assertTrue(np.all(np.isfinite(lo[:, reachable])))
        self.assertTrue(np.all(np.isnan(lo[:, ~reachable])))
        np.testing.assert_allclose(lo[:, reachable], hi[:, reachable], atol=1e-9)
        self.assertLessEqual(solver.max_gap, 1e-9)
        rows, cols, _, x_lo, _, _, _ = strategies
        self.assertEqual((len(rows), len(cols)), (2, 2))
        np.testing.assert_allclose(x_lo.sum(-1), 1.0)
        self.assertTrue(np.all(np.abs(lo[:, reachable]) <= 1.0 + 1e-12))

    def test_wider_boxes_give_a_wider_interval_containing_the_point_value(self):
        _, lo_p, hi_p, _ = self.subtree(np.random.default_rng(10), point=True)
        _, lo_w, hi_w, _ = self.subtree(np.random.default_rng(10), point=False)
        finite = np.isfinite(lo_p)
        self.assertTrue(np.all(lo_w[finite] <= lo_p[finite] + 1e-9))
        self.assertTrue(np.all(hi_w[finite] >= hi_p[finite] - 1e-9))
        self.assertGreater(float((hi_w - lo_w)[finite].mean()), 0.01)


if __name__ == "__main__":
    unittest.main()
