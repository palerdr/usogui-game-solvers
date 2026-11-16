"""The Shapley-Snow solver brackets the LP value of every small matrix game."""

import unittest

import numpy as np
from scipy.optimize import linprog

from src.solver.matrix_game import solve_matrix_game


def lp_value(matrix: np.ndarray) -> float:
    """Game value by linear program: maximize v subject to M'x >= v, sum x = 1."""
    n, m = matrix.shape
    c = np.zeros(n + 1)
    c[-1] = -1.0
    a_ub = np.hstack([-matrix.T, np.ones((m, 1))])
    result = linprog(c, A_ub=a_ub, b_ub=np.zeros(m), A_eq=np.hstack([np.ones((1, n)), [[0.0]]]),
                     b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)], method="highs")
    assert result.success
    return float(result.x[-1])


class MatrixGameTest(unittest.TestCase):
    def check(self, matrix: np.ndarray) -> None:
        x, y, low, high = solve_matrix_game(matrix)
        value = lp_value(matrix)
        scale = max(1.0, np.abs(matrix).max())
        self.assertLessEqual(low, value + 1e-9 * scale, matrix)
        self.assertGreaterEqual(high, value - 1e-9 * scale, matrix)
        self.assertLessEqual(high - low, 1e-9 * scale, matrix)
        self.assertAlmostEqual(x.sum(), 1.0, places=12)
        self.assertAlmostEqual(y.sum(), 1.0, places=12)
        self.assertTrue(np.all(x >= 0) and np.all(y >= 0))
        np.testing.assert_allclose((x @ matrix).min(), low, rtol=0, atol=1e-12 * scale)
        np.testing.assert_allclose((matrix @ y).max(), high, rtol=0, atol=1e-12 * scale)

    def test_random_real_matrices_of_every_shape(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            n, m = rng.integers(1, 6, size=2)
            self.check(rng.normal(size=(n, m)) * rng.choice([1.0, 0.01, 100.0]))

    def test_degenerate_integer_matrices(self):
        rng = np.random.default_rng(2)
        for _ in range(300):
            n, m = rng.integers(1, 6, size=2)
            self.check(rng.integers(-1, 2, size=(n, m)).astype(float))
        self.check(np.zeros((5, 5)))
        self.check(np.ones((3, 4)) * 0.25)
        self.check(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        self.check(np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]))
        self.check(np.array([[0.3, 0.3], [0.3, 0.3], [0.1, 0.9]]))

    def test_matching_pennies_and_pure_saddle(self):
        x, y, low, high = solve_matrix_game(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        np.testing.assert_allclose(x, [0.5, 0.5])
        np.testing.assert_allclose(y, [0.5, 0.5])
        self.assertEqual((low, high), (0.0, 0.0))
        x, y, low, high = solve_matrix_game(np.array([[3.0, 5.0], [1.0, 2.0]]))
        np.testing.assert_allclose(x, [1.0, 0.0])
        np.testing.assert_allclose(y, [1.0, 0.0])
        self.assertEqual((low, high), (3.0, 3.0))


if __name__ == "__main__":
    unittest.main()
