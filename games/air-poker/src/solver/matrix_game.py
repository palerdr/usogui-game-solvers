"""Exact small zero-sum matrix games by Shapley-Snow kernel enumeration.

For a payoff matrix ``M`` (row player maximizes, at most 5 x 5) every matrix
game has an optimal pair of strategies supported on a square submatrix ``S``
with ``1' adj(S) 1 != 0`` (Shapley and Snow 1950):

    v = det(S) / (1' adj(S) 1),  x_I = 1' adj(S) / (1' adj(S) 1),  y_J = adj(S) 1 / (1' adj(S) 1).

The solver enumerates the at most 251 square submatrices, keeps the first pair
that is nonnegative and optimal against every pure reply within a tolerance,
and reports the values each strategy guarantees without the tolerance:
``min_j (x' M)_j`` for the row player and ``max_i (M y)_i`` for the column
player. Those two numbers bracket the game value, and their gap is the saddle
gap the certificate reports.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MAX_SIZE = 5
TOLERANCE = 1e-9


@njit(cache=True)
def determinant(a, k):
    """Determinant of the leading ``k x k`` block of ``a`` by elimination with pivoting."""
    m = a[:k, :k].copy()
    det = 1.0
    for col in range(k):
        pivot = col
        for row in range(col + 1, k):
            if abs(m[row, col]) > abs(m[pivot, col]):
                pivot = row
        if m[pivot, col] == 0.0:
            return 0.0
        if pivot != col:
            for j in range(k):
                m[col, j], m[pivot, j] = m[pivot, j], m[col, j]
            det = -det
        det *= m[col, col]
        for row in range(col + 1, k):
            factor = m[row, col] / m[col, col]
            for j in range(col, k):
                m[row, j] -= factor * m[col, j]
    return det


@njit(cache=True)
def adjugate(s, k, out):
    """Write ``adj(s)`` of the ``k x k`` matrix ``s`` into ``out``; return ``det(s)``."""
    if k == 1:
        out[0, 0] = 1.0
        return s[0, 0]
    minor = np.empty((MAX_SIZE, MAX_SIZE))
    for i in range(k):
        for j in range(k):
            r = 0
            for a in range(k):
                if a == i:
                    continue
                c = 0
                for b in range(k):
                    if b == j:
                        continue
                    minor[r, c] = s[a, b]
                    c += 1
                r += 1
            sign = -1.0 if (i + j) % 2 else 1.0
            out[j, i] = sign * determinant(minor, k - 1)  # transpose of the cofactor matrix
    return determinant(s, k)


@njit(cache=True)
def guarantees(m, n, k_cols, x, y):
    """Return ``(min_j (x'M)_j, max_i (My)_i)``."""
    low = 1e300
    for j in range(k_cols):
        total = 0.0
        for i in range(n):
            total += x[i] * m[i, j]
        low = min(low, total)
    high = -1e300
    for i in range(n):
        total = 0.0
        for j in range(k_cols):
            total += m[i, j] * y[j]
        high = max(high, total)
    return low, high


@njit(cache=True)
def solve(m, n, k_cols):
    """Solve the ``n x k_cols`` game ``m``; return ``(x, y, row_guarantee, column_guarantee)``."""
    best_x = np.zeros(MAX_SIZE)
    best_y = np.zeros(MAX_SIZE)
    best_low = -1e300
    best_high = 1e300
    best_gap = 1e300
    sub = np.empty((MAX_SIZE, MAX_SIZE))
    adj = np.empty((MAX_SIZE, MAX_SIZE))
    rows = np.empty(MAX_SIZE, np.int64)
    cols = np.empty(MAX_SIZE, np.int64)
    x = np.zeros(MAX_SIZE)
    y = np.zeros(MAX_SIZE)
    scale = 1.0
    for i in range(n):
        for j in range(k_cols):
            scale = max(scale, abs(m[i, j]))
    for row_mask in range(1, 1 << n):
        k = 0
        for i in range(n):
            if (row_mask >> i) & 1:
                rows[k] = i
                k += 1
        for col_mask in range(1, 1 << k_cols):
            kc = 0
            for j in range(k_cols):
                if (col_mask >> j) & 1:
                    cols[kc] = j
                    kc += 1
            if kc != k:
                continue
            for a in range(k):
                for b in range(k):
                    sub[a, b] = m[rows[a], cols[b]]
            det = adjugate(sub, k, adj)
            denominator = 0.0
            for a in range(k):
                for b in range(k):
                    denominator += adj[a, b]
            if abs(denominator) <= 1e-14 * max(1.0, scale) ** (k - 1):
                continue
            x[:] = 0.0
            y[:] = 0.0
            feasible = True
            for b in range(k):
                column_sum = 0.0
                row_sum = 0.0
                for a in range(k):
                    column_sum += adj[a, b]
                    row_sum += adj[b, a]
                xv = column_sum / denominator
                yv = row_sum / denominator
                if xv < -TOLERANCE or yv < -TOLERANCE:
                    feasible = False
                    break
                x[rows[b]] = max(xv, 0.0)
                y[cols[b]] = max(yv, 0.0)
            if not feasible:
                continue
            x[:n] /= x[:n].sum()
            y[:k_cols] /= y[:k_cols].sum()
            low, high = guarantees(m, n, k_cols, x, y)
            gap = high - low
            if gap < best_gap:
                best_gap = gap
                best_low = low
                best_high = high
                best_x[:] = x
                best_y[:] = y
                if gap <= TOLERANCE * scale:
                    return best_x, best_y, best_low, best_high
            _ = det
    return best_x, best_y, best_low, best_high


def solve_matrix_game(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Python entry: ``(x, y, row_guarantee, column_guarantee)`` for an ``n x m`` matrix, ``n, m <= 5``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n, k_cols = matrix.shape
    padded = np.zeros((MAX_SIZE, MAX_SIZE))
    padded[:n, :k_cols] = matrix
    x, y, low, high = solve(padded, n, k_cols)
    return x[:n], y[:k_cols], float(low), float(high)
