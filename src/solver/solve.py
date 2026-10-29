"""Primal and dual sequence-form linear programs solved with HiGHS."""

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack

from src.solver.constraints import build_flow
from src.solver.payoff import build_payoff
from src.solver.sequences import SequenceIndex
from src.solver.tree import GameTree


@dataclass(frozen=True, slots=True)
class Solution:
    primal_value: float  # P0's guaranteed payoff from the primal LP
    dual_value: float  # P0's payoff cap from the dual LP
    p0_plan: np.ndarray  # realization plan over P0 sequences
    p1_plan: np.ndarray  # realization plan over P1 sequences
    seconds: float


def _lp(c, a_ub, b_ub, a_eq, b_eq, bounds):
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")
    return result


def solve(tree: GameTree, index: SequenceIndex) -> Solution:
    start = time.perf_counter()
    payoff = build_payoff(tree, index)
    e_mat, e_rhs = build_flow(index.players[0])
    f_mat, f_rhs = build_flow(index.players[1])
    n0, n1 = payoff.shape
    m0, m1 = e_mat.shape[0], f_mat.shape[0]

    # P0: max u_0  s.t.  F^T u - A^T x <= 0,  E x = e,  x >= 0,  u free.
    c = np.zeros(n0 + m1)
    c[n0] = -1.0
    a_ub = hstack([-payoff.T.tocsr(), f_mat.T.tocsr()]).tocsr()
    a_eq = hstack([e_mat, csr_matrix((m0, m1))]).tocsr()
    bounds = [(0, None)] * n0 + [(None, None)] * m1
    primal = _lp(c, a_ub, np.zeros(n1), a_eq, e_rhs, bounds)
    p0_plan = primal.x[:n0]
    primal_value = -primal.fun

    # P1: min w_0  s.t.  A y - E^T w <= 0,  F y = f,  y >= 0,  w free.
    c = np.zeros(n1 + m0)
    c[n1] = 1.0
    a_ub = hstack([payoff, -e_mat.T.tocsr()]).tocsr()
    a_eq = hstack([f_mat, csr_matrix((m1, m0))]).tocsr()
    bounds = [(0, None)] * n1 + [(None, None)] * m0
    dual = _lp(c, a_ub, np.zeros(n0), a_eq, f_rhs, bounds)
    p1_plan = dual.x[:n1]
    dual_value = dual.fun

    return Solution(primal_value, dual_value, p0_plan, p1_plan, time.perf_counter() - start)
