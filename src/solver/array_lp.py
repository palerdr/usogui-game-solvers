"""Sequence-form LP built from the public-tree arrays.

Sequence 0 is empty and sequences 1..10 choose beads. The guess sequences of
actor slot k on layer d are offset[d] + 10 k + a, where action a announces
own beads + a + 1. The LP is P0's primal of solve.py with one reduction: a
last-layer P1 sequence that hits no alive pair only states u_I <= 0, which
every alive sequence of the same information set implies, so it is dropped.
HiGHS returns P1's plan as the multipliers of the remaining rows.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack

from src.solver.public_tree import HANDS, Layer


@dataclass(frozen=True, slots=True)
class SequenceForm:
    offset: tuple[list, list]  # offset[p][d] of player p's sequences on layer d
    count: tuple[int, int]
    payoff: csr_matrix  # P0's payoff by (P0 sequence, P1 sequence)
    flow: tuple[csr_matrix, csr_matrix]  # row 0 fixes the empty sequence; one row per information set


def current_sequence(layers: list[Layer], offset, d: int, q: int) -> np.ndarray:
    """Player q's sequence at each of its slots on layer d, before layer d acts."""
    layer = layers[d]
    hand = layer.slot_hand[q].astype(np.int64)
    if d == 0:
        return hand  # sequence x chooses x beads
    up = layer.slot_parent[q]
    if layers[d - 1].actor == q:
        action = layer.total[layer.slot_node[q]].astype(np.int64) - hand - 1
        return offset[q][d - 1] + HANDS * up.astype(np.int64) + action
    return current_sequence(layers, offset, d - 1, q)[up]


def build_sequence_form(layers: list[Layer]) -> SequenceForm:
    offset = ([None] * len(layers), [None] * len(layers))
    count = [1 + HANDS, 1 + HANDS]
    for d, layer in enumerate(layers):
        p = layer.actor
        offset[p][d] = count[p]
        count[p] += HANDS * len(layer.slot_node[p])

    rows, cols, vals = [], [], []
    for d, layer in enumerate(layers):
        p, q = layer.actor, 1 - layer.actor
        own = offset[p][d] + HANDS * layer.pair_slot[p].astype(np.int64) + layer.pair_hand[q] - 1
        opp = current_sequence(layers, offset, d, q)[layer.pair_slot[q]]
        seq = (own, opp) if p == 0 else (opp, own)
        rows.append(seq[0]), cols.append(seq[1])
        vals.append(np.full(layer.pairs, 1.0 if p == 0 else -1.0))
    payoff = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=tuple(count)
    ).tocsr()

    flows = []
    for p in (0, 1):
        # Row 0: empty sequence = 1. Row 1: the bead choice sums to the empty sequence.
        r = [np.array([0, 1] + [1] * HANDS)]
        c = [np.array([0, 0] + list(range(1, 1 + HANDS)))]
        v = [np.array([1.0, -1.0] + [1.0] * HANDS)]
        row = 2
        for d, layer in enumerate(layers):
            if layer.actor != p:
                continue
            n = len(layer.slot_node[p])
            info_rows = row + np.arange(n)
            parent = current_sequence(layers, offset, d, p)
            r.append(info_rows), c.append(parent), v.append(np.full(n, -1.0))
            r.append(np.repeat(info_rows, HANDS))
            c.append(offset[p][d] + np.arange(n * HANDS))
            v.append(np.ones(n * HANDS))
            row += n
        flows.append(
            coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(row, count[p])).tocsr()
        )
    return SequenceForm(offset, (count[0], count[1]), payoff, (flows[0], flows[1]))


@dataclass(frozen=True, slots=True)
class LPSolution:
    value: float
    p0_plan: np.ndarray
    p1_plan: np.ndarray
    rows: int
    columns: int
    nonzeros: int
    seconds: float
    status: str


def lp_dimensions(form: SequenceForm, layers: list[Layer]) -> tuple[np.ndarray, int, int, int]:
    """Kept P1 rows and the LP's rows, columns and nonzeros, without building it."""
    keep = np.ones(form.count[1], dtype=bool)
    d = len(layers) - 1
    last = slice(form.offset[1][d], form.count[1])
    keep[last] = np.diff(form.payoff.tocsc().indptr)[last] > 0
    n0, (m1, _) = form.count[0], form.flow[1].shape
    ft = form.flow[1].T.tocsr()[keep]
    rows = int(keep.sum()) + form.flow[0].shape[0]
    nnz = int(form.payoff.T.tocsr()[keep].nnz + ft.nnz + form.flow[0].nnz)
    return keep, rows, n0 + m1, nnz


def solve_lp(form: SequenceForm, layers: list[Layer], **options) -> LPSolution:
    start = time.perf_counter()
    keep, rows, columns, nnz = lp_dimensions(form, layers)
    e_mat, f_mat = form.flow
    n0, m0, m1 = form.count[0], e_mat.shape[0], f_mat.shape[0]
    c = np.zeros(n0 + m1)
    c[n0] = -1.0
    a_ub = hstack([-form.payoff.T.tocsr()[keep], f_mat.T.tocsr()[keep]]).tocsr()
    a_eq = hstack([e_mat, csr_matrix((m0, m1))]).tocsr()
    b_eq = np.zeros(m0)
    b_eq[0] = 1.0
    bounds = [(0, None)] * n0 + [(None, None)] * m1
    method = options.pop("method", "highs")
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(a_ub.shape[0]), A_eq=a_eq, b_eq=b_eq,
                  bounds=bounds, method=method, options=options)
    if res.x is None:
        return LPSolution(float("nan"), None, None, rows, columns, nnz, time.perf_counter() - start, res.message)
    p1_plan = np.zeros(form.count[1])
    p1_plan[keep] = -res.ineqlin.marginals
    return LPSolution(-res.fun, res.x[:n0], p1_plan, rows, columns, nnz, time.perf_counter() - start, res.message)


def plan_behavior(form: SequenceForm, layers: list[Layer], p: int, plan: np.ndarray, behavior: list, choose: list) -> None:
    """Write player p's behavior from a realization plan into behavior and choose.

    Negative noise is clipped. An unreached information set plays uniformly.
    """
    from src.solver.cfr_plus import normalize

    plan = np.maximum(np.asarray(plan, dtype=float), 0.0)
    choose[p] = normalize(plan[1 : 1 + HANDS])
    for d, layer in enumerate(layers):
        if layer.actor == p:
            n = len(layer.slot_node[p])
            behavior[d] = normalize(plan[form.offset[p][d] : form.offset[p][d] + n * HANDS].reshape(n, HANDS))
