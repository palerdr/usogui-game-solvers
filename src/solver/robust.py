"""Robust backward induction over public states (GOALS.md step 4).

Public round-start state: round ``r``, number prefix ``P`` (the pairs played),
public class prefix ``C`` (base-6 code, hidden = 5), sealed BIOS after the
ante ``(s0, s1)``. The first bettor is fixed by the round-1 pair and alternates.

    V_r(P, C, s0, s1) = matrix game over the remaining pairs (a, b) of W_r(P + (a, b), C, s0, s1)
    W_r(P', C, s0, s1) = t * N_value + (1 - t) * betting value, t = P(no judgement)

The betting value is the perfect-information value of the betting DAG whose
showdown terminals pay the expectation over the four win classes and whose fold
terminals continue with the hidden class. Nature picks inside the kernel entry's
confidence box: adversarially for the lower bound, favorably for the upper
bound, independently at every chance node (rectangular uncertainty), which
brackets the true value because the true kernel lies in every box.

Rounds are solved depth-first over number prefixes, so live memory is one
value table per level. Values and number strategies of rounds 1 to 3 are kept
for the forward pass; deeper decisions are cheap to recompute on demand.
"""

from __future__ import annotations

import json
import time

import numpy as np
from numba import get_num_threads, njit, prange

from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.kernel.sampler import DIGIT
from src.kernel.table import HIDDEN, KernelCounts, Sequences
from src.sensitivity.sampler import CLASS_NAMES, NOJ, W0, W0C, W1, W1C
from src.solver.betting import (
    ANTE,
    CHIPS,
    ROUNDS,
    BettingDag,
    evaluate,
    reachable_configurations,
    settle,
)
from src.solver.matrix_game import MAX_SIZE, solve

WIN_CLASSES = np.array([W0, W0C, W1, W1C], dtype=np.int64)
ROUND_ONE_FIRST = np.array([[0 if a < b else 1 for b in TEAM_1_METAL_CARDS] for a in TEAM_0_METAL_CARDS], dtype=np.int64)


class Arena:
    """Every reachable betting DAG, flattened for numba."""

    def __init__(self) -> None:
        reach = reachable_configurations()
        self.chips = [[sorted({(s0, s1) for s0, s1, f in reach[r] if f == first} if r else {(23, 23)})
                       for first in range(2)] for r in range(ROUNDS)]
        self.config = -np.ones((ROUNDS, CHIPS, CHIPS, 2), dtype=np.int64)
        node_start, node_count, actor, action_start, action_target = [0], [], [], [0], []
        node_c0, node_c1, node_checked = [], [], []
        show_start, show_c0, show_c1 = [0], [], []
        fold_start, fold_c0, fold_c1, fold_who = [0], [], [], []
        for r in range(ROUNDS):
            for first in range(2):
                for s0, s1 in self.chips[r][first]:
                    dag = BettingDag(s0, s1, int(ANTE[r]), first)
                    self.config[r, s0, s1, first] = len(node_count)
                    node_count.append(len(dag))
                    node_start.append(node_start[-1] + len(dag))
                    actor.extend(dag.actor.tolist())
                    node_c0.extend(k[0] for k in dag.keys)
                    node_c1.extend(k[1] for k in dag.keys)
                    node_checked.extend(int(k[3]) for k in dag.keys)
                    base = len(action_target)
                    action_start.extend((dag.action_start[1:] + base).tolist())
                    action_target.extend(dag.action_target.tolist())
                    show_start.append(show_start[-1] + len(dag.showdowns))
                    show_c0.extend(dag.showdowns[:, 0].tolist())
                    show_c1.extend(dag.showdowns[:, 1].tolist())
                    fold_start.append(fold_start[-1] + len(dag.folds))
                    fold_c0.extend(dag.folds[:, 0].tolist())
                    fold_c1.extend(dag.folds[:, 1].tolist())
                    fold_who.extend(dag.folds[:, 2].tolist())
        as_int = lambda xs: np.array(xs, dtype=np.int64)
        self.node_start, self.actor = as_int(node_start), as_int(actor)
        self.node_c0, self.node_c1, self.node_checked = as_int(node_c0), as_int(node_c1), as_int(node_checked)
        self.action_start, self.action_target = as_int(action_start), as_int(action_target)
        self.show_start, self.show_c0, self.show_c1 = as_int(show_start), as_int(show_c0), as_int(show_c1)
        self.fold_start, self.fold_c0, self.fold_c1, self.fold_who = (as_int(fold_start), as_int(fold_c0),
                                                                      as_int(fold_c1), as_int(fold_who))
        self.max_nodes = max(node_count)
        self.max_terminals = max(max(np.diff(self.show_start)), max(np.diff(self.fold_start)))

    def arrays(self) -> tuple:
        return (self.config, self.node_start, self.actor, self.action_start, self.action_target,
                self.show_start, self.show_c0, self.show_c1, self.fold_start, self.fold_c0, self.fold_c1,
                self.fold_who)

    def policy_arrays(self) -> tuple:
        return (self.config, self.node_start, self.actor, self.action_start, self.action_target,
                self.node_c0, self.node_c1, self.node_checked, self.show_start, self.show_c0, self.show_c1,
                self.fold_start, self.fold_c0, self.fold_c1, self.fold_who, self.max_terminals, self.max_nodes)

    def chip_array(self, r: int, first: int) -> np.ndarray:
        return np.array(self.chips[r][first], dtype=np.int64).reshape(-1, 2)


@njit(cache=True)
def robust_expectation(u, q_lo, q_hi, minimize):
    """Extreme of ``sum q[c] u[c]`` over the box ``[q_lo, q_hi]`` intersected with the simplex.

    Mass starts at the lower ends and the remainder goes to the classes in
    order of payoff, worst first when minimizing, up to their upper ends.
    """
    n = u.shape[0]
    remaining = 1.0
    total = 0.0
    for c in range(n):
        remaining -= q_lo[c]
        total += q_lo[c] * u[c]
    taken = 0
    while remaining > 0.0 and taken < n:
        best = -1
        for c in range(n):
            if q_hi[c] > q_lo[c] and (best < 0 or (u[c] < u[best] if minimize else u[c] > u[best])):
                best = c
        if best < 0:
            break
        extra = min(q_hi[best] - q_lo[best], remaining)
        total += extra * u[best]
        remaining -= extra
        q_hi[best] = q_lo[best]  # consumed; restored by the caller's next fill
        taken += 1
    return total


@njit(cache=True)
def continuation(r, s0, s1, c0, c1, result, code, vnext_lo, vnext_hi):
    """Lower and upper value of one settled outcome: terminal utility or the next round's entry."""
    terminal, utility, t0, t1 = settle(r, s0, s1, c0, c1, result)
    if terminal:
        return utility, utility
    return vnext_lo[code, t0, t1], vnext_hi[code, t0, t1]


@njit(cache=True)
def terminal_payoffs(r, first, s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi,
                     config, show_start, show_c0, show_c1, fold_start, fold_c0, fold_c1, fold_who,
                     show_lo, show_hi, fold_lo, fold_hi, u_lo, u_hi, q_lo, q_hi, q_work):
    """Fill the showdown and fold payoffs of the configuration's DAG; return ``(t_lo, t_hi, n_lo, n_hi)``."""
    cfg = config[r, s0, s1, first]
    t_lo = box_lo[NOJ]
    t_hi = box_hi[NOJ]
    for k in range(4):
        c = WIN_CLASSES[k]
        q_lo[k] = 0.0 if t_lo >= 1.0 else box_lo[c] / (1.0 - t_lo)
        q_hi[k] = 1.0 if t_hi >= 1.0 else min(1.0, box_hi[c] / (1.0 - t_hi))
    for k in range(show_start[cfg], show_start[cfg + 1]):
        c0 = show_c0[k]
        c1 = show_c1[k]
        for j in range(4):
            c = WIN_CLASSES[j]
            u_lo[j], u_hi[j] = continuation(r, s0, s1, c0, c1, c, code * 6 + DIGIT[c], vnext_lo, vnext_hi)
        q_work[:] = q_hi
        show_lo[k - show_start[cfg]] = robust_expectation(u_lo, q_lo, q_work, True)
        q_work[:] = q_hi
        show_hi[k - show_start[cfg]] = robust_expectation(u_hi, q_lo, q_work, False)
    for k in range(fold_start[cfg], fold_start[cfg + 1]):
        lo, hi = continuation(r, s0, s1, fold_c0[k], fold_c1[k], -1 - fold_who[k], code * 6 + HIDDEN,
                              vnext_lo, vnext_hi)
        fold_lo[k - fold_start[cfg]] = lo
        fold_hi[k - fold_start[cfg]] = hi
    ante = ANTE[r]
    n_lo, n_hi = continuation(r, s0, s1, ante, ante, NOJ, code * 6 + DIGIT[NOJ], vnext_lo, vnext_hi)
    return t_lo, t_hi, n_lo, n_hi


@njit(cache=True)
def w_value(r, first, s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi,
            config, node_start, actor, action_start, action_target, show_start, show_c0, show_c1,
            fold_start, fold_c0, fold_c1, fold_who, show_lo, show_hi, fold_lo, fold_hi, scratch, u_lo, u_hi,
            q_lo, q_hi, q_work):
    """``(W_lo, W_hi)`` of one round after the pair is revealed; ``code`` is the class prefix."""
    t_lo, t_hi, n_lo, n_hi = terminal_payoffs(r, first, s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi,
                                              config, show_start, show_c0, show_c1, fold_start, fold_c0,
                                              fold_c1, fold_who, show_lo, show_hi, fold_lo, fold_hi, u_lo,
                                              u_hi, q_lo, q_hi, q_work)
    cfg = config[r, s0, s1, first]
    a0 = node_start[cfg]
    a1 = node_start[cfg + 1]
    b_lo = evaluate(actor[a0:a1], action_start[a0:a1 + 1] - action_start[a0], action_target[action_start[a0]:action_start[a1]],
                    show_lo, fold_lo, scratch)
    b_hi = evaluate(actor[a0:a1], action_start[a0:a1 + 1] - action_start[a0], action_target[action_start[a0]:action_start[a1]],
                    show_hi, fold_hi, scratch)
    w_lo = min(t_lo * n_lo + (1.0 - t_lo) * b_lo, t_hi * n_lo + (1.0 - t_hi) * b_lo)
    w_hi = max(t_lo * n_hi + (1.0 - t_lo) * b_hi, t_hi * n_hi + (1.0 - t_hi) * b_hi)
    return w_lo, w_hi


@njit(cache=True)
def betting_policy(r, first, s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi, use_hi, c0, c1, checked,
                   config, node_start, actor, action_start, action_target, node_c0, node_c1, node_checked,
                   show_start, show_c0, show_c1, fold_start, fold_c0, fold_c1, fold_who, max_terminals, max_nodes):
    """Index, in DAG action order, of the best action at the betting node ``(c0, c1, checked)``.

    ``use_hi`` selects the upper-bound payoffs (Team 1's robust game) instead of the lower.
    """
    show_lo = np.empty(max_terminals)
    show_hi = np.empty(max_terminals)
    fold_lo = np.empty(max_terminals)
    fold_hi = np.empty(max_terminals)
    values = np.empty(max_nodes)
    u_lo = np.empty(4)
    u_hi = np.empty(4)
    q_lo = np.empty(4)
    q_hi = np.empty(4)
    q_work = np.empty(4)
    terminal_payoffs(r, first, s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi, config, show_start, show_c0,
                     show_c1, fold_start, fold_c0, fold_c1, fold_who, show_lo, show_hi, fold_lo, fold_hi,
                     u_lo, u_hi, q_lo, q_hi, q_work)
    show = show_hi if use_hi else show_lo
    fold = fold_hi if use_hi else fold_lo
    cfg = config[r, s0, s1, first]
    a0 = node_start[cfg]
    a1 = node_start[cfg + 1]
    evaluate(actor[a0:a1], action_start[a0:a1 + 1] - action_start[a0], action_target[action_start[a0]:action_start[a1]],
             show, fold, values)
    node = -1
    for n in range(a0, a1):
        if node_c0[n] == c0 and node_c1[n] == c1 and node_checked[n] == checked:
            node = n - a0
            break
    best = -1
    best_value = 0.0
    for slot, a in enumerate(range(action_start[a0 + node], action_start[a0 + node + 1])):
        target = action_target[a]
        if target >= 0:
            v = values[target]
        elif target > -1 - CHIPS * CHIPS:
            v = show[-1 - target]
        else:
            v = fold[-1 - CHIPS * CHIPS - target]
        if best < 0 or (v > best_value if actor[a0 + node] == 0 else v < best_value):
            best = slot
            best_value = v
    return best


@njit(parallel=True, cache=True)
def w_table(r, first, box_lo, box_hi, vnext_lo, vnext_hi, chips, max_nodes, max_terminals,
            config, node_start, actor, action_start, action_target, show_start, show_c0, show_c1,
            fold_start, fold_c0, fold_c1, fold_who, out_lo, out_hi):
    """Fill ``out[code, s0, s1]`` for every class prefix code and reachable chip pair."""
    for code in prange(box_lo.shape[0]):
        show_lo = np.empty(max_terminals)
        show_hi = np.empty(max_terminals)
        fold_lo = np.empty(max_terminals)
        fold_hi = np.empty(max_terminals)
        scratch = np.empty(max_nodes)
        u_lo = np.empty(4)
        u_hi = np.empty(4)
        q_lo = np.empty(4)
        q_hi = np.empty(4)
        q_work = np.empty(4)
        for i in range(chips.shape[0]):
            s0 = chips[i, 0]
            s1 = chips[i, 1]
            lo, hi = w_value(r, first, s0, s1, box_lo[code], box_hi[code], code, vnext_lo, vnext_hi,
                             config, node_start, actor, action_start, action_target, show_start, show_c0,
                             show_c1, fold_start, fold_c0, fold_c1, fold_who, show_lo, show_hi, fold_lo,
                             fold_hi, scratch, u_lo, u_hi, q_lo, q_hi, q_work)
            out_lo[code, s0, s1] = lo
            out_hi[code, s0, s1] = hi


@njit(parallel=True, cache=True)
def v_table(k, w_lo, w_hi, chips, out_lo, out_hi, x_lo, y_lo, x_hi, y_hi, gaps):
    """Solve the ``k x k`` number game at every (code, chip pair); ``w[pair, code, s0, s1]`` with pair = i * k + j."""
    for code in prange(out_lo.shape[0]):
        m = np.zeros((MAX_SIZE, MAX_SIZE))
        worst = 0.0
        for n in range(chips.shape[0]):
            s0 = chips[n, 0]
            s1 = chips[n, 1]
            if k == 1:
                out_lo[code, s0, s1] = w_lo[0, code, s0, s1]
                out_hi[code, s0, s1] = w_hi[0, code, s0, s1]
                x_lo[code, n, 0] = y_lo[code, n, 0] = x_hi[code, n, 0] = y_hi[code, n, 0] = 1.0
                continue
            for i in range(k):
                for j in range(k):
                    m[i, j] = w_lo[i * k + j, code, s0, s1]
            x, y, low, high = solve(m, k, k)
            out_lo[code, s0, s1] = low
            worst = max(worst, high - low)
            x_lo[code, n, :] = x
            y_lo[code, n, :] = y
            for i in range(k):
                for j in range(k):
                    m[i, j] = w_hi[i * k + j, code, s0, s1]
            x, y, low, high = solve(m, k, k)
            out_hi[code, s0, s1] = high
            worst = max(worst, high - low)
            x_hi[code, n, :] = x
            y_hi[code, n, :] = y
        gaps[code] = worst


class RobustSolver:
    """Depth-first robust backward induction with the kernel boxes of a ``KernelCounts`` table."""

    def __init__(self, sequences: Sequences, arena: Arena, round_one: str = "docs/results/kernel-round1.json") -> None:
        self.sequences = sequences
        self.arena = arena
        with open(round_one) as handle:
            entries = json.load(handle)["entries"]
        exact = np.zeros((25, 1, 6))
        for e in entries:
            exact[sequences.prefix_index[0][((e["numbers"][0], e["numbers"][1]),)], 0] = [
                e["probabilities_float"][name] for name in CLASS_NAMES]
        self.box_lo = [exact]
        self.box_hi = [exact.copy()]
        self.pairs = [[(a, b) for a in TEAM_0_METAL_CARDS for b in TEAM_1_METAL_CARDS]]

    def set_boxes(self, table: KernelCounts, delta: float) -> None:
        """Kernel boxes for rounds 2 to 5 from the sampled counts."""
        self.box_lo = self.box_lo[:1]
        self.box_hi = self.box_hi[:1]
        for r in range(1, ROUNDS):
            lo, hi = table.boxes(r, delta)
            self.box_lo.append(lo)
            self.box_hi.append(hi)

    def first(self, prefix: tuple, r: int) -> int:
        return int(ROUND_ONE_FIRST[TEAM_0_METAL_CARDS.index(prefix[0][0]), TEAM_1_METAL_CARDS.index(prefix[0][1])]) ^ (r & 1)

    def remaining_pairs(self, prefix: tuple) -> tuple[list[int], list[int]]:
        used_0 = {a for a, _ in prefix}
        used_1 = {b for _, b in prefix}
        return [a for a in TEAM_0_METAL_CARDS if a not in used_0], [b for b in TEAM_1_METAL_CARDS if b not in used_1]

    def solve(self, keep_depth: int = 2) -> dict:
        """Return the root interval, root strategies and statistics; keep tables up to ``keep_depth``."""
        self.kept_values: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.kept_strategies: dict[tuple, tuple] = {}
        self.max_gap = 0.0
        self.keep_depth = keep_depth
        start = time.perf_counter()
        lo, hi, strategies = self._v((), 0)
        return {"root_lo": float(lo[0, 23, 23]), "root_hi": float(hi[0, 23, 23]), "max_saddle_gap": self.max_gap,
                "strategies": strategies, "seconds": time.perf_counter() - start}

    def _w(self, prefix: tuple, r: int) -> tuple[np.ndarray, np.ndarray]:
        """``W_r`` tables ``[6**r, CHIPS, CHIPS]`` for the number prefix of length ``r + 1``."""
        first = self.first(prefix, r)
        if r + 1 < ROUNDS:
            vnext_lo, vnext_hi, _ = self._v(prefix, r + 1)
        else:
            vnext_lo = vnext_hi = np.zeros((1, 1, 1))
        index = self.sequences.prefix_index[r][prefix]
        out_lo = np.full((6 ** r, CHIPS, CHIPS), np.nan)
        out_hi = np.full((6 ** r, CHIPS, CHIPS), np.nan)
        w_table(r, first, self.box_lo[r][index], self.box_hi[r][index], vnext_lo, vnext_hi,
                self.arena.chip_array(r, first), self.arena.max_nodes, self.arena.max_terminals,
                *self.arena.arrays(), out_lo, out_hi)
        return out_lo, out_hi

    def _v(self, prefix: tuple, r: int):
        """``V_r`` tables ``[6**r, CHIPS, CHIPS]`` for the number prefix of length ``r``, plus strategies."""
        rows, cols = self.remaining_pairs(prefix)
        k = len(rows)
        w_lo = np.empty((k * k, 6 ** r, CHIPS, CHIPS))
        w_hi = np.empty((k * k, 6 ** r, CHIPS, CHIPS))
        for i, a in enumerate(rows):
            for j, b in enumerate(cols):
                w_lo[i * k + j], w_hi[i * k + j] = self._w(prefix + ((a, b),), r)
        first = self.first(prefix, r) if prefix else 0
        chips = self.arena.chip_array(r, first) if prefix else np.array([[23, 23]], dtype=np.int64)
        out_lo = np.full((6 ** r, CHIPS, CHIPS), np.nan)
        out_hi = np.full((6 ** r, CHIPS, CHIPS), np.nan)
        x_lo = np.zeros((6 ** r, chips.shape[0], MAX_SIZE))
        y_lo = np.zeros_like(x_lo)
        x_hi = np.zeros_like(x_lo)
        y_hi = np.zeros_like(x_lo)
        gaps = np.zeros(6 ** r)
        v_table(k, w_lo, w_hi, chips, out_lo, out_hi, x_lo, y_lo, x_hi, y_hi, gaps)
        self.max_gap = max(self.max_gap, float(gaps.max()))
        strategies = (rows, cols, chips, x_lo[..., :k].copy(), y_lo[..., :k].copy(), x_hi[..., :k].copy(),
                      y_hi[..., :k].copy())
        if r <= self.keep_depth:
            self.kept_values[prefix] = (out_lo, out_hi)
            self.kept_strategies[prefix] = strategies
        return out_lo, out_hi, strategies


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--prefix", type=int, default=0, help="solve one round-5 subtree by sequence index")
    args = parser.parse_args()
    sequences = Sequences()
    arena = Arena()
    solver = RobustSolver(sequences, arena)
    solver.set_boxes(KernelCounts(sequences), args.delta)
    start = time.perf_counter()
    prefix = tuple((int(a), int(b)) for a, b in sequences.targets[args.prefix][:5])
    lo, _ = solver._w(prefix, 4)
    reached = np.isfinite(lo)
    print(f"W_5 for {prefix}: {reached.sum()} values in {time.perf_counter() - start:.2f}s on {get_num_threads()} threads, "
          f"lo in [{lo[reached].min():.3f}, {lo[reached].max():.3f}]")


if __name__ == "__main__":
    main()
