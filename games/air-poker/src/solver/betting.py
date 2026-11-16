"""The betting street as a DAG per chip configuration, and round settlement.

A betting state is ``(contribution_0, contribution_1, actor, previous check)``;
the engine's legal actions depend on nothing else once the round's ante and the
players' sealed BIOS at round start are fixed. Each reachable configuration
``(round, sealed_0, sealed_1, first bettor)`` gets one DAG, built once from the
same rules as ``legal_actions`` and checked against the engine by test. Nodes
are stored in topological order; an action's target is a node id, or a
negative code naming a showdown terminal ``(c0, c1)`` or a fold terminal
``(c0, c1, folder)``.

``evaluate`` computes the perfect-information value of one DAG from the
payoffs of its terminals, and ``settle`` applies the engine's settlement
arithmetic to one showdown or fold outcome.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from src.engine.game import ANTES, STARTING_BIOS
from src.sensitivity.sampler import NOJ, W0, W0C, W1, W1C

ROUNDS = len(ANTES)
MAX_SEALED = 2 * (STARTING_BIOS - 1) - 2 * ANTES[0]  # 46: every BIOS one side could hold after round 1
CHIPS = MAX_SEALED + 1
ANTE = np.array(ANTES, dtype=np.int64)


class BettingDag:
    """One configuration's betting DAG."""

    def __init__(self, sealed_0: int, sealed_1: int, ante: int, first: int) -> None:
        self.sealed = (sealed_0, sealed_1)
        self.ante = ante
        self.first = first
        nodes: dict[tuple[int, int, int, bool], int] = {}
        showdowns: dict[tuple[int, int], int] = {}
        folds: dict[tuple[int, int, int], int] = {}
        actions: dict[int, list[int]] = {}
        order = []
        stack = [(ante, ante, first, False)]
        nodes[stack[0]] = 0
        order.append(stack[0])
        while stack:
            key = stack.pop()
            c0, c1, actor, checked = key
            node = nodes[key]
            targets = []
            c = (c0, c1)
            i, j = actor, 1 - actor
            sealed = (sealed_0 - (c0 - ante), sealed_1 - (c1 - ante))
            to_call = max(c) - c[i]
            if to_call == 0:
                if checked:
                    targets.append(-1 - showdowns.setdefault((c0, c1), len(showdowns)))
                else:
                    nxt = (c0, c1, j, True)
                    if nxt not in nodes:
                        nodes[nxt] = len(order)
                        order.append(nxt)
                        stack.append(nxt)
                    targets.append(nodes[nxt])
            else:
                targets.append(-1 - CHIPS * CHIPS - folds.setdefault((c0, c1, i), len(folds)))
            if sealed[i] >= to_call:
                if to_call:
                    called = (c0 + to_call, c1) if i == 0 else (c0, c1 + to_call)
                    targets.append(-1 - showdowns.setdefault(called, len(showdowns)))
                max_extra = min(sealed[i] - to_call, sealed[j], (c0 + c1 + to_call) // 2)
                for extra in range(1, max_extra + 1):
                    pay = to_call + extra
                    nxt = (c0 + pay, c1, j, False) if i == 0 else (c0, c1 + pay, j, False)
                    if nxt not in nodes:
                        nodes[nxt] = len(order)
                        order.append(nxt)
                        stack.append(nxt)
                    targets.append(nodes[nxt])
            actions[node] = targets
        # Topological order: contributions grow or the check flag turns on.
        rank = sorted(range(len(order)), key=lambda n: (order[n][0] + order[n][1], order[n][3]))
        position = {old: new for new, old in enumerate(rank)}
        self.keys = [order[old] for old in rank]
        self.actor = np.array([k[2] for k in self.keys], dtype=np.int64)
        starts = [0]
        flat = []
        for old in rank:
            for t in actions[old]:
                flat.append(position[t] if t >= 0 else t)
            starts.append(len(flat))
        self.action_start = np.array(starts, dtype=np.int64)
        self.action_target = np.array(flat, dtype=np.int64)
        self.showdowns = np.array(sorted(showdowns, key=lambda k: showdowns[k]), dtype=np.int64).reshape(-1, 2)
        self.folds = np.array(sorted(folds, key=lambda k: folds[k]), dtype=np.int64).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.keys)


@njit(cache=True)
def evaluate(actor, action_start, action_target, showdown_value, fold_value, values):
    """Value of the DAG root for Team 0; ``values`` is a scratch array of node values."""
    n = actor.shape[0]
    for node in range(n - 1, -1, -1):
        best = -1e300 if actor[node] == 0 else 1e300
        for a in range(action_start[node], action_start[node + 1]):
            target = action_target[a]
            if target >= 0:
                v = values[target]
            elif target > -1 - CHIPS * CHIPS:
                v = showdown_value[-1 - target]
            else:
                v = fold_value[-1 - CHIPS * CHIPS - target]
            if actor[node] == 0:
                best = max(best, v)
            else:
                best = min(best, v)
        values[node] = best
    return values[0]


@njit(cache=True)
def settle(round_index, sealed_0, sealed_1, c0, c1, result):
    """Apply the engine's settlement to one outcome.

    ``result`` is a class for a showdown, or ``-1`` / ``-2`` for a fold by Team 0 /
    Team 1. ``sealed`` are the BIOS at round start after the ante. Returns
    ``(terminal, utility, next_sealed_0, next_sealed_1)``: when ``terminal`` the
    utility for Team 0 is final; otherwise the next round starts with the
    returned sealed BIOS after its ante.
    """
    ante = ANTE[round_index]
    s0 = sealed_0 - (c0 - ante)
    s1 = sealed_1 - (c1 - ante)
    if result == NOJ:
        s0 += c0
        s1 += c1
    else:
        winner = 0 if (result == W0 or result == W0C or result == -2) else 1
        if winner == 0:
            s0 += c0 + c1
        else:
            s1 += c0 + c1
        if result == W0C:
            if s1 < c1:
                return True, 1.0, 0, 0
            s1 -= c1
        elif result == W1C:
            if s0 < c0:
                return True, -1.0, 0, 0
            s0 -= c0
    if round_index == ROUNDS - 1:
        return True, (1.0 if s0 > s1 else -1.0 if s0 < s1 else 0.0), s0, s1
    next_ante = ANTE[round_index + 1]
    can_0 = s0 >= next_ante
    can_1 = s1 >= next_ante
    if not (can_0 and can_1):
        return True, (0.0 if can_0 == can_1 else 1.0 if can_0 else -1.0), s0, s1
    return False, 0.0, s0 - next_ante, s1 - next_ante


def reachable_configurations() -> list[set[tuple[int, int, int]]]:
    """Per round, the reachable ``(sealed_0, sealed_1, first)`` after the ante; round 1 has both firsts."""
    start = STARTING_BIOS - 1 - ANTES[0]
    reach: list[set[tuple[int, int, int]]] = [{(start, start, 0), (start, start, 1)}]
    for r in range(ROUNDS - 1):
        nxt = set()
        for s0, s1, first in reach[r]:
            dag = BettingDag(s0, s1, ANTES[r], first)
            outcomes = [(c0, c1, result) for c0, c1 in dag.showdowns for result in (W0, W0C, W1, W1C, NOJ)]
            outcomes += [(c0, c1, -1 - folder) for c0, c1, folder in dag.folds]
            for c0, c1, result in outcomes:
                terminal, _, t0, t1 = settle(r, s0, s1, c0, c1, result)
                if not terminal:
                    nxt.add((int(t0), int(t1), 1 - first))
        reach.append(nxt)
    return reach
