"""Best responses that walk the frozen rules engine.

Every transition comes from primitives.apply, every action set from
legal_actions, and every payoff from terminal_returns. The opponent's behavior
is read from a StrategyTable by information state. Nothing here reads the
public-tree arrays, so the certificate is independent of the solver.

Histories are grouped by public transcript. A group's value for the
responding player is a vector over its own beads: the sum, over alive
opponent hands, of opponent reach times payoff. The responder's information
set is (own beads, transcript), so a max per bead count is a best response.
Arithmetic is float64.
"""

from collections import defaultdict
from multiprocessing import get_context
from multiprocessing.pool import AsyncResult

import numpy as np

from src.solver.primitives import (
    GameState,
    Phase,
    apply,
    current_actor,
    information_state,
    initial_state,
    legal_actions,
    terminal_returns,
)
from src.solver.strategy_table import StrategyTable, info_key

type Group = list[tuple[GameState, float]]

_TABLE: StrategyTable | None = None
_ROWS: dict[int, int] = {}


def _load(path: str) -> None:
    global _TABLE, _ROWS
    _TABLE = StrategyTable.load(path)
    _ROWS = _TABLE.rows()


def _behavior(state: GameState, actor: int) -> np.ndarray:
    info = information_state(state, actor)
    key = info_key(info.own_beads, tuple(m.total for m in info.public_history))
    return _TABLE.probs[_ROWS[key]]


def _step(group: Group, player: int):
    actor = int(current_actor(group[0][0]))
    immediate: dict[tuple[int, int], float] = defaultdict(float)
    children: dict[int, Group] = defaultdict(list)
    legal: dict[int, list[int]] = {}
    for state, r in group:
        if r == 0.0:
            continue
        own = state.beads[player]
        actions = legal_actions(state)
        if actor == player:
            legal.setdefault(own, [a.total for a in actions])
            probs = None
        else:
            probs = _behavior(state, actor)
        for action in actions:
            w = r if probs is None else r * float(probs[action.total - state.beads[actor] - 1])
            if w == 0.0:
                continue
            child = apply(state, action)
            if child.phase is Phase.TERMINAL:
                immediate[(own, action.total)] += w * terminal_returns(child)[player]
            else:
                children[action.total].append((child, w))
    return actor, immediate, children, legal


def _combine(actor, player, immediate, legal, child_values) -> dict[int, float]:
    if actor == player:
        return {
            x: max(immediate.get((x, g), 0.0) + child_values.get(g, {}).get(x, 0.0) for g in totals)
            for x, totals in legal.items()
        }
    out: dict[int, float] = defaultdict(float)
    for (x, _), v in immediate.items():
        out[x] += v
    for values in child_values.values():
        for x, v in values.items():
            out[x] += v
    return out


def _value(group: Group, player: int) -> dict[int, float]:
    actor, immediate, children, legal = _step(group, player)
    child_values = {g: _value(c, player) for g, c in children.items()}
    return dict(_combine(actor, player, immediate, legal, child_values))


def _plan(group: Group, player: int, levels: int, pool):
    if levels == 0 or pool is None:
        return pool.apply_async(_value, (group, player)) if pool is not None else _value(group, player)
    actor, immediate, children, legal = _step(group, player)
    kids = {g: _plan(c, player, levels - 1, pool) for g, c in children.items()}
    return actor, immediate, legal, kids


def _resolve(node, player: int) -> dict[int, float]:
    if isinstance(node, AsyncResult):
        return node.get()
    if isinstance(node, dict):
        return node
    actor, immediate, legal, kids = node
    return _combine(actor, player, immediate, legal, {g: _resolve(k, player) for g, k in kids.items()})


def best_response_value(horizon: int, player: int, table_path: str, workers: int = 0, levels: int = 2) -> float:
    """Value to `player` of its best response to the opponent table at table_path."""
    _load(table_path)
    root = initial_state(horizon)
    group: Group = []
    for c0 in legal_actions(root):
        after = apply(root, c0)
        for c1 in legal_actions(after):
            opp_choice = (c1 if player == 0 else c0).count
            group.append((apply(after, c1), float(_TABLE.choose[opp_choice - 1])))
    if workers:
        with get_context("spawn").Pool(workers, initializer=_load, initargs=(table_path,)) as pool:
            values = _resolve(_plan(group, player, levels, pool), player)
    else:
        values = _value(group, player)
    beads = [c.count for c in legal_actions(root)]
    return max(values.get(x, 0.0) for x in beads)


def certify(horizon: int, p0_path: str, p1_path: str, workers: int = 0) -> tuple[float, float]:
    """Return (P0's guarantee, P0's cap) for the profile stored at the two paths."""
    lower = -best_response_value(horizon, 1, p0_path, workers)
    upper = best_response_value(horizon, 0, p1_path, workers)
    return lower, upper
