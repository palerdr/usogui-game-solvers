"""Coarse keys for a player's last decision layer on the public-tree arrays.

This is the array form of abstraction.py, with CFR+ in place of the LPs.
Player i's last layer plays one shared behavior per key.

- merged_best_response fixes the opponent and the earlier layers and picks
  the best action per key. Its value bounds the merge step from above.
- MergedCFRPlus solves the merge step: earlier layers fixed, last layer
  merged, opponent free.
- FixedLastCFRPlus solves the earlier-layers step: last layer fixed, all
  other decisions free.
- alternate runs the two steps in rounds. The engine certificate then proves
  the guarantee of the best lifted strategy.
"""

import numpy as np

from src.solver.array_best_response import best_response_value
from src.solver.cfr_plus import CFRPlus, normalize, regret_matching
from src.solver.public_traversal import Behavior, Choose, reach, values
from src.solver.public_tree import HANDS, Layer


def last_layer(layers: list[Layer], i: int) -> int:
    return max(d for d, layer in enumerate(layers) if layer.actor == i)


def transcripts(layers: list[Layer]) -> list[np.ndarray]:
    """Announced totals of each public node, shaped (nodes, depth)."""
    out = [np.zeros((1, 0), dtype=np.int8)]
    for layer in layers[1:]:
        out.append(np.concatenate([out[-1][layer.parent], layer.total[:, None]], axis=1))
    return out


def _bits(totals: np.ndarray) -> np.ndarray:
    """Set of announced totals as a 19-bit mask per row."""
    if totals.shape[1] == 0:
        return np.zeros(len(totals), dtype=np.int64)
    return np.bitwise_or.reduce(np.int64(1) << (totals.astype(np.int64) - 2), axis=1)


def slot_keys(layers: list[Layer], i: int, name: str) -> tuple[np.ndarray, int]:
    """Dense key id of each slot on player i's last layer, and the number of keys.

    full: the information set itself. support: own beads and the opponent hands
    still alive. hard: own beads, set of own totals, set of opponent totals.
    opening+hard: hard plus P0's first announced total.
    """
    d = last_layer(layers, i)
    layer = layers[d]
    hand = layer.slot_hand[i].astype(np.int64)
    if name == "full":
        return np.arange(len(hand)), len(hand)
    if name == "support":
        n = len(hand)
        bit = np.int64(1) << (layer.pair_hand[1 - i].astype(np.int64) - 1)
        mask = np.bincount(layer.pair_slot[i], weights=bit, minlength=n).astype(np.int64)
        raw = hand + HANDS * mask
    else:
        totals = transcripts(layers)[d][layer.slot_node[i]]
        own = _bits(totals[:, i::2])
        opp = _bits(totals[:, 1 - i :: 2])
        raw = hand + HANDS * (own + (opp << 19))
        if name == "opening+hard":
            raw = raw + (totals[:, 0].astype(np.int64) << 45)
        elif name != "hard":
            raise ValueError(f"unknown key {name!r}")
    uniq, key = np.unique(raw, return_inverse=True)
    return key, len(uniq)


def _key_sums(key: np.ndarray, n_keys: int, weighted: np.ndarray) -> np.ndarray:
    flat = (key[:, None] * HANDS + np.arange(HANDS)).ravel()
    return np.bincount(flat, weights=weighted.ravel(), minlength=n_keys * HANDS).reshape(n_keys, HANDS)


def merged_best_response(
    layers: list[Layer], i: int, behavior: Behavior, choose: Choose, key: np.ndarray, n_keys: int
) -> tuple[float, np.ndarray]:
    """Player i's best value with earlier layers fixed and one action per key.

    Returns i's expected payoff against the opponent part of behavior, and
    the chosen behavior per key.
    """
    last = last_layer(layers, i)
    opp = reach(layers, behavior, choose, 1 - i)
    own = reach(layers, behavior, choose, i)
    best = np.zeros((n_keys, HANDS))

    def own_layer(d: int, cfv: np.ndarray) -> np.ndarray:
        if d != last:
            return np.einsum("ij,ij->i", behavior[d], cfv)
        totals = _key_sums(key, n_keys, own[d][:, None] * cfv)
        best[np.arange(n_keys), totals.argmax(axis=1)] = 1.0
        return np.einsum("ij,ij->i", best[key], cfv)

    root = values(layers, i, opp, behavior, own_layer)
    return float(choose[i] @ root), best


def lift(behavior: Behavior, layers: list[Layer], i: int, key: np.ndarray, merged: np.ndarray) -> Behavior:
    """Copy of behavior with player i's last layer played by key."""
    out = list(behavior)
    out[last_layer(layers, i)] = merged[key]
    return out


class MergedCFRPlus(CFRPlus):
    """CFR+ with player i fixed before its last layer and merged on it.

    The opponent is free. Player i then has one decision per play, so the
    restricted game has perfect recall and CFR+ solves it.
    """

    def __init__(self, layers: list[Layer], i: int, plan: Behavior, plan_choose: Choose, key: np.ndarray, n_keys: int):
        super().__init__(layers)
        self.i = i
        self.plan = plan
        self.plan_choose = plan_choose
        self.key = key
        self.last = last_layer(layers, i)
        self.key_regret = np.zeros((n_keys, HANDS))
        self.key_weight = np.zeros((n_keys, HANDS))

    def _assemble(self, merged: np.ndarray, base: tuple[Behavior, Choose]) -> tuple[Behavior, Choose]:
        behavior, choose = base
        behavior = [
            (self.plan[d] if d != self.last else merged[self.key]) if layer.actor == self.i else behavior[d]
            for d, layer in enumerate(self.layers)
        ]
        choose = tuple(self.plan_choose[q] if q == self.i else choose[q] for q in (0, 1))
        return behavior, choose

    def current(self) -> tuple[Behavior, Choose]:
        return self._assemble(regret_matching(self.key_regret), super().current())

    def average(self) -> tuple[Behavior, Choose]:
        return self._assemble(self.merged_average(), super().average())

    def merged_average(self) -> np.ndarray:
        return normalize(self.key_weight)

    def _own_layer(self, d, cfv, sigma, own, t):
        if self.layers[d].actor != self.i:
            return super()._own_layer(d, cfv, sigma, own, t)
        if d == self.last:
            totals = _key_sums(self.key, len(self.key_regret), own[:, None] * cfv)
            merged = regret_matching(self.key_regret)
            v = np.einsum("ij,ij->i", merged, totals)
            self.key_regret += totals - v[:, None]
            np.maximum(self.key_regret, 0.0, out=self.key_regret)
            self.key_weight += t * merged
        return np.einsum("ij,ij->i", sigma, cfv)

    def _own_root(self, i, root, sigma, t):
        if i != self.i:
            super()._own_root(i, root, sigma, t)


class FixedLastCFRPlus(CFRPlus):
    """CFR+ with player i's last layer fixed; all other decisions are free."""

    def __init__(self, layers: list[Layer], i: int, last_behavior: np.ndarray):
        super().__init__(layers)
        self.last = last_layer(layers, i)
        self.fixed = last_behavior

    def _fix(self, profile: tuple[Behavior, Choose]) -> tuple[Behavior, Choose]:
        behavior, choose = profile
        behavior[self.last] = self.fixed
        return behavior, choose

    def current(self) -> tuple[Behavior, Choose]:
        return self._fix(super().current())

    def average(self) -> tuple[Behavior, Choose]:
        return self._fix(super().average())

    def _own_layer(self, d, cfv, sigma, own, t):
        if d == self.last:
            return np.einsum("ij,ij->i", sigma, cfv)
        return super()._own_layer(d, cfv, sigma, own, t)


def guarantee(layers: list[Layer], i: int, behavior: Behavior, choose: Choose) -> float:
    """Player i's worst-case payoff with i playing its part of behavior."""
    return -best_response_value(layers, 1 - i, behavior, choose)


def alternate(layers, i, behavior, choose, key, n_keys, rounds=3, iterations=1000, log=print):
    """Alternate the merge step and the earlier-layers step, as abstraction.py does.

    Returns the steps as (name, guarantee, cap) in player i's payoffs and the
    profile whose lifted strategy for i has the best guarantee. cap bounds the
    merge step's restricted value from above; the earlier-layers step has none.
    """
    plan, plan_choose = behavior, choose
    steps, best = [], (-np.inf, None)
    for r in range(rounds):
        merge = MergedCFRPlus(layers, i, plan, plan_choose, key, n_keys)
        for _ in range(iterations):
            merge.iterate()
        b, c = merge.average()
        g = guarantee(layers, i, b, c)
        cap, _ = merged_best_response(layers, i, b, c, key, n_keys)
        steps.append((f"merge{r + 1}", g, cap))
        log(steps[-1])
        if g > best[0]:
            best = (g, (b, c))
        early = FixedLastCFRPlus(layers, i, merge.merged_average()[key])
        for _ in range(iterations):
            early.iterate()
        plan, plan_choose = early.average()
        g = guarantee(layers, i, plan, plan_choose)
        steps.append((f"early{r + 1}", g, float("nan")))
        log(steps[-1])
        if g > best[0]:
            best = (g, (plan, plan_choose))
    return steps, best[1]
