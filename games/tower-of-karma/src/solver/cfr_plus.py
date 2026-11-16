"""Exact full-traversal CFR+ on the public-tree arrays.

Updates alternate: P0 updates against P1's current strategy, then P1 updates
against P0's new one. Regret matching+ clips cumulative regrets at zero. The
average strategy weights iteration t by t and by the player's own reach.
"""

import numpy as np

from src.solver.public_traversal import Behavior, Choose, reach, values
from src.solver.public_tree import HANDS, Layer


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Normalize positive regrets per row; a row without positive regret is uniform."""
    total = regret.sum(axis=-1, keepdims=True)
    out = np.full(regret.shape, 1.0 / HANDS)
    np.divide(regret, total, out=out, where=total > 0)
    return out


def normalize(weights: np.ndarray) -> np.ndarray:
    return regret_matching(np.maximum(weights, 0.0))


class CFRPlus:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        shapes = [(len(layer.slot_node[layer.actor]), HANDS) for layer in layers]
        self.regret = [np.zeros(s) for s in shapes]
        self.weight = [np.zeros(s) for s in shapes]
        self.choose_regret = [np.zeros(HANDS), np.zeros(HANDS)]
        self.choose_weight = [np.zeros(HANDS), np.zeros(HANDS)]
        self.iterations = 0

    def current(self) -> tuple[Behavior, Choose]:
        behavior = [regret_matching(r) for r in self.regret]
        return behavior, (regret_matching(self.choose_regret[0]), regret_matching(self.choose_regret[1]))

    def average(self) -> tuple[Behavior, Choose]:
        behavior = [normalize(w) for w in self.weight]
        return behavior, (normalize(self.choose_weight[0]), normalize(self.choose_weight[1]))

    def _own_layer(self, d: int, cfv: np.ndarray, sigma: np.ndarray, own: np.ndarray, t: int) -> np.ndarray:
        """Regret-matching+ update on the traverser's layer d; return the slot values."""
        v = np.einsum("ij,ij->i", sigma, cfv)
        r = self.regret[d]
        r += cfv
        r -= v[:, None]
        np.maximum(r, 0.0, out=r)
        self.weight[d] += (t * own)[:, None] * sigma
        return v

    def _own_root(self, i: int, root: np.ndarray, sigma: np.ndarray, t: int) -> None:
        r = self.choose_regret[i]
        r += root - sigma @ root
        np.maximum(r, 0.0, out=r)
        self.choose_weight[i] += t * sigma

    def _update(self, i: int) -> None:
        t = self.iterations + 1
        behavior, choose = self.current()
        opp = reach(self.layers, behavior, choose, 1 - i)
        own = reach(self.layers, behavior, choose, i)
        root = values(self.layers, i, opp, behavior, lambda d, cfv: self._own_layer(d, cfv, behavior[d], own[d], t))
        self._own_root(i, root, choose[i], t)

    def iterate(self) -> None:
        self._update(0)
        self._update(1)
        self.iterations += 1
