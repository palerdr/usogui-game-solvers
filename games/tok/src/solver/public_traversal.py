"""Forward reach and backward counterfactual values on the public-tree arrays.

A behavior is one array per layer, shaped (actor slots, 10), plus the bead
choice of each player. Column a announces own beads + a + 1. No chance node
exists, so the reach of an alive pair is r0(x0) * r1(x1).
"""

from collections.abc import Callable

import numpy as np

from src.solver.public_tree import HANDS, Layer

type Behavior = list[np.ndarray]
type Choose = tuple[np.ndarray, np.ndarray]


def step_index(layers: list[Layer], d: int, q: int) -> np.ndarray:
    """Flat (slot, action) index on layer d - 1 that leads to each q-slot on layer d.

    Valid only when q acts on layer d - 1.
    """
    layer = layers[d]
    action = layer.total[layer.slot_node[q]].astype(np.int64) - layer.slot_hand[q] - 1
    return HANDS * layer.slot_parent[q].astype(np.int64) + action


def hit_index(layer: Layer) -> np.ndarray:
    """Flat (actor slot, action) index of the announcement that hits each alive pair."""
    p, q = layer.actor, 1 - layer.actor
    return HANDS * layer.pair_slot[p].astype(np.int64) + layer.pair_hand[q] - 1


def reach(layers: list[Layer], behavior: Behavior, choose: Choose, q: int) -> list[np.ndarray]:
    """Probability of player q's own actions to each of q's slots, layer by layer."""
    out = [choose[q][layers[0].slot_hand[q] - 1]]
    for d in range(1, len(layers)):
        up = out[-1][layers[d].slot_parent[q]]
        if layers[d - 1].actor == q:
            up = up * behavior[d - 1].ravel()[step_index(layers, d, q)]
        out.append(up)
    return out


def values(
    layers: list[Layer],
    i: int,
    opp_reach: list[np.ndarray],
    behavior: Behavior,
    own_layer: Callable[[int, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Player i's counterfactual value at each of its root slots (indexed by beads - 1).

    The value of slot (h, x) sums the opponent's reach times i's payoff over the
    opponent hands alive at h. On i's layers, own_layer(d, cfv) receives the
    action values (slots, 10) and returns the slot values; it may update
    regrets. The opponent plays `behavior` on its layers.
    """
    below = None
    for d in reversed(range(len(layers))):
        layer = layers[d]
        p = layer.actor
        n = len(layer.slot_node[i])
        if p == i:
            cfv = np.zeros(n * HANDS)
            cfv[hit_index(layer)] = opp_reach[d][layer.pair_slot[1 - i]]
            cfv = cfv.reshape(n, HANDS)
            if below is not None:
                child = layer.child_slot
                live = child >= 0
                cfv[live] += below[child[live]]
            below = own_layer(d, cfv)
        else:
            v = np.zeros(n) if below is None else np.bincount(layers[d + 1].slot_parent[i], weights=below, minlength=n)
            lost = opp_reach[d][layer.pair_slot[p]] * behavior[d].ravel()[hit_index(layer)]
            below = v - np.bincount(layer.pair_slot[i], weights=lost, minlength=n)
    return below
