"""Array form of the D_K game tree, organized by public transcript.

A public node is an ordered transcript of misses. It stores the bead pairs
(x0, x1) that are still alive there: every announcement so far was legal for
its actor and missed. Layer d holds the decision nodes of turn d // 2 + 1;
P0 acts on even layers and P1 on odd layers.

A slot is a (public node, own beads) pair for one player. The actor's slots
are its information sets. Each full-tree decision node is one alive pair.
"""

from dataclasses import dataclass

import numpy as np

HANDS = 10
TOTALS = 19  # announcements 2..20


@dataclass(frozen=True, slots=True)
class Layer:
    actor: int
    turn: int
    parent: np.ndarray  # public node -> parent public node (-1 at the root)
    total: np.ndarray  # public node -> announcement that led to it (0 at the root)
    pair_node: np.ndarray  # alive pair -> public node
    pair_hand: tuple[np.ndarray, np.ndarray]  # alive pair -> (x0, x1)
    pair_slot: tuple[np.ndarray, np.ndarray]  # alive pair -> slot of each player
    slot_node: tuple[np.ndarray, np.ndarray]  # slot -> public node
    slot_hand: tuple[np.ndarray, np.ndarray]  # slot -> own beads
    slot_parent: tuple[np.ndarray, np.ndarray]  # slot -> same player's slot one layer up
    child_slot: np.ndarray  # actor slot x action -> actor slot one layer down, or -1

    @property
    def nodes(self) -> int:
        return len(self.parent)

    @property
    def pairs(self) -> int:
        return len(self.pair_node)


def _slots(pair_node: np.ndarray, hand: np.ndarray):
    key = pair_node.astype(np.int64) * HANDS + (hand - 1)
    uniq, inverse = np.unique(key, return_inverse=True)
    return (uniq // HANDS).astype(np.int32), (uniq % HANDS + 1).astype(np.int8), inverse.astype(np.int32), uniq


def _layer(actor, turn, parent, total, pair_node, x0, x1, prev_keys, prev_parent_of_node):
    hands = (x0, x1)
    slot_node, slot_hand, pair_slot, keys = [], [], [], []
    for q in (0, 1):
        n, h, inv, k = _slots(pair_node, hands[q])
        slot_node.append(n), slot_hand.append(h), pair_slot.append(inv), keys.append(k)
    slot_parent = []
    for q in (0, 1):
        if prev_keys is None:
            slot_parent.append(np.full(len(slot_node[q]), -1, dtype=np.int32))
        else:
            up = prev_parent_of_node[slot_node[q]].astype(np.int64) * HANDS + (slot_hand[q] - 1)
            slot_parent.append(np.searchsorted(prev_keys[q], up).astype(np.int32))
    layer = Layer(
        actor=actor,
        turn=turn,
        parent=parent,
        total=total,
        pair_node=pair_node,
        pair_hand=(x0, x1),
        pair_slot=(pair_slot[0], pair_slot[1]),
        slot_node=(slot_node[0], slot_node[1]),
        slot_hand=(slot_hand[0], slot_hand[1]),
        slot_parent=(slot_parent[0], slot_parent[1]),
        child_slot=np.full((len(slot_node[actor]), HANDS), -1, dtype=np.int32),
    )
    return layer, keys


def build_public_tree(horizon: int) -> list[Layer]:
    """Return the 2 * horizon decision layers of D_horizon."""
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    grid = np.arange(1, HANDS + 1, dtype=np.int8)
    x0 = np.repeat(grid, HANDS)
    x1 = np.tile(grid, HANDS)
    layer, keys = _layer(
        0, 1, np.array([-1], np.int32), np.array([0], np.int8),
        np.zeros(HANDS * HANDS, np.int32), x0, x1, None, None,
    )
    layers = [layer]
    for d in range(1, 2 * horizon):
        prev = layers[-1]
        p = prev.actor
        own = prev.pair_hand[p].astype(np.int16)
        g = own[:, None] + np.arange(1, HANDS + 1, dtype=np.int16)[None, :]  # pair x action
        hit = (prev.pair_hand[0] + prev.pair_hand[1]).astype(np.int16)
        miss = g != hit[:, None]
        rows, acts = np.nonzero(miss)
        g_child = g[rows, acts]
        node_key = prev.pair_node[rows].astype(np.int64) * TOTALS + (g_child - 2)
        uniq, child_node = np.unique(node_key, return_inverse=True)
        parent = (uniq // TOTALS).astype(np.int32)
        total = (uniq % TOTALS + 2).astype(np.int8)
        layer, new_keys = _layer(
            1 - p, d // 2 + 1, parent, total, child_node.astype(np.int32),
            prev.pair_hand[0][rows], prev.pair_hand[1][rows], keys, parent,
        )
        # Link each actor slot one layer up to the actor's slot after each action.
        k = layer.slot_parent[p]
        a = layer.total[layer.slot_node[p]].astype(np.int16) - layer.slot_hand[p] - 1
        prev.child_slot[k, a] = np.arange(len(k), dtype=np.int32)
        layers.append(layer)
        keys = new_keys
    return layers


@dataclass(frozen=True, slots=True)
class Sizes:
    nodes: int
    hit_leaves: int
    draw_leaves: int
    public_nodes: int
    information_sets: tuple[int, int]
    sequences: tuple[int, int]


def sizes(layers: list[Layer]) -> Sizes:
    """Count the full extensive-form tree that the layers represent."""
    decisions = sum(layer.pairs for layer in layers)
    draws = layers[-1].pairs * (HANDS - 1)
    infosets = [1, 1]  # the bead choice of each player
    for layer in layers:
        infosets[layer.actor] += len(layer.slot_node[layer.actor])
    sequences = tuple(1 + HANDS + HANDS * (n - 1) for n in infosets)
    return Sizes(
        nodes=1 + HANDS + 2 * decisions + draws,
        hit_leaves=decisions,
        draw_leaves=draws,
        public_nodes=sum(layer.nodes for layer in layers),
        information_sets=(infosets[0], infosets[1]),
        sequences=sequences,
    )
