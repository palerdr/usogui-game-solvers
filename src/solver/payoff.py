"""Sequence-form payoff matrix: P0's payoff indexed by (P0 sequence, P1 sequence)."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from src.solver.sequences import SequenceIndex
from src.solver.tree import GameTree


def build_payoff(tree: GameTree, index: SequenceIndex) -> csr_matrix:
    """Entry (s0, s1) is P0's return at the leaf reached by that sequence pair.

    Draws pay zero and are omitted, so the matrix holds one nonzero per hit leaf.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for node_id, node in enumerate(tree.nodes):
        if node.returns is None or node.returns[0] == 0:
            continue
        s0, s1 = index.node_sequences[node_id]
        rows.append(s0)
        cols.append(s1)
        vals.append(float(node.returns[0]))
    shape = (len(index.players[0].sequences), len(index.players[1].sequences))
    return coo_matrix((np.array(vals), (rows, cols)), shape=shape).tocsr()
