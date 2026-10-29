"""Sequence-form flow constraints: E x = e for one player's realization plan."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from src.solver.sequences import PlayerSequences


def build_flow(player: PlayerSequences) -> tuple[csr_matrix, np.ndarray]:
    """Row 0 fixes the empty sequence to 1; row 1+i balances information set i.

    Balance at information set I: sum of the action sequences of I equals the
    realization weight of the sequence that leads to I.
    """
    rows = [0]
    cols = [0]
    vals = [1.0]
    for info_id, info in enumerate(player.information_sets):
        rows.append(1 + info_id)
        cols.append(info.parent_sequence_id)
        vals.append(-1.0)
        for seq_id in info.action_sequence_ids:
            rows.append(1 + info_id)
            cols.append(seq_id)
            vals.append(1.0)
    shape = (1 + len(player.information_sets), len(player.sequences))
    matrix = coo_matrix((np.array(vals), (rows, cols)), shape=shape).tocsr()
    rhs = np.zeros(shape[0])
    rhs[0] = 1.0
    return matrix, rhs
