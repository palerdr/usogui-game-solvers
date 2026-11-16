"""Test whether a player loses value when its last decisions are keyed coarsely.

The player's last layer of information sets is merged by `key`. Earlier layers
are fixed to a plan and the merged behavior is optimized by LP; then the merged
behavior is fixed and the earlier layers are re-optimized. A guarantee equal to
the game value proves the coarser key loses nothing for that player. A worse
guarantee is evidence of a loss, not a proof.
"""

from collections.abc import Callable, Hashable

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack

from src.solver.constraints import build_flow
from src.solver.payoff import build_payoff
from src.solver.primitives import InformationState, Player
from src.solver.sequences import PlayerSequences, SequenceIndex
from src.solver.tree import GameTree

type Key = Callable[[InformationState], Hashable]


def last_layer(player: PlayerSequences) -> list[int]:
    """Information sets none of whose action sequences lead to a later one."""
    parents = {info.parent_sequence_id for info in player.information_sets}
    return [
        i
        for i, info in enumerate(player.information_sets)
        if not any(s in parents for s in info.action_sequence_ids)
    ]


def _lp(c, a_ub, b_ub, a_eq, b_eq, n_free):
    bounds = [(0, None)] * (len(c) - n_free) + [(None, None)] * n_free
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")
    return result


def merged_guarantee(
    tree: GameTree, index: SequenceIndex, player: Player, plan, key: Key, rounds: int = 3
) -> tuple[list[float], np.ndarray]:
    """Alternate LPs; return P0-values per step and the final full realization plan."""
    own = index.players[player]
    payoff = build_payoff(tree, index)
    own_flow, own_rhs = build_flow(own)
    opp_flow, _ = build_flow(index.players[1 - player])
    # Guarantee LP in unified form: min w_0  s.t.  G r - H^T w <= 0 and own flow.
    gmat = payoff.tocsr() if player == Player.P1 else (-payoff.T).tocsr()
    hmat_t = opp_flow.T.tocsr()
    sign = 1 if player == Player.P1 else -1  # converts w_0 to a P0-value
    n_own, m_opp = len(own.sequences), opp_flow.shape[0]

    layer = last_layer(own)
    keys: dict[Hashable, int] = {}
    actions_of: dict[Hashable, tuple] = {}
    for i in layer:
        info = own.information_sets[i]
        k = key(info.information_state)
        acts = tuple(own.sequences[s].action for s in info.action_sequence_ids)
        if keys.setdefault(k, len(keys)) is not None and actions_of.setdefault(k, acts) != acts:
            raise RuntimeError(f"merged information sets differ in actions at {k!r}")
    n_act = len(actions_of[next(iter(keys))])
    n_s = len(keys) * n_act
    last_seqs = {s for i in layer for s in own.information_sets[i].action_sequence_ids}
    early = [s for s in range(n_own) if s not in last_seqs]
    early_pos = {s: j for j, s in enumerate(early)}

    def s_col(i: int, a: int) -> int:
        return keys[key(own.information_sets[i].information_state)] * n_act + a

    plan = np.asarray(plan, dtype=float)
    values: list[float] = []
    s_vec = np.zeros(n_s)
    for _ in range(rounds):
        # LP1: merged last-layer behavior s, earlier layers fixed to plan.
        base = plan.copy()
        base[list(last_seqs)] = 0.0
        rows, cols, vals = [], [], []
        for i in layer:
            info = own.information_sets[i]
            for a, s in enumerate(info.action_sequence_ids):
                rows.append(s), cols.append(s_col(i, a)), vals.append(plan[info.parent_sequence_id])
        mmat = coo_matrix((vals, (rows, cols)), shape=(n_own, n_s)).tocsr()
        c = np.zeros(n_s + m_opp)
        c[n_s] = 1.0
        a_ub = hstack([gmat @ mmat, -hmat_t]).tocsr()
        eq_rows = [k * n_act + a for k in range(len(keys)) for a in range(n_act)]
        a_eq = hstack([coo_matrix((np.ones(n_s), ([c_ // n_act for c_ in eq_rows], eq_rows)), shape=(len(keys), n_s)).tocsr(), csr_matrix((len(keys), m_opp))]).tocsr()
        res = _lp(c, a_ub, -(gmat @ base), a_eq, np.ones(len(keys)), m_opp)
        s_vec = res.x[:n_s]
        values.append(sign * res.x[n_s])
        plan = base + mmat @ s_vec

        # LP2: earlier layers z, merged behavior fixed to s.
        rows, cols, vals = [], [], []
        for s in early:
            rows.append(s), cols.append(early_pos[s]), vals.append(1.0)
        for i in layer:
            info = own.information_sets[i]
            for a, s in enumerate(info.action_sequence_ids):
                rows.append(s), cols.append(early_pos[info.parent_sequence_id]), vals.append(s_vec[s_col(i, a)])
        nmat = coo_matrix((vals, (rows, cols)), shape=(n_own, len(early))).tocsr()
        keep_rows = [0] + [1 + i for i in range(len(own.information_sets)) if i not in set(layer)]
        flow_z = own_flow[keep_rows][:, early]
        c = np.zeros(len(early) + m_opp)
        c[len(early)] = 1.0
        a_ub = hstack([gmat @ nmat, -hmat_t]).tocsr()
        a_eq = hstack([flow_z, csr_matrix((len(keep_rows), m_opp))]).tocsr()
        res = _lp(c, a_ub, np.zeros(gmat.shape[0]), a_eq, own_rhs[keep_rows], m_opp)
        values.append(sign * res.x[len(early)])
        plan = nmat @ res.x[: len(early)]
    return values, plan
