"""Exact best responses against a fixed behavioral strategy.

Arithmetic is rational, so a returned guarantee is a proof about that strategy.
"""

from fractions import Fraction

from src.solver.primitives import Action, InformationState, Player
from src.solver.sequences import SequenceIndex
from src.solver.tree import GameTree

type Behavior = dict[InformationState, dict[Action, Fraction]]


def behavior_from_plan(index: SequenceIndex, player: Player, plan) -> Behavior:
    """Normalize a realization plan into exact per-information-set distributions.

    Negative numerical noise is clipped. An unreached information set gets a
    uniform distribution; its choice never affects the opponent's best response.
    """
    seqs = index.players[player]
    behavior: Behavior = {}
    for info in seqs.information_sets:
        weights = [max(Fraction(float(plan[s])), Fraction(0)) for s in info.action_sequence_ids]
        total = sum(weights)
        if total == 0:
            weights = [Fraction(1)] * len(weights)
            total = Fraction(len(weights))
        behavior[info.information_state] = {
            seqs.sequences[s].action: w / total for s, w in zip(info.action_sequence_ids, weights)
        }
    return behavior


def best_response_value(tree: GameTree, player: Player, opponent: Behavior) -> Fraction:
    """Exact value, to `player`, of the best response against `opponent`."""
    n = len(tree.nodes)
    reach = [Fraction(0)] * n
    reach[tree.root_id] = Fraction(1)
    depth = [0] * n
    for node_id, node in enumerate(tree.nodes):  # preorder: parents precede children
        if node.actor is None:
            continue
        probs = None if node.actor == player else opponent[node.information_set]
        for edge in node.edges:
            depth[edge.child_id] = depth[node_id] + 1
            reach[edge.child_id] = reach[node_id] * (1 if probs is None else probs[edge.action])

    by_depth: dict[int, list[int]] = {}
    for node_id in range(n):
        by_depth.setdefault(depth[node_id], []).append(node_id)

    value: list[Fraction | None] = [None] * n
    for d in sorted(by_depth, reverse=True):
        action_values: dict[InformationState, dict[Action, Fraction]] = {}
        for node_id in by_depth[d]:
            node = tree.nodes[node_id]
            if node.actor is None:
                value[node_id] = Fraction(node.returns[player])
            elif node.actor != player:
                probs = opponent[node.information_set]
                value[node_id] = sum(probs[e.action] * value[e.child_id] for e in node.edges)
            else:
                acc = action_values.setdefault(node.information_set, {})
                for e in node.edges:
                    acc[e.action] = acc.get(e.action, Fraction(0)) + reach[node_id] * value[e.child_id]
        best = {info: max(vals, key=vals.get) for info, vals in action_values.items()}
        for node_id in by_depth[d]:
            node = tree.nodes[node_id]
            if node.actor == player:
                chosen = best[node.information_set]
                value[node_id] = next(value[e.child_id] for e in node.edges if e.action == chosen)
    return value[tree.root_id]


def certify(tree: GameTree, p0: Behavior, p1: Behavior) -> tuple[Fraction, Fraction]:
    """Return (P0's guarantee, P0's cap): lower and upper bounds on the game value."""
    lower = -best_response_value(tree, Player.P1, p0)
    upper = best_response_value(tree, Player.P0, p1)
    return lower, upper
