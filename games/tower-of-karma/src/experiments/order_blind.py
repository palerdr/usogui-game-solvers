"""Solve D_K, then test whether each player keeps full value when its last
decisions are keyed by hard state, surviving pairs, or hard support only.
Run: PYTHONPATH=. uv run python src/experiments/order_blind.py 2"""
import sys, time; sys.setrecursionlimit(10000)
from src.solver.build_tree import build_tree
from src.solver.sequences import build_sequence_index
from src.solver.solve import solve
from src.solver.abstraction import merged_guarantee, last_layer
from src.solver.best_response import behavior_from_plan, best_response_value
from src.solver.primitives import Player

K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
tree = build_tree(K); index = build_sequence_index(tree)
sol = solve(tree, index)
v = sol.primal_value
print(f"K={K} value={v:+.9f}")

# surviving pair set per public node
pairs = {}
for n in tree.nodes:
    if n.actor is not None and n.state.beads[1] is not None:
        pairs.setdefault((n.state.phase, n.state.turn, n.state.public_history), set()).add(n.state.beads)
pairs = {k: frozenset(s) for k, s in pairs.items()}

def key_hard(info):
    own = frozenset(m.total for m in info.public_history if m.player == info.player)
    opp = frozenset(m.total for m in info.public_history if m.player != info.player)
    return (info.own_beads, own, opp)
def key_pairs(info):
    return (info.own_beads, pairs[(info.phase, info.turn, info.public_history)])
def key_support(info):
    S = pairs[(info.phase, info.turn, info.public_history)]
    return (info.own_beads, frozenset(p[1 - info.player] for p in S if p[info.player] == info.own_beads))

for name, key in (("order-blind", key_hard), ("pairs", key_pairs), ("support", key_support)):
    for player, plan in ((Player.P0, sol.p0_plan), (Player.P1, sol.p1_plan)):
        own = index.players[player]
        layer = last_layer(own)
        nkeys = len({key(own.information_sets[i].information_state) for i in layer})
        t = time.perf_counter()
        vals, plan_out = merged_guarantee(tree, index, player, plan, key, rounds=3)
        beh = behavior_from_plan(index, player, plan_out)
        exact = -best_response_value(tree, Player.P1, beh) if player == Player.P0 else best_response_value(tree, Player.P0, beh)
        loss = (v - float(exact)) if player == Player.P0 else (float(exact) - v)
        print(f"{name:11s} {player.name}: last-layer infosets={len(layer)} -> keys={nkeys}  "
              f"guarantees={[f'{x:+.6f}' for x in vals]}  exact={float(exact):+.9f}  loss={loss:+.2e}  {time.perf_counter()-t:.0f}s")
