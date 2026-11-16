"""Test a coarse key for one player's last decision layer of D_K on the arrays.
Run: PYTHONPATH=. uv run python src/experiments/order_blind_k3.py K SOURCE PLAYER KEY [ITERATIONS] [ROUNDS] [WORKERS]
SOURCE names the certified equilibrium (lp or cfr; certify.py must have run).
KEY is support, hard, or opening+hard. The player's earlier layers start at
the equilibrium. Each round runs ITERATIONS of merged CFR+ (last layer by key,
earlier layers fixed) and then ITERATIONS of CFR+ on the earlier layers (last
layer fixed). The best lifted strategy is certified on the full game by the
engine certificate. The one-sided check fixes the opponent to the equilibrium
and bounds the round-1 merge step from above."""
import json
import resource
import sys
import time

from src.solver.artifacts import ROOT, load_certificate, load_profile
from src.solver.engine_certificate import best_response_value
from src.solver.public_abstraction import alternate, merged_best_response, slot_keys
from src.solver.public_tree import build_public_tree
from src.solver.strategy_table import export_table

if __name__ == "__main__":
    K, source, i, name = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    rounds = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 14
    start = time.perf_counter()
    layers = build_public_tree(K)
    behavior, choose = load_profile(layers, K, source)
    cert = load_certificate(K, source)
    v_lo, v_hi = (cert["lower"], cert["upper"]) if i == 0 else (-cert["upper"], -cert["lower"])
    key, n_keys = slot_keys(layers, i, name)
    full, n_full = slot_keys(layers, i, "full")
    reference, _ = merged_best_response(layers, i, behavior, choose, full, n_full)
    cap, _ = merged_best_response(layers, i, behavior, choose, key, n_keys)
    tag = f"K={K} P{i} key={name}"
    print(f"{tag} infosets={n_full} keys={n_keys} value=[{v_lo:+.9f},{v_hi:+.9f}] "
          f"one_sided_full={reference:+.9f} one_sided_key={cap:+.9f} round1_loss>={v_lo - cap:+.3e}", flush=True)

    def log(step):
        print(f"{tag} {step[0]} guarantee={step[1]:+.9f} loss={v_hi - step[1]:+.3e} "
              f"cap={step[2]:+.9f} t={time.perf_counter() - start:.0f}s", flush=True)

    t = time.perf_counter()
    steps, (b, c) = alternate(layers, i, behavior, choose, key, n_keys, rounds, iterations, log)
    cfr_seconds = time.perf_counter() - t
    path = f"{ROOT}/abstraction_{K}_p{i}_{name}.npz"
    export_table(layers, b, c, i).save(path)
    del layers, behavior, b
    t = time.perf_counter()
    certified = -best_response_value(K, 1 - i, path, workers)
    cert_seconds = time.perf_counter() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    worker_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    record = {
        "K": K, "player": i, "key": name, "infosets": n_full, "keys": n_keys,
        "value": [v_lo, v_hi], "one_sided_key": cap, "one_sided_full": reference,
        "round1_loss_lower_bound": v_lo - cap, "steps": steps, "certified_guarantee": certified,
        "loss": [v_lo - certified, v_hi - certified], "iterations": iterations, "rounds": rounds,
        "cfr_seconds": cfr_seconds, "certificate_seconds": cert_seconds, "peak_GiB": peak, "worker_peak_GiB": worker_peak,
    }
    print(f"{tag} certified_guarantee={certified:+.9f} loss in [{v_lo - certified:+.3e}, {v_hi - certified:+.3e}] "
          f"cfr_seconds={cfr_seconds:.0f} certificate_seconds={cert_seconds:.0f} peak_GiB={peak:.2f} worker_peak_GiB={worker_peak:.2f}", flush=True)
    with open(f"{ROOT}/abstraction_{K}_p{i}_{name}.json", "w") as f:
        json.dump(record, f, indent=1)
