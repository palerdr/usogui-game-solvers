"""Run CFR+ on D_K until the array saddle gap reaches a target or time runs out.
Run: PYTHONPATH=. uv run python src/experiments/solve_cfr.py K TARGET SECONDS [CHECK_EVERY]
Saves regrets and average weights to src/artifacts/cfr_K.npz at each check."""
import resource
import sys
import time

from src.solver.array_best_response import saddle_gap
from src.solver.artifacts import save_cfr
from src.solver.cfr_plus import CFRPlus
from src.solver.public_tree import build_public_tree


K = int(sys.argv[1])
target = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
every = int(sys.argv[4]) if len(sys.argv) > 4 else 250
path = f"src/artifacts/cfr_{K}.npz"
layers = build_public_tree(K)
cfr = CFRPlus(layers)
start = time.perf_counter()
solve_seconds = 0.0
while True:
    t = time.perf_counter()
    for _ in range(every):
        cfr.iterate()
    solve_seconds += time.perf_counter() - t
    behavior, choose = cfr.average()
    lower, upper = saddle_gap(layers, behavior, choose)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(f"iterations={cfr.iterations} lower={lower:+.12f} upper={upper:+.12f} gap={upper - lower:.3e} "
          f"cfr_seconds={solve_seconds:.0f} wall={time.perf_counter() - start:.0f} peak_GiB={peak:.2f}", flush=True)
    done = upper - lower <= target or time.perf_counter() - start > limit
    if done or cfr.iterations % (10 * every) == 0:
        save_cfr(cfr, path)
    if done:
        break
