"""Solve D_K with the array sequence-form LP and save both plans.
Run: PYTHONPATH=. uv run python src/experiments/solve_lp.py K METHOD SECONDS
METHOD is highs, highs-ds or highs-ipm. Plans go to src/artifacts/lp_K.npz."""
import resource
import sys
import time

import numpy as np

from src.solver.array_lp import build_sequence_form, solve_lp
from src.solver.public_tree import build_public_tree

K = int(sys.argv[1])
method = sys.argv[2] if len(sys.argv) > 2 else "highs"
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
t = time.perf_counter()
layers = build_public_tree(K)
form = build_sequence_form(layers)
print(f"K={K} built in {time.perf_counter() - t:.1f}s", flush=True)
options = {"time_limit": limit, "disp": True}
if method == "highs-ipm":
    options["run_crossover"] = "off"
sol = solve_lp(form, layers, method=method, **options)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
print(f"value={sol.value:+.12f} rows={sol.rows} columns={sol.columns} nonzeros={sol.nonzeros} "
      f"lp_seconds={sol.seconds:.1f} peak_GiB={peak:.2f} status={sol.status}", flush=True)
if sol.p0_plan is not None:
    np.savez(f"src/artifacts/lp_{K}.npz", p0=sol.p0_plan, p1=sol.p1_plan, value=sol.value)
