"""Certify a solved D_K profile with the engine certificate.
Run: PYTHONPATH=. uv run python src/experiments/certify.py K SOURCE [WORKERS]
SOURCE is lp (src/artifacts/lp_K.npz) or cfr (src/artifacts/cfr_K.npz). The
exported strategy tables go to src/artifacts/SOURCE_K_p0.npz and _p1.npz."""
import resource
import sys
import time

from src.solver.artifacts import load_profile, save_certificate
from src.solver.engine_certificate import certify
from src.solver.public_tree import build_public_tree
from src.solver.strategy_table import export_table


if __name__ == "__main__":
    K, source = int(sys.argv[1]), sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    layers = build_public_tree(K)
    behavior, choose = load_profile(layers, K, source)
    paths = [f"src/artifacts/{source}_{K}_p{p}.npz" for p in (0, 1)]
    for p in (0, 1):
        export_table(layers, behavior, choose, p).save(paths[p])
    del layers, behavior
    t = time.perf_counter()
    lower, upper = certify(K, paths[0], paths[1], workers)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    worker_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"K={K} source={source} lower={lower:+.12f} upper={upper:+.12f} gap={upper - lower:.3e} "
          f"certificate_seconds={time.perf_counter() - t:.0f} main_peak_GiB={peak:.2f} worker_peak_GiB={worker_peak:.2f}", flush=True)
    save_certificate(K, source, {"lower": lower, "upper": upper, "seconds": time.perf_counter() - t,
                                     "peak_GiB": peak, "worker_peak_GiB": worker_peak})
