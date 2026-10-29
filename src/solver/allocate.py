"""The adaptive allocation loop (KERNEL.md slice K5).

Each iteration solves the robust game on the current kernel boxes, estimates
the reach of every number sequence by self-play under both robust equilibria,
allocates the next sampling budget with a uniform floor plus a share
proportional to reach, samples, and records the root interval. The confidence
sequences stay valid under this allocation, so the interval of every iteration
is a certificate on its own.

Usage:

    uv run python -B -m src.solver.allocate --iterations 3 --budget 1e8 --games 2000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np

from src.kernel.sampler import sample_sequence
from src.kernel.table import KernelCounts, Sequences
from src.solver.robust import Arena, RobustSolver
from src.solver.selfplay import sequence_reach


def allocation(reach: np.ndarray, budget: int, floor_fraction: float) -> np.ndarray:
    """Samples per sequence: a uniform floor plus the rest in proportion to ``reach``."""
    n = reach.shape[0]
    floor = int(budget * floor_fraction) // n
    rest = budget - floor * n
    share = reach / reach.sum() if reach.sum() > 0 else np.full(n, 1.0 / n)
    return floor + np.floor(rest * share).astype(np.int64)


def run(iterations: int, budget: int, floor_fraction: float, games: int, seed: int, delta: float,
        out: str) -> dict:
    sequences = Sequences()
    arena = Arena()
    table = KernelCounts(sequences)
    solver = RobustSolver(sequences, arena)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True).stdout.strip()
    report = {"commit": commit, "delta": delta, "budget_per_iteration": budget, "floor_fraction": floor_fraction,
              "games_per_equilibrium": games, "seed": seed, "iterations": []}
    samples = 0
    for it in range(iterations):
        start = time.perf_counter()
        solver.set_boxes(table, delta)
        result = solver.solve(keep_depth=2)
        solve_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reach, returns = sequence_reach(solver, sequences, games, seed * 1000 + it)
        reach_seconds = time.perf_counter() - start
        plan = allocation(reach, budget, floor_fraction)
        start = time.perf_counter()
        for s in np.flatnonzero(plan):
            counts, errors = sample_sequence(sequences.targets[s], seed, it * len(sequences) + int(s), int(plan[s]))
            assert errors == 0
            table.add(int(s), counts)
        samples += int(plan.sum())
        sample_seconds = time.perf_counter() - start
        top = np.argsort(reach)[::-1][:10]
        record = {"iteration": it, "samples_before": samples - int(plan.sum()), "root_lo": result["root_lo"],
                  "root_hi": result["root_hi"], "width": result["root_hi"] - result["root_lo"],
                  "max_saddle_gap": result["max_saddle_gap"], "self_play_mean_return": returns,
                  "sequences_reached": int((reach > 0).sum()),
                  "top_sequences": [{"sequence": sequences.targets[s].tolist(), "reach": round(float(reach[s]), 2)} for s in top],
                  "solve_seconds": round(solve_seconds, 1), "reach_seconds": round(reach_seconds, 1),
                  "sample_seconds": round(sample_seconds, 1), "samples_after": samples}
        report["iterations"].append(record)
        with open(out, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"iteration {it}: root in [{result['root_lo']:.4f}, {result['root_hi']:.4f}] "
              f"width {record['width']:.4f} from {record['samples_before']:.2e} samples; solve {solve_seconds:.0f}s, "
              f"reach {reach_seconds:.0f}s ({record['sequences_reached']} sequences), sampling {sample_seconds:.0f}s",
              flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--budget", type=float, default=1e8)
    parser.add_argument("--floor", type=float, default=0.2)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--out", default="docs/results/allocation.json")
    args = parser.parse_args()
    run(args.iterations, int(args.budget), args.floor, args.games, args.seed, args.delta, args.out)


if __name__ == "__main__":
    main()
