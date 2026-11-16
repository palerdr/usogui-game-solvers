"""Measure how much past result classes change the result kernel.

For a fixed number sequence the kernel at round ``k`` is the distribution of the
result class given the chance prefix. This script estimates, per round, the
reach-weighted total variation between the kernel conditioned on all earlier
result classes and the kernel conditioned on the numbers alone. Conditioning on
the finest partition bounds every coarser one (a fold hides a class), so the
reported number is the bias a number-only kernel would carry.

Usage:

    uv run python -B -m src.sensitivity.measure --mode round2 --trajectories 20000
    uv run python -B -m src.sensitivity.measure --mode full --sequences 60 --trajectories 100000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import time
from itertools import permutations

import numpy as np

from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.sensitivity.sampler import CLASS_NAMES, HandTables, sample_trajectory

TABLES: HandTables | None = None


def _worker(task):
    index, numbers, trajectories, seed = task
    global TABLES
    if TABLES is None:
        TABLES = HandTables()
    rng = np.random.default_rng([seed, index])
    rounds = len(numbers)
    counts: list[dict[tuple[int, ...], np.ndarray]] = [{} for _ in range(rounds)]
    for _ in range(trajectories):
        classes = sample_trajectory(TABLES, numbers, rng)
        for k in range(len(classes)):
            prefix = classes[:k]
            row = counts[k].get(prefix)
            if row is None:
                row = counts[k][prefix] = np.zeros(len(CLASS_NAMES), dtype=np.int64)
            row[classes[k]] += 1
    return index, numbers, [{" ".join(CLASS_NAMES[c] for c in p): r.tolist() for p, r in c.items()}
                            for c in counts]


def weighted_tv(round_counts: dict[str, list[int]]) -> dict:
    """Reach-weighted TV between conditional and marginal kernels at one round."""
    rows = np.array(list(round_counts.values()), dtype=np.float64)
    total = rows.sum()
    marginal = rows.sum(axis=0) / total
    reach = rows.sum(axis=1) / total
    conditional = rows / rows.sum(axis=1, keepdims=True)
    tv = 0.5 * np.abs(conditional - marginal[None, :]).sum(axis=1)
    no_judgement = marginal[CLASS_NAMES.index("N")]
    big = rows.sum(axis=1) >= 200
    # Null baseline: the same statistic when every prefix draws from the marginal.
    # Finite samples inflate TV, so the measured value is read against this floor.
    null_rng = np.random.default_rng(0)
    sizes = rows.sum(axis=1).astype(np.int64)
    null_values = []
    for _ in range(20):
        fake = np.array([null_rng.multinomial(n, marginal) for n in sizes], dtype=np.float64)
        fake_tv = 0.5 * np.abs(fake / sizes[:, None] - marginal[None, :]).sum(axis=1)
        null_values.append((reach * fake_tv).sum())
    return {
        "weighted_tv": float((reach * tv).sum()),
        "null_weighted_tv": float(np.mean(null_values)),
        "no_judgement_rate": float(no_judgement),
        "max_tv_prefixes_with_200_samples": float(tv[big].max()) if big.any() else None,
        "prefixes": int(rows.shape[0]),
        "marginal": {name: float(p) for name, p in zip(CLASS_NAMES, marginal)},
        "samples": int(total),
    }


def round2_sequences() -> list[tuple[tuple[int, int], ...]]:
    out = []
    for n0 in TEAM_0_METAL_CARDS:
        for n1 in TEAM_1_METAL_CARDS:
            for m0 in TEAM_0_METAL_CARDS:
                if m0 == n0:
                    continue
                for m1 in TEAM_1_METAL_CARDS:
                    if m1 != n1:
                        out.append(((n0, n1), (m0, m1)))
    return out


def full_sequences(count: int, seed: int) -> list[tuple[tuple[int, int], ...]]:
    rng = np.random.default_rng(seed)
    perms_0 = list(permutations(TEAM_0_METAL_CARDS))
    perms_1 = list(permutations(TEAM_1_METAL_CARDS))
    out = []
    for _ in range(count):
        p0 = perms_0[rng.integers(len(perms_0))]
        p1 = perms_1[rng.integers(len(perms_1))]
        out.append(tuple(zip(p0, p1)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("round2", "full"), required=True)
    parser.add_argument("--trajectories", type=int, default=20_000)
    parser.add_argument("--sequences", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sequences = round2_sequences() if args.mode == "round2" else full_sequences(args.sequences, args.seed)
    tasks = [(i, seq, args.trajectories, args.seed) for i, seq in enumerate(sequences)]
    start = time.monotonic()
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_worker, tasks, chunksize=1)
    elapsed = time.monotonic() - start

    per_sequence = []
    by_round: dict[int, list[float]] = {}
    for _, numbers, counts in sorted(results):
        stats = [weighted_tv(c) for c in counts]
        per_sequence.append({"numbers": numbers, "rounds": stats})
        for k, s in enumerate(stats, start=1):
            if k > 1:
                by_round.setdefault(k, []).append((s["weighted_tv"], s["null_weighted_tv"]))
    summary = {k: {"mean_weighted_tv": float(np.mean([a for a, _ in v])),
                   "max_weighted_tv": float(np.max([a for a, _ in v])),
                   "mean_null_weighted_tv": float(np.mean([b for _, b in v])),
                   "sequences": len(v)} for k, v in sorted(by_round.items())}
    for entry in per_sequence:
        for k, s in enumerate(entry["rounds"], start=1):
            summary.setdefault(k, {}).setdefault("rates", []).append(s["no_judgement_rate"])
    for k, s in summary.items():
        rates = s.pop("rates", [])
        s["mean_no_judgement_rate"] = float(np.mean(rates)) if rates else 0.0
        s["max_no_judgement_rate"] = float(np.max(rates)) if rates else 0.0
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True,
                            text=True).stdout.strip()
    report = {
        "mode": args.mode, "seed": args.seed, "trajectories_per_sequence": args.trajectories,
        "sequences": len(sequences), "workers": args.workers, "wall_seconds": round(elapsed, 1),
        "trajectories_per_second": round(len(sequences) * args.trajectories / elapsed),
        "commit": commit, "summary_by_round": summary, "per_sequence": per_sequence,
    }
    out = args.out or f"docs/results/sensitivity-{args.mode}.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"{len(sequences)} sequences x {args.trajectories} trajectories in {elapsed:.0f}s "
          f"({report['trajectories_per_second']} traj/s) -> {out}")
    for k, s in summary.items():
        tv = (f"mean weighted TV {s['mean_weighted_tv']:.4f} (null {s['mean_null_weighted_tv']:.4f}), "
              f"max {s['max_weighted_tv']:.4f}, " if "mean_weighted_tv" in s else "")
        print(f"round {k}: {tv}no-judgement rate mean {s['mean_no_judgement_rate']:.4f} "
              f"max {s['max_no_judgement_rate']:.4f}")


if __name__ == "__main__":
    main()
