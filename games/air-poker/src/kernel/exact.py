"""Exact result kernel of one round by complete pair enumeration.

For a deck ``D`` and targets ``(n0, n1)`` the uniform upper policies give every
pair in ``L(D,n0) x L(D,n1)`` the same probability, so the result kernel is a
ratio of integer counts. This module counts the five result classes over the
full cross product with vectorized NumPy. On the full deck the 25 canonical
pairs total 52,341,125,760 hand pairs.

Usage:

    uv run python -B -m src.kernel.exact            # all 25 pairs, full deck
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import time
from fractions import Fraction

import numpy as np

from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.engine.poker_math import FULL_DECK_MASK
from src.sensitivity.sampler import CLASS_NAMES, NOJ, TIE, W0, W0C, W1, W1C, HandTables

TABLES: HandTables | None = None
CHUNK = 256


def exact_kernel_counts(tables: HandTables, deck: int, target_0: int, target_1: int) -> np.ndarray:
    """Return integer counts of each result class over ``L(deck,n0) x L(deck,n1)``.

    The counts divide by their sum to give the kernel. A one-sided no-rank gives
    the whole mass to the other side's plain win; a two-sided no-rank gives it
    to ``NOJ`` with a count of one, matching ``iter_chance_outcomes``.
    """
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    legal_0 = tables.legal_indices(target_0, deck)
    legal_1 = tables.legal_indices(target_1, deck)
    if legal_0.size == 0 and legal_1.size == 0:
        counts[NOJ] = 1
        return counts
    if legal_0.size == 0:
        counts[W1] = legal_1.size
        return counts
    if legal_1.size == 0:
        counts[W0] = legal_0.size
        return counts
    masks_0 = tables.masks[target_0][legal_0]
    ranks_0 = tables.ranks[target_0][legal_0]
    masks_1 = tables.masks[target_1][legal_1]
    ranks_1 = tables.ranks[target_1][legal_1]
    for start in range(0, masks_0.size, CHUNK):
        m0 = masks_0[start:start + CHUNK, None]
        r0 = ranks_0[start:start + CHUNK, None]
        overlap = (m0 & masks_1[None, :]) != 0
        win_0 = r0 > ranks_1[None, :]
        tie = r0 == ranks_1[None, :]
        counts[W0C] += np.count_nonzero(win_0 & overlap)
        counts[W0] += np.count_nonzero(win_0) - np.count_nonzero(win_0 & overlap)
        counts[TIE] += np.count_nonzero(tie)
        lose = ~win_0 & ~tie
        counts[W1C] += np.count_nonzero(lose & overlap)
        counts[W1] += np.count_nonzero(lose) - np.count_nonzero(lose & overlap)
    return counts


def _worker(task):
    target_0, target_1 = task
    global TABLES
    if TABLES is None:
        TABLES = HandTables()
    start = time.monotonic()
    counts = exact_kernel_counts(TABLES, FULL_DECK_MASK, target_0, target_1)
    return target_0, target_1, counts.tolist(), time.monotonic() - start


def main() -> None:
    tasks = [(a, b) for a in TEAM_0_METAL_CARDS for b in TEAM_1_METAL_CARDS]
    workers = max(1, (os.cpu_count() or 2) - 1)
    start = time.monotonic()
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = pool.map(_worker, tasks, chunksize=1)
    elapsed = time.monotonic() - start
    tables = HandTables()
    entries = []
    for target_0, target_1, counts, seconds in sorted(results):
        total = sum(counts)
        support = (int(tables.masks[target_0].size), int(tables.masks[target_1].size))
        assert total == support[0] * support[1], (target_0, target_1, total, support)
        entries.append({
            "numbers": [target_0, target_1], "support": support, "pairs": total,
            "counts": dict(zip(CLASS_NAMES, counts)),
            "probabilities": {name: str(Fraction(c, total)) for name, c in zip(CLASS_NAMES, counts)},
            "probabilities_float": {name: c / total for name, c in zip(CLASS_NAMES, counts)},
            "seconds": round(seconds, 2),
        })
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False,
                            capture_output=True, text=True).stdout.strip()
    report = {"model": "canonical-uniform-v1", "round": 1, "deck": "full", "commit": commit,
              "workers": workers, "wall_seconds": round(elapsed, 1),
              "total_pairs": sum(e["pairs"] for e in entries), "entries": entries}
    out = "docs/results/kernel-round1.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"{len(entries)} pairs, {report['total_pairs']} hand pairs, {elapsed:.0f}s wall -> {out}")
    print("Baku\\Lalo  " + "".join(f"{b:>8}" for b in TEAM_1_METAL_CARDS) + "   P(Team 0 wins)")
    for a in TEAM_0_METAL_CARDS:
        row = [e for e in entries if e["numbers"][0] == a]
        print(f"{a:>9}  " + "".join(f"{e['probabilities_float']['W0'] + e['probabilities_float']['W0c']:8.4f}"
                                   for e in row))


if __name__ == "__main__":
    main()
