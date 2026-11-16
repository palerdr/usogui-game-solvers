"""Write the golden fixtures of crates/ap/tests/data from the Python reference.

Run from `games/air-poker` (see README.md in this directory).
"""

import json
import os
import sys

import numpy as np

from src.engine.poker_math import FULL_DECK_MASK, make_card
from src.kernel.confidence import boxes, interval
from src.kernel.dp import build_table, count, unrank, value_counts
from src.kernel.sampler import bounded, sample_round_hands, sample_rounds, sample_sequence, stream
from src.kernel.table import KernelCounts, Sequences
from src.sensitivity.sampler import CANONICAL_TARGETS
from src.solver.robust import Arena, RobustSolver

OUT = os.path.dirname(os.path.abspath(__file__))
REDUCED = sum(1 << make_card(r, 0) for r in range(13)) | sum(1 << make_card(r, 1) for r in range(6))


def write(name, data):
    with open(os.path.join(OUT, name), "w") as handle:
        json.dump(data, handle, separators=(",", ":"))
        handle.write("\n")


def decks(seed, n, low, high):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kept = rng.choice(52, size=int(rng.integers(low, high + 1)), replace=False)
        out.append(int(sum(1 << int(c) for c in kept)))
    return out


def dp():
    cases = []
    for deck in [FULL_DECK_MASK, REDUCED, *decks(1, 10, 12, 52)]:
        a = value_counts(deck)
        table = build_table(a)
        counts = [int(count(table, n)) for n in range(70)]
        ranks = []
        for n in CANONICAL_TARGETS:
            c = counts[n]
            for u in sorted({0, c // 3, c // 2, c - 1}) if c else []:
                ranks.append([n, u, int(unrank(table, a, deck, n, u))])
        cases.append({"deck": deck, "counts": counts, "unrank": ranks})
    write("dp.json", cases)


def sampler():
    streams = [[s, b, i, int(stream(s, b, i))] for s in (0, 1, 20261029, 20261029001) for b in (0, 7) for i in (0, 1, 123456)]
    draws = []
    for s, b, i, n in ((5, 0, 0, 1), (5, 0, 1, 2), (5, 0, 2, 3), (5, 0, 3, 61892), (9, 3, 4, 2598960), (1, 1, 1, 100)):
        state = stream(s, b, i)
        us = []
        for _ in range(5):
            u, state = bounded(np.uint64(state), n)
            us.append(int(u))
        draws.append([s, b, i, n, us])
    sequences = []
    for targets, seed, batch, n in (([(25, 8), (26, 15), (36, 44), (39, 47), (45, 63)], 20261029, 0, 20000),
                                    ([(39, 47), (45, 8), (36, 15), (26, 44), (25, 63)], 3, 11, 5000)):
        counts, errors = sample_sequence(targets, seed, batch, n)
        cells = [[int(r), int(c), int(counts[r, c])] for r, c in zip(*np.nonzero(counts))]
        sequences.append({"targets": targets, "seed": seed, "batch": batch, "n": n, "errors": errors, "counts": cells})
    tiny = REDUCED & ~sum(1 << make_card(r, 0) for r in range(6, 13))
    rounds = []
    for deck, a, b, seed, batch, n in ((FULL_DECK_MASK, 36, 15, 20261029, 4, 100000), (REDUCED, 26, 8, 2, 0, 1000),
                                       (decks(3, 1, 20, 20)[0], 45, 44, 7, 2, 50000), (tiny, 36, 63, 1, 0, 10)):
        rounds.append({"deck": deck, "t0": a, "t1": b, "seed": seed, "batch": batch, "n": n,
                       "counts": sample_rounds(deck, a, b, seed, batch, n).tolist()})
    hands = []
    rng = np.random.default_rng(5)
    pairs = [(a, b) for a in (25, 26, 36, 39, 45) for b in (8, 15, 44, 47, 63)]
    for index, deck in enumerate([FULL_DECK_MASK, REDUCED, tiny, *decks(6, 17, 10, 40)]):
        a, b = pairs[int(rng.integers(25))]
        result, next_deck, h0, h1 = sample_round_hands(deck, a, b, 77, index)
        hands.append({"deck": deck, "t0": a, "t1": b, "seed": 77, "index": index, "class": result,
                      "next": next_deck, "hands": [h0, h1]})
    write("sampler.json", {"streams": streams, "bounded": draws, "sequences": sequences, "rounds": rounds,
                           "hands": hands})


def confidence():
    rows = []
    for n in (0, 1, 2, 10, 100, 1000, 12345, 10**6, 10**8, 3 * 10**8):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            if 0 <= k <= n:
                for alpha in (0.01, 1e-3 / 21_904_800 / 5, 1e-11):
                    lo, hi = interval(k, n, alpha)
                    rows.append([k, n, alpha, lo, hi])
    box_rows = []
    for counts in ([900, 50, 40, 10, 0, 0], [0, 0, 0, 0, 0, 1000], [0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 0, 5],
                   [12345678, 23456, 3456789, 456, 0, 7]):
        lo, hi = boxes(np.array([counts]), 1e-3 / 21_904_800)
        box_rows.append({"counts": counts, "lo": lo[0].tolist(), "hi": hi[0].tolist()})
    write("confidence.json", {"intervals": rows, "boxes": box_rows})


def round_one():
    with open("docs/results/kernel-round1.json") as handle:
        report = json.load(handle)
    names = ("W0", "W0c", "W1", "W1c", "T", "N")
    write("kernel_round1.json", {"wall_seconds": report["wall_seconds"], "workers": report["workers"],
                                 "entries": [[*e["numbers"], [e["counts"][k] for k in names]] for e in report["entries"]]})


def robust():
    sequences = Sequences()
    arena = Arena()
    table = KernelCounts(sequences)
    prefix = tuple((int(a), int(b)) for a, b in sequences.targets[7][:3])
    sampled = []
    for s in range(len(sequences)):
        if tuple(map(tuple, sequences.targets[s][:3].tolist())) == prefix:
            counts, errors = sample_sequence(sequences.targets[s], 9, s, 30000)
            assert errors == 0
            table.add(s, counts)
            sampled.append(s)
    solver = RobustSolver(sequences, arena)
    solver.set_boxes(table, 1e-3)
    solver.kept_values, solver.kept_strategies, solver.max_gap, solver.keep_depth = {}, {}, 0.0, -1
    lo, hi, strategies = solver._v(prefix, 3)
    rng = np.random.default_rng(1)
    first = solver.first(prefix, 3)
    chips = arena.chips[3][first]
    values = []
    for _ in range(400):
        code = int(rng.integers(6 ** 3))
        s0, s1 = chips[int(rng.integers(len(chips)))]
        values.append([code, s0, s1, float(lo[code, s0, s1]), float(hi[code, s0, s1])])
    rows, cols, chip_array, x_lo, y_lo, x_hi, y_hi = strategies
    mixes = []
    for _ in range(100):
        code = int(rng.integers(6 ** 3))
        n = int(rng.integers(len(chip_array)))
        mixes.append([code, int(chip_array[n, 0]), int(chip_array[n, 1]), x_lo[code, n].tolist(),
                      y_lo[code, n].tolist(), x_hi[code, n].tolist(), y_hi[code, n].tolist()])
    full = prefix + ((rows[0], cols[1]), (rows[1], cols[0]))
    w_lo, w_hi = solver._w(full, 4)
    chips5 = arena.chips[4][solver.first(full, 4)]
    w = []
    for _ in range(400):
        code = int(rng.integers(6 ** 4))
        s0, s1 = chips5[int(rng.integers(len(chips5)))]
        w.append([code, s0, s1, float(w_lo[code, s0, s1]), float(w_hi[code, s0, s1])])
    write("robust.json", {"delta": 1e-3, "seed": 9, "n": 30000, "sampled": sampled, "prefix": prefix,
                          "max_gap": solver.max_gap, "v3": values, "strategies": mixes, "w4_prefix": full, "w4": w})


if __name__ == "__main__":
    for name in sys.argv[1:] or ("dp", "sampler", "confidence", "round_one", "robust"):
        globals()[name]()
