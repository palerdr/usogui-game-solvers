"""Compiled trajectory sampler with dense prefix counts (KERNEL.md, slice K2).

A trajectory plays the five rounds of one number sequence from the full deck:
each round builds the deck's DP table, draws each hand by one bounded random
integer and ``unrank``, classifies the pair exactly as the engine does, and
removes the union. The class of round ``r`` is added to the row of the revealed
prefix (the classes of rounds ``1..r-1`` as a base-5 code), so one trajectory
samples every prefix on its path at its chance reach.

Random numbers come from a counter-based generator: the stream of trajectory
``i`` of batch ``b`` under ``seed`` starts at ``mix64`` of ``(seed, b, i)`` and
steps by the splitmix64 increment, so a batch is reproducible on any device and
independent across batches.

Usage:

    uv run python -B -m src.kernel.sampler bench                  # trajectories/s
    uv run python -B -m src.kernel.sampler round1 --per-pair 1e8  # gate 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.engine.poker_math import FULL_DECK_MASK, NUM_CARDS
from src.kernel.dp import (
    HAND,
    MAX_SUM,
    VALUES,
    count,
    fill_table,
    fill_value_counts,
    unrank,
)
from src.sensitivity.sampler import CLASS_NAMES, NOJ, TIE, W0, W0C, W1, W1C

ROUNDS = 5
CLASSES = len(CLASS_NAMES)
# Row of the revealed prefix before round r (0-based) is ROW_OFFSET[r] + its
# base-5 code; the rows of all five rounds total ROWS.
ROW_OFFSET = np.array([0, 1, 6, 31, 156], dtype=np.int64)
ROWS = 781
# Base-5 digit of each class in a prefix code; T is unreachable and has none.
DIGIT = np.array([0, 1, 2, 3, -1, 4], dtype=np.int64)
FULL = np.int64(FULL_DECK_MASK)

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
MIX_2 = np.uint64(0x94D049BB133111EB)


@njit(cache=True)
def mix64(z):
    """The splitmix64 output function, a bijection on 64-bit words."""
    z = (z ^ (z >> np.uint64(30))) * MIX_1
    z = (z ^ (z >> np.uint64(27))) * MIX_2
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def stream_state(seed, batch, index):
    """Initial generator state of trajectory ``index`` in ``batch`` under ``seed``."""
    state = mix64(np.uint64(seed) + GOLDEN)
    state = mix64(state + np.uint64(batch) * GOLDEN)
    return mix64(state + np.uint64(index) * GOLDEN)


@njit(cache=True)
def next64(state):
    """Return ``(word, state)``: the next 64-bit word and the advanced state."""
    state = state + GOLDEN
    return mix64(state), state


@njit(cache=True)
def bounded(state, n):
    """Return ``(u, state)`` with ``u`` exactly uniform in ``[0, n)``, ``n >= 1``.

    Takes the top ``b`` bits of a word, ``2**b >= n``, and rejects values at or
    above ``n``; fewer than two words are consumed on average.
    """
    if n == 1:
        return np.int64(0), state
    b = 1
    while (np.int64(1) << b) < n:
        b += 1
    shift = np.uint64(64 - b)
    while True:
        word, state = next64(state)
        u = np.int64(word >> shift)
        if u < n:
            return u, state


@njit(cache=True)
def evaluate5(mask):
    """Numba port of ``poker_math._evaluate_five_unchecked``; same packed rank."""
    rank_counts = 0
    rank_mask = 0
    for card in range(NUM_CARDS):
        if (mask >> card) & 1:
            rank = card % 13
            rank_counts += 1 << (rank << 2)
            rank_mask |= 1 << rank
    is_flush = False
    for suit in range(4):
        if (mask & (0x1FFF << (13 * suit))) == mask:
            is_flush = True
    straight_code = 0
    if (rank_mask & 0x100F) == 0x100F:
        straight_code = 4
    else:
        for high in range(12, 3, -1):
            if ((rank_mask >> (high - 4)) & 0x1F) == 0x1F:
                straight_code = high + 1
                break
    straight_high = straight_code - 1

    single0 = single1 = single2 = single3 = single4 = 0
    pair0 = pair1 = trip = quad = 0
    has_trip = has_quad = False
    singles = pairs = 0
    for rank in range(12, -1, -1):
        c = (rank_counts >> (rank << 2)) & 0xF
        if c == 4:
            quad = rank
            has_quad = True
        elif c == 3:
            trip = rank
            has_trip = True
        elif c == 2:
            if pairs == 0:
                pair0 = rank
            else:
                pair1 = rank
            pairs += 1
        elif c == 1:
            if singles == 0:
                single0 = rank
            elif singles == 1:
                single1 = rank
            elif singles == 2:
                single2 = rank
            elif singles == 3:
                single3 = rank
            else:
                single4 = rank
            singles += 1

    if is_flush and straight_code:
        return (8 << 24) | (straight_high << 20)
    if has_quad:
        return (7 << 24) | (quad << 20) | (single0 << 16)
    if has_trip:
        if pairs:
            return (6 << 24) | (trip << 20) | (pair0 << 16)
        return (3 << 24) | (trip << 20) | (single0 << 16) | (single1 << 12)
    if pairs == 2:
        return (2 << 24) | (pair0 << 20) | (pair1 << 16) | (single0 << 12)
    if pairs == 1:
        return (1 << 24) | (pair0 << 20) | (single0 << 16) | (single1 << 12) | (single2 << 8)
    kickers = (single0 << 20) | (single1 << 16) | (single2 << 12) | (single3 << 8) | (single4 << 4)
    if is_flush:
        return (5 << 24) | kickers
    if straight_code:
        return (4 << 24) | (straight_high << 20)
    return kickers


@njit(cache=True)
def evaluate_many(masks):
    out = np.empty(masks.shape[0], np.int64)
    for i in range(masks.shape[0]):
        out[i] = evaluate5(masks[i])
    return out


@njit(cache=True)
def play_round(deck, target_0, target_1, a, table, state):
    """Play one round on ``deck``; return ``(class, next_deck, hand_0, hand_1, state)``.

    A hand is ``-1`` when its target is impossible. Both impossible is no
    judgement with the deck unchanged; one impossible gives the other side a
    plain win and removes its hand, as the engine does.
    """
    fill_value_counts(deck, a)
    fill_table(a, table)
    c0 = count(table, target_0)
    c1 = count(table, target_1)
    if c0 == 0 and c1 == 0:
        return NOJ, deck, np.int64(-1), np.int64(-1), state
    hand_0 = np.int64(-1)
    hand_1 = np.int64(-1)
    if c0 > 0:
        u, state = bounded(state, c0)
        hand_0 = unrank(table, a, deck, target_0, u)
    if c1 > 0:
        u, state = bounded(state, c1)
        hand_1 = unrank(table, a, deck, target_1, u)
    if hand_0 < 0:
        return W1, deck & ~hand_1, hand_0, hand_1, state
    if hand_1 < 0:
        return W0, deck & ~hand_0, hand_0, hand_1, state
    rank_0 = evaluate5(hand_0)
    rank_1 = evaluate5(hand_1)
    overlap = (hand_0 & hand_1) != 0
    if rank_0 == rank_1:
        result = TIE
    elif rank_0 > rank_1:
        result = W0C if overlap else W0
    else:
        result = W1C if overlap else W1
    return result, deck & ~(hand_0 | hand_1), hand_0, hand_1, state


@njit(parallel=True, cache=True)
def _count_sequence(targets, seed, batch, n, counts, values, tables, errors):
    """Add ``n`` trajectories of the number sequence ``targets`` (5 x 2) to ``counts``.

    ``counts[thread, row, class]`` receives one per round; ``errors[thread]``
    counts trajectories cut by an unreachable class.
    """
    for i in prange(n):
        thread = get_thread_id()
        state = stream_state(seed, batch, i)
        deck = FULL
        prefix = 0
        for r in range(ROUNDS):
            result, deck, _, _, state = play_round(deck, targets[r, 0], targets[r, 1],
                                                   values[thread], tables[thread], state)
            counts[thread, ROW_OFFSET[r] + prefix, result] += 1
            digit = DIGIT[result]
            if digit < 0:
                errors[thread] += 1
                break
            prefix = prefix * 5 + digit


@njit(parallel=True, cache=True)
def _count_rounds(deck, target_0, target_1, seed, batch, n, counts, values, tables):
    """Add ``n`` single rounds on ``deck`` to ``counts[thread, class]``."""
    for i in prange(n):
        thread = get_thread_id()
        state = stream_state(seed, batch, i)
        result, _, _, _, _ = play_round(deck, target_0, target_1, values[thread], tables[thread], state)
        counts[thread, result] += 1


@njit(parallel=True, cache=True)
def _round_classes(deck, target_0, target_1, seed, batch, out, values, tables):
    """Write the class of single round ``i`` on ``deck`` to ``out[i]``."""
    for i in prange(out.shape[0]):
        thread = get_thread_id()
        state = stream_state(seed, batch, i)
        out[i] = play_round(deck, target_0, target_1, values[thread], tables[thread], state)[0]


def _workspaces():
    threads = get_num_threads()
    return (np.empty((threads, VALUES + 1), np.int64),
            np.empty((threads, VALUES + 2, HAND + 1, MAX_SUM + 1), np.int64))


def sample_sequence(targets, seed: int, batch: int, n: int) -> tuple[np.ndarray, int]:
    """Return ``(counts[ROWS, CLASSES], errors)`` for ``n`` trajectories of ``targets``."""
    values, tables = _workspaces()
    counts = np.zeros((values.shape[0], ROWS, CLASSES), np.int64)
    errors = np.zeros(values.shape[0], np.int64)
    _count_sequence(np.asarray(targets, dtype=np.int64), seed, batch, n, counts, values, tables, errors)
    return counts.sum(axis=0), int(errors.sum())


def sample_rounds(deck: int, target_0: int, target_1: int, seed: int, batch: int, n: int) -> np.ndarray:
    """Return class counts of ``n`` single rounds on ``deck``."""
    values, tables = _workspaces()
    counts = np.zeros((values.shape[0], CLASSES), np.int64)
    _count_rounds(np.int64(deck), target_0, target_1, seed, batch, n, counts, values, tables)
    return counts.sum(axis=0)


def stream(seed: int, batch: int, index: int) -> np.uint64:
    """Python-facing ``stream_state``: the word stays ``uint64`` across the call boundary."""
    return np.uint64(stream_state(seed, batch, index))


def sample_round_classes(deck: int, target_0: int, target_1: int, seed: int, batch: int, n: int) -> np.ndarray:
    """Return the class of each of ``n`` single rounds on ``deck``, in stream order."""
    values, tables = _workspaces()
    out = np.empty(n, np.int8)
    _round_classes(np.int64(deck), target_0, target_1, seed, batch, out, values, tables)
    return out


def sample_round_hands(deck: int, target_0: int, target_1: int, seed: int, index: int):
    """Return ``(class, next_deck, hand_0, hand_1)`` of one round; hands ``None`` when no-rank."""
    values, tables = _workspaces()
    result, next_deck, hand_0, hand_1 = play_round(np.int64(deck), target_0, target_1, values[0],
                                                   tables[0], stream(seed, 0, index))[:4]
    return int(result), int(next_deck), (None if hand_0 < 0 else int(hand_0)), (None if hand_1 < 0 else int(hand_1))


def row(round_index: int, prefix: tuple[int, ...]) -> int:
    """Row of the entry for round ``round_index`` (0-based) after revealed ``prefix``."""
    code = 0
    for result in prefix:
        code = code * 5 + int(DIGIT[result])
    return int(ROW_OFFSET[round_index]) + code


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False,
                          capture_output=True, text=True).stdout.strip()


def bench(n: int) -> None:
    from numba import set_num_threads

    targets = [(25, 8), (26, 15), (36, 44), (39, 47), (45, 63)]
    sample_sequence(targets, 1, 0, 1000)
    for threads in (1, get_num_threads()):
        set_num_threads(threads)
        start = time.perf_counter()
        counts, errors = sample_sequence(targets, 1, 0, n)
        seconds = time.perf_counter() - start
        print(f"{threads:>2} threads: {n / seconds:12,.0f} trajectories/s, errors {errors}, "
              f"round-1 row {counts[0].tolist()}")


def round1(per_pair: int, seed: int) -> None:
    """Gate 3: single rounds on the full deck against ``docs/results/kernel-round1.json``."""
    with open("docs/results/kernel-round1.json") as handle:
        exact = {tuple(e["numbers"]): e for e in json.load(handle)["entries"]}
    sample_rounds(FULL_DECK_MASK, 25, 8, seed, 0, 1000)
    start = time.perf_counter()
    entries = []
    worst = 0.0
    for batch, (a, b) in enumerate((a, b) for a in TEAM_0_METAL_CARDS for b in TEAM_1_METAL_CARDS):
        counts = sample_rounds(FULL_DECK_MASK, a, b, seed, batch, per_pair)
        z = {}
        for index, name in enumerate(CLASS_NAMES):
            p = exact[(a, b)]["probabilities_float"][name]
            if p in (0.0, 1.0):
                z[name] = 0.0 if counts[index] == round(p * per_pair) else float("inf")
            else:
                z[name] = (counts[index] - per_pair * p) / np.sqrt(per_pair * p * (1 - p))
        worst = max(worst, max(abs(v) for v in z.values()))
        entries.append({"numbers": [a, b], "n": per_pair, "counts": dict(zip(CLASS_NAMES, counts.tolist())),
                        "z": {k: round(v, 3) for k, v in z.items()}})
        print(f"({a:>2},{b:>2}) max |z| {max(abs(v) for v in z.values()):5.2f}", flush=True)
    elapsed = time.perf_counter() - start
    report = {"gate": 3, "seed": seed, "per_pair": per_pair, "commit": _commit(), "threads": get_num_threads(),
              "wall_seconds": round(elapsed, 1), "max_abs_z": round(worst, 3), "passed": bool(worst < 4),
              "entries": entries}
    out = "docs/results/kernel-round1-parity.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"max |z| {worst:.3f} over 150 cells, {25 * per_pair:.2e} rounds, {elapsed:.0f}s wall -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("bench", "round1"))
    parser.add_argument("--per-pair", type=float, default=1e8)
    parser.add_argument("--trajectories", type=float, default=2e6)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.command == "bench":
        bench(int(args.trajectories))
    else:
        round1(int(args.per_pair), args.seed)


if __name__ == "__main__":
    main()
