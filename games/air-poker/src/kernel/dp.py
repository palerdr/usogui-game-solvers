"""Count and unrank legal hands by a dynamic program over card values.

``L(D, n)`` is the set of five-card subsets of deck ``D`` whose Air Poker values
sum to ``n``. Enumerating it costs at least its size; the solver's sampler must
draw from it in microseconds. This module counts it and maps an integer in
``[0, |L(D, n)|)`` to a distinct legal hand, so one bounded random integer is one
exact uniform draw.

For value ``v = 1..13`` the deck holds ``a[v]`` cards. The table

    C[v, k, s] = number of k-card subsets of the cards with value >= v whose sum is s
    C[14, 0, 0] = 1

obeys ``C[v, k, s] = sum_j binom(a[v], j) * C[v + 1, k - j, s - j * v]`` and gives
``|L(D, n)| = C[1, 5, n]``. The table depends on the deck only, so one build
serves both targets of a round. Unranking walks ``v = 1..13``, chooses the
multiplicity ``j`` by cumulative counts, then the ``j`` suits by the
combinatorial number system. Every legal mask has exactly one rank.

The functions are numba hot loops that take and return plain integers and
NumPy arrays, so the same code runs in the trajectory sampler's compiled loop.

Usage:

    uv run python -B -m src.kernel.dp            # throughput of build + unrank
"""

from __future__ import annotations

import time

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

VALUES = 13
HAND = 5
MAX_SUM = 64
NUM_CARDS = 52
# BINOM[a, j] = binom(a, j) for a = 0..4 cards of one value and j = 0..5 chosen.
BINOM = np.array([[1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0],
                  [1, 2, 1, 0, 0, 0],
                  [1, 3, 3, 1, 0, 0],
                  [1, 4, 6, 4, 1, 0]], dtype=np.int64)


@njit(cache=True)
def fill_value_counts(deck, a):
    """Overwrite ``a`` (length 14) with ``a[v]`` = cards of value ``v`` in ``deck``."""
    a[:] = 0
    for card in range(NUM_CARDS):
        if (deck >> card) & 1:
            rank = card % 13
            a[1 if rank == 12 else rank + 2] += 1


@njit(cache=True)
def value_counts(deck):
    """Return a new ``a`` with ``a[v]`` the number of cards of value ``v`` in ``deck``."""
    a = np.empty(VALUES + 1, np.int64)
    fill_value_counts(deck, a)
    return a


@njit(cache=True)
def fill_table(a, table):
    """Overwrite ``table`` of shape ``(15, 6, 65)`` with ``C`` for the value counts ``a``.

    The hot loop takes a caller-owned table so a thread reuses one workspace
    across trajectories instead of allocating per round.
    """
    table[VALUES + 1] = 0
    table[VALUES + 1, 0, 0] = 1
    for v in range(VALUES, 0, -1):
        av = a[v]
        for k in range(HAND + 1):
            top = min(av, k)
            for s in range(MAX_SUM + 1):
                total = 0
                for j in range(top + 1):
                    rest = s - j * v
                    if rest >= 0:
                        total += BINOM[av, j] * table[v + 1, k - j, rest]
                table[v, k, s] = total


@njit(cache=True)
def build_table(a):
    """Return a new ``C`` of shape ``(15, 6, 65)`` for the value counts ``a``."""
    table = np.empty((VALUES + 2, HAND + 1, MAX_SUM + 1), np.int64)
    fill_table(a, table)
    return table


@njit(cache=True)
def count(table, n):
    """Return ``|L(D, n)|`` from the deck's table; zero outside ``6..64``."""
    if n < 0 or n > MAX_SUM:
        return 0
    return table[1, HAND, n]


@njit(cache=True)
def unrank(table, a, deck, n, u):
    """Return the mask of the ``u``-th hand of ``L(deck, n)``, ``0 <= u < count``.

    Returns ``-1`` when ``u`` is out of range.
    """
    mask = 0
    k = HAND
    s = n
    for v in range(1, VALUES + 1):
        av = a[v]
        j = 0
        while True:
            if j > av or j > k or s - j * v < 0:
                return -1
            ways = BINOM[av, j] * table[v + 1, k - j, s - j * v]
            if u < ways:
                break
            u -= ways
            j += 1
        combos = BINOM[av, j]
        r = u % combos
        u = u // combos
        rank = 12 if v == 1 else v - 2
        remaining = av
        need = j
        for suit in range(4):
            if need == 0:
                break
            card = suit * 13 + rank
            if not (deck >> card) & 1:
                continue
            including = BINOM[remaining - 1, need - 1]
            if r < including:
                mask |= 1 << card
                need -= 1
            else:
                r -= including
            remaining -= 1
        k -= j
        s -= j * v
    return mask


def hand_count(deck: int, n: int) -> int:
    """Python convenience: ``|L(deck, n)|``."""
    return int(count(build_table(value_counts(deck)), n))


def unrank_hand(deck: int, n: int, u: int) -> int:
    """Python convenience: the ``u``-th legal hand of ``L(deck, n)`` as a mask."""
    a = value_counts(deck)
    return int(unrank(build_table(a), a, deck, n, u))


@njit(parallel=True, cache=True)
def _bench(decks, targets, words, counts, tables):
    """Build one table per deck and unrank one hand per target; return a checksum.

    ``counts`` and ``tables`` hold one workspace per thread.
    """
    checksum = 0
    for i in prange(decks.shape[0]):
        thread = get_thread_id()
        deck = decks[i]
        a = counts[thread]
        table = tables[thread]
        fill_value_counts(deck, a)
        fill_table(a, table)
        for t in range(targets.shape[0]):
            c = count(table, targets[t])
            if c > 0:
                checksum += unrank(table, a, deck, targets[t], words[i, t] % c)
    return checksum


def main() -> None:
    from numba import set_num_threads

    from src.engine.game import TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS

    rng = np.random.default_rng(20261029)
    trials = 200_000
    targets = np.array(sorted(TEAM_0_METAL_CARDS + TEAM_1_METAL_CARDS), dtype=np.int64)
    # Decks of 52 down to 12 cards, the range the five rounds pass through.
    sizes = rng.integers(12, 53, size=trials)
    decks = np.empty(trials, dtype=np.int64)
    for i in range(trials):
        kept = rng.choice(NUM_CARDS, size=sizes[i], replace=False)
        decks[i] = int(np.sum(np.int64(1) << kept.astype(np.int64)))
    words = rng.integers(0, 2**62, size=(trials, targets.size), dtype=np.int64)
    counts = np.empty((get_num_threads(), VALUES + 1), np.int64)
    tables = np.empty((get_num_threads(), VALUES + 2, HAND + 1, MAX_SUM + 1), np.int64)
    _bench(decks[:16], targets, words[:16], counts, tables)  # compile
    for threads in (1, get_num_threads()):
        set_num_threads(threads)
        start = time.perf_counter()
        checksum = _bench(decks, targets, words, counts, tables)
        seconds = time.perf_counter() - start
        print(f"{threads:>2} threads: {trials / seconds:12,.0f} tables/s, "
              f"{trials * targets.size / seconds:12,.0f} unranks/s, checksum {checksum}")


if __name__ == "__main__":
    main()
