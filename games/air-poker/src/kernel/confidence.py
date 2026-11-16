"""Time-uniform confidence sets for kernel entries from class counts alone.

Each class indicator of an entry is a Bernoulli variable with mean ``p``, so the
likelihood of ``k`` successes in ``n`` draws is known exactly and the mixture
likelihood ratio

    M_n(p) = m(k, n) / (p**k * (1 - p)**(n - k)),
    m(k, n) = B(k + 1/2, n - k + 1/2) / B(1/2, 1/2)

is a nonnegative martingale with mean one under ``p`` (Robbins' Beta mixture
with the Krichevsky-Trofimov prior). By Ville's inequality
``P(exists n: M_n(p) >= 1/alpha) <= alpha``, so

    C_n = { p : p**k * (1 - p)**(n - k) >= alpha * m(k, n) }

contains ``p`` at every ``n`` simultaneously with probability at least
``1 - alpha``, whatever rule decides how many samples an entry receives. The
set is an interval around ``k / n``. At ``k = 0`` it is
``[0, 1 - (alpha * m(0, n))**(1/n)]``, about ``(log(1/alpha) + log(n)/2) / n``.

A box for one entry takes each of the five reachable classes at level
``alpha / 5`` and sets the unreachable tie class to zero; ``tighten`` then
intersects the box with the simplex, which is implied and loses no validity.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from src.sensitivity.sampler import TIE

ITERATIONS = 64


@njit(cache=True)
def log_mixture(k, n):
    """``log m(k, n)`` for the Beta(1/2, 1/2) mixture of Bernoulli likelihoods."""
    return math.lgamma(k + 0.5) + math.lgamma(n - k + 0.5) - math.lgamma(n + 1.0) - math.log(math.pi)


@njit(cache=True)
def log_likelihood(p, k, n):
    return k * math.log(p) + (n - k) * math.log1p(-p)


@njit(cache=True)
def interval(k, n, alpha):
    """Return ``(lo, hi)`` of ``C_n`` for ``k`` successes in ``n`` draws at level ``alpha``.

    The endpoints are the outer ends of the final bisection brackets, so the
    returned interval contains the exact set.
    """
    if n == 0:
        return 0.0, 1.0
    threshold = math.log(alpha) + log_mixture(k, n)
    mode = k / n
    if k == 0:
        lo = 0.0
    else:
        a, b = 0.0, mode  # likelihood below threshold at a, above at b
        for _ in range(ITERATIONS):
            mid = 0.5 * (a + b)
            if log_likelihood(mid, k, n) >= threshold:
                b = mid
            else:
                a = mid
        lo = a
    if k == n:
        hi = 1.0
    else:
        a, b = mode, 1.0
        for _ in range(ITERATIONS):
            mid = 0.5 * (a + b)
            if log_likelihood(mid, k, n) >= threshold:
                a = mid
            else:
                b = mid
        hi = b
    return lo, hi


@njit(parallel=True, cache=True)
def _intervals(k, n, alpha, lo, hi):
    for i in prange(k.shape[0]):
        lo[i], hi[i] = interval(k[i], n[i], alpha)


def intervals(k: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise ``interval`` over arrays of equal shape."""
    k = np.ascontiguousarray(k, dtype=np.int64)
    n = np.ascontiguousarray(np.broadcast_to(n, k.shape), dtype=np.int64)
    lo = np.empty(k.size)
    hi = np.empty(k.size)
    _intervals(k.ravel(), n.ravel(), alpha, lo, hi)
    return lo.reshape(k.shape), hi.reshape(k.shape)


def tighten(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intersect a box on the last axis with the simplex: ``p_i >= 1 - sum_j!=i hi_j`` and ``<= 1 - sum_j!=i lo_j``."""
    other_hi = hi.sum(axis=-1, keepdims=True) - hi
    other_lo = lo.sum(axis=-1, keepdims=True) - lo
    return np.maximum(lo, 1.0 - other_hi), np.minimum(hi, 1.0 - other_lo)


def boxes(counts: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` boxes for entries whose last axis holds the six class counts.

    ``alpha`` is the level of one entry; each reachable class gets ``alpha / 5``.
    """
    n = counts.sum(axis=-1, keepdims=True)
    lo, hi = intervals(counts, n, alpha / 5)
    lo[..., TIE] = 0.0
    hi[..., TIE] = 0.0
    return tighten(lo, hi)
