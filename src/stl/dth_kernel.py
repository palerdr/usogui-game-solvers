# Copyright (c) 2026 James Cenawood.
"""Build the DTH table: one backward pass of the numba ladder over the potential layers
phi = 1200..0.

The sweep writes artifacts/dth/V.npy (17,011 x 17,012 float64, sha256 e43977b7...), the
value the leap
manifest binds. Run it with `uv run python -m stl dth [--output PATH]`. The rules, the
scalar
ladder and the checks live in stl.dth, which the hosted game imports without numba.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numba as nb
import numpy as np
from numba import njit, prange
from numba.np.ufunc.parallel import get_thread_id

from stl.dth import (
    CLASSES,
    DEFAULT_PATH,
    LAGS,
    LP,
    MAX_LAYER,
    MAX_PHI,
    MAX_SADDLE_GAP,
    PURE,
    RECHECK_SAMPLES,
    SUPPORT,
    UNSOLVED,
    WIN,
    N,
    ProfileTable,
    class_values,
    decode_class,
    recheck,
    try_rung3,
)


# values and layers
def values_array() -> np.ndarray:
    """V[pc, pd] = value; NaN until solved; the extra column WIN holds -1.0."""
    V = np.full((N, N + 1), np.nan, dtype=np.float64)  # noqa: N806
    V[:, WIN] = -1.0
    return V


def kinds_array() -> np.ndarray:
    return np.full((N, N), UNSOLVED, dtype=np.uint8)


def layer_pairs(P: int, table: ProfileTable) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
    """All classes (pc, pd) with phi[pc] + phi[pd] == P, pd-major so neighbours read the
    same row of V."""
    pcs, pds = [], []
    for a in range(max(0, P - MAX_PHI), min(MAX_PHI, P) + 1):
        bc, bd = table.bucket[a], table.bucket[P - a]
        if bc.size and bd.size:
            pds.append(np.repeat(bd, bc.size))
            pcs.append(np.tile(bc, bd.size))
    if not pcs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(pcs), np.concatenate(pds)


# solver ladder, numba kernel (rungs 1-2 fused per class; the LP stays in scipy)
@njit(cache=True)
def _ladder(s: np.ndarray, f: float, r: np.ndarray, q: np.ndarray) -> tuple[float, int]:
    """Rungs 1-2 on one class (~120 comparisons, then ~3.6k multiply-adds) in
    L1-resident scratch.
    Returns (value, kind); kind == LP means 'not certified here, hand it to the LP'."""
    n = s.shape[0]
    lo = s[0]
    hi = s[0]
    for k in range(1, n):
        lo = min(lo, s[k])
        hi = max(hi, s[k])
    s0 = s[0]
    maximin = max(lo, min(f, s0))
    minimax = min(hi, max(f, s0))
    if minimax - maximin <= MAX_SADDLE_GAP:
        return 0.5 * (maximin + minimax), PURE

    d0 = s0 - f
    if abs(d0) < 1e-12:  # noqa: PLR2004
        return 0.0, LP
    # r[k] = -sum_{m<k} (s[m+1] - s[m]) r[k-1-m] / d0
    r[0] = 1.0
    for k in range(1, n):
        acc = 0.0
        for m in range(k):
            acc += (s[m + 1] - s[m]) * r[k - 1 - m]
        r[k] = -acc / d0
    # Normalize the clipped reversed weights.
    qsum = 0.0
    for k in range(n):
        rk = r[n - 1 - k]
        if not np.isfinite(rk):
            return 0.0, LP
        qk = max(0.0, rk)
        q[k] = qk
        qsum += qk
    if qsum <= 0.0:
        return 0.0, LP
    for k in range(n):
        q[k] /= qsum
    # (M q)[i] = f * sum_{j<i} q[j] + sum_{k>=i} s[k-i] q[k]; track only its min and max
    cum = 0.0
    mn = np.inf
    mx = -np.inf
    for i in range(n):
        acc = f * cum
        for k in range(i, n):
            acc += s[k - i] * q[k]
        mn = min(mn, acc)
        mx = max(mx, acc)
        cum += q[i]
    if mx - mn <= MAX_SADDLE_GAP:
        return 0.5 * (mn + mx), SUPPORT
    return 0.0, LP


@njit(parallel=True, cache=True)
def _solve_layer_kernel(
    pcs: np.ndarray,
    pds: np.ndarray,
    V: np.ndarray,  # noqa: N803
    K: np.ndarray,  # noqa: N803
    succ: np.ndarray,
    fail: np.ndarray,
    rev: np.ndarray,
    scratch: np.ndarray,
    need_lp: np.ndarray,
) -> None:
    """One layer in parallel: gather the 61 children of each class, run the fused
    ladder, write V and K.
    Every read is a finished layer and every write is this class's own cell, so there
    are no races.
    prange splits the layer statically across threads (numba's workqueue layer; tbb has
    no macOS-arm64 wheel)."""
    for i in prange(pcs.shape[0]):
        tid = get_thread_id()
        s = scratch[tid, 0]
        r = scratch[tid, 1]
        q = scratch[tid, 2]
        pc = pcs[i]
        pd = pds[i]
        for k in range(LAGS):
            s[k] = -V[pd, succ[pc, k]]
        p = rev[pc]
        f = p * (-V[pd, fail[pc]]) + (1.0 - p)
        value, kind = _ladder(s, f, r, q)  # pyright: ignore[reportArgumentType] -- Indexed payoff is scalar.
        if kind == LP:
            need_lp[i] = True
        else:
            need_lp[i] = False
            V[pc, pd] = value
            K[pc, pd] = kind


def solve_layer(
    P: int,  # noqa: N803
    V: np.ndarray,  # noqa: N803
    K: np.ndarray,  # noqa: N803
    table: ProfileTable,
    scratch: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Solve layer P: the kernel for rungs 1-2, then the LP worklist. Returns (classes,
    counts by rung)."""
    pcs, pds = layer_pairs(P, table)
    if pcs.size == 0:
        return 0, np.zeros(3, dtype=np.int64)
    need_lp = np.empty(pcs.size, dtype=np.bool_)
    _solve_layer_kernel(
        pcs,
        pds,
        V,
        K,
        table.success_children,
        table.fail_child,
        table.rev,
        scratch,
        need_lp,
    )
    for i in np.flatnonzero(need_lp):
        pc, pd = int(pcs[i]), int(pds[i])
        res = try_rung3(*class_values(pc, pd, V, table))
        if not res.certified:
            message = f"uncertified class ({pc}, {pd}) (gap {res.saddle_gap}) - aborting build"  # noqa: E501 -- Preserve the diagnostic text.
            raise RuntimeError(message)
        V[pc, pd] = res.value
        K[pc, pd] = LP
    return pcs.size, np.bincount(K[pcs, pds], minlength=3).astype(np.int64)


# sweep
def finalize(
    V: np.ndarray,  # noqa: N803
    K: np.ndarray,  # noqa: N803
    table: ProfileTable,
    log: Callable[[str], object] = print,
) -> None:
    """Every class solved and in range, every rung recorded, and 1,200 strided classes
    re-derived (4.9)."""
    core = V[:, :N]
    if np.isnan(core).any():
        message = "unsolved classes remain"
        raise AssertionError(message)
    if not ((np.abs(core) <= 1 + 1e-9).all()):
        message = "a value left [-1, 1]"
        raise AssertionError(message)
    if not ((K <= LP).all()):
        message = "a class has no rung recorded"
        raise AssertionError(message)
    worst = 0.0
    for c in np.linspace(0, CLASSES - 1, RECHECK_SAMPLES).astype(np.int64):
        pc, pd = decode_class(c)
        err = abs(recheck(c, V, table).value - V[pc, pd])
        worst = max(worst, err)
        if not (err <= MAX_SADDLE_GAP):
            message = f"recheck mismatch at class {c}: {err}"
            raise AssertionError(message)
    log(f"finalize: {RECHECK_SAMPLES} rechecks ok, worst |diff| = {worst:.3e}")


def build(
    table: ProfileTable, stop_at: int = 0, log: Callable[[str], object] = print
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole sweep (DTH.md 4.8): layers MAX_LAYER down to stop_at, N^2 = 289,374,121
    classes
    times the constant per-class ladder. Returns V, K, counts by rung."""
    V, K = values_array(), kinds_array()  # noqa: N806
    scratch = np.empty((nb.get_num_threads(), 3, LAGS), dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    t0 = time.perf_counter()
    for P in range(MAX_LAYER, stop_at - 1, -1):  # noqa: N806
        n, c = solve_layer(P, V, K, table, scratch)
        counts += c
        if n and P % 25 == 0:
            log(
                f"layer {P:4d}: {n:8d} classes | {counts.sum():10d} solved | {time.perf_counter() - t0:7.1f}s"  # noqa: E501 -- Preserve the diagnostic text.
            )
    return V, K, counts


def write(V: np.ndarray, path: str | Path = DEFAULT_PATH) -> None:  # noqa: N803
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, V)
