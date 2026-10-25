"""Build the DTH table: one backward pass of the numba ladder over the potential layers phi = 1200..0.

The sweep writes artifacts/dth/V.npy (17,011 x 17,012 float64, sha256 e43977b7...), the value the leap
manifest binds. Run it with `uv run python -m stl.dth_kernel [--output PATH]`. The rules, the scalar
ladder and the checks live in stl.dth, which the hosted game imports without numba.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numba as nb
import numpy as np
from numba import njit, prange

from stl.dth import (
    CLASSES, DEFAULT_PATH, LAGS, LP, MAX_LAYER, MAX_PHI, MAX_SADDLE_GAP, N, PURE, RECHECK_SAMPLES, SUPPORT,
    UNSOLVED, WIN, ProfileTable, build_table, class_values, decode_class, recheck, try_rung3, verify_anchors,
)


# ---- values and layers -------------------------------------------------------------------------
def values_array() -> np.ndarray:
    """V[pc, pd] = value; NaN until solved; the extra column WIN holds -1.0."""
    V = np.full((N, N + 1), np.nan, dtype=np.float64)
    V[:, WIN] = -1.0
    return V


def kinds_array() -> np.ndarray:
    return np.full((N, N), UNSOLVED, dtype=np.uint8)


def layer_pairs(P: int, table: ProfileTable) -> tuple[np.ndarray, np.ndarray]:
    """All classes (pc, pd) with phi[pc] + phi[pd] == P, pd-major so neighbours read the same row of V."""
    pcs, pds = [], []
    for a in range(max(0, P - MAX_PHI), min(MAX_PHI, P) + 1):
        bc, bd = table.bucket[a], table.bucket[P - a]
        if bc.size and bd.size:
            pds.append(np.repeat(bd, bc.size))
            pcs.append(np.tile(bc, bd.size))
    if not pcs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(pcs), np.concatenate(pds)


# ---- solver ladder, numba kernel (rungs 1-2 fused per class; the LP stays in scipy) -------------
@njit(cache=True)
def _ladder(s, f, r, q):
    """Rungs 1-2 on one class (~120 comparisons, then ~3.6k multiply-adds) in L1-resident scratch.
    Returns (value, kind); kind == LP means 'not certified here, hand it to the LP'."""
    n = s.shape[0]
    lo = s[0]
    hi = s[0]
    for k in range(1, n):
        if s[k] < lo:
            lo = s[k]
        if s[k] > hi:
            hi = s[k]
    s0 = s[0]
    maximin = max(lo, min(f, s0))
    minimax = min(hi, max(f, s0))
    if minimax - maximin <= MAX_SADDLE_GAP:
        return 0.5 * (maximin + minimax), PURE

    d0 = s0 - f
    if abs(d0) < 1e-12:
        return 0.0, LP
    # r[k] = -sum_{m<k} (s[m+1] - s[m]) r[k-1-m] / d0
    r[0] = 1.0
    for k in range(1, n):
        acc = 0.0
        for m in range(k):
            acc += (s[m + 1] - s[m]) * r[k - 1 - m]
        r[k] = -acc / d0
    # q = clip(reverse(r), 0) / sum
    qsum = 0.0
    for k in range(n):
        rk = r[n - 1 - k]
        if not np.isfinite(rk):
            return 0.0, LP
        qk = rk if rk > 0.0 else 0.0
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
        if acc < mn:
            mn = acc
        if acc > mx:
            mx = acc
        cum += q[i]
    if mx - mn <= MAX_SADDLE_GAP:
        return 0.5 * (mn + mx), SUPPORT
    return 0.0, LP


@njit(parallel=True, cache=True)
def _solve_layer_kernel(pcs, pds, V, K, succ, fail, rev, scratch, need_lp):
    """One layer in parallel: gather the 61 children of each class, run the fused ladder, write V and K.
    Every read is a finished layer and every write is this class's own cell, so there are no races.
    prange splits the layer statically across threads (numba's workqueue layer; tbb has no macOS-arm64 wheel)."""
    for i in prange(pcs.shape[0]):
        tid = nb.get_thread_id()
        s = scratch[tid, 0]
        r = scratch[tid, 1]
        q = scratch[tid, 2]
        pc = pcs[i]
        pd = pds[i]
        for k in range(LAGS):
            s[k] = -V[pd, succ[pc, k]]
        p = rev[pc]
        f = p * (-V[pd, fail[pc]]) + (1.0 - p)
        value, kind = _ladder(s, f, r, q)
        if kind == LP:
            need_lp[i] = True
        else:
            need_lp[i] = False
            V[pc, pd] = value
            K[pc, pd] = kind


def solve_layer(P: int, V: np.ndarray, K: np.ndarray, table: ProfileTable, scratch: np.ndarray) -> tuple[int, np.ndarray]:
    """Solve layer P: the kernel for rungs 1-2, then the LP worklist. Returns (classes, counts by rung)."""
    pcs, pds = layer_pairs(P, table)
    if pcs.size == 0:
        return 0, np.zeros(3, dtype=np.int64)
    need_lp = np.empty(pcs.size, dtype=np.bool_)
    _solve_layer_kernel(pcs, pds, V, K, table.success_children, table.fail_child, table.rev, scratch, need_lp)
    for i in np.flatnonzero(need_lp):
        pc, pd = int(pcs[i]), int(pds[i])
        res = try_rung3(*class_values(pc, pd, V, table))
        if not res.certified:
            raise RuntimeError(f"uncertified class ({pc}, {pd}) (gap {res.saddle_gap}) - aborting build")
        V[pc, pd] = res.value
        K[pc, pd] = LP
    return pcs.size, np.bincount(K[pcs, pds], minlength=3).astype(np.int64)



# ---- sweep -------------------------------------------------------------------------------------
def finalize(V: np.ndarray, K: np.ndarray, table: ProfileTable, log=print) -> None:
    """Every class solved and in range, every rung recorded, and 1,200 strided classes re-derived (4.9)."""
    core = V[:, :N]
    assert not np.isnan(core).any(), "unsolved classes remain"
    assert (np.abs(core) <= 1 + 1e-9).all(), "a value left [-1, 1]"
    assert (K <= LP).all(), "a class has no rung recorded"
    worst = 0.0
    for c in np.linspace(0, CLASSES - 1, RECHECK_SAMPLES).astype(np.int64):
        pc, pd = decode_class(c)
        err = abs(recheck(c, V, table).value - V[pc, pd])
        worst = max(worst, err)
        assert err <= MAX_SADDLE_GAP, f"recheck mismatch at class {c}: {err}"
    log(f"finalize: {RECHECK_SAMPLES} rechecks ok, worst |diff| = {worst:.3e}")


def build(table: ProfileTable, stop_at: int = 0, log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole sweep (architecture.md 4.8): layers MAX_LAYER down to stop_at, N^2 = 289,374,121 classes
    times the constant per-class ladder. Returns V, K, counts by rung."""
    V, K = values_array(), kinds_array()
    scratch = np.empty((nb.get_num_threads(), 3, LAGS), dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    t0 = time.perf_counter()
    for P in range(MAX_LAYER, stop_at - 1, -1):
        n, c = solve_layer(P, V, K, table, scratch)
        counts += c
        if n and P % 25 == 0:
            log(f"layer {P:4d}: {n:8d} classes | {counts.sum():10d} solved | {time.perf_counter() - t0:7.1f}s")
    return V, K, counts



def write(V: np.ndarray, path: str | Path = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, V)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m stl.dth_kernel", description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    output = parser.parse_args(argv).output
    table = build_table()
    V, K, counts = build(table)
    finalize(V, K, table)
    for state, expected, got, ok in verify_anchors(V, table):
        print(f"anchor {state}: expected {expected:.16f} got {got:.16f} {'ok' if ok else 'MISMATCH'}")
    print(f"pure={counts[0]} support={counts[1]} lp={counts[2]} total={counts.sum()}")
    write(V, output)


if __name__ == "__main__":
    main()
