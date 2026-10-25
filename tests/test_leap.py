# Copyright 2026 The STL authors
"""We test stl.leap: clocks, reachability, the scalar oracle, the HiGHS
fallback, the sweep scheduler, the packed audit and the verified read path."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, linprog

from stl.dth import DEFAULT_PATH
from stl.leap import StageResult, profile_index
from tests.conftest import solve_stage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import SimpleNamespace

    from stl.leap import PackedReader

DTH_PATH = DEFAULT_PATH
ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "leap"


# ──────────────────────────────────────────────────────────────────────
# Protect the leap kernel, its clocks and its sweep scheduler.
# ──────────────────────────────────────────────────────────────────────


def evaluate_states(
    key: tuple[str, int], pairs: np.ndarray, dth: np.ndarray
) -> np.ndarray:
    """Evaluate bounded test subtrees without sweeping a key."""
    from stl.leap import WIN, child_key, is_window, profiles  # noqa: PLC0415

    p = profiles()

    @lru_cache(None)
    def value(key: tuple[str, int] | None, pc: int, pd: int) -> float:
        if key is None:
            return float(dth[pc, pd])
        success = child_key(key)
        s = np.array(
            [1.0 if ch == WIN else -value(success, pd, int(ch)) for ch in p.succ[pc]]
        )
        f = 1.0
        if p.rev[pc]:
            f = (
                p.rev[pc]
                * -value(child_key(key, int(p.st[pc]) + 60), pd, int(p.fail[pc]))
                + 1
                - p.rev[pc]
            )
        return solve_stage(s, f, is_window(key)).value

    return np.array([value(key, int(pc), int(pd)) for pc, pd in pairs])


def lp(s: np.ndarray, f: float, window: bool) -> float:  # noqa: FBT001
    n = 60 + int(window)
    a = np.full((n, 60), f)
    for d in range(60):
        a[d, d:] = s[: 60 - d]
    r = linprog(
        np.r_[np.zeros(n), -1.0],
        A_ub=np.c_[-a.T, np.ones(60)],
        b_ub=np.zeros(60),
        A_eq=[np.r_[np.ones(n), 0.0]],
        b_eq=[1.0],
        bounds=[(0, None)] * n + [(None, None)],
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    assert r.success
    p = np.maximum(r.x[:-1], 0)
    p /= p.sum()
    q = np.maximum(-r.ineqlin.marginals, 0)
    q /= q.sum()
    lo, hi = (p @ a).min(), (a @ q).max()
    assert hi - lo <= 1e-6  # noqa: PLR2004
    return (lo + hi) / 2


def test_clock_matches_engine() -> None:
    from stl.game import Game, Player, Referee  # noqa: PLC0415
    from stl.leap import child_clock, snap  # noqa: PLC0415

    rng = random.Random(381)  # noqa: S311
    for clock in [720, 3420, 3479, 3480, 3539, 3540, 3599, 3600, 3601]:
        for half in (1, 2):
            for fail in (False, True):
                g = Game(Player("Hal"), Player("Baku"), Referee())
                g.seed(381)
                g.game_clock = clock
                g.current_half = half
                _, checker = g.get_roles_for_half(half)
                checker.cylinder = rng.randrange(180)
                q = int(checker.cylinder + 60) if fail else 0
                g.resolve_half_round(60 if fail else 1, 1, survived_outcome=True)
                assert child_clock(half, clock, q) == g.game_clock
        g.game_clock = clock
        g.snap_clock_to_next_minute()
        assert snap(clock) == g.game_clock


def test_leap_window_is_closed_and_turn_duration_agrees() -> None:
    from stl.game import (  # noqa: PLC0415
        LS_WINDOW_END,
        LS_WINDOW_START,
        TURN_DURATION_LEAP,
        TURN_DURATION_NORMAL,
    )
    from stl.leap import is_leap_window, turn_duration  # noqa: PLC0415

    expected = {
        LS_WINDOW_START - 1: False,
        LS_WINDOW_START: True,
        LS_WINDOW_END: True,
        LS_WINDOW_END + 1: False,
    }

    for clock, inside in expected.items():
        assert is_leap_window(clock) is inside
        assert turn_duration(clock) == (
            TURN_DURATION_LEAP if inside else TURN_DURATION_NORMAL
        )


@pytest.mark.parametrize(("limit", "expected"), [(1500, 570612), (1800, 12971150)])
def test_forward_counts(limit: int, expected: int) -> None:
    from stl.leap import forward_reachability  # noqa: PLC0415

    assert sum(forward_reachability(limit=limit).counts.values()) == expected


@pytest.mark.slow
def test_forward_counts_full(tmp_path: Path) -> None:
    from stl.leap import forward_reachability  # noqa: PLC0415

    r = forward_reachability(directory=tmp_path)
    assert sum(r.counts.values()) == 9453333117  # noqa: PLR2004
    assert r.by_type == {"H1": 4180634990, "H2": 4188677815, "REV": 1084020312}
    assert r.window_count == 470336555  # noqa: PLR2004


@pytest.mark.parametrize("window", [False, True])
def test_oracle_vs_lp(window: bool) -> None:  # noqa: FBT001

    rng = np.random.default_rng(593)
    stages = [(rng.uniform(-1, 1, 60), rng.uniform(-1, 1)) for _ in range(30)]
    stages += [(np.linspace(0.8, -0.7, 60), -0.9), (np.ones(60) * 0.3, 0.2)]
    for s, f in stages:
        result = solve_stage(s, f, window)
        assert result.gap <= 1e-6  # noqa: PLR2004
        assert abs(result.value - lp(s, f, window)) <= 1e-8  # noqa: PLR2004


def synthetic(
    n: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(1981)
    pcs = np.arange(n, dtype=np.int32) % 17
    pds = np.arange(n, dtype=np.int32) % 23
    # Monotone payoffs exercise the nonnegative equalizer certificate.
    children = np.sort(rng.uniform(-0.7, 0.7, (23, 61)), axis=1)
    fail = np.full((23, 2), 0.8)
    cols = np.tile(np.arange(60, dtype=np.int32), (17, 1))
    fcols = np.zeros(17, dtype=np.int32)
    rev = np.linspace(0.91, 0.99, 17)
    return pcs, pds, children, fail, cols, fcols, rev


def run_kernel(
    args: Sequence[np.ndarray],
    window: bool = False,  # noqa: FBT001, FBT002
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    out = np.empty(len(args[0]))
    kind = np.empty(len(out), dtype=np.uint8)
    pcs, pds, children, failed_children, cols, fcols, rev = args
    failed = sweep_batch(
        pcs,
        pds,
        children,
        failed_children,
        cols,
        fcols,
        rev,
        window,
        tolerance,
        out,
        kind,
    )
    return out, kind, failed


@pytest.mark.parametrize("n", [1, 15, 16, 17, 1000])
@pytest.mark.parametrize("window", [False, True])
def test_kernel_vs_oracle(n: int, window: bool) -> None:  # noqa: FBT001

    args = synthetic(n)
    out, kind, failed = run_kernel(args, window)
    assert failed == 0
    pc, pd, st, ft, sc, fc, rev = args
    for i in range(n):
        result = solve_stage(
            -st[pd[i], sc[pc[i]]],
            float(rev[pc[i]] * -ft[pd[i], fc[pc[i]]] + 1 - rev[pc[i]]),
            window,
        )
        assert abs(out[i] - result.value) <= 1e-9  # noqa: PLR2004
        assert kind[i] == result.kind


@pytest.mark.parametrize(
    "bad", ["nan", "tolerance", "column", "rev", "shape", "row", "fortran"]
)
def test_kernel_rejects_bad_input(bad: str) -> None:
    args = list(synthetic(17))
    tol = 1e-6
    if bad == "nan":
        args[2][0, 0] = np.nan
    if bad == "tolerance":
        tol = 1e-5
    if bad == "column":
        args[4][0, 0] = 61
    if bad == "rev":
        args[6][0] = 1.01
    if bad == "shape":
        args[1] = args[1][:-1]
    if bad == "row":
        args[1][0] = 23
    if bad == "fortran":
        args[2] = np.asfortranarray(args[2])
    with pytest.raises((ValueError, RuntimeError)):
        run_kernel(args, tolerance=tol)


def test_kernel_reports_failures() -> None:

    rng = np.random.default_rng(492)
    for _ in range(100):
        s = rng.uniform(-1, 1, 60)
        f = float(rng.uniform(-1, 1))
        result = solve_stage(s, f, False, allow_lp=False)  # noqa: FBT003
        if result.kind == 255:  # noqa: PLR2004
            break
    else:
        pytest.fail("seed must exercise the LP rung")
    args = (
        np.zeros(1, np.int32),
        np.zeros(1, np.int32),
        -s[None, :],
        np.array([[-f]]),
        np.arange(60, dtype=np.int32)[None, :],
        np.zeros(1, np.int32),
        np.ones(1),
    )
    out, kind, failed = run_kernel(args)
    assert failed == 1
    assert kind[0] == 255  # noqa: PLR2004
    assert np.isnan(out[0])
    result = solve_stage(s, f, False)  # noqa: FBT003
    assert result.kind == 3  # noqa: PLR2004
    assert result.gap <= 1e-6  # noqa: PLR2004
    assert abs(result.value - lp(s, f, False)) <= 1e-9  # noqa: FBT003, PLR2004


@pytest.fixture
def dth() -> np.ndarray:
    if not DTH_PATH.exists():
        pytest.skip("DTH artifact unavailable")
    return np.load(DTH_PATH, mmap_mode="r")


@pytest.mark.slow
def test_dth_full_parity(dth: np.ndarray) -> None:
    from stl.leap import N, profiles  # noqa: PLC0415

    p = profiles()
    start = time.perf_counter()
    worst = 0.0
    for lo in range(0, N, 64):
        pd = np.repeat(np.arange(lo, min(N, lo + 64), dtype=np.int32), N)
        pc = np.tile(np.arange(N, dtype=np.int32), min(64, N - lo))
        out, _kind, failed = run_kernel((pc, pd, dth, dth, p.succ, p.fail, p.rev))
        assert failed == 0
        worst = max(worst, float(np.max(np.abs(out - dth[pc, pd]))))
    elapsed = time.perf_counter() - start
    print(f"DTH: {N * N / elapsed:.0f} classes/s, max error {worst:.3g}", flush=True)  # noqa: T201
    assert worst <= 1e-9  # noqa: PLR2004


@pytest.mark.parametrize("key", [("H1", 59), ("H2", 56), ("H1", 56)])
def test_exact_dth_keys(key: tuple[str, int], dth: np.ndarray) -> None:
    from stl.leap import N  # noqa: PLC0415

    rng = np.random.default_rng(826)
    pairs = rng.integers(0, N, (200, 2), dtype=np.int32)
    values = evaluate_states(key, pairs, dth)
    assert np.max(np.abs(values - dth[pairs[:, 0], pairs[:, 1]])) <= 1e-9  # noqa: PLR2004


def test_window_states_vs_lp(dth: np.ndarray) -> None:
    from stl.leap import N, profiles  # noqa: PLC0415

    p = profiles()
    rng = np.random.default_rng(296)
    pairs = rng.integers(0, N, (200, 2), dtype=np.int32)
    out = evaluate_states(("H2", 57), pairs, dth)
    for value, (pc, pd) in zip(out, pairs, strict=False):
        s = -dth[pd, p.succ[pc]]
        f = p.rev[pc] * -dth[pd, p.fail[pc]] + 1 - p.rev[pc]
        assert abs(value - lp(s, float(f), True)) <= 1e-8  # noqa: FBT003, PLR2004


def test_subtree_vs_naive(dth: np.ndarray) -> None:
    from stl.leap import N_ALIVE, WIN, child_clock, profiles  # noqa: PLC0415

    p = profiles()

    @lru_cache(None)
    def naive(half: int, clock: int, pc: int, pd: int) -> float:
        if clock > 3600:  # noqa: PLR2004
            return float(dth[pc, pd])
        next_clock = child_clock(half, clock, 0)
        s = np.array(
            [
                1.0 if ch == WIN else -naive(3 - half, next_clock, pd, int(ch))
                for ch in p.succ[pc]
            ]
        )
        f = 1.0
        if p.rev[pc]:
            f -= p.rev[pc] * (
                1
                + naive(
                    3 - half,
                    child_clock(half, clock, int(p.st[pc]) + 60),
                    pd,
                    int(p.fail[pc]),
                )
            )
        return lp(s, f, half == 2 and 3540 <= clock <= 3600)  # noqa: PLR2004

    pair = np.array([[N_ALIVE + 294, N_ALIVE + 294]], dtype=np.int32)
    expected = naive(1, 3180, *map(int, pair[0]))
    assert naive.cache_info().currsize < 20000  # noqa: PLR2004
    assert abs(evaluate_states(("H1", 53), pair, dth)[0] - expected) <= 1e-8  # noqa: PLR2004


def test_store_pack_checkpoint(tmp_path: Path) -> None:
    from stl.leap import N, TableStore  # noqa: PLC0415

    store = TableStore(tmp_path)
    a = store.create(("REV", 3420))
    a[3, 4] = 0.25
    a[8, 9] = -0.4
    a.flush()
    store.pack(("REV", 3420))
    restored = store.load(("REV", 3420))
    assert restored.shape == (N, 184)
    assert restored[3, 4] == 0.25  # noqa: PLR2004
    assert restored[8, 9] == -0.4  # noqa: PLR2004
    assert np.isnan(restored[0, 0])
    assert np.all(restored[:, -1] == -1.0)


def test_kernel_rounding_mode() -> None:
    import ctypes  # noqa: PLC0415
    import platform  # noqa: PLC0415

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("this test uses Darwin ARM64 fenv constants")
    libc = ctypes.CDLL(None)
    original = libc.fegetround()
    try:
        assert libc.fesetround(0x400000) == 0
        with pytest.raises(ValueError, match="floating-point"):
            run_kernel(synthetic(1))
    finally:
        assert libc.fesetround(original) == 0


def test_reachability_cache_hashes(tmp_path: Path) -> None:
    from stl.leap import file_hash, validate_files  # noqa: PLC0415

    path = tmp_path / "reach.npy"
    path.write_bytes(b"reach bitmap")
    hashes = {"reach.npy": file_hash(path)}
    validate_files(tmp_path, hashes)
    path.write_bytes(b"changed bitmap")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_files(tmp_path, hashes)


def test_hot_horizon(tmp_path: Path) -> None:
    from stl.leap import TableStore  # noqa: PLC0415

    store = TableStore(tmp_path)
    a = store.create(("REV", 3600))
    a[0, 0] = 0.75
    a.flush()
    del a
    assert store.path(("REV", 3600)).exists()
    store.pack(("REV", 3600))
    assert not store.path(("REV", 3600)).exists()
    assert store.load(("REV", 3600))[0, 0] == 0.75  # noqa: PLR2004


def test_reachability_source_guard(tmp_path: Path) -> None:
    import json  # noqa: PLC0415

    from stl.leap import _load_reach, forward_reachability  # noqa: PLC0415

    forward_reachability(limit=720, directory=tmp_path)
    assert sum(_load_reach(tmp_path).counts.values()) == 1
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["builder_sha256"] = "stale"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source mismatch"):
        _load_reach(tmp_path)


def test_oracle_recurrence_overflow_uses_lp() -> None:

    s = np.full(60, -0.9)
    s[0] = 2e-6
    failed = solve_stage(s, 0.0, False, allow_lp=False)  # noqa: FBT003
    assert failed.kind == 255  # noqa: PLR2004
    assert np.isnan(failed.value)
    result = solve_stage(s, 0.0, False)  # noqa: FBT003
    assert result.kind == 3  # noqa: PLR2004
    assert result.gap <= 1e-6  # noqa: PLR2004
    assert abs(result.value - lp(s, 0.0, False)) <= 1e-9  # noqa: FBT003, PLR2004
    args = (
        np.zeros(1, np.int32),
        np.zeros(1, np.int32),
        -s[None, :],
        np.zeros((1, 1)),
        np.arange(60, dtype=np.int32)[None, :],
        np.zeros(1, np.int32),
        np.ones(1),
    )
    out, kind, count = run_kernel(args)
    assert count == 1
    assert kind[0] == 255  # noqa: PLR2004
    assert np.isnan(out[0])


def test_scheduler_sends_kernel_failures_to_highs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dth: np.ndarray
) -> None:
    from stl import leap as build  # noqa: PLC0415
    from stl.leap import N, solve_lp  # noqa: PLC0415

    calls = []

    def fail_kernel(*args: Any) -> int:  # noqa: ANN401
        out, kind = args[-2:]
        out[:] = np.nan
        kind[:] = 255
        return len(out)

    def highs(*args: Any) -> StageResult:  # noqa: ANN401
        calls.append(args[2])
        return solve_lp(*args)

    monkeypatch.setattr(build, "sweep_batch", fail_kernel)
    monkeypatch.setattr(build, "solve_lp", highs)
    bits = np.zeros((N, 23), np.uint8)
    bits[0, 0] = 1
    record = build.sweep_key(("REV", 3600), bits, build.TableStore(tmp_path), dth)
    assert calls == [True]
    assert record["states"] == record["failures"] == 1
    assert record["lp_seconds"] > 0


# ──────────────────────────────────────────────────────────────────────
# We check warm HiGHS solves and exact fallback reuse against scalar LPs.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("window", [False, True])
def test_warm_highs_matches_scalar_lp(window: bool) -> None:  # noqa: FBT001
    from stl.leap import WarmLP, solve_lp  # noqa: PLC0415

    solver = WarmLP(window)
    rng = np.random.default_rng(7941)
    for _ in range(40):
        s = rng.uniform(-1, 1, 60)
        f = rng.uniform(-1, 1)
        result = solver.solve(s, f)
        expected = solve_lp(s, f, window)
        assert result.gap <= 1e-6  # noqa: PLR2004
        assert abs(result.value - expected.value) < 1e-8  # noqa: PLR2004
    with pytest.raises(ValueError, match="finite"):
        solver.solve(np.full(60, np.nan), 0.5)


def test_fallback_cache_requires_same_children_and_roles() -> None:
    from stl.leap import RevivalCache  # noqa: PLC0415

    cache = RevivalCache()
    pcs = np.array([0, 1], np.int32)
    pds = np.array([0, 0], np.int32)
    assert np.isnan(cache.get(("H1", 51), ("H1", 54), False, pcs, pds)).all()
    cache.put(("H1", 51), ("H1", 54), False, pcs, pds, np.array([0.1, 0.2]))  # noqa: FBT003
    assert np.array_equal(
        cache.get(("H1", 51), ("H1", 54), False, pcs, pds), [0.1, 0.2]
    )
    assert np.isnan(cache.get(("H1", 51), ("H1", 55), False, pcs, pds)).all()
    assert np.isnan(cache.get(("H1", 51), ("H1", 54), True, pcs, pds)).all()
    assert np.isnan(cache.get(("H1", 52), ("H1", 54), False, pcs, pds)).all()


def test_parallel_fallback_matches_oracle(tmp_path: Path) -> None:
    from stl.leap import FallbackPool, N, profiles, solve_lp  # noqa: PLC0415

    rng = np.random.default_rng(452)
    p = profiles()
    succ = rng.uniform(-1, 1, (2, N + 1))
    fail = rng.uniform(-1, 1, (2, N + 1))
    succ[:, -1] = fail[:, -1] = -1
    np.save(tmp_path / "succ.npy", succ)
    np.save(tmp_path / "fail.npy", fail)
    pcs = np.arange(100, dtype=np.int32)
    pds = np.arange(100, dtype=np.int32) % 2
    with FallbackPool(workers=2) as pool:
        result = pool.solve(
            tmp_path / "succ.npy",
            tmp_path / "fail.npy",
            pcs,
            pds,
            False,  # noqa: FBT003
            False,  # noqa: FBT003
        )
    expected = [
        solve_lp(
            -succ[pd, p.succ[pc]],
            p.rev[pc] * -fail[pd, p.fail[pc]] + 1 - p.rev[pc],
            False,  # noqa: FBT003
        ).value
        for pc, pd in zip(pcs, pds, strict=False)
    ]
    assert np.max(np.abs(cast("np.ndarray", result) - expected)) < 1e-8  # noqa: PLR2004


def test_fallback_reuses_only_certified_identical_stages(tmp_path: Path) -> None:
    from stl.leap import Fallback, N  # noqa: PLC0415

    for name in ("succ", "fail"):
        table = np.lib.format.open_memmap(
            tmp_path / f"{name}.npy", mode="w+", dtype=float, shape=(1, N + 1)
        )
        table[:] = 0.2
        table[:, -1] = -1.0
        table.flush()
    succ = np.load(tmp_path / "succ.npy", mmap_mode="r")
    fail = np.load(tmp_path / "fail.npy", mmap_mode="r")
    pcs = pds = np.array([0], np.int32)
    with Fallback(workers=1) as fallback:
        first, solved, cached = fallback.solve(
            ("H1", 51),
            ("H1", 55),
            succ,
            fail,
            pcs,
            pds,
            False,  # noqa: FBT003
        )
        assert solved == 1
        assert cached == 0
        second, solved, cached = fallback.solve(
            ("H1", 51),
            ("H1", 55),
            succ,
            fail,
            pcs,
            pds,
            False,  # noqa: FBT003
        )
        assert solved == 0
        assert cached == 1
        assert np.array_equal(first, second)


def test_scheduler_routes_failed_classes_to_parallel_highs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stl import leap as b  # noqa: PLC0415
    from stl.leap import Fallback  # noqa: PLC0415

    rng = np.random.default_rng(87)
    child = np.lib.format.open_memmap(
        tmp_path / "child.npy", mode="w+", shape=(1, b.N + 1), dtype=float
    )
    child[:] = rng.uniform(-1, 1, child.shape)
    child[:, -1] = -1.0
    child.flush()
    output = np.lib.format.open_memmap(
        tmp_path / "output.npy", mode="w+", shape=(1, 1), dtype=float
    )
    store = b.TableStore(tmp_path / "tables")
    monkeypatch.setattr(store, "create", lambda key: output)  # noqa: ARG005
    monkeypatch.setattr(store, "load", lambda key: child)  # noqa: ARG005

    def reject(*args: Any) -> int:  # noqa: ANN401
        args[-2][:] = np.nan
        args[-1][:] = 255
        return len(args[0])

    monkeypatch.setattr(b, "sweep_batch", reject)
    bitmap = np.zeros((b.N, (b.N + 7) // 8), np.uint8)
    bitmap[0, 0] = 1
    with Fallback(workers=1) as fallback:
        record = b.sweep_key(
            ("H1", 56),
            bitmap,
            store,
            child,
            min_clock=720,
            full=True,
            fallback=fallback,
        )
    p = b.profiles()
    expected = b.solve_lp(
        -child[0, p.succ[0]],
        p.rev[0] * -child[0, p.fail[0]] + 1 - p.rev[0],
        False,  # noqa: FBT003
    )
    assert abs(output[0, 0] - expected.value) < 1e-8  # noqa: PLR2004
    assert record["failures"] == record["lp_solves"] == 1
    assert record["cached_failures"] == 0


def test_lp_retries_unknown_simplex_status_with_ipm() -> None:
    from stl.leap import solve_lp  # noqa: PLC0415

    # We retain the REV[2933] matrix that triggered a HiGHS Unknown status.
    s = np.array(
        [
            0.5607424418891126,
            0.5655518491845215,
            0.5703557003984117,
            0.5751538917730719,
            0.5799463026905874,
            0.584732834250341,
            0.6483151058350379,
            0.7086023579242888,
            0.47224604612927623,
            0.4767158090516911,
            0.48118325337218537,
            0.4856483277407043,
            0.4901109783322807,
            0.49457114897107246,
            0.4990287813388775,
            0.5034838152362253,
            0.5079361889304526,
            0.512385839581746,
            0.5168327037531836,
            0.5212767180054091,
            0.5257178195730805,
            0.5301559879950309,
            0.5345910960381096,
            0.5390231125559077,
            0.5434519856808143,
            0.5478776693322658,
            0.5523001279516182,
            0.5567193234040602,
            0.5611352345647862,
            0.565547849433565,
            0.5699571654486212,
            0.5743631934830495,
            0.5787659574049986,
            0.5831654953098447,
            0.5875618603403369,
            0.5919551211579275,
            0.596417074977373,
            0.6008109522654486,
            0.6052003990242172,
            0.6095857644792779,
            0.6139671369133468,
            0.6183446727965163,
            0.6227185920631692,
            0.6270891738258304,
            0.631456746960816,
            0.6358216779805593,
            0.6401957524718873,
            0.6445576496959142,
            0.648917109413709,
            0.6532747511938127,
            0.6576311560582244,
            0.661988792656643,
            0.6663442467358592,
            0.6706995720939722,
            0.6750555527746332,
            0.679411789005464,
            0.6837688775376115,
            0.6881268553096092,
            0.692485841342741,
            0.6968458766397327,
        ]
    )
    result = solve_lp(s, 0.5693978099931765, False)  # noqa: FBT003
    assert result.gap <= 1e-8  # noqa: PLR2004
    assert abs(result.value - 0.5693978096030541) < 1e-8  # noqa: PLR2004


def test_audit_retries_unknown_status_with_ipm(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from stl import leap as audit  # noqa: PLC0415

    original = audit.linprog
    methods = []

    def first_unknown(*args: Any, **kwargs: Any) -> OptimizeResult | SimpleNamespace:  # noqa: ANN401
        methods.append(kwargs["method"])
        if len(methods) == 1:
            return SimpleNamespace(success=False, message="Unknown")
        return original(*args, **kwargs)

    monkeypatch.setattr(audit, "linprog", first_unknown)
    result = audit.certify_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert methods == ["highs", "highs-ipm"]
    assert result["gap"] < 1e-8  # noqa: PLR2004
    assert abs(result["value"]) < 1e-8  # noqa: PLR2004


def test_lp_and_audit_retry_centered_matrix() -> None:
    from stl.leap import certify_matrix, solve_lp, stage_matrix  # noqa: PLC0415

    # We retain the H2[46] matrix that defeated both unshifted HiGHS methods.
    s = np.array(
        [
            -0.6599336293341775,
            -0.6598371595049548,
            -0.6596519227840301,
            -0.6595688060954965,
            -0.6594876102392648,
            -0.6593841293334477,
            -0.6593010610142643,
            -0.6592300217172435,
            -0.6591642708311296,
            -0.6591013652732551,
            -0.6590401720591859,
            -0.6589796617506387,
            -0.6576852759956346,
            -0.6576660175796247,
            -0.657647521262862,
            -0.6576198116525441,
            -0.6576045536907242,
            -0.657587172316132,
            -0.6575733314790461,
            -0.6575611009335036,
            -0.6575499418399386,
            -0.6575391689487855,
            -0.6574303903079805,
            -0.65742772235835,
            -0.6574255733427329,
            -0.6574237748632162,
            -0.6574825326479494,
            -0.6574284951528107,
            -0.6574252466275179,
            -0.6574224647918139,
            -0.6574135443965261,
            -0.657412881064743,
            -0.6574113827295205,
            -0.657411064120967,
            -0.6574110056117766,
            -0.6574110112173694,
            -0.6574110407250207,
            -0.6574109953379883,
            -0.6574109935884253,
            -0.657410993581152,
            -0.657410993581152,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
            -0.6574109935811518,
        ]
    )
    f = 0.6619631342243604
    result = solve_lp(s, f, False)  # noqa: FBT003
    audited = certify_matrix(stage_matrix(s, f, False))  # noqa: FBT003
    assert result.gap <= 1e-8  # noqa: PLR2004
    assert audited["gap"] <= 1e-8  # noqa: PLR2004
    assert abs(result.value - (-0.657410993831167)) < 1e-8  # noqa: PLR2004
    assert abs(result.value - audited["value"]) < 1e-8  # noqa: PLR2004


def test_lp_retries_first_success_offset_and_returns_support() -> None:
    from stl.leap import certify_matrix, solve_lp, stage_matrix  # noqa: PLC0415

    s = np.array(
        [
            0.30501604147760514,
            0.3054351558364657,
            0.30585071224070093,
            0.3066373930873052,
            0.3070028766823562,
            0.3076328036693285,
            0.3081612091176772,
            0.30861711515298623,
            0.3090184553403741,
            0.3094632378905324,
            0.3098809265627512,
            0.3102710519555468,
            0.3106331298093312,
            0.29784711454978263,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
            0.31127112835744686,
        ]
    )
    f = 0.9999999999999999
    result = solve_lp(s, f, False, support=True)  # noqa: FBT003
    audit = certify_matrix(stage_matrix(s, f, False))  # noqa: FBT003
    assert result.gap <= 1e-8  # noqa: PLR2004
    assert audit["gap"] <= 1e-8  # noqa: PLR2004
    assert abs(result.value - 0.31127112833305104) < 1e-9  # noqa: PLR2004
    assert result.support is not None
    assert all(0 < x < (1 << 60) for x in result.support)


def test_parallel_pool_returns_seed_supports(tmp_path: Path) -> None:
    from stl.leap import FallbackPool, N, profiles, solve_lp  # noqa: PLC0415

    p = profiles()
    rng = np.random.default_rng(907)
    a = rng.uniform(-0.5, 0.5, (2, N + 1))
    a[:, -1] = -1
    np.save(tmp_path / "a.npy", a)
    pc = np.arange(12, dtype=np.int32)
    pd = pc % 2
    with FallbackPool(workers=2) as pool:
        values, supports = pool.solve(
            tmp_path / "a.npy",
            tmp_path / "a.npy",
            pc,
            pd,
            False,  # noqa: FBT003
            False,  # noqa: FBT003
            support=True,
        )
    assert supports is not None
    assert supports.shape == (len(pc), 2)
    assert supports.dtype == np.uint64
    assert np.all(supports > 0)
    assert np.all(supports < (1 << 60))
    expected = [
        solve_lp(
            -a[d, p.succ[c]],
            p.rev[c] * -a[d, p.fail[c]] + 1 - p.rev[c],
            False,  # noqa: FBT003
        ).value
        for c, d in zip(pc, pd, strict=False)
    ]
    assert np.max(np.abs(cast("np.ndarray", values) - expected)) < 1e-8  # noqa: PLR2004


def test_support_refresh_batches_use_available_workers(tmp_path: Path) -> None:
    from stl.leap import FallbackPool  # noqa: PLC0415

    jobs = []

    class Executor:
        def map(self, function: Callable, submitted: list) -> list:  # noqa: ARG002
            jobs.extend(submitted)
            return [
                (np.zeros(len(job[2])), np.ones((len(job[2]), 2), np.uint64))
                for job in jobs
            ]

    pool = FallbackPool(workers=4)
    pool.executor = Executor()  # pyright: ignore[reportAttributeAccessIssue] - synchronous executor double
    ids = np.arange(4, dtype=np.int32)
    pool.solve(
        tmp_path / "succ.npy",
        tmp_path / "fail.npy",
        ids,
        ids,
        False,  # noqa: FBT003
        False,  # noqa: FBT003
        support=True,
    )
    assert len(jobs) == 4  # noqa: PLR2004


# ──────────────────────────────────────────────────────────────────────
# We test the full sweep and packed-value audit before running the opening.
# ──────────────────────────────────────────────────────────────────────


def test_full_mode_crosses_calibration_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stl import leap as b  # noqa: PLC0415

    path = DTH_PATH
    if not path.exists():
        pytest.skip("DTH artifact unavailable")
    dth = np.load(path, mmap_mode="r")
    store = b.TableStore(tmp_path)
    output = np.lib.format.open_memmap(
        tmp_path / "one.npy", mode="w+", shape=(1, 1), dtype=float
    )
    monkeypatch.setattr(store, "create", lambda key: output)  # noqa: ARG005
    monkeypatch.setattr(store, "load", lambda key: dth)  # noqa: ARG005
    bits = np.zeros((b.N, (b.N + 7) // 8), np.uint8)
    bits[0, 0] = 1
    with pytest.raises(ValueError, match="3420"):
        b.sweep_key(("H1", 56), bits, store, dth, min_clock=720)
    result = b.sweep_key(("H1", 56), bits, store, dth, min_clock=720, full=True)
    assert result["states"] == 1
    assert result["failures"] == 0
    assert abs(output[0, 0] - dth[0, 0]) < 1e-9  # noqa: PLR2004
    with pytest.raises(ValueError, match="720"):
        b.sweep_key(("H1", 11), bits, store, dth, min_clock=660, full=True)


@pytest.fixture
def small_packed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    from stl import leap  # noqa: PLC0415

    a = b = leap
    monkeypatch.setattr(leap, "N", 400)
    key = ("REV", 3600)
    store = b.TableStore(tmp_path / "tables")
    table = store.create(key)
    rng = np.random.default_rng(384)
    mask = rng.random((400, 183)) < 0.17  # noqa: PLR2004
    data = rng.uniform(-1, 1, (400, 183))
    table[:, :183][mask] = data[mask]
    table.flush()
    del table
    store.pack(key)
    return a, b, store, key, mask, data


def test_packed_reader_rank_and_select(small_packed: tuple) -> None:
    a, _b, store, key, mask, data = small_packed
    reader = a.PackedReader(store.directory)
    rows, cols = np.nonzero(mask)
    assert np.array_equal(reader.get(key, rows, cols), data[mask])
    assert np.array_equal(reader.get(key, [0, 399], [183, 183]), [-1.0, -1.0])
    sampled_rows, sampled_cols = reader.sample(key, 200, np.random.default_rng(52))
    assert len(sampled_rows) == 200  # noqa: PLR2004
    assert mask[sampled_rows, sampled_cols].all()
    assert (
        len(set(zip(sampled_rows.tolist(), sampled_cols.tolist(), strict=False))) == 200  # noqa: PLR2004
    )
    assert np.array_equal(
        reader.get(key, sampled_rows, sampled_cols), data[sampled_rows, sampled_cols]
    )
    row, col = np.argwhere(~mask)[0]
    with pytest.raises(ValueError, match="unreachable"):
        reader.get(key, int(row), int(col))
    with pytest.raises(ValueError, match="range"):
        reader.get(key, -1, 0)


def test_packed_audit_checks_membership_and_values(small_packed: tuple) -> None:
    a, _b, store, key, mask, _data = small_packed
    reached = np.packbits(mask, axis=1, bitorder="little")
    result = a.audit_packed_key(store.directory, key, reached, int(mask.sum()))
    assert result["states"] == int(mask.sum())
    changed = mask.copy()
    row, col = np.argwhere(~mask)[0]
    changed[row, col] = True
    with pytest.raises(ValueError, match="reachability"):
        a.audit_packed_key(
            store.directory,
            key,
            np.packbits(changed, axis=1, bitorder="little"),
            int(changed.sum()),
        )
    values = np.load(store.directory / "REV_3600.values.npy", mmap_mode="r+")
    values[0] = np.nan
    values.flush()
    with pytest.raises(ValueError, match="finite"):
        a.audit_packed_key(store.directory, key, reached, int(mask.sum()))


def test_cold_pack_commits_before_dense_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stl import leap as b  # noqa: PLC0415

    monkeypatch.setattr(b, "N", 8)
    store = b.TableStore(tmp_path / "tables")
    key = ("REV", 3600)
    table = store.create(key)
    table[0, 0] = 0.3
    table.flush()
    del table
    files = {"tables/REV_3600.npy": b.file_hash(store.path(key))}
    records = [{"key": list(key), "states": 1}]
    b.write_json(tmp_path / "checkpoint.json", {"files": files})
    old_checkpoint = (tmp_path / "checkpoint.json").read_bytes()
    real_write = b.write_json

    def fail_write(*args: Any) -> None:  # noqa: ANN401, ARG001
        msg = "injected checkpoint failure"
        raise OSError(msg)

    monkeypatch.setattr(b, "write_json", fail_write)
    with pytest.raises(OSError, match="injected"):
        b.commit_full_checkpoint(tmp_path, {}, records, files.copy(), store, 3059, 1.0)
    assert store.path(key).exists()
    assert (tmp_path / "checkpoint.json").read_bytes() == old_checkpoint
    b.validate_files(tmp_path, files)
    monkeypatch.setattr(b, "write_json", real_write)
    b.commit_full_checkpoint(tmp_path, {}, records, files, store, 3059, 1.0)
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    b.validate_files(tmp_path, saved["files"])
    assert "tables/REV_3600.npy" not in saved["files"]
    assert "tables/REV_3600.values.npy" in saved["files"]
    assert not store.path(key).exists()


def test_explicit_lp_certifies_both_policies() -> None:
    from stl.leap import certify_matrix  # noqa: PLC0415

    result = certify_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert abs(result["value"]) < 1e-12  # noqa: PLR2004
    assert result["gap"] <= 1e-6  # noqa: PLR2004
    assert np.allclose(result["drop"], [0.5, 0.5])
    assert np.allclose(result["check"], [0.5, 0.5])


def test_audit_uses_root_roles_and_reset_columns() -> None:
    from stl.leap import stage_inputs  # noqa: PLC0415

    calls = []

    class Reader:
        def get(
            self, key: tuple[str, int], rows: np.ndarray, cols: np.ndarray
        ) -> np.ndarray:
            calls.append((key, np.asarray(rows), np.asarray(cols)))
            return np.full(np.broadcast_arrays(rows, cols)[0].shape, 0.25)

    s, f = stage_inputs(
        cast("PackedReader", Reader()), ("H1", 12), 0, 0, cast("np.ndarray", None)
    )
    assert np.array_equal(s, np.full(60, -0.25))
    assert abs(f - (-0.25 * 0.95 + 0.05)) < 1e-15  # noqa: PLR2004
    assert calls[0][0] == ("H2", 12)
    assert np.array_equal(calls[0][2], np.arange(1, 61))
    assert calls[1][0] == ("REV", 1020)
    assert calls[1][1] == 0
    assert calls[1][2] == 1


def test_full_sweep_groups_compressed_failure_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stl import leap as b  # noqa: PLC0415
    from stl.leap import profiles  # noqa: PLC0415

    path = DTH_PATH
    if not path.exists():
        pytest.skip("DTH artifact unavailable")
    dth = np.load(path, mmap_mode="r")
    p = profiles()
    first = np.full((b.N, 184), 0.1)
    first[:, -1] = -1.0
    second = np.full((b.N, 184), 0.7)
    second[:, -1] = -1.0
    store = b.TableStore(tmp_path)
    output = np.lib.format.open_memmap(
        tmp_path / "out.npy", mode="w+", shape=(2, 1), dtype=float
    )
    monkeypatch.setattr(store, "create", lambda key: output)  # noqa: ARG005
    children = {("H2", 12): dth, ("REV", 1020): first, ("REV", 1021): second}
    monkeypatch.setattr(store, "load", lambda key: children[key])
    bits = np.zeros((b.N, (b.N + 7) // 8), np.uint8)
    bits[:2, 0] = 1
    result = b.sweep_key(("H1", 12), bits, store, dth, min_clock=720, full=True)
    assert result["states"] == 2  # noqa: PLR2004
    for pc, fail_value in [(0, 0.1), (1, 0.7)]:
        expected = solve_stage(
            -dth[0, p.succ[pc]],
            p.rev[pc] * -fail_value + 1 - p.rev[pc],
            False,  # noqa: FBT003
        )
        assert abs(output[pc, 0] - expected.value) < 1e-9  # noqa: PLR2004


# ──────────────────────────────────────────────────────────────────────
# We test open_leap, the verified read path into a completed leap artifact.
# ──────────────────────────────────────────────────────────────────────

needs_artifact = pytest.mark.skipif(
    not (ARTIFACT / "manifest.json").is_file() or not DTH_PATH.is_file(),
    reason="artifacts/leap or artifacts/dth/V.npy is unavailable",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fake_artifact(tmp_path: Path) -> tuple[Path, Path, dict]:
    """A manifest, one listed file, and a DTH tail; the tables stay unread."""
    dth = tmp_path / "V.npy"
    np.save(dth, np.arange(6, dtype=np.float64).reshape(2, 3))
    artifact = tmp_path / "leap"
    artifact.mkdir()
    (artifact / "audit.json").write_text("{}\n")
    manifest = {
        "schema": "stl-leap-values-v1",
        "complete": True,
        "builder_sha256": "b" * 64,
        "dth_sha256": _sha256(dth),
        "files": {"audit.json": _sha256(artifact / "audit.json")},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    return artifact, dth, manifest


def _rewrite(artifact: Path, manifest: dict, **changes: object) -> None:
    (artifact / "manifest.json").write_text(json.dumps({**manifest, **changes}))


def test_open_leap_checks_schema_completion_and_dth_hash(
    fake_artifact: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    from stl.leap import open_leap  # noqa: PLC0415

    artifact, dth, manifest = fake_artifact
    identity = open_leap(artifact, dth=dth).identity
    assert identity.artifact == artifact.resolve()
    assert identity.dth_values == dth.resolve()
    assert identity.builder_sha256 == "b" * 64
    assert identity.dth_sha256 == manifest["dth_sha256"]
    assert identity.schema == "stl-leap-values-v1"
    assert identity.verified == "manifest"
    with pytest.raises(ValueError, match="verify"):
        open_leap(artifact, dth=dth, verify="none")
    with pytest.raises(FileNotFoundError, match="manifest"):
        open_leap(tmp_path / "missing", dth=dth)
    _rewrite(artifact, manifest, schema="stl-leap-values-v0")
    with pytest.raises(ValueError, match="schema"):
        open_leap(artifact, dth=dth)
    _rewrite(artifact, manifest, complete=False)
    with pytest.raises(ValueError, match="complete"):
        open_leap(artifact, dth=dth)
    _rewrite(artifact, manifest, dth_sha256="0" * 64)
    with pytest.raises(ValueError, match="DTH tail"):
        open_leap(artifact, dth=dth)


def test_files_mode_rehashes_every_listed_file(
    fake_artifact: tuple[Path, Path, dict],
) -> None:
    from stl.leap import open_leap  # noqa: PLC0415

    artifact, dth, _ = fake_artifact
    assert open_leap(artifact, dth=dth, verify="files").identity.verified == "files"
    (artifact / "audit.json").write_text('{"changed": true}\n')
    assert open_leap(artifact, dth=dth).identity.verified == "manifest"
    with pytest.raises(ValueError, match=r"audit.json"):
        open_leap(artifact, dth=dth, verify="files")


def test_default_paths_come_from_the_repository() -> None:
    from stl import leap  # noqa: PLC0415

    root = Path(leap.__file__).resolve().parents[2]
    assert root / "artifacts" / "leap" == leap.DEFAULT_ARTIFACT
    assert root / "artifacts" / "dth" / "V.npy" == DEFAULT_PATH
    defaults = inspect.signature(leap.open_leap).parameters
    assert defaults["artifact"].default == leap.DEFAULT_ARTIFACT
    assert defaults["dth"].default == DEFAULT_PATH


def test_profile_and_column_lookups_reject_missing_profiles(
    fake_artifact: tuple[Path, Path, dict],
) -> None:
    from stl.leap import WIN, open_leap, profiles  # noqa: PLC0415

    artifact, dth, _ = fake_artifact
    table = open_leap(artifact, dth=dth)
    p = profiles()
    index = profile_index(st=60, ttd=120)
    assert (p.st[index], p.ttd[index]) == (60, 120)
    with pytest.raises(ValueError, match="no profile"):
        profile_index(st=299, ttd=0)
    with pytest.raises(ValueError, match="no column"):
        table.value(("REV", 3600), 0, index)
    with pytest.raises(ValueError, match="out of range"):
        table.value(("H1", 44), 0, WIN + 1)


def test_stage_enforces_the_1e6_residual_and_gap_gates(
    fake_artifact: tuple[Path, Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive each gate with synthetic certificates.

    A stored 0.0 makes the 1e-6 residual exact.
    """
    from stl.leap import open_leap  # noqa: PLC0415

    artifact, dth, _ = fake_artifact
    table = open_leap(artifact, dth=dth)
    monkeypatch.setattr(table, "value", lambda key, checker, dropper: 0.0)  # noqa: ARG005
    monkeypatch.setattr("stl.leap.stage_inputs", lambda *args: (None, None))  # noqa: ARG005
    monkeypatch.setattr("stl.leap.explicit_matrix", lambda *args: None)  # noqa: ARG005

    def certify(certificate: dict) -> None:
        monkeypatch.setattr("stl.leap.certify_matrix", lambda matrix: dict(certificate))  # noqa: ARG005

    certify({"value": 2e-6, "gap": 0.0})
    with pytest.raises(ValueError, match="Bellman residual"):
        table.stage(("H1", 44), 0, 0)
    certify({"value": 0.0, "gap": 2e-6})
    with pytest.raises(ValueError, match="saddle gap"):
        table.stage(("H1", 44), 0, 0)
    certify({"value": 1e-6, "gap": 1e-6})
    assert table.stage(("H1", 44), 0, 0) == {
        "stored_value": 0.0,
        "bellman_residual": 1e-6,
        "value": 1e-6,
        "gap": 1e-6,
    }

    def lp_failure(matrix: np.ndarray) -> None:  # noqa: ARG001
        msg = "opening/audit LP failed the 1e-6 gate"
        raise RuntimeError(msg)

    monkeypatch.setattr("stl.leap.certify_matrix", lp_failure)
    with pytest.raises(
        ValueError,
        match="H1_44 checker 0 dropper 0: opening/audit LP failed the 1e-6 gate",
    ):
        table.stage(("H1", 44), 0, 0)


@needs_artifact
def test_reader_reproduces_the_h1_strategy_records() -> None:
    """Reproduce the 16 archived Figure 3 records, minutes 44-59, via the reader."""
    from stl.leap import (  # noqa: PLC0415
        DEFAULT_ARTIFACT,
        PackedReader,
        certify_matrix,
        explicit_matrix,
        open_leap,
        stage_inputs,
    )

    table = open_leap()
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    assert table.identity.artifact == DEFAULT_ARTIFACT == ARTIFACT
    assert table.identity.builder_sha256 == manifest["builder_sha256"]
    assert table.identity.dth_sha256 == manifest["dth_sha256"]
    checker = profile_index(st=60, ttd=120)
    dropper = profile_index(st=0, ttd=180)
    # We rebuild each record from the packed tables and the audit LP directly.
    packed = PackedReader(ARTIFACT / "tables")
    dth = np.load(DTH_PATH, mmap_mode="r")
    for minute in range(44, 60):
        key = ("H1", minute)
        record = table.stage(key, checker, dropper)
        stored = float(packed.get(key, checker, dropper))
        certificate = certify_matrix(
            explicit_matrix(*stage_inputs(packed, key, checker, dropper, dth), False)  # noqa: FBT003
        )
        assert record == {
            "stored_value": stored,
            "bellman_residual": abs(stored - certificate["value"]),
            **certificate,
        }
        assert record["bellman_residual"] <= 1e-6  # noqa: PLR2004
        assert record["gap"] <= 1e-6  # noqa: PLR2004
        assert len(record["drop"]) == len(record["check"]) == 60  # noqa: PLR2004
        assert abs(sum(record["drop"]) - 1) < 1e-9  # noqa: PLR2004
        assert abs(sum(record["check"]) - 1) < 1e-9  # noqa: PLR2004


@needs_artifact
def test_reader_maps_revival_columns_and_certifies_window_stages() -> None:
    from stl.leap import PackedReader, is_window, open_leap, profiles  # noqa: PLC0415

    table = open_leap(ARTIFACT, dth=DTH_PATH)
    packed = PackedReader(ARTIFACT / "tables")
    rng = np.random.default_rng(93482)
    for key in (("REV", 3600), ("H2", 58), ("H1", 12)):
        (checker,), (column,) = packed.sample(key, 1, rng)
        dropper = int(profiles().s0[column]) if key[0] == "REV" else int(column)
        assert table.value(key, int(checker), dropper) == float(
            packed.get(key, checker, column)
        )
        record = table.stage(key, int(checker), dropper)
        assert len(record["drop"]) == 60 + int(is_window(key))
        assert record["bellman_residual"] <= 1e-6  # noqa: PLR2004
        assert record["gap"] <= 1e-6  # noqa: PLR2004
