# Copyright (c) 2026 James Cenawood.
"""Build, audit and read the certified leap table of the public STL game.

The paper ``stl.tex`` reports this table. ``run_full`` sweeps every reachable
state back to the clock-720 opening and audits the result; ``open_leap`` is the
one read path into a completed artifact. The numba sweep lives in
``leap_kernel.py``. The LP code here certifies single stages. The independent scalar
oracle in tests/conftest.py checks the kernel. ``docs/LEAP.md``
holds the certificate equations.

Run ``python -m stl leap [--dth artifacts/dth/V.npy] [--output artifacts/leap]``
to build the table in an empty output directory. The completed report records
coverage and numerical audit results.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import resource
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Self, cast

import highspy
import numpy as np
from scipy.optimize import linprog

from stl.dth import DEFAULT_PATH, Table
from stl.game import (
    LS_WINDOW_END,
    LS_WINDOW_START,
    TURN_DURATION_LEAP,
    TURN_DURATION_NORMAL,
    Player,
    Referee,
)
from stl.leap_kernel import solve_packing, sweep_batch

__all__ = ["LeapTable", "open_leap", "profile_index"]

type Key = tuple[str, int]
type PathLike = str | os.PathLike[str]

# The completed leap artifact (82 GB, gitignored).
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "leap"


# ────────────────────────────────────────────────────────────
# Quotient profiles and clocks
# ────────────────────────────────────────────────────────────


def is_leap_window(clock: float) -> bool:
    """Return whether a half-round starting at ``clock`` spans second 61."""

    return LS_WINDOW_START <= clock <= LS_WINDOW_END


def turn_duration(clock: float) -> int:
    """Return the number of literal action seconds at ``clock``."""

    return TURN_DURATION_LEAP if is_leap_window(clock) else TURN_DURATION_NORMAL


N_ALIVE = 16_711


N = 17_011


WIN = N


def snap(clock: int) -> int:
    if clock < 3600:
        next_minute = (clock // 60 + 1) * 60
        return 3601 if next_minute == 3600 else next_minute
    return 3601 if clock == 3600 else 3601 + ((clock - 3601) // 60 + 1) * 60


def child_clock(half: int, clock: int, q: int = 0) -> int:
    if half not in (1, 2) or q < 0:
        message = "invalid half or dose"
        raise ValueError(message)
    end = clock + turn_duration(clock) + (q + 120 if q else 0)
    return end + 60 if half == 1 else snap(end)


def eligible(s: int, t: int) -> bool:
    return s + 60 < 300 and s + t + 60 <= 300


@dataclass(frozen=True)
class Profiles:
    st: np.ndarray
    ttd: np.ndarray
    succ: np.ndarray
    fail: np.ndarray
    rev: np.ndarray
    s0: np.ndarray
    idx0: np.ndarray
    blocks: tuple


@lru_cache(None)
def profiles() -> Profiles:
    ids = {}
    blocks = []
    st = []
    ttd = []
    for t in [0, *range(60, 301)]:
        start = len(st)
        for s in range(300):
            if eligible(s, t):
                ids[s, t] = len(st)
                st.append(s)
                ttd.append(t)
        if len(st) > start:
            blocks.append((start, len(st)))
    if len(st) != N_ALIVE:
        message = "numerical invariant failed"
        raise AssertionError(message)
    st.extend(range(300))
    ttd.extend([301] * 300)
    if len(st) != N:
        message = "numerical invariant failed"
        raise AssertionError(message)
    succ = np.empty((N, 60), np.int32)
    fail = np.full(N, WIN, np.int32)
    rev = np.zeros(N)
    referee = Referee()
    for pc, (s, t) in enumerate(zip(st, ttd, strict=True)):
        for lag in range(1, 61):
            grown = s + lag
            succ[pc, lag - 1] = (
                WIN if grown >= 300 else ids.get((grown, t), N_ALIVE + grown)
            )
        if pc < N_ALIVE:
            fail[pc] = ids.get((0, s + t + 60), N_ALIVE)
            rev[pc] = referee.compute_survival_probability(
                Player("Checker", ttd=t), s + 60
            )
    s0 = np.array([ids[0, t] for t in [0, *range(60, 241)]] + [N_ALIVE], np.int32)
    if len(s0) != 183:
        message = "numerical invariant failed"
        raise AssertionError(message)
    idx0 = np.full(N + 1, -1, np.int32)
    idx0[s0] = np.arange(183)
    idx0[WIN] = 183
    return Profiles(
        np.array(st, np.int32),
        np.array(ttd, np.int32),
        succ,
        fail,
        rev,
        s0,
        idx0,
        tuple(blocks),
    )


# ────────────────────────────────────────────────────────────
# Scalar stage oracle
# ────────────────────────────────────────────────────────────

GAP = 1e-6


@dataclass(frozen=True)
class StageResult:
    value: float
    kind: int
    lower: float
    upper: float
    support: tuple[int, int] | None = None

    @property
    def gap(self) -> float:
        return self.upper - self.lower


def stage_matrix(s: np.ndarray | list[float], f: float, window: bool) -> np.ndarray:
    matrix = np.full((60 + int(window), 60), f, dtype=np.float64)
    for row in range(60):
        matrix[row, row:] = s[: 60 - row]
    return matrix


def policy_masks(p: np.ndarray, q: np.ndarray) -> tuple[int, int]:
    return cast(
        "tuple[int, int]",
        tuple(
            int.from_bytes(
                np.packbits(np.asarray(x) > 0, bitorder="little").tobytes(), "little"
            )
            for x in (p, q)
        ),
    )


def solve_lp(
    s: np.ndarray | list[float],
    f: float,
    window: bool,
    *,
    support: bool = False,
) -> StageResult:
    matrix = stage_matrix(s, f, window)
    rows = len(matrix)
    failure = "HiGHS failed the 1e-6 saddle gate"
    # We retain the successful path of each prior retry. A common offset
    # changes the LP coordinates; we certify against the original matrix.
    attempts = [
        ("highs", None),
        ("highs-ipm", None),
        ("highs-ipm", f),
        ("highs-ipm", float(matrix[0, 0])),
    ]
    for method, offset in attempts:
        lp_matrix = matrix if offset is None else matrix - offset
        result = linprog(
            np.r_[np.zeros(rows), -1.0],
            A_ub=np.c_[-lp_matrix.T, np.ones(60)],
            b_ub=np.zeros(60),
            A_eq=[np.r_[np.ones(rows), 0.0]],
            b_eq=[1.0],
            bounds=[(0, None)] * rows + [(None, None)],
            method=method,
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            failure = f"HiGHS failed: {result.message}"
            continue
        p = np.maximum(result.x[:-1], 0)
        q = np.maximum(-result.ineqlin.marginals, 0)
        p /= p.sum()
        q /= q.sum()
        lower = float((p @ matrix).min())
        upper = float((matrix @ q).max())
        if not math.isfinite(upper - lower) or upper - lower > GAP:
            failure = "HiGHS failed the 1e-6 saddle gate"
            continue
        return StageResult(
            (lower + upper) / 2,
            3,
            lower,
            upper,
            policy_masks(p, q) if support else None,
        )
    raise RuntimeError(failure)


# ────────────────────────────────────────────────────────────
# Reachability, value storage and the full sweep
# ────────────────────────────────────────────────────────────

POP = np.array([i.bit_count() for i in range(256)], np.uint8)


TOTALS = {"H1": 4180634990, "H2": 4188677815, "REV": 1084020312}


# The full sweep's residue path. Each option keeps the 1e-6 full-matrix gate.
# benchmark_leap_replay selected one kink attempt; more attempts ran slower.
RESIDUE = {"crash": True, "kernel_native": True, "kink_attempts": 1}


def key_clock(key: Key) -> int:
    kind, index = key
    return index if kind == "REV" else 60 * index + (120 if kind == "H2" else 0)


def key_name(key: Key) -> str:
    return f"{key[0]}_{key[1]}"


def key_columns(key: Key) -> int:
    return 183 if key[0] == "REV" else N


def all_keys() -> list[Key]:
    return sorted(
        [("H1", m) for m in range(12, 60)]
        + [("H2", m) for m in range(12, 59)]
        + [("REV", c) for c in range(1020, 3601)],
        key=key_clock,
    )


def child_key(key: Key, q: int = 0) -> Key | None:
    half = 1 if key[0] == "H1" else 2
    clock = child_clock(half, key_clock(key), q)
    if clock > 3600:
        return None
    if half == 1:
        return ("REV", clock) if q else ("H2", key[1])
    return ("H1", clock // 60)


def is_window(key: Key) -> bool:
    return key[0] != "H1" and 3540 <= key_clock(key) <= 3600


def _transpose(bits: np.ndarray, columns: int) -> np.ndarray:
    return np.packbits(
        np.unpackbits(bits, axis=1, count=columns, bitorder="little").T,
        axis=1,
        bitorder="little",
    )


def _success_bits(bits: np.ndarray) -> np.ndarray:
    p = profiles()
    target = np.zeros_like(bits)
    for start, stop in (*p.blocks, (N_ALIVE, N)):
        length = stop - start
        expanded = np.zeros((min(300, length + 60), bits.shape[1]), np.uint8)
        expanded[1 : min(length + 1, 300)] = bits[start : start + min(length, 299)]
        for shift in (1, 2, 4, 8, 16, 28):
            expanded[shift:] |= expanded[:-shift]
        if start == N_ALIVE:
            target[N_ALIVE:] |= expanded
        else:
            target[start:stop] |= expanded[:length]
            target[N_ALIVE + length : N_ALIVE + len(expanded)] |= expanded[length:]
    return target


@dataclass
class Reachability:
    counts: dict
    directory: Path | None

    @property
    def by_type(self) -> dict[str, int]:
        return {
            kind: sum(v for k, v in self.counts.items() if k[0] == kind)
            for kind in TOTALS
        }

    @property
    def window_count(self) -> int:
        return sum(v for k, v in self.counts.items() if is_window(k))

    def load(self, key: Key) -> np.memmap:
        return np.load(
            cast("Path", self.directory) / f"{key_name(key)}.npy", mmap_mode="r"
        )


def forward_reachability(
    limit: int = 3600, directory: PathLike | None = None
) -> Reachability:
    """Union successor sets before counting each quotient state once."""
    if limit > 3600:
        message = "reachability ends at 3600"
        raise ValueError(message)
    directory = Path(directory) if directory is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    p = profiles()
    pending = {}
    counts = {}

    def target(key: Key) -> np.ndarray:
        if key not in pending:
            pending[key] = np.zeros((N, (key_columns(key) + 7) // 8), np.uint8)
        return pending[key]

    target(("H1", 12))[0, 0] = 1
    for key in all_keys():
        if key_clock(key) > limit:
            break
        if key not in pending:
            continue
        bits = pending.pop(key)
        count = int(POP[bits].sum())
        if not count:
            continue
        counts[key] = count
        if directory is not None:
            np.save(directory / f"{key_name(key)}.npy", bits)
        ncols = key_columns(key)
        rows = p.s0 if key[0] == "REV" else slice(None)
        success = child_key(key)
        if success is not None and key_clock(success) <= limit:
            target(success)[rows] |= _transpose(_success_bits(bits), ncols)
        # We group by the destination clock before merging revival paths.
        groups = {}
        for pc in np.flatnonzero(np.any(bits[:N_ALIVE], axis=1)):
            dest = child_key(key, int(p.st[pc]) + 60)
            if dest is not None and key_clock(dest) <= limit:
                groups.setdefault(dest, []).append(int(pc))
        for dest, pcs in groups.items():
            merged = np.zeros((183, bits.shape[1]), np.uint8)
            np.bitwise_or.at(merged, p.idx0[p.fail[pcs]], bits[pcs])
            transposed = _transpose(merged, ncols)
            if dest[0] == "REV":
                target(dest)[rows] |= transposed
            else:
                # We place the 183 reset-profile columns in their full profile
                # positions.
                dense = np.unpackbits(transposed, axis=1, count=183, bitorder="little")
                dest_bits = target(dest)
                for j, col in enumerate(p.s0):
                    dest_bits[rows, col // 8] |= dense[:, j] << int(col % 8)
        if directory is not None and key[0] == "H1":
            print(f"reach {key_name(key)}: {count:,}", flush=True)
    return Reachability(counts, directory)


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class TableStore:
    def __init__(self, directory: PathLike) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.write_bytes = 0
        self.write_seconds = 0.0
        self.pack_bytes = 0
        self.pack_seconds = 0.0

    def path(self, key: Key) -> Path:
        return self.directory / f"{key_name(key)}.npy"

    def create(self, key: Key) -> np.memmap:
        start = time.perf_counter()
        table = np.lib.format.open_memmap(
            self.path(key), mode="w+", dtype=np.float64, shape=(N, key_columns(key) + 1)
        )
        table[:] = np.nan
        table[:, -1] = -1.0
        table.flush()
        self.write_seconds += time.perf_counter() - start
        self.write_bytes += table.nbytes
        return table

    def pack(self, key: Key) -> None:
        start = time.perf_counter()
        table = np.load(self.path(key), mmap_mode="r")
        cols = key_columns(key)
        count = sum(
            int(np.isfinite(table[lo : lo + 64, :cols]).sum()) for lo in range(0, N, 64)
        )
        prefix = self.directory / key_name(key)
        bitmap_path = prefix.with_suffix(".bits.npy")
        values_path = prefix.with_suffix(".values.npy")
        bitmap = np.lib.format.open_memmap(
            bitmap_path, mode="w+", dtype=np.uint8, shape=((N * cols + 7) // 8,)
        )
        values = np.lib.format.open_memmap(
            values_path, mode="w+", dtype=np.float64, shape=(count,)
        )
        offset = 0
        for lo in range(0, N, 64):
            block = table[lo : lo + 64, :cols]
            mask = np.isfinite(block)
            packed = np.packbits(mask.reshape(-1), bitorder="little")
            bitmap[lo * cols // 8 : lo * cols // 8 + len(packed)] = packed
            data = block[mask]
            values[offset : offset + len(data)] = data
            offset += len(data)
        bitmap.flush()
        values.flush()
        self.pack_bytes += bitmap.nbytes + values.nbytes
        self.pack_seconds += time.perf_counter() - start
        del bitmap, values, table
        self.path(key).unlink()

    def pack_after(self, clock: int) -> None:
        """Pack dense tables beyond the remaining sweep's dependency horizon."""
        for path in self.directory.glob("*.npy"):
            if "." in path.stem:
                continue
            kind, index = path.stem.split("_")
            key = (kind, int(index))
            if key_clock(key) > clock:
                self.pack(key)

    def load(self, key: Key) -> np.memmap:
        if self.path(key).exists():
            return np.load(self.path(key), mmap_mode="r")
        prefix = self.directory / key_name(key)
        bitmap = np.load(prefix.with_suffix(".bits.npy"), mmap_mode="r")
        values = np.load(prefix.with_suffix(".values.npy"), mmap_mode="r")
        table = self.create(key)
        cols = key_columns(key)
        offset = 0
        for lo in range(0, N, 64):
            rows = min(64, N - lo)
            length = rows * cols
            mask = (
                np.unpackbits(
                    bitmap[lo * cols // 8 : (lo * cols + length + 7) // 8],
                    count=length,
                    bitorder="little",
                )
                .reshape(rows, cols)
                .astype(bool)
            )
            count = int(mask.sum())
            table[lo : lo + rows, :cols][mask] = values[offset : offset + count]
            offset += count
        if offset != len(values):
            message = "packed value count mismatch"
            raise ValueError(message)
        table.flush()
        return table


def sweep_key(
    key: Key,
    bitmap: np.ndarray,
    store: TableStore,
    dth: np.ndarray,
    *,
    min_clock: int = 3420,
    full: bool = False,
    fallback: Fallback | None = None,
) -> dict:
    floor = 720 if full else 3420
    if min_clock < floor or key_clock(key) < min_clock:
        message = f"key sweeps must stop at clock {floor}"
        raise ValueError(message)
    p = profiles()
    table = store.create(key)
    cols = key_columns(key)
    success_key = child_key(key)
    success_table = dth if success_key is None else store.load(success_key)
    full_pds = p.s0 if key[0] == "REV" else np.arange(N, dtype=np.int32)
    states = failures = lp_solves = cached_failures = 0
    kernel_seconds = lp_seconds = 0.0
    kernel_native = support_solves = 0
    # The kernel solves H1 and H2 residue itself when the fallback asks for it.
    # Its LP time then counts in kernel_seconds, so compare residue paths on
    # sweep time; native_solves and native_seconds cover the Python path only.
    options = (
        cast("NativeFallback", fallback).kernel_options(key)
        if hasattr(fallback, "kernel_options")
        else {}
    )
    native_before = getattr(fallback, "native_solves", 0)
    native_time_before = getattr(fallback, "native_seconds", 0.0)
    highs_before = getattr(fallback, "highs_seconds", None)
    groups = {}
    for checker in np.flatnonzero(np.any(bitmap, axis=1)):
        dest = None if not p.rev[checker] else child_key(key, int(p.st[checker]) + 60)
        groups.setdefault(dest, []).append(int(checker))
    # We group by child table before batching to avoid repeated map validation
    # for small fragments of the same revival group. We retain Dropper row order.
    for dest, checker_group in groups.items():
        checkers = np.array(checker_group, np.int32)
        reached = np.unpackbits(
            bitmap[checkers], axis=1, count=cols, bitorder="little"
        ).T
        fail_table = dth if dest is None else store.load(dest)
        fail_col = p.idx0[p.fail] if dest is not None and dest[0] == "REV" else p.fail
        rows_per_batch = max(64, min(cols, 1_048_576 // len(checkers)))
        for start in range(0, cols, rows_per_batch):
            js, local_pc = np.nonzero(reached[start : start + rows_per_batch])
            js += start
            if not len(js):
                continue
            pcs = np.ascontiguousarray(checkers[local_pc])
            pds = np.ascontiguousarray(full_pds[js], dtype=np.int32)
            out = np.empty(len(pcs))
            kind = np.empty(len(pcs), np.uint8)
            tick = time.perf_counter()
            failed = sweep_batch(
                pcs,
                pds,
                success_table,
                fail_table,
                p.succ,
                fail_col,
                p.rev,
                is_window(key),
                1e-6,
                out,
                kind,
                **options,
            )
            kernel_seconds += time.perf_counter() - tick
            if options:
                # Kinds 3, 4, and 5 are residue that the kernel certified itself.
                solved = int(np.count_nonzero(kind == 3))
                guessed = int(np.count_nonzero((kind == 4) | (kind == 5)))
                kernel_native += solved
                support_solves += guessed
                failures += solved + guessed
            failures += failed
            if failed:
                tick = time.perf_counter()
                indices = np.flatnonzero(kind == 255)
                if fallback is None:
                    for i in indices:
                        checker, dropper = int(pcs[i]), int(pds[i])
                        successes = -success_table[dropper, p.succ[checker]]
                        failure = (
                            p.rev[checker] * -fail_table[dropper, fail_col[checker]]
                            + 1
                            - p.rev[checker]
                        )
                        out[i] = solve_lp(successes, failure, is_window(key)).value
                    lp_solves += failed
                else:
                    out[indices], solved, cached = fallback.solve(
                        success_key,
                        dest,
                        cast("np.memmap", success_table),
                        cast("np.memmap", fail_table),
                        pcs[indices],
                        pds[indices],
                        is_window(key),
                    )
                    lp_solves += solved
                    cached_failures += cached
                    if failed > 1000:
                        native = failed - solved - cached
                        print(
                            f"fallback {key_name(key)}: {native:,} native, {solved:,} HiGHS, {cached:,} cached",  # noqa: E501 -- Preserve the diagnostic text.
                            flush=True,
                        )
                lp_seconds += time.perf_counter() - tick
            if not np.isfinite(out).all():
                message = "uncertified class aborts the build"
                raise RuntimeError(message)
            tick = time.perf_counter()
            table[pcs, js] = out
            store.write_seconds += time.perf_counter() - tick
            states += len(pcs)
    tick = time.perf_counter()
    table.flush()
    store.write_seconds += time.perf_counter() - tick
    if states != int(POP[bitmap].sum()):
        message = "sweep missed reachable classes"
        raise RuntimeError(message)
    return {
        "key": list(key),
        "clock": key_clock(key),
        "states": states,
        "kernel_seconds": kernel_seconds,
        "classes_per_second": states / kernel_seconds,
        "failures": failures,
        "fallback_seconds": lp_seconds,
        "lp_seconds": lp_seconds
        if highs_before is None
        else cast("NativeFallback", fallback).highs_seconds - highs_before,
        "native_solves": getattr(fallback, "native_solves", 0) - native_before,
        "native_seconds": getattr(fallback, "native_seconds", 0.0) - native_time_before,
        "kernel_native_solves": kernel_native,
        "support_solves": support_solves,
        "lp_solves": lp_solves,
        "cached_failures": cached_failures,
    }


def _rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def run_full(dth_path: PathLike, directory: PathLike) -> dict:
    """We sweep the public game to its clock-720 opening and audit the result."""
    started = time.perf_counter()
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        message = f"{directory}: leap builds require an empty output directory"
        raise FileExistsError(message)
    dth = Table.open(Path(dth_path)).values
    if not np.all(dth[:, -1] == -1):
        message = "DTH table sentinel mismatch"
        raise ValueError(message)
    store = TableStore(directory / "tables")
    records = []
    reach = forward_reachability(directory=directory / "reachability")
    if reach.by_type != TOTALS or reach.window_count != 470336555:
        message = "full reachability count mismatch"
        raise RuntimeError(message)
    with NativeFallback(**RESIDUE) as fallback:
        for key in sorted(reach.counts, key=key_clock, reverse=True):
            tick = time.perf_counter()
            record = sweep_key(
                key,
                reach.load(key),
                store,
                dth,
                min_clock=720,
                full=True,
                fallback=fallback,
            )
            record["sweep_seconds"] = time.perf_counter() - tick
            records.append(record)
            store.pack_after(key_clock(key) + 540)
            print(json.dumps(record), flush=True)
    store.pack_after(0)
    if sum(r["states"] for r in records) != sum(TOTALS.values()) or len(records) != len(
        reach.counts
    ):
        message = "full sweep coverage mismatch"
        raise RuntimeError(message)
    print(
        "Sweep reached the opening; starting packed coverage and Bellman audit.",
        flush=True,
    )
    audit_started = time.perf_counter()
    audit = audit_full(directory, reach, records, dth)
    write_json(directory / "audit.json", audit)
    rows = {}
    for kind in TOTALS:
        selected = [r for r in records if r["key"][0] == kind]
        seconds = sum(r["kernel_seconds"] for r in selected)
        rows[kind] = {
            "states": sum(r["states"] for r in selected),
            "kernel_seconds": seconds,
            "classes_per_second": sum(r["states"] for r in selected) / seconds,
            "failures": sum(r["failures"] for r in selected),
        }
    report = {
        "root": audit["root"],
        "rows": rows,
        "states": sum(TOTALS.values()),
        "keys": len(records),
        "failures": sum(r["failures"] for r in records),
        "lp_solves": sum(r["lp_solves"] for r in records),
        "native_solves": sum(r.get("native_solves", 0) for r in records),
        "kernel_native_solves": sum(r.get("kernel_native_solves", 0) for r in records),
        "support_solves": sum(r.get("support_solves", 0) for r in records),
        "native_seconds": sum(r.get("native_seconds", 0.0) for r in records),
        "fallback_seconds": sum(
            r.get("fallback_seconds", r["lp_seconds"]) for r in records
        ),
        "cached_failures": sum(r["cached_failures"] for r in records),
        "kernel_seconds": sum(r["kernel_seconds"] for r in records),
        "lp_seconds": sum(r["lp_seconds"] for r in records),
        "wall_seconds": time.perf_counter() - started,
        "audit_seconds": time.perf_counter() - audit_started,
        "peak_rss_bytes": _rss_bytes(),
        "memmap_write_MB_s": store.write_bytes / store.write_seconds / 1e6,
        "pack_MB_s": store.pack_bytes / store.pack_seconds / 1e6,
        "lp_audit_samples": audit["lp_samples"],
        "max_bellman_residual": audit["max_bellman_residual"],
        "changed_states": audit["changed_states"],
        "mean_baku_gain_over_state_classes": audit["mean_baku_gain_over_state_classes"],
        "max_baku_gain": audit["max_baku_gain"],
        "min_baku_gain": audit["min_baku_gain"],
    }
    write_json(directory / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


# ────────────────────────────────────────────────────────────
# HiGHS fallback for kernel residue
# ────────────────────────────────────────────────────────────


class WarmLP:
    """We solve each explicit matrix with HiGHS, then check both saddle bounds."""

    def __init__(self, window: bool) -> None:
        self.window = window
        self.basis = None
        self.highs = highspy.Highs()
        for name, value in [
            ("output_flag", False),
            ("threads", 1),
            ("solver", "simplex"),
            ("presolve", "off"),
            ("primal_feasibility_tolerance", 1e-9),
            ("dual_feasibility_tolerance", 1e-9),
        ]:
            if self.highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                message = f"HiGHS rejected option {name}"
                raise RuntimeError(message)
        rows = 60 + int(window)
        self.lp = highspy.HighsLp()
        self.lp.num_col_ = rows + 1
        self.lp.num_row_ = 61
        self.lp.col_cost_ = np.r_[np.zeros(rows), -1.0]
        self.lp.col_lower_ = np.r_[np.zeros(rows), -np.inf]
        self.lp.col_upper_ = np.full(rows + 1, np.inf)
        self.lp.row_lower_ = np.r_[np.full(60, -np.inf), 1.0]
        self.lp.row_upper_ = np.r_[np.zeros(60), 1.0]
        self.lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        self.lp.a_matrix_.start_ = np.arange(rows + 2, dtype=np.int32) * 61
        self.lp.a_matrix_.index_ = np.tile(np.arange(61, dtype=np.int32), rows + 1)

    def solve(
        self, s: np.ndarray | list[float], f: float, *, support: bool = False
    ) -> StageResult:
        s = np.asarray(s, np.float64)
        if s.shape != (60,) or not np.isfinite(s).all() or not np.isfinite(f):
            message = "stage needs finite payoffs"
            raise ValueError(message)
        matrix = stage_matrix(s, f, self.window)
        coefficients = np.empty((len(matrix) + 1, 61))
        coefficients[:-1, :60] = -matrix
        coefficients[:-1, 60] = 1.0
        coefficients[-1, :60] = 1.0
        coefficients[-1, 60] = 0.0
        self.lp.a_matrix_.value_ = coefficients.ravel()
        status = self.highs.passModel(self.lp)
        if status == highspy.HighsStatus.kError:
            message = "HiGHS rejected the explicit stage matrix"
            raise RuntimeError(message)
        if (
            self.basis is not None
            and self.highs.setBasis(self.basis) == highspy.HighsStatus.kError
        ):
            message = "HiGHS rejected its saved basis"
            raise RuntimeError(message)
        self.highs.run()
        if self.highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            self.basis = None
            return solve_lp(s, f, self.window, support=support)
        solution = self.highs.getSolution()
        p = np.maximum(solution.col_value[:-1], 0.0)
        q = np.maximum(-np.asarray(solution.row_dual[:60]), 0.0)
        p /= p.sum()
        q /= q.sum()
        lower = float(np.min(p @ matrix))
        upper = float(np.max(matrix @ q))
        if not np.isfinite(upper - lower) or upper - lower > 1e-8:
            # We discard an inaccurate warm basis and use the scalar LP authority.
            self.basis = None
            return solve_lp(s, f, self.window, support=support)
        self.basis = self.highs.getBasis()
        return StageResult(
            (lower + upper) / 2,
            3,
            lower,
            upper,
            policy_masks(p, q) if support else None,
        )


class RevivalCache:
    """We reuse a certified value only for identical child tables and profiles."""

    def __init__(self) -> None:
        self.groups = OrderedDict()
        self.tokens = {}

    def _token(self, fail: Key | None) -> int:
        return self.tokens.setdefault(fail, len(self.tokens))

    def get(
        self,
        success: Key | None,
        fail: Key | None,
        window: bool,
        pcs: np.ndarray,
        pds: np.ndarray,
    ) -> np.ndarray:
        output = np.full(len(pcs), np.nan)
        key = (success, window)
        if key not in self.groups:
            return output
        self.groups.move_to_end(key)
        values, tokens = self.groups[key]
        cols = profiles().idx0[pds]
        eligible = (cols >= 0) & (cols < 183)
        indices = np.flatnonzero(eligible)
        indices = indices[tokens[pcs[indices], cols[indices]] == self._token(fail)]
        output[indices] = values[pcs[indices], cols[indices]]
        return output

    def put(
        self,
        success: Key | None,
        fail: Key | None,
        window: bool,
        pcs: np.ndarray,
        pds: np.ndarray,
        output: np.ndarray,
    ) -> None:
        cols = profiles().idx0[pds]
        eligible = (cols >= 0) & (cols < 183)
        if not eligible.any():
            return
        if not np.isfinite(output).all():
            message = "cache requires finite certified values"
            raise ValueError(message)
        key = (success, window)
        if key not in self.groups:
            if len(self.groups) == 2:
                self.groups.popitem(last=False)
            self.groups[key] = (
                np.full((N, 183), np.nan),
                np.full((N, 183), -1, np.int32),
            )
        values, tokens = self.groups[key]
        values[pcs[eligible], cols[eligible]] = output[eligible]
        tokens[pcs[eligible], cols[eligible]] = self._token(fail)


@lru_cache(maxsize=2)
def _table(path: PathLike) -> np.memmap:
    return np.load(path, mmap_mode="r")


@lru_cache(maxsize=2)
def _solver(window: bool) -> WarmLP:
    return WarmLP(window)


def _solve_chunk(
    job: tuple[str, str, np.ndarray, np.ndarray, bool, bool, bool],
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
    succ_path, fail_path, pcs, pds, window, compressed, support = job
    succ = _table(succ_path)
    fail = _table(fail_path)
    p = profiles()
    solver = _solver(False if support else window)
    result = np.empty(len(pcs))
    masks = np.zeros((len(pcs), 2), np.uint64) if support else None
    for i, (pc, pd) in enumerate(zip(pcs, pds, strict=True)):
        col = p.idx0[p.fail[pc]] if compressed else p.fail[pc]
        s = -succ[pd, p.succ[pc]]
        f = p.rev[pc] * -fail[pd, col] + 1 - p.rev[pc]
        solved = solver.solve(s, cast("float", f), support=support)
        result[i] = max(solved.value, f) if support and window else solved.value
        if masks is not None:
            masks[i] = cast("tuple[int, int]", solved.support)
    return (result, masks) if support else result


class FallbackPool:
    """We send independent LP batches to processes with one HiGHS thread each."""

    def __init__(self, workers: int | None = None) -> None:
        self.workers = workers or min(16, os.cpu_count() or 1)
        self.executor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def solve(
        self,
        succ_path: PathLike,
        fail_path: PathLike,
        pcs: np.ndarray,
        pds: np.ndarray,
        window: bool,
        compressed: bool,
        *,
        support: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray | None]:
        if self.executor is None:
            self.executor = ProcessPoolExecutor(
                self.workers, mp_context=multiprocessing.get_context("spawn")
            )
        chunk = max(1 if support else 32, min(2048, len(pcs) // self.workers))
        jobs = [
            (
                str(succ_path),
                str(fail_path),
                pcs[i : i + chunk],
                pds[i : i + chunk],
                window,
                compressed,
                support,
            )
            for i in range(0, len(pcs), chunk)
        ]
        output = np.empty(len(pcs))
        offset = 0
        masks = np.zeros((len(pcs), 2), np.uint64) if support else None
        for response in self.executor.map(_solve_chunk, jobs):
            values, policies = (
                cast("tuple[np.ndarray, np.ndarray]", response)
                if support
                else (cast("np.ndarray", response), None)
            )
            output[offset : offset + len(values)] = values
            if masks is not None:
                masks[offset : offset + len(values)] = cast("np.ndarray", policies)
            offset += len(values)
        return (output, masks) if support else output


class Fallback:
    def __init__(self, workers: int | None = None) -> None:
        self.pool = FallbackPool(workers)
        self.cache = RevivalCache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.pool.__exit__(*args)

    def solve(
        self,
        success: Key | None,
        failure: Key | None,
        succ_table: np.memmap,
        fail_table: np.memmap,
        pcs: np.ndarray,
        pds: np.ndarray,
        window: bool,
    ) -> tuple[np.ndarray, int, int]:
        output = self.cache.get(success, failure, window, pcs, pds)
        missing = np.flatnonzero(np.isnan(output))
        if len(missing):
            output[missing] = self.pool.solve(
                cast("str", succ_table.filename),
                cast("str", fail_table.filename),
                pcs[missing],
                pds[missing],
                window,
                failure is not None and failure[0] == "REV",
            )
            self.cache.put(
                success, failure, window, pcs[missing], pds[missing], output[missing]
            )
        return output, len(missing), len(pcs) - len(missing)


class NativeFallback(Fallback):
    """We try certified native pivots, then send the remaining residue to HiGHS.

    With crash, each packing LP starts from the recurrence basis. With
    kernel_native, sweep_key asks the kernel to solve H1 and H2 residue inside
    its workers, and to try kink_attempts moved supports before each LP. REV
    keys keep this Python path so they can reuse cached revival values.
    """

    def __init__(
        self,
        workers: int | None = None,
        *,
        crash: bool = False,
        kernel_native: bool = False,
        kink_attempts: int = 0,
    ) -> None:
        super().__init__(workers)
        if kink_attempts < 0 or (kink_attempts and not kernel_native):
            message = "kink attempts need the kernel residue mode"
            raise ValueError(message)
        self.crash = crash
        self.kernel_native = kernel_native
        self.kink_attempts = kink_attempts
        self.native_solves = 0
        self.native_seconds = self.highs_seconds = 0.0

    @property
    def config(self) -> dict[str, bool | int]:
        return {
            "crash": self.crash,
            "kernel_native": self.kernel_native,
            "kink_attempts": self.kink_attempts,
        }

    def kernel_options(self, key: Key) -> dict:
        """We return the kernel keywords for one key; REV keys keep the cache path."""
        if not self.kernel_native or key[0] == "REV":
            return {}
        return {
            "native": True,
            "crash": self.crash,
            "kink_attempts": self.kink_attempts,
        }

    def solve(
        self,
        success: Key | None,
        failure: Key | None,
        succ_table: np.memmap,
        fail_table: np.memmap,
        pcs: np.ndarray,
        pds: np.ndarray,
        window: bool,
    ) -> tuple[np.ndarray, int, int]:
        output = self.cache.get(success, failure, window, pcs, pds)
        missing = np.flatnonzero(np.isnan(output))
        cached = len(pcs) - len(missing)
        if not len(missing):
            return output, 0, cached
        tick = time.perf_counter()
        p = profiles()
        cs = pcs[missing]
        ds = pds[missing]
        compressed = failure is not None and failure[0] == "REV"
        columns = p.idx0[p.fail[cs]] if compressed else p.fail[cs]
        s = np.ascontiguousarray(-succ_table[ds[:, None], p.succ[cs]])
        f = np.ascontiguousarray(p.rev[cs] * -fail_table[ds, columns] + 1 - p.rev[cs])
        values = np.empty(len(missing))
        kind = np.empty(len(missing), np.uint8)
        failed = solve_packing(s, f, window, values, kind, crash=self.crash)
        self.native_seconds += time.perf_counter() - tick
        self.native_solves += len(missing) - failed
        residual = np.flatnonzero(kind == 255)
        if len(residual):
            tick = time.perf_counter()
            values[residual] = self.pool.solve(
                cast("str", succ_table.filename),
                cast("str", fail_table.filename),
                cs[residual],
                ds[residual],
                window,
                compressed,
            )
            self.highs_seconds += time.perf_counter() - tick
        if not np.isfinite(values).all():
            message = "uncertified native/HiGHS residue"
            raise RuntimeError(message)
        output[missing] = values
        self.cache.put(success, failure, window, cs, ds, values)
        return output, len(residual), cached


# ────────────────────────────────────────────────────────────
# Packed-value audit
# ────────────────────────────────────────────────────────────


class PackedReader:
    """We gather values by bitmap rank without restoring dense tables."""

    def __init__(self, directory: PathLike) -> None:
        self.directory = Path(directory)

    @lru_cache(maxsize=32)  # noqa: B019 The bounded cache retains 32 read-only table maps.
    def _open(self, key: Key) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prefix = self.directory / key_name(key)
        bits = np.load(prefix.with_suffix(".bits.npy"), mmap_mode="r")
        values = np.load(prefix.with_suffix(".values.npy"), mmap_mode="r")
        if bits.dtype != np.uint8 or bits.shape != ((N * key_columns(key) + 7) // 8,):
            message = "packed bitmap schema mismatch"
            raise ValueError(message)
        if values.dtype != np.float64 or values.ndim != 1:
            message = "packed value schema mismatch"
            raise ValueError(message)
        totals = np.add.reduceat(
            POP[bits].astype(np.int64), np.arange(0, len(bits), 4096)
        )
        ranks = np.r_[np.int64(0), np.cumsum(totals)]
        if ranks[-1] != len(values):
            message = "packed bitmap and value count mismatch"
            raise ValueError(message)
        return bits, values, ranks

    def get(
        self,
        key: Key,
        rows: np.ndarray | list[int] | int | np.integer,
        cols: np.ndarray | list[int] | int | np.integer,
    ) -> np.ndarray:
        rows, cols = np.broadcast_arrays(
            np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        )
        width = key_columns(key)
        if np.any((rows < 0) | (rows >= N) | (cols < 0) | (cols > width)):
            message = "packed index out of range"
            raise ValueError(message)
        output = np.full(rows.shape, -1.0, dtype=np.float64)
        live = cols != width
        if not live.any():
            return output
        flat = rows[live] * width + cols[live]
        byte, bit = flat // 8, flat % 8
        bits, values, ranks = self._open(tuple(key))
        if np.any((bits[byte] & (1 << bit)) == 0):
            message = "unreachable packed state"
            raise ValueError(message)
        blocks = byte // 4096
        offsets = np.empty(len(byte), np.int64)
        for block in np.unique(blocks):
            selected = blocks == block
            start = int(block) * 4096
            prefix = np.r_[
                np.int64(0), np.cumsum(POP[bits[start : start + 4096]], dtype=np.int64)
            ]
            offsets[selected] = ranks[block] + prefix[byte[selected] - start]
        offsets += POP[(bits[byte] & ((1 << bit) - 1)).astype(np.uint8)]
        output[live] = values[offsets]
        if not np.isfinite(output).all():
            message = "packed values must be finite"
            raise ValueError(message)
        return output

    def sample(
        self, key: Key, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        bits, values, ranks = self._open(tuple(key))
        targets = rng.choice(len(values), min(count, len(values)), replace=False)
        flat = []
        for target in targets:
            block = int(np.searchsorted(ranks, target, side="right") - 1)
            start = block * 4096
            within = int(target - ranks[block])
            cumulative = np.cumsum(POP[bits[start : start + 4096]], dtype=np.int64)
            byte = int(np.searchsorted(cumulative, within, side="right"))
            before = int(cumulative[byte - 1]) if byte else 0
            set_bits = np.flatnonzero(
                np.unpackbits(np.array([bits[start + byte]]), bitorder="little")
            )
            flat.append((start + byte) * 8 + int(set_bits[within - before]))
        return np.divmod(np.array(flat, np.int64), key_columns(key))


def audit_packed_key(
    directory: PathLike,
    key: Key,
    reached: np.ndarray,
    expected: int,
    dth: np.ndarray | None = None,
) -> dict:
    prefix = Path(directory) / key_name(key)
    cols = key_columns(key)
    bits = np.load(prefix.with_suffix(".bits.npy"), mmap_mode="r")
    values = np.load(prefix.with_suffix(".values.npy"), mmap_mode="r")
    if bits.dtype != np.uint8 or bits.shape != ((N * cols + 7) // 8,):
        message = "packed bitmap schema mismatch"
        raise ValueError(message)
    if values.dtype != np.float64 or values.shape != (expected,):
        message = "packed reachability/value count mismatch"
        raise ValueError(message)
    offset = 0
    gain_sum = 0.0
    changed = 0
    min_gain = np.inf
    max_gain = -np.inf
    largest = None
    for lo in range(0, N, 64):
        mask = np.unpackbits(
            reached[lo : lo + 64], axis=1, count=cols, bitorder="little"
        ).astype(bool)
        packed = np.packbits(mask.reshape(-1), bitorder="little")
        if not np.array_equal(
            bits[lo * cols // 8 : lo * cols // 8 + len(packed)], packed
        ):
            message = f"packed reachability mismatch: {key}"
            raise ValueError(message)
        count = int(mask.sum())
        data = values[offset : offset + count]
        offset += count
        if not np.isfinite(data).all():
            message = f"packed values must be finite: {key}"
            raise ValueError(message)
        if np.any(np.abs(data) > 1 + 1e-9):
            message = f"packed value outside utility range: {key}"
            raise ValueError(message)
        if dth is not None and count:
            baseline = dth[lo : lo + len(mask)]
            baseline = (
                baseline[:, profiles().s0] if key[0] == "REV" else baseline[:, :N]
            )
            gain = baseline[mask] - data if key[0] == "H1" else data - baseline[mask]
            gain_sum += float(gain.sum())
            changed += int((gain > 1e-8).sum())
            min_gain = min(min_gain, float(gain.min()))
            if float(gain.max()) > max_gain:
                max_gain = float(gain.max())
                position = int(np.flatnonzero(mask)[int(gain.argmax())])
                row, col = divmod(position, cols)
                largest = [
                    lo + row,
                    int(profiles().s0[col]) if key[0] == "REV" else col,
                ]
    if offset != expected:
        message = f"packed reachability count mismatch: {key}"
        raise ValueError(message)
    # We allow the accumulated prefix error here; each stage still uses 1e-6.
    if dth is not None and min_gain < -1.3e-5:
        message = f"Baku action-set monotonicity failed: {key}: {min_gain}"
        raise ValueError(message)
    return {
        "states": offset,
        "baku_gain_sum": gain_sum,
        "changed_states": changed,
        "min_baku_gain": min_gain if dth is not None else None,
        "max_baku_gain": max_gain if dth is not None else None,
        "largest_gain_profiles": largest,
    }


def certify_matrix(matrix: np.ndarray | list) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        message = "matrix must contain finite payoffs"
        raise ValueError(message)
    rows, cols = matrix.shape
    failure = "opening/audit LP failed the 1e-6 gate"
    attempts = [
        ("highs", None),
        ("highs-ipm", None),
        ("highs-ipm", float(matrix[-1, 0])),
        ("highs-ipm", float(matrix[0, 0])),
    ]
    for method, offset in attempts:
        lp_matrix = matrix if offset is None else matrix - offset
        result = linprog(
            np.r_[np.zeros(rows), -1.0],
            A_ub=np.c_[-lp_matrix.T, np.ones(cols)],
            b_ub=np.zeros(cols),
            A_eq=[np.r_[np.ones(rows), 0.0]],
            b_eq=[1.0],
            bounds=[(0, None)] * rows + [(None, None)],
            method=method,
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            failure = f"opening/audit HiGHS failure: {result.message}"
            continue
        drop = np.maximum(result.x[:-1], 0.0)
        check = np.maximum(-result.ineqlin.marginals, 0.0)
        drop /= drop.sum()
        check /= check.sum()
        lower = float(np.min(drop @ matrix))
        upper = float(np.max(matrix @ check))
        if not np.isfinite(upper - lower) or upper - lower > 1e-6:
            failure = "opening/audit LP failed the 1e-6 gate"
            continue
        return {
            "value": (lower + upper) / 2,
            "lower": lower,
            "upper": upper,
            "gap": upper - lower,
            "drop": drop.tolist(),
            "check": check.tolist(),
        }
    raise RuntimeError(failure)


def explicit_matrix(s: np.ndarray | list[float], f: float, window: bool) -> np.ndarray:
    # We assemble cells from the action comparison for an independent LP audit.
    drop, check = np.indices((60 + int(window), 60))
    matrix = np.full(drop.shape, f, dtype=np.float64)
    success = check >= drop
    matrix[success] = s[(check - drop)[success]]
    return matrix


def stage_inputs(
    reader: PackedReader, key: Key, pc: int, pd: int, dth: np.ndarray
) -> tuple[np.ndarray, float]:
    p = profiles()

    def child_values(
        dest: Key | None, columns: np.ndarray | int | np.integer
    ) -> np.ndarray:
        if dest is None:
            return dth[pd, columns]
        mapped = p.idx0[columns] if dest[0] == "REV" else columns
        return reader.get(dest, pd, mapped)

    s = -child_values(child_key(key), p.succ[pc])
    f = 1.0
    if p.rev[pc]:
        f = p.rev[pc] * -float(
            child_values(child_key(key, int(p.st[pc]) + 60), p.fail[pc])
        ) + (1 - p.rev[pc])
    return s, f


def audit_full(
    directory: PathLike,
    reach: Reachability,
    records: list[dict],
    dth: np.ndarray,
    samples_per_key: int = 2,
) -> dict:
    directory = Path(directory)
    reader = PackedReader(directory / "tables")
    rng = np.random.default_rng(93482)
    checked = samples = 0
    worst_residual = worst_gap = 0.0
    results = {}
    for key in reach.counts:
        info = audit_packed_key(
            reader.directory, key, reach.load(key), reach.counts[key], dth
        )
        checked += info["states"]
        results[key_name(key)] = info
        rows, cols = reader.sample(
            key, 200 if key == ("H2", 57) else samples_per_key, rng
        )
        for pc, col in zip(rows, cols, strict=True):
            pd = int(profiles().s0[col]) if key[0] == "REV" else int(col)
            s, f = stage_inputs(reader, key, int(pc), pd, dth)
            certificate = certify_matrix(explicit_matrix(s, f, is_window(key)))
            stored = float(reader.get(key, pc, col))
            residual = abs(stored - certificate["value"])
            if residual > 1e-6:
                message = f"Bellman audit failed: {key} {pc} {pd}: {residual}"
                raise RuntimeError(message)
            worst_residual = max(worst_residual, residual)
            worst_gap = max(worst_gap, certificate["gap"])
            samples += 1
        if key[0] == "H1":
            print(f"audit {key_name(key)}: {info['states']:,} states", flush=True)
    if checked != sum(TOTALS.values()) or checked != sum(r["states"] for r in records):
        message = "full audit state count mismatch"
        raise RuntimeError(message)
    p = profiles()
    s, f = stage_inputs(reader, ("H1", 12), 0, 0, dth)
    root = certify_matrix(explicit_matrix(s, f, False))
    root["stored_value"] = float(reader.get(("H1", 12), 0, 0))
    root["hal_win_probability"] = (1 + root["stored_value"]) / 2
    root["baku_win_probability"] = (1 - root["stored_value"]) / 2
    root["dth_value"] = float(dth[0, 0])
    root["baku_gain_probability_points"] = 50 * (
        root["dth_value"] - root["stored_value"]
    )
    root["success_payoffs"] = s.tolist()
    root["failure_payoff"] = f
    root["dth_policies"] = certify_matrix(
        explicit_matrix(
            -dth[0, p.succ[0]],
            p.rev[0] * -dth[0, p.fail[0]] + (1 - p.rev[0]),
            False,
        )
    )
    return {
        "states_checked": checked,
        "lp_samples": samples,
        "max_bellman_residual": worst_residual,
        "max_lp_gap": worst_gap,
        "root": root,
        "keys": results,
        "changed_states": sum(r["changed_states"] for r in results.values()),
        "mean_baku_gain_over_state_classes": sum(
            r["baku_gain_sum"] for r in results.values()
        )
        / checked,
        "min_baku_gain": min(r["min_baku_gain"] for r in results.values()),
        "max_baku_gain": max(r["max_baku_gain"] for r in results.values()),
    }


# ────────────────────────────────────────────────────────────
# Verified read path
# ────────────────────────────────────────────────────────────

# A completed report records full coverage and a successful Bellman audit.
# A key names a table as ("H1", minute), ("H2", minute) or ("REV", clock).
# checker and dropper are profile indices that profile_index returns.

TOLERANCE = 1e-6


def profile_index(st: int, ttd: int) -> int:
    """Return the quotient profile of a player with squandered time st and TTD ttd."""

    p = profiles()
    found = np.flatnonzero((p.st == st) & (p.ttd == ttd))
    if len(found) != 1:
        message = f"no profile has st={st} and ttd={ttd}"
        raise ValueError(message)
    return int(found[0])


class LeapTable:
    """Stored values and stage certificates of one verified leap artifact."""

    def __init__(self, artifact: Path, dth: np.ndarray) -> None:
        self._reader = PackedReader(artifact / "tables")
        self._dth = dth

    def value(self, key: tuple[str, int], checker: int, dropper: int) -> float:
        """Return the stored value of one state, from the Dropper's side.

        The implicit WIN dropper profile reads as -1, as the builder stores it.
        """

        key = (str(key[0]), int(key[1]))
        if not 0 <= dropper <= WIN:
            message = f"dropper profile {dropper} is out of range"
            raise ValueError(message)
        column = int(profiles().idx0[dropper]) if key[0] == "REV" else dropper
        if column < 0:
            message = f"{key_name(key)} stores no column for dropper profile {dropper}"
            raise ValueError(message)
        return float(self._reader.get(key, checker, column))

    def stage(self, key: tuple[str, int], checker: int, dropper: int) -> dict:
        """Re-certify one stored value with an independent LP on its stage matrix.

        The stage matrix comes from the stored child values and the DTH tail.
        The method raises ValueError when the LP finds no certificate with a
        saddle gap of at most 1e-6, or when the certified value differs from
        the stored value by more than 1e-6. The result holds the stored value,
        the Bellman residual and the certificate of
        ``certify_matrix``.
        """

        key = (str(key[0]), int(key[1]))
        stored = self.value(key, checker, dropper)
        success, failure = stage_inputs(self._reader, key, checker, dropper, self._dth)
        where = f"{key_name(key)} checker {checker} dropper {dropper}"
        # certify_matrix raises RuntimeError when no LP attempt passes its 1e-6 gap
        # gate.
        try:
            certificate = certify_matrix(
                explicit_matrix(success, failure, is_window(key))
            )
        except RuntimeError as error:
            message = f"{where}: {error}"
            raise ValueError(message) from error
        residual = abs(stored - certificate["value"])
        if residual > TOLERANCE:
            message = f"{where}: Bellman residual {residual:.3g} exceeds {TOLERANCE:g}"
            raise ValueError(message)
        if certificate["gap"] > TOLERANCE:
            message = (
                f"{where}: saddle gap {certificate['gap']:.3g} exceeds {TOLERANCE:g}"
            )
            raise ValueError(message)
        return {"stored_value": stored, "bellman_residual": residual, **certificate}


def open_leap(
    artifact: str | os.PathLike[str] = DEFAULT_ARTIFACT,
    *,
    dth: str | os.PathLike[str] = DEFAULT_PATH,
) -> LeapTable:
    """Open a completed leap artifact and validate its DTH tail."""
    root = Path(artifact).resolve()
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("states") != sum(TOTALS.values())
        or {kind: row["states"] for kind, row in report.get("rows", {}).items()}
        != TOTALS
        or not 0 <= report.get("max_bellman_residual", math.inf) <= TOLERANCE
    ):
        message = f"{report_path}: no complete, numerically checked leap result"
        raise ValueError(message)
    return LeapTable(root, Table.open(Path(dth)).values)
