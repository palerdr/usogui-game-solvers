"""Build, audit and read the certified leap table of the public STL game.

The paper ``stl.tex`` reports this table. ``run_full`` sweeps every reachable
state back to the clock-720 opening and audits the result; ``open_leap`` is the
one read path into a completed artifact. The numba sweep lives in
``leap_kernel.py``. The scalar oracle here certifies single stages, and the
tests hold the kernel to it. ``src/stl/docs/LEAP_CERTIFICATE.md``
holds the certificate equations.

Run ``python -m stl.leap [--dth artifacts/dth/V.npy] [--output artifacts/leap]``
to build or resume the table. Resume needs the same builder hash, DTH table
and checkpoint file hashes.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

import highspy
import numba
from numba import njit, types
from numba.extending import intrinsic
import numpy as np
from scipy.optimize import linprog

from stl.dth import DEFAULT_PATH
from stl.game import (
    LS_WINDOW_END,
    LS_WINDOW_START,
    TURN_DURATION_LEAP,
    TURN_DURATION_NORMAL,
    Player,
    Referee,
)
from stl.leap_kernel import solve_packing, sweep_batch

# The completed leap artifact (82 GB, gitignored). Its manifest binds the
# sha256 of the DTH table at DEFAULT_PATH.
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "leap"


# ────────────────────────────────────────────────────────────
# One-rounding fma for the scalar oracle
# ────────────────────────────────────────────────────────────

# We compute a*b+c with one rounding, as math.fma does, on Python 3.12.
# math.fma arrived in Python 3.13. The scalar oracle below and the support
# oracle in tests/test_leap_kernel.py call `fma`. It emits `llvm.fma`, the
# instruction that the kernel's `_fma` emits, so it returns the bits that
# math.fma returns. It also raises where math.fma raises: ValueError when the
# result is NaN and no operand is, and OverflowError when finite operands give
# an infinite result. The oracles do not use the kernel's `_fma`, the code
# that they check, so this module keeps its own copy of the intrinsic.

@intrinsic
def _fma(typingctx, a, b, c):
    """We return a*b+c with one rounding."""
    if not (a == types.float64 and b == types.float64 and c == types.float64):
        return None

    def codegen(context, builder, signature, args):
        return builder.fma(*args)
    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True)
def _fused(a, b, c):
    return _fma(a, b, c)


def fma(a, b, c):
    """We return a*b+c with one rounding and raise the errors of math.fma."""
    x, y, z = float(a), float(b), float(c)
    r = _fused(x, y, z)
    if math.isnan(r):
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            raise ValueError('invalid operation in fma')
    elif math.isinf(r) and math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
        raise OverflowError('overflow in fma')
    return r

# ────────────────────────────────────────────────────────────
# Quotient profiles and clocks
# ────────────────────────────────────────────────────────────

def is_leap_window(clock: int | float) -> bool:
    """Return whether a half-round starting at ``clock`` spans second 61."""

    return LS_WINDOW_START <= clock <= LS_WINDOW_END


def turn_duration(clock: int | float) -> int:
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
        raise ValueError('invalid half or dose')
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
    ids = {}; blocks = []; st = []; ttd = []
    for t in [0, *range(60, 301)]:
        start = len(st)
        for s in range(300):
            if eligible(s, t):
                ids[s, t] = len(st); st.append(s); ttd.append(t)
        if len(st) > start:
            blocks.append((start, len(st)))
    assert len(st) == N_ALIVE
    st.extend(range(300)); ttd.extend([301]*300)
    assert len(st) == N
    succ = np.empty((N, 60), np.int32)
    fail = np.full(N, WIN, np.int32)
    rev = np.zeros(N)
    referee = Referee()
    for pc, (s, t) in enumerate(zip(st, ttd)):
        for lag in range(1, 61):
            grown = s + lag
            succ[pc, lag-1] = WIN if grown >= 300 else ids.get((grown, t), N_ALIVE+grown)
        if pc < N_ALIVE:
            fail[pc] = ids.get((0, s+t+60), N_ALIVE)
            rev[pc] = referee.compute_survival_probability(Player('Checker', ttd=t), s+60)
    s0 = np.array([ids[0, t] for t in [0, *range(60, 241)]]+[N_ALIVE], np.int32)
    assert len(s0) == 183
    idx0 = np.full(N+1, -1, np.int32); idx0[s0] = np.arange(183); idx0[WIN] = 183
    return Profiles(np.array(st, np.int32), np.array(ttd, np.int32), succ, fail, rev,
                    s0, idx0, tuple(blocks))

# ────────────────────────────────────────────────────────────
# Scalar stage oracle
# ────────────────────────────────────────────────────────────

GAP = 1e-6


CERT_RADIUS = 1e-10


@dataclass(frozen=True)
class StageResult:
    value: float
    kind: int
    lower: float
    upper: float
    support: tuple[int, int] | None = None

    @property
    def gap(self):
        return self.upper - self.lower


def stage_matrix(s, f, window):
    matrix = np.full((60+int(window), 60), f, dtype=np.float64)
    for row in range(60):
        matrix[row, row:] = s[:60-row]
    return matrix


def policy_masks(p, q):
    return tuple(int.from_bytes(np.packbits(np.asarray(x) > 0, bitorder='little').tobytes(), 'little')
                 for x in (p, q))


def solve_lp(s, f, window, *, support=False):
    matrix = stage_matrix(s, f, window)
    rows = len(matrix)
    failure = 'HiGHS failed the 1e-6 saddle gate'
    # We retain the successful path of each prior retry. A common offset
    # changes the LP coordinates; we certify against the original matrix.
    attempts = [('highs', None), ('highs-ipm', None),
                ('highs-ipm', f), ('highs-ipm', float(matrix[0, 0]))]
    for method, offset in attempts:
        lp_matrix = matrix if offset is None else matrix-offset
        result = linprog(np.r_[np.zeros(rows), -1.],
                         A_ub=np.c_[-lp_matrix.T, np.ones(60)], b_ub=np.zeros(60),
                         A_eq=[np.r_[np.ones(rows), 0.]], b_eq=[1.],
                         bounds=[(0, None)]*rows+[(None, None)], method=method,
                         options={'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        if not result.success:
            failure = f'HiGHS failed: {result.message}'
            continue
        p = np.maximum(result.x[:-1], 0); q = np.maximum(-result.ineqlin.marginals, 0)
        p /= p.sum(); q /= q.sum()
        lower = float((p @ matrix).min()); upper = float((matrix @ q).max())
        if not math.isfinite(upper-lower) or upper-lower > GAP:
            failure = 'HiGHS failed the 1e-6 saddle gate'
            continue
        return StageResult((lower+upper)/2, 3, lower, upper, policy_masks(p, q) if support else None)
    raise RuntimeError(failure)


def solve_stage(s, f, window, *, allow_lp=True):
    s = np.asarray(s, dtype=np.float64)
    if s.shape != (60,) or not np.isfinite(s).all() or not math.isfinite(f):
        raise ValueError('stage needs 60 finite successes and a finite failure')
    f = float(f); s = list(map(float, s))
    lo, hi = min(s), max(s)
    lower, upper = max(lo, min(f, s[0])), min(hi, max(f, s[0]))
    if window:
        lower, upper = max(lower, f), max(upper, f)
    kind = 0
    if upper-lower <= GAP:
        value = (lower+upper)/2
        if window and f >= (max(lo, min(f, s[0]))+min(hi, max(f, s[0])))/2:
            kind = 2
        return StageResult(value, kind, lower, upper)
    d = s[0]-f
    if abs(d) >= 1e-12:
        b = [0.] + [(s[k-1]-s[k])/d for k in range(1, 60)]
        r = [1.]
        for k in range(1, 60):
            acc = 0.
            try:
                for j in range(k):
                    acc = fma(b[k-j], r[j], acc)
            except (OverflowError, ValueError):
                acc = math.nan
            r.append(acc)
            if not math.isfinite(acc):
                break
        q = [max(x, 0.) if math.isfinite(x) else math.nan for x in reversed(r)]
        total = sum(q)
        if math.isfinite(total) and total > 0:
            if min(r) >= 0 and lo >= -2 and hi <= 2 and abs(f) <= 2 and abs(d) >= GAP:
                value = f+d/total
                lower, upper = value-CERT_RADIUS, value+CERT_RADIUS
            else:
                q = np.array(q)/total
                payoffs = stage_matrix(s, f, False) @ q
                lower, upper = float(payoffs.min()), float(payoffs.max())
                value = (lower+upper)/2
            if window:
                lower, upper = max(lower, f), max(upper, f)
                kind = 2 if f >= value else 1
                value = max(value, f)
            else:
                kind = 1
            if math.isfinite(upper-lower) and upper-lower <= GAP:
                return StageResult(value, kind, lower, upper)
    if allow_lp:
        return solve_lp(s, f, window)
    return StageResult(math.nan, 255, -math.inf, math.inf)

# ────────────────────────────────────────────────────────────
# Reachability, value storage and the full sweep
# ────────────────────────────────────────────────────────────

POP = np.array([i.bit_count() for i in range(256)], np.uint8)


TOTALS = {'H1': 4180634990, 'H2': 4188677815, 'REV': 1084020312}


# The full sweep's residue path. Each option keeps the 1e-6 full-matrix gate.
# benchmark_leap_replay selected one kink attempt; more attempts ran slower.
RESIDUE = {'crash': True, 'kernel_native': True, 'kink_attempts': 1}


def key_clock(key):
    kind, index = key
    return index if kind == 'REV' else 60*index + (120 if kind == 'H2' else 0)


def key_name(key):
    return f'{key[0]}_{key[1]}'


def key_columns(key):
    return 183 if key[0] == 'REV' else N


def all_keys():
    return sorted([('H1', m) for m in range(12, 60)] +
                  [('H2', m) for m in range(12, 59)] +
                  [('REV', c) for c in range(1020, 3601)], key=key_clock)


def child_key(key, q=0):
    half = 1 if key[0] == 'H1' else 2
    clock = child_clock(half, key_clock(key), q)
    if clock > 3600:
        return None
    if half == 1:
        return ('REV', clock) if q else ('H2', key[1])
    return ('H1', clock//60)


def is_window(key):
    return key[0] != 'H1' and 3540 <= key_clock(key) <= 3600


def _transpose(bits, columns):
    return np.packbits(np.unpackbits(bits, axis=1, count=columns, bitorder='little').T,
                       axis=1, bitorder='little')


def _success_bits(bits):
    p = profiles()
    target = np.zeros_like(bits)
    for start, stop in (*p.blocks, (N_ALIVE, N)):
        length = stop-start
        expanded = np.zeros((min(300, length+60), bits.shape[1]), np.uint8)
        expanded[1:min(length+1, 300)] = bits[start:start+min(length, 299)]
        for shift in (1, 2, 4, 8, 16, 28):
            expanded[shift:] |= expanded[:-shift]
        if start == N_ALIVE:
            target[N_ALIVE:] |= expanded
        else:
            target[start:stop] |= expanded[:length]
            target[N_ALIVE+length:N_ALIVE+len(expanded)] |= expanded[length:]
    return target


@dataclass
class Reachability:
    counts: dict
    directory: Path | None

    @property
    def total(self):
        return sum(self.counts.values())

    @property
    def by_type(self):
        return {kind: sum(v for k, v in self.counts.items() if k[0] == kind) for kind in TOTALS}

    @property
    def window_count(self):
        return sum(v for k, v in self.counts.items() if is_window(k))

    def load(self, key):
        return np.load(self.directory / f'{key_name(key)}.npy', mmap_mode='r')


def forward_reachability(limit=3600, directory=None):
    """Union successor sets before counting each quotient state once."""
    if limit > 3600:
        raise ValueError('reachability ends at 3600')
    directory = Path(directory) if directory is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    p = profiles(); pending = {}; counts = {}

    def target(key):
        if key not in pending:
            pending[key] = np.zeros((N, (key_columns(key)+7)//8), np.uint8)
        return pending[key]

    target(('H1', 12))[0, 0] = 1
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
            np.save(directory/f'{key_name(key)}.npy', bits)
        ncols = key_columns(key)
        rows = p.s0 if key[0] == 'REV' else slice(None)
        success = child_key(key)
        if success is not None and key_clock(success) <= limit:
            target(success)[rows] |= _transpose(_success_bits(bits), ncols)
        # We group by the destination clock before merging revival paths.
        groups = {}
        for pc in np.flatnonzero(np.any(bits[:N_ALIVE], axis=1)):
            dest = child_key(key, int(p.st[pc])+60)
            if dest is not None and key_clock(dest) <= limit:
                groups.setdefault(dest, []).append(int(pc))
        for dest, pcs in groups.items():
            merged = np.zeros((183, bits.shape[1]), np.uint8)
            np.bitwise_or.at(merged, p.idx0[p.fail[pcs]], bits[pcs])
            transposed = _transpose(merged, ncols)
            if dest[0] == 'REV':
                target(dest)[rows] |= transposed
            else:
                # We place the 183 reset-profile columns in their full profile positions.
                dense = np.unpackbits(transposed, axis=1, count=183, bitorder='little')
                dest_bits = target(dest)
                for j, col in enumerate(p.s0):
                    dest_bits[rows, col//8] |= dense[:, j] << int(col % 8)
        if directory is not None and key[0] == 'H1':
            print(f'reach {key_name(key)}: {count:,}', flush=True)
    result = Reachability(counts, directory)
    if directory is not None:
        (directory/'counts.json').write_text(json.dumps({key_name(k): v for k, v in counts.items()}))
        files = {path.name: file_hash(path) for path in directory.glob('*') if path.name != 'manifest.json'}
        write_json(directory/'manifest.json', {'builder_sha256': builder_hash(), 'files': files, 'limit': limit})
    return result


def evaluate_states(key, pairs, dth):
    """Evaluate bounded test subtrees without sweeping a key."""
    p = profiles()
    @lru_cache(None)
    def value(key, pc, pd):
        if key is None:
            return float(dth[pc, pd])
        success = child_key(key)
        s = np.array([1. if ch == WIN else -value(success, pd, int(ch)) for ch in p.succ[pc]])
        f = 1.
        if p.rev[pc]:
            f = p.rev[pc] * -value(child_key(key, int(p.st[pc])+60), pd, int(p.fail[pc])) + 1-p.rev[pc]
        return solve_stage(s, f, is_window(key)).value
    return np.array([value(key, int(pc), int(pd)) for pc, pd in pairs])


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def builder_hash():
    """Hash the sources that decide a leap value, by package-relative name."""
    package = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ('game.py', 'leap.py', 'leap_kernel.py'):
        encoded = name.encode(); data = (package/name).read_bytes()
        h.update(len(encoded).to_bytes(8, 'big')); h.update(encoded)
        h.update(len(data).to_bytes(8, 'big')); h.update(data)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


class TableStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.write_bytes = 0; self.write_seconds = 0.
        self.pack_bytes = 0; self.pack_seconds = 0.

    def path(self, key):
        return self.directory/f'{key_name(key)}.npy'

    def create(self, key):
        start = time.perf_counter()
        table = np.lib.format.open_memmap(self.path(key), mode='w+', dtype=np.float64,
                                          shape=(N, key_columns(key)+1))
        table[:] = np.nan; table[:, -1] = -1.; table.flush()
        self.write_seconds += time.perf_counter()-start
        self.write_bytes += table.nbytes
        return table

    def pack(self, key, *, retain_dense=False):
        start = time.perf_counter(); table = np.load(self.path(key), mmap_mode='r')
        cols = key_columns(key)
        count = sum(int(np.isfinite(table[lo:lo+64, :cols]).sum()) for lo in range(0, N, 64))
        prefix = self.directory/key_name(key)
        bitmap_path = prefix.with_suffix('.bits.npy'); values_path = prefix.with_suffix('.values.npy')
        bitmap = np.lib.format.open_memmap(bitmap_path, mode='w+', dtype=np.uint8,
                                           shape=((N*cols+7)//8,))
        values = np.lib.format.open_memmap(values_path, mode='w+', dtype=np.float64, shape=(count,))
        offset = 0
        for lo in range(0, N, 64):
            block = table[lo:lo+64, :cols]; mask = np.isfinite(block)
            packed = np.packbits(mask.reshape(-1), bitorder='little')
            bitmap[lo*cols//8:lo*cols//8+len(packed)] = packed
            data = block[mask]; values[offset:offset+len(data)] = data; offset += len(data)
        bitmap.flush(); values.flush()
        self.pack_bytes += bitmap.nbytes+values.nbytes
        self.pack_seconds += time.perf_counter()-start
        del bitmap, values, table
        if not retain_dense:
            self.path(key).unlink()

    def pack_cold(self, clock):
        packed = []
        for path in self.directory.glob('*.npy'):
            if '.' in path.stem:
                continue
            kind, index = path.stem.split('_'); key = (kind, int(index))
            if key_clock(key) > clock+540:
                self.pack(key); packed.append(key)
        return packed

    def load(self, key):
        if self.path(key).exists():
            return np.load(self.path(key), mmap_mode='r')
        prefix = self.directory/key_name(key)
        bitmap = np.load(prefix.with_suffix('.bits.npy'), mmap_mode='r')
        values = np.load(prefix.with_suffix('.values.npy'), mmap_mode='r')
        table = self.create(key); cols = key_columns(key); offset = 0
        for lo in range(0, N, 64):
            rows = min(64, N-lo); length = rows*cols
            mask = np.unpackbits(bitmap[lo*cols//8:(lo*cols+length+7)//8],
                                 count=length, bitorder='little').reshape(rows, cols).astype(bool)
            count = int(mask.sum()); table[lo:lo+rows, :cols][mask] = values[offset:offset+count]
            offset += count
        if offset != len(values):
            raise ValueError('packed value count mismatch')
        table.flush()
        return table


def sweep_key(key, bitmap, store, dth, *, min_clock=3420, full=False, fallback=None):
    floor = 720 if full else 3420
    if min_clock < floor or key_clock(key) < min_clock:
        raise ValueError(f'key sweeps must stop at clock {floor}')
    p = profiles(); table = store.create(key); cols = key_columns(key)
    success_key = child_key(key)
    success_table = dth if success_key is None else store.load(success_key)
    full_pds = p.s0 if key[0] == 'REV' else np.arange(N, dtype=np.int32)
    states = failures = lp_solves = cached_failures = 0; kernel_seconds = lp_seconds = 0.
    kernel_native = support_solves = 0
    # The kernel solves H1 and H2 residue itself when the fallback asks for it.
    # Its LP time then counts in kernel_seconds, so compare residue paths on
    # sweep time; native_solves and native_seconds cover the Python path only.
    options = fallback.kernel_options(key) if hasattr(fallback, 'kernel_options') else {}
    native_before = getattr(fallback, 'native_solves', 0)
    native_time_before = getattr(fallback, 'native_seconds', 0.)
    highs_before = getattr(fallback, 'highs_seconds', None)
    groups = {}
    for checker in np.flatnonzero(np.any(bitmap, axis=1)):
        dest = None if not p.rev[checker] else child_key(key, int(p.st[checker])+60)
        groups.setdefault(dest, []).append(int(checker))
    # We group by child table before batching to avoid repeated map validation
    # for small fragments of the same revival group. We retain Dropper row order.
    for dest, checkers in groups.items():
        checkers = np.array(checkers, np.int32)
        reached = np.unpackbits(bitmap[checkers], axis=1, count=cols, bitorder='little').T
        fail_table = dth if dest is None else store.load(dest)
        fail_col = p.idx0[p.fail] if dest is not None and dest[0] == 'REV' else p.fail
        rows_per_batch = max(64, min(cols, 1_048_576//len(checkers)))
        for start in range(0, cols, rows_per_batch):
            js, local_pc = np.nonzero(reached[start:start+rows_per_batch]); js += start
            if not len(js):
                continue
            pcs = np.ascontiguousarray(checkers[local_pc])
            pds = np.ascontiguousarray(full_pds[js], dtype=np.int32)
            out = np.empty(len(pcs)); kind = np.empty(len(pcs), np.uint8)
            tick = time.perf_counter()
            failed = sweep_batch(pcs, pds, success_table, fail_table, p.succ,
                                 fail_col, p.rev, is_window(key), 1e-6, out, kind, **options)
            kernel_seconds += time.perf_counter()-tick
            if options:
                # Kinds 3, 4, and 5 are residue that the kernel certified itself.
                solved = int(np.count_nonzero(kind == 3)); guessed = int(np.count_nonzero((kind == 4) | (kind == 5)))
                kernel_native += solved; support_solves += guessed; failures += solved+guessed
            failures += failed
            if failed:
                tick = time.perf_counter()
                indices = np.flatnonzero(kind == 255)
                if fallback is None:
                    for i in indices:
                        checker, dropper = int(pcs[i]), int(pds[i])
                        successes = -success_table[dropper, p.succ[checker]]
                        failure = p.rev[checker] * -fail_table[dropper, fail_col[checker]] + 1-p.rev[checker]
                        out[i] = solve_lp(successes, failure, is_window(key)).value
                    lp_solves += failed
                else:
                    out[indices], solved, cached = fallback.solve(success_key, dest, success_table,
                            fail_table, pcs[indices], pds[indices], is_window(key))
                    lp_solves += solved; cached_failures += cached
                    if failed > 1000:
                        native = failed-solved-cached
                        print(f'fallback {key_name(key)}: {native:,} native, {solved:,} HiGHS, {cached:,} cached', flush=True)
                lp_seconds += time.perf_counter()-tick
            if not np.isfinite(out).all():
                raise RuntimeError('uncertified class aborts the build')
            tick = time.perf_counter()
            table[pcs, js] = out
            store.write_seconds += time.perf_counter()-tick
            states += len(pcs)
    tick = time.perf_counter(); table.flush()
    store.write_seconds += time.perf_counter()-tick
    if states != int(POP[bitmap].sum()):
        raise RuntimeError('sweep missed reachable classes')
    return {'key': list(key), 'clock': key_clock(key), 'states': states,
            'kernel_seconds': kernel_seconds, 'classes_per_second': states/kernel_seconds,
            'failures': failures, 'fallback_seconds': lp_seconds,
            'lp_seconds': lp_seconds if highs_before is None else fallback.highs_seconds-highs_before,
            'native_solves': getattr(fallback, 'native_solves', 0)-native_before,
            'native_seconds': getattr(fallback, 'native_seconds', 0.)-native_time_before,
            'kernel_native_solves': kernel_native, 'support_solves': support_solves,
            'lp_solves': lp_solves, 'cached_failures': cached_failures}


def _load_reach(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['builder_sha256'] != builder_hash():
        raise ValueError('reachability source mismatch')
    validate_files(directory, manifest['files'])
    data = json.loads((directory/'counts.json').read_text())
    return Reachability({(name.split('_')[0], int(name.split('_')[1])): n
                         for name, n in data.items()}, directory)


def _rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def commit_full_checkpoint(directory, identity, records, files, store, clock, elapsed, *, finalize=False):
    directory = Path(directory); cold = []
    for path in store.directory.glob('*.npy'):
        if '.' in path.stem:
            continue
        kind, index = path.stem.split('_'); key = (kind, int(index))
        if finalize or key_clock(key) > clock+540:
            # We retain the dense source until the checkpoint owns both packed files.
            store.pack(key, retain_dense=True)
            for suffix in ('.bits.npy', '.values.npy'):
                packed = store.directory/f'{key_name(key)}{suffix}'
                files[str(packed.relative_to(directory))] = file_hash(packed)
            files.pop(str(path.relative_to(directory)), None)
            cold.append(path)
    io = {name: getattr(store, name) for name in ('write_bytes', 'write_seconds', 'pack_bytes', 'pack_seconds')}
    write_json(directory/'checkpoint.json', {'identity': identity, 'records': records,
               'files': files, 'io': io, 'elapsed_seconds': elapsed,
               'peak_rss_bytes': _rss_bytes(), 'sweep_complete': finalize})
    for path in cold:
        path.unlink()


def run_full(dth_path, directory):
    """We sweep the public game to its clock-720 opening and audit the result."""
    started = time.perf_counter(); directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dth_path = Path(dth_path); dth = np.load(dth_path, mmap_mode='r')
    if dth.shape != (N, N+1) or dth.dtype != np.float64 or not np.all(dth[:, -1] == -1):
        raise ValueError('DTH table schema mismatch')
    identity = {'schema': 'stl-leap-values-v1', 'builder_sha256': builder_hash(),
                'dth_sha256': file_hash(dth_path), 'saddle_tolerance': 1e-6,
                'minimum_clock': 720, 'mode': 'full', 'python': sys.version, 'numpy': np.__version__,
                'numba': numba.__version__, 'highs': highspy.Highs().version(), 'residue': RESIDUE}
    checkpoint_path = directory/'checkpoint.json'; store = TableStore(directory/'tables')
    records = []; files = {}; elapsed_base = 0.; prior_peak = 0
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint['identity'] != identity:
            raise ValueError('checkpoint source or DTH digest mismatch')
        manifest_path = directory/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('complete') and manifest.get('builder_sha256') == identity['builder_sha256']:
                validate_files(directory, manifest['files'])
                report = json.loads((directory/'report.json').read_text())
                print(json.dumps(report, indent=2), flush=True)
                return report
        files = checkpoint['files']; validate_files(directory, files)
        records = checkpoint['records']; elapsed_base = checkpoint.get('elapsed_seconds', 0.)
        prior_peak = checkpoint.get('peak_rss_bytes', 0)
        for name, value in checkpoint['io'].items():
            setattr(store, name, value)
        # We remove a dense copy left by an interruption after checkpoint commit.
        for path in store.directory.glob('*.npy'):
            if '.' not in path.stem and str(path.relative_to(directory)) not in files:
                packed = str(path.with_suffix('.values.npy').relative_to(directory))
                if packed in files:
                    path.unlink()
    reach_dir = directory/'reachability'
    reach = _load_reach(reach_dir) if (reach_dir/'counts.json').exists() else forward_reachability(directory=reach_dir)
    if reach.by_type != TOTALS or reach.window_count != 470336555:
        raise RuntimeError('full reachability count mismatch')
    if not files:
        files = {str(path.relative_to(directory)): file_hash(path) for path in reach_dir.glob('*')}
    done = {tuple(r['key']) for r in records}
    with NativeFallback(**RESIDUE) as fallback:
        for key in sorted(reach.counts, key=key_clock, reverse=True):
            if key in done:
                continue
            tick = time.perf_counter()
            record = sweep_key(key, reach.load(key), store, dth, min_clock=720, full=True, fallback=fallback)
            record['sweep_seconds'] = time.perf_counter()-tick
            path = store.path(key); files[str(path.relative_to(directory))] = file_hash(path)
            records.append(record)
            commit_full_checkpoint(directory, identity, records, files, store, key_clock(key),
                                   elapsed_base+time.perf_counter()-started)
            print(json.dumps(record), flush=True)
    commit_full_checkpoint(directory, identity, records, files, store, 720,
                           elapsed_base+time.perf_counter()-started, finalize=True)
    if sum(r['states'] for r in records) != sum(TOTALS.values()) or len(records) != len(reach.counts):
        raise RuntimeError('full sweep coverage mismatch')
    print('Sweep reached the opening; starting packed coverage and Bellman audit.', flush=True)
    audit_started = time.perf_counter()
    audit = audit_full(directory, reach, records, dth)
    write_json(directory/'audit.json', audit)
    rows = {}
    for kind in TOTALS:
        selected = [r for r in records if r['key'][0] == kind]
        seconds = sum(r['kernel_seconds'] for r in selected)
        rows[kind] = {'states': sum(r['states'] for r in selected),
                      'kernel_seconds': seconds,
                      'classes_per_second': sum(r['states'] for r in selected)/seconds,
                      'failures': sum(r['failures'] for r in selected)}
    report = {'root': audit['root'], 'rows': rows, 'states': sum(TOTALS.values()),
              'keys': len(records), 'failures': sum(r['failures'] for r in records),
              'lp_solves': sum(r['lp_solves'] for r in records),
              'native_solves': sum(r.get('native_solves', 0) for r in records),
              'kernel_native_solves': sum(r.get('kernel_native_solves', 0) for r in records),
              'support_solves': sum(r.get('support_solves', 0) for r in records),
              'native_seconds': sum(r.get('native_seconds', 0.) for r in records),
              'fallback_seconds': sum(r.get('fallback_seconds', r['lp_seconds']) for r in records),
              'cached_failures': sum(r['cached_failures'] for r in records),
              'kernel_seconds': sum(r['kernel_seconds'] for r in records),
              'lp_seconds': sum(r['lp_seconds'] for r in records),
              'wall_seconds': elapsed_base+time.perf_counter()-started,
              'audit_seconds': time.perf_counter()-audit_started,
              'peak_rss_bytes': max(prior_peak, _rss_bytes()),
              'memmap_write_MB_s': store.write_bytes/store.write_seconds/1e6,
              'pack_MB_s': store.pack_bytes/store.pack_seconds/1e6,
              'lp_audit_samples': audit['lp_samples'],
              'max_bellman_residual': audit['max_bellman_residual'],
              'changed_states': audit['changed_states'],
              'mean_baku_gain_over_state_classes': audit['mean_baku_gain_over_state_classes'],
              'max_baku_gain': audit['max_baku_gain'], 'min_baku_gain': audit['min_baku_gain']}
    write_json(directory/'report.json', report)
    for name in ('checkpoint.json', 'audit.json', 'report.json'):
        files[name] = file_hash(directory/name)
    write_json(directory/'manifest.json', {**identity, 'files': files, 'complete': True,
               'stopped_at_clock': 720, 'states': sum(TOTALS.values()),
               'audit': {'packed_coverage': 'all reachable states', 'lp_samples': audit['lp_samples'],
                         'max_bellman_residual': audit['max_bellman_residual']}})
    print(json.dumps(report, indent=2), flush=True)
    return report


def validate_files(directory, files):
    for name, digest in files.items():
        path = Path(directory)/name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'checkpoint file hash mismatch: {name}')

# ────────────────────────────────────────────────────────────
# HiGHS fallback for kernel residue
# ────────────────────────────────────────────────────────────

class WarmLP:
    """We solve each explicit matrix with HiGHS, then check both saddle bounds."""
    def __init__(self, window):
        self.window = window; self.basis = None
        self.highs = highspy.Highs()
        for name, value in [('output_flag', False), ('threads', 1), ('solver', 'simplex'),
                            ('presolve', 'off'), ('primal_feasibility_tolerance', 1e-9),
                            ('dual_feasibility_tolerance', 1e-9)]:
            if self.highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f'HiGHS rejected option {name}')
        rows = 60+int(window)
        self.lp = highspy.HighsLp()
        self.lp.num_col_ = rows+1; self.lp.num_row_ = 61
        self.lp.col_cost_ = np.r_[np.zeros(rows), -1.]
        self.lp.col_lower_ = np.r_[np.zeros(rows), -np.inf]
        self.lp.col_upper_ = np.full(rows+1, np.inf)
        self.lp.row_lower_ = np.r_[np.full(60, -np.inf), 1.]
        self.lp.row_upper_ = np.r_[np.zeros(60), 1.]
        self.lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        self.lp.a_matrix_.start_ = np.arange(rows+2, dtype=np.int32)*61
        self.lp.a_matrix_.index_ = np.tile(np.arange(61, dtype=np.int32), rows+1)

    def solve(self, s, f, *, support=False):
        s = np.asarray(s, np.float64)
        if s.shape != (60,) or not np.isfinite(s).all() or not np.isfinite(f):
            raise ValueError('stage needs finite payoffs')
        matrix = stage_matrix(s, f, self.window)
        coefficients = np.empty((len(matrix)+1, 61))
        coefficients[:-1, :60] = -matrix; coefficients[:-1, 60] = 1.
        coefficients[-1, :60] = 1.; coefficients[-1, 60] = 0.
        self.lp.a_matrix_.value_ = coefficients.ravel()
        status = self.highs.passModel(self.lp)
        if status == highspy.HighsStatus.kError:
            raise RuntimeError('HiGHS rejected the explicit stage matrix')
        if self.basis is not None:
            if self.highs.setBasis(self.basis) == highspy.HighsStatus.kError:
                raise RuntimeError('HiGHS rejected its saved basis')
        self.highs.run()
        if self.highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            self.basis = None
            return solve_lp(s, f, self.window, support=support)
        solution = self.highs.getSolution()
        p = np.maximum(solution.col_value[:-1], 0.)
        q = np.maximum(-np.asarray(solution.row_dual[:60]), 0.)
        p /= p.sum(); q /= q.sum()
        lower = float(np.min(p @ matrix)); upper = float(np.max(matrix @ q))
        if not np.isfinite(upper-lower) or upper-lower > 1e-8:
            # We discard an inaccurate warm basis and use the scalar LP authority.
            self.basis = None
            return solve_lp(s, f, self.window, support=support)
        self.basis = self.highs.getBasis()
        return StageResult((lower+upper)/2, 3, lower, upper, policy_masks(p, q) if support else None)


class RevivalCache:
    """We reuse a certified value only for identical child tables and profiles."""
    def __init__(self):
        self.groups = OrderedDict(); self.tokens = {}

    def _token(self, fail):
        return self.tokens.setdefault(fail, len(self.tokens))

    def get(self, success, fail, window, pcs, pds):
        output = np.full(len(pcs), np.nan)
        key = (success, window)
        if key not in self.groups:
            return output
        self.groups.move_to_end(key)
        values, tokens = self.groups[key]; cols = profiles().idx0[pds]
        eligible = (cols >= 0) & (cols < 183)
        indices = np.flatnonzero(eligible)
        indices = indices[tokens[pcs[indices], cols[indices]] == self._token(fail)]
        output[indices] = values[pcs[indices], cols[indices]]
        return output

    def put(self, success, fail, window, pcs, pds, output):
        cols = profiles().idx0[pds]; eligible = (cols >= 0) & (cols < 183)
        if not eligible.any():
            return
        if not np.isfinite(output).all():
            raise ValueError('cache requires finite certified values')
        key = (success, window)
        if key not in self.groups:
            if len(self.groups) == 2:
                self.groups.popitem(last=False)
            self.groups[key] = (np.full((N, 183), np.nan), np.full((N, 183), -1, np.int32))
        values, tokens = self.groups[key]
        values[pcs[eligible], cols[eligible]] = output[eligible]
        tokens[pcs[eligible], cols[eligible]] = self._token(fail)


@lru_cache(maxsize=2)
def _table(path):
    return np.load(path, mmap_mode='r')


@lru_cache(maxsize=2)
def _solver(window):
    return WarmLP(window)


def _solve_chunk(job):
    succ_path, fail_path, pcs, pds, window, compressed, support = job
    succ = _table(succ_path); fail = _table(fail_path)
    p = profiles(); solver = _solver(False if support else window); result = np.empty(len(pcs))
    masks = np.zeros((len(pcs), 2), np.uint64) if support else None
    for i, (pc, pd) in enumerate(zip(pcs, pds)):
        col = p.idx0[p.fail[pc]] if compressed else p.fail[pc]
        s = -succ[pd, p.succ[pc]]
        f = p.rev[pc]*-fail[pd, col]+1-p.rev[pc]
        solved = solver.solve(s, f, support=support)
        result[i] = max(solved.value, f) if support and window else solved.value
        if support:
            masks[i] = solved.support
    return (result, masks) if support else result


class FallbackPool:
    """We send independent LP batches to processes with one HiGHS thread each."""
    def __init__(self, workers=None):
        self.workers = workers or min(16, os.cpu_count() or 1)
        self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def solve(self, succ_path, fail_path, pcs, pds, window, compressed, *, support=False):
        if self.executor is None:
            self.executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'))
        chunk = max(1 if support else 32, min(2048, len(pcs)//self.workers))
        jobs = [(str(succ_path), str(fail_path), pcs[i:i+chunk], pds[i:i+chunk], window, compressed, support)
                for i in range(0, len(pcs), chunk)]
        output = np.empty(len(pcs)); offset = 0
        masks = np.zeros((len(pcs), 2), np.uint64) if support else None
        for response in self.executor.map(_solve_chunk, jobs):
            values, policies = response if support else (response, None)
            output[offset:offset+len(values)] = values
            if support:
                masks[offset:offset+len(values)] = policies
            offset += len(values)
        return (output, masks) if support else output


class Fallback:
    def __init__(self, workers=None):
        self.pool = FallbackPool(workers); self.cache = RevivalCache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pool.__exit__(*args)

    def solve(self, success, failure, succ_table, fail_table, pcs, pds, window):
        output = self.cache.get(success, failure, window, pcs, pds)
        missing = np.flatnonzero(np.isnan(output))
        if len(missing):
            output[missing] = self.pool.solve(succ_table.filename, fail_table.filename,
                    pcs[missing], pds[missing], window, failure is not None and failure[0] == 'REV')
            self.cache.put(success, failure, window, pcs[missing], pds[missing], output[missing])
        return output, len(missing), len(pcs)-len(missing)


class NativeFallback(Fallback):
    """We try certified native pivots, then send the remaining residue to HiGHS.

    With crash, each packing LP starts from the recurrence basis. With
    kernel_native, sweep_key asks the kernel to solve H1 and H2 residue inside
    its workers, and to try kink_attempts moved supports before each LP. REV
    keys keep this Python path so they can reuse cached revival values.
    """
    def __init__(self, workers=None, *, crash=False, kernel_native=False, kink_attempts=0):
        super().__init__(workers)
        if kink_attempts < 0 or (kink_attempts and not kernel_native):
            raise ValueError('kink attempts need the kernel residue mode')
        self.crash = crash; self.kernel_native = kernel_native; self.kink_attempts = kink_attempts
        self.native_solves = 0
        self.native_seconds = self.highs_seconds = 0.

    @property
    def config(self):
        return {'crash': self.crash, 'kernel_native': self.kernel_native,
                'kink_attempts': self.kink_attempts}

    def kernel_options(self, key):
        """We return the kernel keywords for one key; REV keys keep the cache path."""
        if not self.kernel_native or key[0] == 'REV':
            return {}
        return {'native': True, 'crash': self.crash, 'kink_attempts': self.kink_attempts}

    def solve(self, success, failure, succ_table, fail_table, pcs, pds, window):
        output = self.cache.get(success, failure, window, pcs, pds)
        missing = np.flatnonzero(np.isnan(output)); cached = len(pcs)-len(missing)
        if not len(missing):
            return output, 0, cached
        tick = time.perf_counter(); p = profiles()
        cs = pcs[missing]; ds = pds[missing]
        compressed = failure is not None and failure[0] == 'REV'
        columns = p.idx0[p.fail[cs]] if compressed else p.fail[cs]
        s = np.ascontiguousarray(-succ_table[ds[:,None], p.succ[cs]])
        f = np.ascontiguousarray(p.rev[cs]*-fail_table[ds,columns]+1-p.rev[cs])
        values = np.empty(len(missing)); kind = np.empty(len(missing),np.uint8)
        failed = solve_packing(s,f,window,values,kind,crash=self.crash)
        self.native_seconds += time.perf_counter()-tick
        self.native_solves += len(missing)-failed
        residual = np.flatnonzero(kind == 255)
        if len(residual):
            tick = time.perf_counter()
            values[residual] = self.pool.solve(succ_table.filename,fail_table.filename,
                    cs[residual],ds[residual],window,compressed)
            self.highs_seconds += time.perf_counter()-tick
        if not np.isfinite(values).all():
            raise RuntimeError('uncertified native/HiGHS residue')
        output[missing] = values
        self.cache.put(success,failure,window,cs,ds,values)
        return output, len(residual), cached

# ────────────────────────────────────────────────────────────
# Packed-value audit
# ────────────────────────────────────────────────────────────

class PackedReader:
    """We gather values by bitmap rank without restoring dense tables."""
    def __init__(self, directory):
        self.directory = Path(directory)

    @lru_cache(maxsize=32)
    def _open(self, key):
        prefix = self.directory/key_name(key)
        bits = np.load(prefix.with_suffix('.bits.npy'), mmap_mode='r')
        values = np.load(prefix.with_suffix('.values.npy'), mmap_mode='r')
        if bits.dtype != np.uint8 or bits.shape != ((N*key_columns(key)+7)//8,):
            raise ValueError('packed bitmap schema mismatch')
        if values.dtype != np.float64 or values.ndim != 1:
            raise ValueError('packed value schema mismatch')
        totals = np.add.reduceat(POP[bits].astype(np.int64), np.arange(0, len(bits), 4096))
        ranks = np.r_[np.int64(0), np.cumsum(totals)]
        if ranks[-1] != len(values):
            raise ValueError('packed bitmap and value count mismatch')
        return bits, values, ranks

    def get(self, key, rows, cols):
        rows, cols = np.broadcast_arrays(np.asarray(rows, np.int64), np.asarray(cols, np.int64))
        width = key_columns(key)
        if np.any((rows < 0) | (rows >= N) | (cols < 0) | (cols > width)):
            raise ValueError('packed index out of range')
        output = np.full(rows.shape, -1., dtype=np.float64)
        live = cols != width
        if not live.any():
            return output
        flat = rows[live]*width+cols[live]
        byte, bit = flat//8, flat % 8
        bits, values, ranks = self._open(tuple(key))
        if np.any((bits[byte] & (1 << bit)) == 0):
            raise ValueError('unreachable packed state')
        blocks = byte//4096
        offsets = np.empty(len(byte), np.int64)
        for block in np.unique(blocks):
            selected = blocks == block; start = int(block)*4096
            prefix = np.r_[np.int64(0), np.cumsum(POP[bits[start:start+4096]], dtype=np.int64)]
            offsets[selected] = ranks[block]+prefix[byte[selected]-start]
        offsets += POP[(bits[byte] & ((1 << bit)-1)).astype(np.uint8)]
        output[live] = values[offsets]
        if not np.isfinite(output).all():
            raise ValueError('packed values must be finite')
        return output

    def sample(self, key, count, rng):
        bits, values, ranks = self._open(tuple(key))
        targets = rng.choice(len(values), min(count, len(values)), replace=False)
        flat = []
        for target in targets:
            block = int(np.searchsorted(ranks, target, side='right')-1)
            start = block*4096; within = int(target-ranks[block])
            cumulative = np.cumsum(POP[bits[start:start+4096]], dtype=np.int64)
            byte = int(np.searchsorted(cumulative, within, side='right'))
            before = int(cumulative[byte-1]) if byte else 0
            set_bits = np.flatnonzero(np.unpackbits(np.array([bits[start+byte]]), bitorder='little'))
            flat.append((start+byte)*8+int(set_bits[within-before]))
        return np.divmod(np.array(flat, np.int64), key_columns(key))


def audit_packed_key(directory, key, reached, expected, dth=None):
    prefix = Path(directory)/key_name(key); cols = key_columns(key)
    bits = np.load(prefix.with_suffix('.bits.npy'), mmap_mode='r')
    values = np.load(prefix.with_suffix('.values.npy'), mmap_mode='r')
    if bits.dtype != np.uint8 or bits.shape != ((N*cols+7)//8,):
        raise ValueError('packed bitmap schema mismatch')
    if values.dtype != np.float64 or values.shape != (expected,):
        raise ValueError('packed reachability/value count mismatch')
    offset = 0; gain_sum = 0.; changed = 0; min_gain = np.inf; max_gain = -np.inf; largest = None
    for lo in range(0, N, 64):
        mask = np.unpackbits(reached[lo:lo+64], axis=1, count=cols, bitorder='little').astype(bool)
        packed = np.packbits(mask.reshape(-1), bitorder='little')
        if not np.array_equal(bits[lo*cols//8:lo*cols//8+len(packed)], packed):
            raise ValueError(f'packed reachability mismatch: {key}')
        count = int(mask.sum()); data = values[offset:offset+count]; offset += count
        if not np.isfinite(data).all():
            raise ValueError(f'packed values must be finite: {key}')
        if np.any(np.abs(data) > 1+1e-9):
            raise ValueError(f'packed value outside utility range: {key}')
        if dth is not None and count:
            baseline = dth[lo:lo+len(mask)]
            baseline = baseline[:, profiles().s0] if key[0] == 'REV' else baseline[:, :N]
            gain = baseline[mask]-data if key[0] == 'H1' else data-baseline[mask]
            gain_sum += float(gain.sum()); changed += int((gain > 1e-8).sum())
            min_gain = min(min_gain, float(gain.min()))
            if float(gain.max()) > max_gain:
                max_gain = float(gain.max())
                position = int(np.flatnonzero(mask)[int(gain.argmax())])
                row, col = divmod(position, cols)
                largest = [lo+row, int(profiles().s0[col]) if key[0] == 'REV' else col]
    if offset != expected:
        raise ValueError(f'packed reachability count mismatch: {key}')
    # We allow the accumulated prefix error here; each stage still uses 1e-6.
    if dth is not None and min_gain < -1.3e-5:
        raise ValueError(f'Baku action-set monotonicity failed: {key}: {min_gain}')
    return {'states': offset, 'baku_gain_sum': gain_sum, 'changed_states': changed,
            'min_baku_gain': min_gain if dth is not None else None,
            'max_baku_gain': max_gain if dth is not None else None, 'largest_gain_profiles': largest}


def certify_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('matrix must contain finite payoffs')
    rows, cols = matrix.shape
    failure = 'opening/audit LP failed the 1e-6 gate'
    attempts = [('highs', None), ('highs-ipm', None),
                ('highs-ipm', float(matrix[-1, 0])), ('highs-ipm', float(matrix[0, 0]))]
    for method, offset in attempts:
        lp_matrix = matrix if offset is None else matrix-offset
        result = linprog(np.r_[np.zeros(rows), -1.],
                         A_ub=np.c_[-lp_matrix.T, np.ones(cols)], b_ub=np.zeros(cols),
                         A_eq=[np.r_[np.ones(rows), 0.]], b_eq=[1.],
                         bounds=[(0, None)]*rows+[(None, None)], method=method,
                         options={'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        if not result.success:
            failure = f'opening/audit HiGHS failure: {result.message}'
            continue
        drop = np.maximum(result.x[:-1], 0.); check = np.maximum(-result.ineqlin.marginals, 0.)
        drop /= drop.sum(); check /= check.sum()
        lower = float(np.min(drop @ matrix)); upper = float(np.max(matrix @ check))
        if not np.isfinite(upper-lower) or upper-lower > 1e-6:
            failure = 'opening/audit LP failed the 1e-6 gate'
            continue
        return {'value': (lower+upper)/2, 'lower': lower, 'upper': upper, 'gap': upper-lower,
                'drop': drop.tolist(), 'check': check.tolist()}
    raise RuntimeError(failure)


def explicit_matrix(s, f, window):
    # We assemble cells from the action comparison for an independent LP audit.
    drop, check = np.indices((60+int(window), 60))
    matrix = np.full(drop.shape, f, dtype=np.float64)
    success = check >= drop
    matrix[success] = s[(check-drop)[success]]
    return matrix


def stage_inputs(reader, key, pc, pd, dth):
    p = profiles()
    def child_values(dest, columns):
        if dest is None:
            return dth[pd, columns]
        mapped = p.idx0[columns] if dest[0] == 'REV' else columns
        return reader.get(dest, pd, mapped)
    s = -child_values(child_key(key), p.succ[pc])
    f = 1.
    if p.rev[pc]:
        f = p.rev[pc] * -float(child_values(child_key(key, int(p.st[pc])+60), p.fail[pc])) + (1-p.rev[pc])
    return s, f


def audit_full(directory, reach, records, dth, samples_per_key=2):
    directory = Path(directory); reader = PackedReader(directory/'tables')
    rng = np.random.default_rng(93482)
    checked = samples = 0; worst_residual = worst_gap = 0.; results = {}
    for key in reach.counts:
        info = audit_packed_key(reader.directory, key, reach.load(key), reach.counts[key], dth)
        checked += info['states']; results[key_name(key)] = info
        rows, cols = reader.sample(key, 200 if key == ('H2', 57) else samples_per_key, rng)
        for pc, col in zip(rows, cols):
            pd = int(profiles().s0[col]) if key[0] == 'REV' else int(col)
            s, f = stage_inputs(reader, key, int(pc), pd, dth)
            certificate = certify_matrix(explicit_matrix(s, f, is_window(key)))
            stored = float(reader.get(key, pc, col))
            residual = abs(stored-certificate['value'])
            if residual > 1e-6:
                raise RuntimeError(f'Bellman audit failed: {key} {pc} {pd}: {residual}')
            worst_residual = max(worst_residual, residual)
            worst_gap = max(worst_gap, certificate['gap']); samples += 1
        if key[0] == 'H1':
            print(f'audit {key_name(key)}: {info["states"]:,} states', flush=True)
    if checked != sum(TOTALS.values()) or checked != sum(r['states'] for r in records):
        raise RuntimeError('full audit state count mismatch')
    p = profiles(); s, f = stage_inputs(reader, ('H1', 12), 0, 0, dth)
    root = certify_matrix(explicit_matrix(s, f, False))
    root['stored_value'] = float(reader.get(('H1', 12), 0, 0))
    root['hal_win_probability'] = (1+root['stored_value'])/2
    root['baku_win_probability'] = (1-root['stored_value'])/2
    root['dth_value'] = float(dth[0, 0])
    root['baku_gain_probability_points'] = 50*(root['dth_value']-root['stored_value'])
    root['success_payoffs'] = s.tolist(); root['failure_payoff'] = f
    root['dth_policies'] = certify_matrix(explicit_matrix(-dth[0, p.succ[0]],
                            p.rev[0]*-dth[0, p.fail[0]]+(1-p.rev[0]), False))
    return {'states_checked': checked, 'lp_samples': samples,
            'max_bellman_residual': worst_residual, 'max_lp_gap': worst_gap,
            'root': root, 'keys': results,
            'changed_states': sum(r['changed_states'] for r in results.values()),
            'mean_baku_gain_over_state_classes': sum(r['baku_gain_sum'] for r in results.values())/checked,
            'min_baku_gain': min(r['min_baku_gain'] for r in results.values()),
            'max_baku_gain': max(r['max_baku_gain'] for r in results.values())}

# ────────────────────────────────────────────────────────────
# Verified read path
# ────────────────────────────────────────────────────────────

# A completed artifact opens read-only. open_leap checks the manifest before
# it returns a table: the schema is stl-leap-values-v1, the builder marked the
# artifact complete, and the SHA-256 of the DTH table equals dth_sha256. With
# verify="files" it also re-hashes each file that the manifest lists, which
# reads the whole artifact. A key names a table as the builder does:
# ("H1", minute), ("H2", minute) or ("REV", clock). checker and dropper are
# profile indices that LeapTable.profile_index returns.

SCHEMA = "stl-leap-values-v1"


TOLERANCE = 1e-6


VERIFY_MODES = ("manifest", "files")


@dataclass(frozen=True)
class LeapIdentity:
    """The facts that bind a figure or a policy to one artifact."""

    artifact: Path
    dth_values: Path
    schema: str
    builder_sha256: str
    dth_sha256: str
    verified: str


class LeapTable:
    """Stored values and stage certificates of one verified leap artifact."""

    def __init__(self, identity: LeapIdentity, dth: np.ndarray):
        self._identity = identity
        self._reader = PackedReader(identity.artifact / "tables")
        self._dth = dth

    @property
    def identity(self) -> LeapIdentity:
        return self._identity

    def profile_index(self, st: int, ttd: int) -> int:
        """Return the quotient profile of a player with squandered time st and TTD ttd."""

        p = profiles()
        found = np.flatnonzero((p.st == st) & (p.ttd == ttd))
        if len(found) != 1:
            raise ValueError(f"no profile has st={st} and ttd={ttd}")
        return int(found[0])

    def value(self, key: tuple[str, int], checker: int, dropper: int) -> float:
        """Return the stored value of one state, from the Dropper's side.

        The implicit WIN dropper profile reads as -1, as the builder stores it.
        """

        key = (str(key[0]), int(key[1]))
        if not 0 <= dropper <= WIN:
            raise ValueError(f"dropper profile {dropper} is out of range")
        column = int(profiles().idx0[dropper]) if key[0] == "REV" else dropper
        if column < 0:
            raise ValueError(f"{key_name(key)} stores no column for dropper profile {dropper}")
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
        # certify_matrix raises RuntimeError when no LP attempt passes its 1e-6 gap gate.
        try:
            certificate = certify_matrix(explicit_matrix(success, failure, is_window(key)))
        except RuntimeError as error:
            raise ValueError(f"{where}: {error}") from error
        residual = abs(stored - certificate["value"])
        if residual > TOLERANCE:
            raise ValueError(f"{where}: Bellman residual {residual:.3g} exceeds {TOLERANCE:g}")
        if certificate["gap"] > TOLERANCE:
            raise ValueError(f"{where}: saddle gap {certificate['gap']:.3g} exceeds {TOLERANCE:g}")
        return {"stored_value": stored, "bellman_residual": residual, **certificate}


def open_leap(
    artifact: str | os.PathLike[str] = DEFAULT_ARTIFACT,
    *,
    dth: str | os.PathLike[str] = DEFAULT_PATH,
    verify: str = "manifest",
) -> LeapTable:
    """Open a completed leap artifact after its manifest checks pass."""

    if verify not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}, not {verify!r}")
    root = Path(artifact).resolve()
    dth_path = Path(dth).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no leap manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{manifest_path}: schema {manifest.get('schema')!r} is not {SCHEMA!r}")
    if manifest.get("complete") is not True:
        raise ValueError(f"{manifest_path}: the builder did not mark the artifact complete")
    if file_hash(dth_path) != manifest["dth_sha256"]:
        raise ValueError(f"{dth_path}: the DTH tail differs from the audited solve")
    if verify == "files":
        for name, expected in manifest["files"].items():
            if file_hash(root / name) != expected:
                raise ValueError(f"{root / name}: the file differs from its manifest hash")
    identity = LeapIdentity(
        artifact=root,
        dth_values=dth_path,
        schema=SCHEMA,
        builder_sha256=str(manifest["builder_sha256"]),
        dth_sha256=str(manifest["dth_sha256"]),
        verified=verify,
    )
    return LeapTable(identity, np.load(dth_path, mmap_mode="r"))


def main(argv=None):
    """Build, or resume, the full certified sweep to the clock-720 opening."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--dth', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)
    run_full(args.dth, args.output)


if __name__ == '__main__':
    main()
