"""We reuse HiGHS bases and certified revival values during the full sweep."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
import os

import highspy
import numpy as np

from stl.solver.leap_oracle import StageResult, policy_masks, solve_lp, stage_matrix
from stl.solver.leap_profiles import N, profiles


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
        import time
        # The spawn workers of FallbackPool import this module and run only
        # HiGHS, so we import the numba kernel where we call it.
        from stl.solver.leap_kernel import solve_packing
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
