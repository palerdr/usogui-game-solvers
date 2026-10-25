# Copyright 2026 The STL authors
"""Test value storage, layer enumeration, and the fused numba ladder.

Compare the scalar oracle and dead-band sweep with rechecks and anchors.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import pytest

from stl import dth as m
from stl import dth_kernel as kernel

N = m.N
DEAD_ANCHOR = ((240, 0, 240, 0), 0.3372132166291093)
# The dead-band anchor and its subtree occupy layers >= 1082 (dead x dead).
STOP = 1082


@pytest.fixture(scope="module")
def table() -> m.ProfileTable:
    return m.build_table()


@pytest.fixture(scope="module")
def counts(table: m.ProfileTable) -> np.ndarray:
    sizes = np.array([len(b) for b in table.bucket], dtype=np.int64)
    return np.convolve(sizes, sizes)


def quiet(_msg: str) -> None:
    pass


def test_storage_layout() -> None:
    values = kernel.values_array()
    assert values.shape == (N, N + 1)
    assert values.dtype == np.float64
    assert values.flags.c_contiguous
    assert np.isnan(values[:, :N]).all()
    assert (values[:, m.WIN] == -1.0).all()
    kinds = kernel.kinds_array()
    assert kinds.shape == (N, N)
    assert kinds.dtype == np.uint8
    assert (kinds == m.UNSOLVED).all()


def test_layer_pairs(table: m.ProfileTable, counts: np.ndarray) -> None:
    total = 0
    for potential in range(m.MAX_LAYER + 1):
        pcs, pds = kernel.layer_pairs(potential, table)
        assert pcs.dtype == np.int32
        assert pds.dtype == np.int32
        assert pcs.size == counts[potential]
        if pcs.size:
            assert (table.phi[pcs] + table.phi[pds] == potential).all()
            keys = pcs.astype(np.int64) * N + pds
            assert np.unique(keys).size == keys.size
        total += pcs.size
    assert total == m.CLASSES
    pcs, pds = kernel.layer_pairs(374, table)
    a0 = max(0, 374 - m.MAX_PHI)
    while table.bucket[a0].size == 0 or table.bucket[374 - a0].size == 0:
        a0 += 1
    k = table.bucket[a0].size
    assert (pds[:k] == pds[0]).all()
    assert np.array_equal(pcs[:k], table.bucket[a0])


def test_kernel_matches_oracle_on_one_layer(table: m.ProfileTable) -> None:
    # smooth synthetic values so rung 2 certifies often; children live in other
    # layers, so kernel writes
    # inside layer 374 never feed the oracle's reads
    rng = np.random.default_rng(0)
    values = 0.9 * np.sin(
        np.linspace(0, 6, N + 1)[None, :] + np.linspace(0, 3, N)[:, None]
    )
    values += rng.normal(0, 0.005, size=values.shape)
    values[:, m.WIN] = -1.0
    kinds = kernel.kinds_array()
    pcs, pds = kernel.layer_pairs(374, table)
    pick = rng.choice(pcs.size, 3000, replace=False)
    pcs, pds = np.ascontiguousarray(pcs[pick]), np.ascontiguousarray(pds[pick])
    before = values[pcs, pds].copy()
    scratch = np.empty((nb.get_num_threads(), 3, m.LAGS))
    need_lp = np.empty(pcs.size, dtype=np.bool_)
    kernel._solve_layer_kernel(  # noqa: SLF001
        pcs,
        pds,
        values,
        kinds,
        table.success_children,
        table.fail_child,
        table.rev,
        scratch,
        need_lp,
    )
    kinds = kinds[pcs, pds]
    assert np.array_equal(kinds == m.UNSOLVED, need_lp)
    routed = np.bincount(kinds[~need_lp], minlength=3)
    assert routed[m.SUPPORT] > 0  # the smooth data exercises rung 2
    lp_disagreements = 0
    for i in range(pcs.size):
        pc, pd = int(pcs[i]), int(pds[i])
        s, f = m.class_values(
            pc, pd, values, table
        )  # children live in other layers: untouched by these writes
        if need_lp[i]:
            r1 = m.try_rung1(s, f)
            r2 = m.try_rung2(s, f)
            if r1.certified or (r2 is not None and r2.certified):
                lp_disagreements += 1  # rounding at the certificate edge; must be rare
            continue
        expected = m.solve_class(s, f)
        assert abs(values[pc, pd] - expected.value) <= m.MAX_SADDLE_GAP + 1e-12
    assert lp_disagreements <= 3  # noqa: PLR2004
    values[pcs, pds] = before
    print(  # noqa: T201
        "\nkernel routing on 3000 synthetic classes (pure, support, lp): "
        f"{routed[0]} {routed[1]} {need_lp.sum()}"
    )


def random_batch(
    rng: np.random.Generator, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    return rng.uniform(-1, 1, (batch_size, 60)), rng.uniform(-1, 1, batch_size)


def structured_batch(
    rng: np.random.Generator, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    return np.sort(rng.uniform(-1, 1, (batch_size, 60)), axis=1), rng.uniform(
        -1, 1, batch_size
    )


def ladder(s: np.ndarray, f: float) -> tuple[float, int]:
    r = np.empty(60)
    q = np.empty(60)
    return kernel._ladder(np.ascontiguousarray(s, dtype=np.float64), float(f), r, q)  # noqa: SLF001


def test_ladder_matches_scalar_rungs() -> None:
    rng = np.random.default_rng(5)
    edge_cases = 0
    for maker in (random_batch, structured_batch):
        success, failed = maker(rng, 300)
        for i in range(300):
            value, kind = ladder(success[i], failed[i])
            r1 = m.try_rung1(success[i], failed[i])
            r2 = m.try_rung2(success[i], failed[i])
            if kind == m.PURE:
                assert r1.certified
                assert value == pytest.approx(r1.value, abs=1e-12)
            elif kind == m.SUPPORT:
                assert not r1.certified
                if r2 is None or not r2.certified:
                    edge_cases += 1
                else:
                    assert value == pytest.approx(r2.value, abs=1e-9)
            else:
                assert not r1.certified
                if r2 is not None and r2.certified:
                    edge_cases += 1
    assert edge_cases <= 3  # noqa: PLR2004


def test_ladder_matches_solve_class() -> None:
    rng = np.random.default_rng(7)
    success, failed = structured_batch(rng, 200)
    success[0] = 0.5
    failed[0] = 0.5
    success[1] = 1.0
    failed[1] = 1.0
    success[2] = np.linspace(-1, 1, 60)
    failed[2] = -1.0
    kinds = []
    for i in range(200):
        value, kind = ladder(success[i], failed[i])
        kinds.append(kind)
        expected = m.solve_class(success[i], failed[i])
        assert expected.value is not None
        if kind != m.LP:
            assert abs(value - expected.value) <= m.MAX_SADDLE_GAP + 1e-12
    assert kinds[0] == m.PURE
    assert ladder(success[0], failed[0])[0] == 0.5  # noqa: PLR2004
    assert kinds[1] == m.PURE
    assert ladder(success[1], failed[1])[0] == 1.0
    assert kinds[2] == m.PURE
    assert ladder(success[2], failed[2])[0] == -1.0
    print(  # noqa: T201
        "\nladder routing on the structured batch (pure, support, lp): "
        f"{np.bincount(kinds, minlength=3)}"
    )


def test_dead_band_anchor(table: m.ProfileTable, counts: np.ndarray) -> None:
    values, kinds, routed = kernel.build(table, stop_at=STOP, log=quiet)
    pc = pd = m.profile(240, 0, table)
    assert abs(values[pc, pd] - DEAD_ANCHOR[1]) <= m.MAX_SADDLE_GAP
    solved = ~np.isnan(values[:, : m.N])
    assert np.array_equal(solved, kinds != m.UNSOLVED)
    assert solved.sum() == counts[STOP:].sum() == routed.sum()
    pcs, pds = np.nonzero(solved)
    assert (table.phi[pcs] + table.phi[pds] >= STOP).all()
    assert (np.abs(values[pcs, pds]) <= 1 + 1e-9).all()
    # every solved class re-derives independently to within the gate
    for i in range(0, pcs.size, max(1, pcs.size // 60)):
        c = int(pcs[i]) * m.N + int(pds[i])
        assert (
            abs(m.recheck(c, values, table).value - values[pcs[i], pds[i]])
            <= m.MAX_SADDLE_GAP
        )
    ok = {state: good for state, _e, _g, good in m.verify_anchors(values, table)}
    assert ok[DEAD_ANCHOR[0]]
