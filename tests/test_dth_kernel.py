"""stl.dth_kernel: value storage, layer enumeration, the fused numba ladder against the scalar oracle, and the
layer sweep on the dead band with recheck and the anchors."""
import numpy as np
import numba as nb
import pytest

from stl import dth as m
from stl import dth_kernel as kernel

N = m.N
DEAD_ANCHOR = ((240, 0, 240, 0), 0.3372132166291093)
STOP = 1082   # the dead-band anchor's layer; its whole subtree lives in layers >= 1082 (dead x dead only)


@pytest.fixture(scope="module")
def T():
    return m.build_table()


@pytest.fixture(scope="module")
def counts(T):
    sizes = np.array([len(b) for b in T.bucket], dtype=np.int64)
    return np.convolve(sizes, sizes)


def quiet(_msg):
    pass


def test_storage_layout():
    V = kernel.values_array()
    assert V.shape == (N, N + 1) and V.dtype == np.float64 and V.flags.c_contiguous
    assert np.isnan(V[:, :N]).all() and (V[:, m.WIN] == -1.0).all()
    K = kernel.kinds_array()
    assert K.shape == (N, N) and K.dtype == np.uint8 and (K == m.UNSOLVED).all()


def test_layer_pairs(T, counts):
    total = 0
    for P in range(m.MAX_LAYER + 1):
        pcs, pds = kernel.layer_pairs(P, T)
        assert pcs.dtype == np.int32 and pds.dtype == np.int32
        assert pcs.size == counts[P]
        if pcs.size:
            assert (T.phi[pcs] + T.phi[pds] == P).all()
            keys = pcs.astype(np.int64) * N + pds
            assert np.unique(keys).size == keys.size
        total += pcs.size
    assert total == m.CLASSES
    pcs, pds = kernel.layer_pairs(374, T)
    a0 = max(0, 374 - m.MAX_PHI)
    while T.bucket[a0].size == 0 or T.bucket[374 - a0].size == 0:
        a0 += 1
    k = T.bucket[a0].size
    assert (pds[:k] == pds[0]).all() and np.array_equal(pcs[:k], T.bucket[a0])


def test_kernel_matches_oracle_on_one_layer(T):
    # smooth synthetic values so rung 2 certifies often; children live in other layers, so kernel writes
    # inside layer 374 never feed the oracle's reads
    rng = np.random.default_rng(0)
    V = 0.9 * np.sin(np.linspace(0, 6, N + 1)[None, :] + np.linspace(0, 3, N)[:, None])
    V += rng.normal(0, 0.005, size=V.shape)
    V[:, m.WIN] = -1.0
    K = kernel.kinds_array()
    pcs, pds = kernel.layer_pairs(374, T)
    pick = rng.choice(pcs.size, 3000, replace=False)
    pcs, pds = np.ascontiguousarray(pcs[pick]), np.ascontiguousarray(pds[pick])
    before = V[pcs, pds].copy()
    scratch = np.empty((nb.get_num_threads(), 3, m.LAGS))
    need_lp = np.empty(pcs.size, dtype=np.bool_)
    kernel._solve_layer_kernel(pcs, pds, V, K, T.success_children, T.fail_child, T.rev, scratch, need_lp)
    kinds = K[pcs, pds]
    assert np.array_equal(kinds == m.UNSOLVED, need_lp)
    routed = np.bincount(kinds[~need_lp], minlength=3)
    assert routed[m.SUPPORT] > 0                       # the smooth data exercises rung 2
    lp_disagreements = 0
    for i in range(pcs.size):
        pc, pd = int(pcs[i]), int(pds[i])
        s, f = m.class_values(pc, pd, V, T)                # children live in other layers: untouched by these writes
        if need_lp[i]:
            r1 = m.try_rung1(s, f)
            r2 = m.try_rung2(s, f)
            if r1.certified or (r2 is not None and r2.certified):
                lp_disagreements += 1                  # rounding at the certificate edge; must be rare
            continue
        expected = m.solve_class(s, f)
        assert abs(V[pc, pd] - expected.value) <= m.MAX_SADDLE_GAP + 1e-12
    assert lp_disagreements <= 3
    V[pcs, pds] = before
    print(f"\nkernel routing on 3000 synthetic classes (pure, support, lp): {routed[0]} {routed[1]} {need_lp.sum()}")


def random_batch(rng, B):
    return rng.uniform(-1, 1, (B, 60)), rng.uniform(-1, 1, B)


def structured_batch(rng, B):
    return np.sort(rng.uniform(-1, 1, (B, 60)), axis=1), rng.uniform(-1, 1, B)


def ladder(s, f):
    r = np.empty(60)
    q = np.empty(60)
    return kernel._ladder(np.ascontiguousarray(s, dtype=np.float64), float(f), r, q)


def test_ladder_matches_scalar_rungs():
    rng = np.random.default_rng(5)
    edge_cases = 0
    for maker in (random_batch, structured_batch):
        S, F = maker(rng, 300)
        for i in range(300):
            value, kind = ladder(S[i], F[i])
            r1 = m.try_rung1(S[i], F[i])
            r2 = m.try_rung2(S[i], F[i])
            if kind == m.PURE:
                assert r1.certified and value == pytest.approx(r1.value, abs=1e-12)
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
    assert edge_cases <= 3


def test_ladder_matches_solve_class():
    rng = np.random.default_rng(7)
    S, F = structured_batch(rng, 200)
    S[0] = 0.5; F[0] = 0.5
    S[1] = 1.0; F[1] = 1.0
    S[2] = np.linspace(-1, 1, 60); F[2] = -1.0
    kinds = []
    for i in range(200):
        value, kind = ladder(S[i], F[i])
        kinds.append(kind)
        expected = m.solve_class(S[i], F[i])
        if kind != m.LP:
            assert abs(value - expected.value) <= m.MAX_SADDLE_GAP + 1e-12
    assert kinds[0] == m.PURE and ladder(S[0], F[0])[0] == 0.5
    assert kinds[1] == m.PURE and ladder(S[1], F[1])[0] == 1.0
    assert kinds[2] == m.PURE and ladder(S[2], F[2])[0] == -1.0
    print(f"\nladder routing on the structured batch (pure, support, lp): {np.bincount(kinds, minlength=3)}")


def test_dead_band_anchor(T, counts):
    V, K, routed = kernel.build(T, stop_at=STOP, log=quiet)
    pc = pd = m.profile(240, 0, T)
    assert abs(V[pc, pd] - DEAD_ANCHOR[1]) <= m.MAX_SADDLE_GAP
    solved = ~np.isnan(V[:, :m.N])
    assert np.array_equal(solved, K != m.UNSOLVED)
    assert solved.sum() == counts[STOP:].sum() == routed.sum()
    pcs, pds = np.nonzero(solved)
    assert (T.phi[pcs] + T.phi[pds] >= STOP).all()
    assert (np.abs(V[pcs, pds]) <= 1 + 1e-9).all()
    # every solved class re-derives independently to within the gate
    for i in range(0, pcs.size, max(1, pcs.size // 60)):
        c = int(pcs[i]) * m.N + int(pds[i])
        assert abs(m.recheck(c, V, T).value - V[pcs[i], pds[i]]) <= m.MAX_SADDLE_GAP
    ok = {state: good for state, _e, _g, good in m.verify_anchors(V, T)}
    assert ok[DEAD_ANCHOR[0]]
