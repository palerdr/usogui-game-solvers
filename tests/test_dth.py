# Copyright 2026 The STL authors
"""Profile tables, class encoding, rules, scalar ladder, and Table.

Revival parity with stl.game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from stl import dth as m
from stl.game import Player, Referee

if TYPE_CHECKING:
    from pathlib import Path

N = m.N
ALIVE = m.N_ALIVE


@pytest.fixture(scope="module")
def table() -> m.ProfileTable:
    return m.build_table()


def test_census(table: m.ProfileTable) -> None:
    assert table.alive.sum() == 16711
    assert table.st.size == N == 17011
    assert table.alive_id.shape == (300, 301)
    assert table.rev.dtype == np.float64
    assert table.success_children.shape == (N, 60)
    per_ttd = {
        t: int((table.ttd[:ALIVE] == t).sum()) for t in [0, *list(range(60, 301))]
    }
    assert per_ttd[0] == 240
    for t in range(60, 241):
        assert per_ttd[t] == 241 - t
    for t in range(241, 301):
        assert per_ttd[t] == 0
    # normative order: TTD ascending, ST ascending inside each TTD
    order = np.lexsort((table.st[:ALIVE], table.ttd[:ALIVE]))
    assert np.array_equal(order, np.arange(ALIVE))


def test_spot_ids(table: m.ProfileTable) -> None:
    assert table.alive_id[0, 0] == 0
    assert table.alive_id[239, 0] == 239
    assert table.alive_id[0, 60] == 240
    assert table.alive_id[180, 60] == 420
    assert m.profile(0, 0, table) == 0
    assert m.profile(240, 0, table) == 16951
    assert m.profile(299, 0, table) == 17010
    assert m.profile(240, 300, table) == 16951  # dead: TTD discarded
    with pytest.raises(IndexError):
        m.profile(5, 30, table)  # alive TTD in 1..59 is off-domain


def test_phi_and_buckets(table: m.ProfileTable) -> None:
    assert table.phi[table.alive].min() == 0
    assert table.phi[table.alive].max() == 240
    assert table.phi[~table.alive].min() == 301
    assert table.phi[~table.alive].max() == 600
    assert np.array_equal(table.phi[~table.alive], table.st[~table.alive] + 301)
    sizes = np.array([len(b) for b in table.bucket])
    assert sizes.size == 601
    assert sizes.sum() == N
    assert (sizes[241:301] == 0).all()
    assert (sizes[301:] == 1).all()
    for v, b in enumerate(table.bucket):
        assert b.dtype == np.int32
        assert (table.phi[b] == v).all()
        assert np.array_equal(b, np.sort(b))


def test_succ_runs(table: m.ProfileTable) -> None:
    succ = table.success_children
    assert np.array_equal(succ[0], np.arange(1, 61))  # (0,0): every lag survives
    assert np.array_equal(
        succ[239], np.arange(16951, 17011)
    )  # (239,0): every lag lands dead
    assert (succ[17010] == m.WIN).all()  # (299, dead): every lag hits capacity
    # every row: [consecutive alive ids][consecutive dead ids][WIN...]
    kind = np.where(succ < ALIVE, 0, np.where(succ < N, 1, 2))
    assert (np.diff(kind, axis=1) >= 0).all()
    same = (kind[:, 1:] == kind[:, :-1]) & (kind[:, 1:] != 2)
    assert (np.diff(succ, axis=1)[same] == 1).all()
    assert (kind[~table.alive] >= 1).all()  # dead profiles never have alive children
    # a child is the parent grown by lag, with the same TTD when alive
    lag = np.arange(1, 61)
    valid = succ != m.WIN
    assert (table.st[succ[valid]] == (table.st[:, None] + lag)[valid]).all()
    alive_child = succ < ALIVE
    assert (
        table.ttd[succ[alive_child]]
        == np.broadcast_to(table.ttd[:, None], succ.shape)[alive_child]
    ).all()
    # WIN exactly when grown >= 300
    assert np.array_equal(succ == m.WIN, (table.st[:, None] + lag) >= 300)


def test_fail_and_rev(table: m.ProfileTable) -> None:
    p = m.profile(0, 0, table)
    assert table.fail_child[p] == 240
    assert table.rev[p] == pytest.approx(0.95)
    p = m.profile(180, 60, table)
    assert (
        table.fail_child[p] == ALIVE
    )  # t' = 300 cannot survive: dead sentinel at ST 0
    assert table.rev[p] == pytest.approx(0.178125)
    assert (table.fail_child[~table.alive] == m.WIN).all()
    assert (table.rev[~table.alive] == 0.0).all()
    assert (table.rev[table.alive] > 0.0).all()
    assert (table.rev[table.alive] <= 0.95).all()
    # alive fail-children are revived at ST 0 with TTD = t + s + 60
    fc = table.fail_child[table.alive]
    alive_fc = fc < ALIVE
    assert (table.st[fc[alive_fc]] == 0).all()
    assert (
        table.ttd[fc[alive_fc]] == (table.ttd[:ALIVE] + table.st[:ALIVE] + 60)[alive_fc]
    ).all()


def test_potential_strictly_increases(table: m.ProfileTable) -> None:
    succ, phi = table.success_children, table.phi
    ok = succ != m.WIN
    assert (phi[succ[ok]] > np.broadcast_to(phi[:, None], succ.shape)[ok]).all()
    ok = table.fail_child != m.WIN
    assert (phi[table.fail_child[ok]] > phi[ok]).all()


def test_layer_census(table: m.ProfileTable) -> None:
    sizes = np.array([len(b) for b in table.bucket], dtype=np.int64)
    count = np.convolve(sizes, sizes)  # count[P] = sum_a sizes[a] * sizes[P - a]
    assert count.size == 1201
    assert count.sum() == 289_374_121
    assert count[:601].sum() == 283_297_201
    assert count[601:841].sum() == 6_015_600
    assert count[841:].sum() == 61_320
    assert count.argmax() == 374
    assert count[374] == 1_678_715


def test_encode(table: m.ProfileTable) -> None:
    assert m.encode_state(0, 0, 0, 0, table) == 0
    assert m.encode_state(240, 0, 240, 0, table) == 16951 * N + 16951
    assert m.encode_state(10, 60, 200, 0, table) == m.profile(
        10, 60, table
    ) * N + m.profile(200, 0, table)


def structured_batch(
    rng: np.random.Generator, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    return np.sort(rng.uniform(-1, 1, (batch_size, 60)), axis=1), rng.uniform(
        -1, 1, batch_size
    )


def test_solve_result_post_init() -> None:
    r = m.SolveResult(maximin=0.25, minimax=0.25 + 5e-7, kind=m.SUPPORT)
    assert r.certified
    assert r.saddle_gap == pytest.approx(5e-7)
    assert r.value == pytest.approx(0.25 + 2.5e-7)
    assert r.kind == m.SUPPORT
    r = m.SolveResult(maximin=0.0, minimax=1e-3, kind=m.LP)
    assert not r.certified
    assert r.value is None
    r = m.SolveResult(maximin=np.nan, minimax=np.nan, kind=m.LP)
    assert not r.certified
    assert r.value is None
    r = m.SolveResult(maximin=0.3, minimax=0.3 - 1e-15, kind=m.PURE)
    assert r.certified


def test_m_dot_matches_full_matrix() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        s, f = rng.uniform(-1, 1, 60), rng.uniform(-1, 1)
        q = rng.uniform(0, 1, 60)
        q /= q.sum()
        assert np.allclose(
            m._M_dot(s, f, q),
            m.full_matrix(s, f) @ q,
            atol=1e-12,
            rtol=0,
        )


def test_scalar_bounds_bracket_lp_value() -> None:
    rng = np.random.default_rng(6)
    success, failed = structured_batch(rng, 60)
    for i in range(60):
        res = m.try_rung3(success[i], failed[i])
        assert res.certified
        assert res.kind == m.LP
        v = res.value
        assert v is not None
        r1 = m.try_rung1(success[i], failed[i])
        assert r1.maximin <= v + 2e-6
        assert r1.minimax >= v - 2e-6
        r2 = m.try_rung2(success[i], failed[i])
        if r2 is not None:
            assert r2.maximin <= v + 2e-6
            assert r2.minimax >= v - 2e-6


def test_rungs_return_certifying_strategies() -> None:
    rng = np.random.default_rng(8)
    success, failed = structured_batch(rng, 40)
    success[0] = 0.5
    failed[0] = 0.5
    for i in range(40):
        matrix = m.full_matrix(success[i], failed[i])
        for res in (
            m.try_rung1(success[i], failed[i]),
            m.try_rung2(success[i], failed[i]),
            m.try_rung3(success[i], failed[i]),
        ):
            if res is None:
                continue
            assert res.drop is not None
            assert res.check is not None
            assert res.drop.shape == (60,)
            assert res.check.shape == (60,)
            assert res.drop.min() >= 0
            assert res.check.min() >= 0
            assert res.drop.sum() == pytest.approx(1)
            assert res.check.sum() == pytest.approx(1)
            assert (res.drop @ matrix).min() == pytest.approx(
                res.maximin, abs=1e-12
            )  # L from the Dropper's mix
            assert (matrix @ res.check).max() == pytest.approx(
                res.minimax, abs=1e-12
            )  # U from the Checker's mix
            if res.kind == m.SUPPORT:
                assert np.array_equal(res.drop, res.check[::-1])  # the mirror pair


# ---- frozen rules (ported from the retired src/dth solver contract)
# ------------------------------
def test_frozen_revival_surface_and_inclusive_dose_threshold(
    table: m.ProfileTable,
) -> None:
    assert m.revival_probability(0, 0) == pytest.approx(0.95)
    assert m.revival_probability(0, 60) == pytest.approx(0.95 * 0.75)
    assert m.revival_probability(239, 0) == pytest.approx(0.95 / 240.0)
    assert m.revival_probability(240, 0) == 0.0
    assert (
        table.fail_child[m.profile(240, 0, table)] == m.WIN
    )  # a failed check at ST 240 is fatal


def test_resulting_total_ttd_of_300_stays_revival_eligible(
    table: m.ProfileTable,
) -> None:
    p = m.profile(0, 240, table)
    assert table.rev[p] == m.revival_probability(0, 240) > 0.0
    assert (
        table.fail_child[p] == m.profile(0, 300, table) == ALIVE
    )  # revived at ST 0 with TTD 300


def test_resulting_total_ttd_above_300_is_fatal(table: m.ProfileTable) -> None:
    assert m.revival_probability(0, 241) == 0.0
    assert table.fail_child[m.profile(0, 241, table)] == m.WIN


def test_successful_check_dumps_at_the_cylinder_cap(table: m.ProfileTable) -> None:
    p = m.profile(298, 0, table)
    assert table.success_children[p, 0] == m.profile(299, 0, table)  # ST 299 stays live
    assert table.success_children[p, 1] == m.WIN  # exactly 300 dumps
    assert (
        table.success_children[m.profile(299, 0, table)] == m.WIN
    ).all()  # above 300 dumps
    assert (
        table.success_children[m.profile(240, 0, table), 59] == m.WIN
    )  # exactly 300 by a 60-second lag


# ---- Table and Stage
# ---------------------------------------------------------------------------
class _Values:
    """Stands in for V: every class reads `value`."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __getitem__(self, key: object) -> float:
        return self.value


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0),
        (-1, 0, 0, 0),
        (300, 0, 0, 0),
        (0, 301, 0, 0),
        (0, 0, 0, 301),
        (False, 0, 0, 0),
        (0.0, 0, 0, 0),
    ],
)
def test_table_rejects_malformed_states(table: m.ProfileTable, state: tuple) -> None:
    with pytest.raises(ValueError):
        m.Table(cast("np.ndarray", _Values()), table).value(state)


def test_table_rejects_an_alive_ttd_below_60(table: m.ProfileTable) -> None:
    with pytest.raises(LookupError, match="off-domain"):
        m.Table(cast("np.ndarray", _Values()), table).value((5, 30, 0, 0))
    assert (
        m.Table(cast("np.ndarray", _Values(0.5)), table).value((250, 30, 0, 0)) == 0.5
    )  # a dead TTD collapses to its sentinel


def test_stage_is_the_certified_full_matrix(
    table: m.ProfileTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    s, f = np.sort(np.random.default_rng(9).uniform(-1, 1, 60)), 0.3
    monkeypatch.setattr(m, "class_values", lambda pc, pd, values, table: (s, f))
    expected = m.solve_class(s, f)
    assert expected.value is not None
    stage = m.Table(cast("np.ndarray", _Values(expected.value)), table).stage(
        (0, 0, 0, 0)
    )
    assert stage.state == (0, 0, 0, 0)
    assert stage.value == expected.value
    assert np.array_equal(stage.matrix, m.full_matrix(s, f))
    assert stage.gap <= m.MAX_SADDLE_GAP
    assert stage.drop.sum() == pytest.approx(1)
    assert stage.check.sum() == pytest.approx(1)
    assert not stage.matrix.flags.writeable
    assert not stage.drop.flags.writeable
    assert not stage.check.flags.writeable


def test_stage_fails_closed_when_the_saddle_gap_is_too_large(
    table: m.ProfileTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        m,
        "class_values",
        lambda pc, pd, values, table: (np.linspace(-1, 1, 60), 0.0),
    )
    monkeypatch.setattr(
        m,
        "solve_class",
        lambda s, f: m.SolveResult(
            0.0, 0.0, m.PURE, drop=np.eye(60)[0], check=np.eye(60)[0]
        ),
    )
    with pytest.raises(RuntimeError, match="saddle gap"):
        m.Table(cast("np.ndarray", _Values()), table).stage((0, 0, 0, 0))


def test_stage_fails_closed_when_the_stored_value_leaves_its_certificate(
    table: m.ProfileTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        m,
        "class_values",
        lambda pc, pd, values, table: (np.full(60, 0.25), 0.25),
    )
    with pytest.raises(RuntimeError, match="outside its certificate"):
        m.Table(cast("np.ndarray", _Values(0.5)), table).stage((0, 0, 0, 0))


def test_open_fails_closed_on_the_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "V.npy"
    np.save(path, np.zeros((3, 4)))
    with pytest.raises(ValueError, match="float64"):
        m.Table.open(path)


def test_open_fails_closed_on_the_anchors_and_the_rechecks(
    tmp_path: Path, table: m.ProfileTable
) -> None:
    path = tmp_path / "V.npy"
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float64, shape=(N, N + 1)
    )  # sparse: reads 0.0
    del values
    with pytest.raises(ValueError, match="anchors"):
        m.Table.open(path)
    values = np.load(path, mmap_mode="r+")
    for (sc, tc, sd, td), value in m.ANCHORS:
        values[m.profile(sc, tc, table), m.profile(sd, td, table)] = value
    values.flush()
    del values
    assert m.Table.open(path).value((0, 0, 0, 0)) == m.ANCHORS[0][1]
    with pytest.raises(ValueError, match="recheck"):
        m.Table.open(path, rechecks=1)  # class 0's children read 0.0
    path.unlink()


@pytest.mark.skipif(
    not m.DEFAULT_PATH.is_file(), reason="artifacts/dth/V.npy is unavailable"
)
def test_the_real_table_serves_a_certified_root() -> None:
    table = m.Table.open(rechecks=12)
    stage = table.stage((0, 0, 0, 0))
    assert (
        stage.value
        == table.value((0, 0, 0, 0))
        == pytest.approx(m.ANCHORS[0][1], abs=m.MAX_SADDLE_GAP)
    )
    assert stage.gap <= m.MAX_SADDLE_GAP
    assert np.count_nonzero(stage.drop) > 1
    assert np.count_nonzero(stage.check) > 1


def test_stl_and_dth_revival_surfaces_match_over_the_full_domain() -> None:
    referee = Referee(cprs_performed=17)
    player = Player(name="Either", physicality=0.01)
    for st_in_vial in range(300):
        for ttd in range(301):
            player.ttd = ttd
            actual = referee.compute_survival_probability(
                player, death_duration=st_in_vial + 60
            )
            assert actual == m.revival_probability(st_in_vial, ttd)
