# Copyright 2026 The STL authors
"""We test the numba leap kernel in stl.leap_kernel against the retired Rust
kernel's golden outputs and against the scalar oracles."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from stl.leap import GAP, StageResult, solve_lp, stage_matrix
from tests.conftest import fma, solve_stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from stl.leap import Fallback

# ──────────────────────────────────────────────────────────────────────
# The scalar support oracle: certify reduced support guesses before sending a stage to
# HiGHS.
# ──────────────────────────────────────────────────────────────────────

ALL_ACTIONS = (1 << 60) - 1
EDGE_LIMIT = 64


@dataclass(frozen=True)
class SupportResult:
    stage: StageResult
    p: np.ndarray
    q: np.ndarray
    p_mask: int
    q_mask: int
    attempts: int


def reverse_support(mask: int) -> int:
    return int(f"{mask:060b}"[::-1], 2)


def kink_index(s: np.ndarray) -> int | None:
    """We return the first index k of the most negative step s[k]-s[k-1], or None.

    A nondecreasing s with f > s[0] gives nonnegative recurrence weights, so
    a residue stage needs at least one negative step.
    """
    s = np.asarray(s, np.float64)
    steps = s[1:] - s[:-1]
    k = int(np.argmin(steps))
    return k + 1 if steps[k] < 0 else None


def holes(mask: int) -> list[tuple[int, int]]:
    """We list the maximal runs of absent actions as inclusive (start, end) pairs."""
    runs = []
    i = 0
    while i < 60:  # noqa: PLR2004
        if mask >> i & 1:
            i += 1
            continue
        start = i
        while i + 1 < 60 and not mask >> (i + 1) & 1:  # noqa: PLR2004
            i += 1
        runs.append((start, i))
        i += 1
    return runs


def move_holes(mask: int, near: int, old: int, delta: int) -> int:
    """We move each hole with the nearer of the anchors near and 59-old.

    A hole at the first anchor moves by delta and a hole at the second by
    -delta. A hole that starts at action 0 keeps its place. We clip moved
    holes to actions 0..59.
    """
    far = 59 - old
    out = ALL_ACTIONS
    for start, end in holes(mask):
        shift = (
            0
            if start == 0
            else (delta if abs(end - near) <= abs(end - far) else -delta)
        )
        for x in range(max(start + shift, 0), min(end + shift, 59) + 1):
            out &= ~(1 << x)
    return out


def kink_seed(
    p_mask: int, q_mask: int, old: int | None, new: int | None
) -> tuple[int, int]:
    """We move a certified support from kink old to kink new.

    Dropper holes use the anchors old and 59-old; Checker holes use old-1 and
    59-old. We return the masks unchanged when either kink is absent or the
    two kinks agree.
    """
    if old is None or new is None or old == new:
        return p_mask, q_mask
    delta = new - old
    return move_holes(p_mask, old, old, delta), move_holes(q_mask, old - 1, old, delta)


def recurrence(s: np.ndarray, f: float) -> np.ndarray | None:
    d = s[0] - f
    if abs(d) < 1e-12:  # noqa: PLR2004
        return None
    b = np.r_[0.0, (s[:-1] - s[1:]) / d]
    r = np.zeros(60)
    r[0] = 1.0
    for k in range(1, 60):
        acc = 0.0
        for j in range(k):
            acc = fma(float(b[k - j]), float(r[j]), acc)
        r[k] = acc
    return r if np.isfinite(r).all() else None


def reduced_weights(r: np.ndarray, p_mask: int, q_mask: int) -> np.ndarray | None:
    """We solve the hole constraints in g, then reconstruct the Checker mix."""
    if not p_mask or not q_mask:
        return None
    rows = [i for i in range(60) if p_mask >> i & 1]
    columns = np.array([bool(q_mask >> i & 1) for i in range(60)])
    g = np.array(
        [i for i in range(59) if not (p_mask >> i & 1 and p_mask >> (i + 1) & 1)] + [59]
    )
    equations = [
        np.array([r[t - j] if t >= j else 0.0 for t in g])
        for j in np.flatnonzero(~columns)
    ]
    for a, c in itertools.pairwise(rows):
        if c > a + 1:
            equations.append(((g >= a) & (g < c)).astype(float))
    equations.append(np.cumsum(r)[g])
    if len(equations) != len(g):
        return None
    a = np.array(equations)
    rhs = np.zeros(len(g))
    rhs[-1] = 1.0
    try:
        weights = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return None
    q = np.zeros(60)
    for j in np.flatnonzero(columns):
        q[j] = sum(r[t - j] * x for t, x in zip(g, weights, strict=False) if t >= j)
    if not np.isfinite(q).all() or np.min(q) < 0 or q.sum() <= 0:
        return None
    return q / q.sum()


def shifted_isolated_holes(mask: int, mode: int) -> int:
    result = mask
    for i in range(60):
        if mask >> i & 1:
            continue
        if (i > 0 and not (mask >> (i - 1) & 1)) or (
            i < 59 and not (mask >> (i + 1) & 1)  # noqa: PLR2004
        ):
            continue
        direction = (
            (-1 if i < 30 else 1)  # noqa: PLR2004
            if mode == 0
            else ((1 if i < 30 else -1) if mode == 1 else (-1 if mode == 2 else 1))  # noqa: PLR2004
        )
        target = i + direction
        if 0 <= target < 60 and mask >> target & 1:  # noqa: PLR2004
            result = (result | (1 << i)) & ~(1 << target)
    return result


def edge_candidates(p_mask: int, q_mask: int) -> Iterator[tuple[int, int]]:
    """We bound edge probes so an unsuccessful guess still reaches HiGHS."""
    seen = {(p_mask, q_mask)}
    count = 0
    # We move isolated holes together when two edges travel in opposite
    # directions, then try individual paired support boundaries.
    for mode in range(4):
        pair = (
            shifted_isolated_holes(p_mask, mode),
            shifted_isolated_holes(q_mask, mode),
        )
        if pair not in seen:
            seen.add(pair)
            yield pair
            count += 1
    for shift in range(59):
        if ((p_mask >> shift) ^ (p_mask >> (shift + 1))) & 1 and (
            (q_mask >> (58 - shift)) ^ (q_mask >> (59 - shift))
        ) & 1:
            pair = (p_mask ^ (3 << shift), q_mask ^ (3 << (58 - shift)))
            if pair not in seen:
                seen.add(pair)
                yield pair
                count += 1

    def neighbors(mask: int) -> list[int]:
        output = [mask]
        i = 0
        while i < 60:  # noqa: PLR2004
            if mask >> i & 1:
                i += 1
                continue
            start = i
            while i + 1 < 60 and not (mask >> (i + 1) & 1):  # noqa: PLR2004
                i += 1
            end = i
            if start > 0:
                output.append(mask ^ (1 << (start - 1)) ^ (1 << end))
            if end < 59:  # noqa: PLR2004
                output.append(mask ^ (1 << start) ^ (1 << (end + 1)))
            i += 1
        for i in range(60):
            bit = (mask >> i) & 1
            boundary = (i > 0 and ((mask >> (i - 1)) & 1) != bit) or (
                i < 59 and ((mask >> (i + 1)) & 1) != bit  # noqa: PLR2004
            )
            if boundary:
                output.append(mask ^ (1 << i))
            if i < 59 and ((mask >> (i + 1)) & 1) != bit:  # noqa: PLR2004
                output.append(mask ^ (3 << i))
        return output

    for pp in neighbors(p_mask):
        for qq in neighbors(q_mask):
            pair = (pp, qq)
            if pp.bit_count() == qq.bit_count() and pair not in seen:
                if count >= EDGE_LIMIT:
                    return
                seen.add(pair)
                yield pair
                count += 1


def solve_supported(
    s: np.ndarray,
    f: float,
    window: bool,  # noqa: FBT001
    p_mask: int,
    q_mask: int,
    *,
    edges: bool = True,
    max_attempts: int | None = None,
) -> SupportResult | None:
    """We accept a guessed support only after a complete saddle certificate.

    You can cap the number of supports we try, counting the guess as the
    first, with max_attempts.
    """
    s = np.asarray(s, np.float64)
    p_mask = int(p_mask)
    q_mask = int(q_mask)
    if s.shape != (60,) or not np.isfinite(s).all() or not math.isfinite(f):
        msg = "stage needs 60 finite successes and a finite failure"
        raise ValueError(msg)
    if min(p_mask, q_mask) < 0 or (p_mask | q_mask) & ~ALL_ACTIONS:
        msg = "support bits must name actions 0 through 59"
        raise ValueError(msg)
    if not p_mask or not q_mask:
        return None
    r = recurrence(s, f)
    if r is None:
        return None
    candidates = [(p_mask, q_mask)]
    if edges:
        candidates.extend(edge_candidates(p_mask, q_mask))
    matrix = stage_matrix(s, f, False)  # noqa: FBT003
    for attempt, (pp, qq) in enumerate(candidates, 1):
        if max_attempts is not None and attempt > max_attempts:
            return None
        q = reduced_weights(r, pp, qq)
        if q is None:
            continue
        reverse_p = reduced_weights(r, reverse_support(qq), reverse_support(pp))
        if reverse_p is None:
            continue
        p = reverse_p[::-1].copy()
        lower = float((p @ matrix).min())
        upper = float((matrix @ q).max())
        square_value = (lower + upper) / 2
        kind = 4 if attempt == 1 else 5
        if window:
            lower, upper = max(lower, f), max(upper, f)
            if f >= square_value:
                kind = 2
            if f >= float((p @ matrix).min()):
                p = np.r_[np.zeros(60), 1.0]
            else:
                p = np.r_[p, 0.0]
        if math.isfinite(upper - lower) and -1e-12 <= upper - lower <= GAP:  # noqa: PLR2004
            return SupportResult(
                StageResult(
                    max(square_value, f) if window else square_value, kind, lower, upper
                ),
                p,
                q,
                pp,
                qq,
                attempt,
            )
    return None


# ──────────────────────────────────────────────────────────────────────
# We certify reduced support solves against the complete stage matrix.
# ──────────────────────────────────────────────────────────────────────

S = np.array(
    [
        -0.1284722791674977,
        -0.12311238213031911,
        -0.11783938104329245,
        -0.11229663346467983,
        -0.10668796350604809,
        -0.10083315349960814,
        -0.09537470669625181,
        -0.08977130778795045,
        -0.08339761432886156,
        -0.07769516443214197,
        -0.07192164010554408,
        -0.06616436419483152,
        -0.06006093717963651,
        -0.053227655288462405,
        -0.047598391274268514,
        -0.041854762653576905,
        -0.03587453314612943,
        -0.03003252289938163,
        -0.023603057751273637,
        -0.017586484540291714,
        -0.011219842703011795,
        -0.005275671293213546,
        0.0008117282032785167,
        0.007153154358869429,
        0.013428233497556619,
        0.01974489537252866,
        0.026399180249675525,
        0.03279969710950498,
        -0.00047749039889410927,
        0.005131084056217761,
        0.010850843327395387,
        0.01664725521396604,
        0.022484377545262724,
        0.02847221115882731,
        0.03448311001780113,
        0.04064462357322546,
        0.04664102425924954,
        0.05273846839689167,
        0.05896383364466534,
        0.06518391929773504,
        0.07147635977875716,
        0.07779174310726783,
        0.08423778804274584,
        0.09077531440470277,
        0.09702645256663872,
        0.1033539948162649,
        0.1097535240115865,
        0.11616685105592794,
        0.12271761517131707,
        0.1292247029577216,
        0.13584770092279674,
        0.14236099302853944,
        0.14894853172567968,
        0.15561553076972012,
        0.1622855862936653,
        0.17099723315775567,
        0.1901691314113193,
        0.19683910407608857,
        0.20375096839599455,
        0.2099309116592808,
    ]
)
F = 0.813835962781682
P = 1152921500328656895
Q = 1152921500588703743
VALUE = 0.1504047730058836


def test_support_oracle_certifies_known_asymmetric_support() -> None:
    from stl.leap import stage_matrix  # noqa: PLC0415

    for window in (False, True):
        result = solve_supported(S, F, window, P, Q, edges=False)
        assert result is not None
        matrix = stage_matrix(S, F, window)
        assert np.min(result.p) >= 0
        assert np.min(result.q) >= 0
        assert abs(result.p.sum() - 1) < 1e-12  # noqa: PLR2004
        assert abs(result.q.sum() - 1) < 1e-12  # noqa: PLR2004
        assert abs(result.stage.lower - np.min(result.p @ matrix)) < 1e-12  # noqa: PLR2004
        assert abs(result.stage.upper - np.max(matrix @ result.q)) < 1e-12  # noqa: PLR2004
        assert result.stage.gap <= 1e-6  # noqa: PLR2004
        assert abs(result.stage.value - (max(VALUE, F) if window else VALUE)) < 1e-9  # noqa: PLR2004


def test_support_oracle_rejects_invalid_candidates() -> None:
    assert (
        solve_supported(S, F, False, (1 << 60) - 1, (1 << 60) - 1, edges=False) is None  # noqa: FBT003
    )
    assert solve_supported(S, F, False, 0, Q) is None  # noqa: FBT003
    assert solve_supported(np.ones(60), 1.0, False, P, Q) is None  # noqa: FBT003
    with pytest.raises(ValueError, match="support"):
        solve_supported(S, F, False, 1 << 60, Q)  # noqa: FBT003
    with pytest.raises(ValueError, match="finite"):
        solve_supported(np.full(60, np.nan), F, False, P, Q)  # noqa: FBT003


def test_support_oracle_recovers_paired_edge_move() -> None:
    for i in range(59):
        if ((P >> i) ^ (P >> (i + 1))) & 1 and ((Q >> (58 - i)) ^ (Q >> (59 - i))) & 1:
            wrong_p = P ^ (3 << i)
            wrong_q = Q ^ (3 << (58 - i))
            if solve_supported(S, F, False, wrong_p, wrong_q, edges=False) is None:  # noqa: FBT003
                break
    else:
        pytest.fail("fixture must contain a support boundary")
    result = solve_supported(S, F, False, wrong_p, wrong_q, edges=True)  # noqa: FBT003
    assert result is not None
    assert result.stage.kind == 5  # noqa: PLR2004
    assert abs(result.stage.value - VALUE) < 1e-9  # noqa: PLR2004


@pytest.mark.parametrize("n", [1, 15, 16, 17, 1000])
@pytest.mark.parametrize("window", [False, True])
def test_support_kernel_parity_and_spare_lanes(n: int, window: bool) -> None:  # noqa: FBT001
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    scale = np.linspace(0.7, 0.9, n)
    offset = 0.03 * np.sin(np.arange(n))
    successes = scale[:, None] * S + offset[:, None]
    failures = scale * F + offset
    pc = np.arange(n, dtype=np.int32) % 17
    pd = np.arange(n, dtype=np.int32)
    supports = np.tile(np.array([P, Q], np.uint64), (17, 1))
    support_out = np.zeros((n, 2), np.uint64)
    out = np.empty(n)
    kind = np.empty(n, np.uint8)
    failed = sweep_batch(
        pc,
        pd,
        np.ascontiguousarray(-successes),
        np.ascontiguousarray(-failures[:, None]),
        np.tile(np.arange(60, dtype=np.int32), (17, 1)),
        np.zeros(17, np.int32),
        np.ones(17),
        window,
        1e-6,
        out,
        kind,
        supports,
        support_out,
    )
    assert failed == 0
    for i in range(n):
        reference = solve_supported(
            successes[i], failures[i], window, P, Q, edges=False
        )
        assert reference is not None
        assert abs(out[i] - reference.stage.value) < 1e-9  # noqa: PLR2004
    if not window:
        assert np.all(kind == 4)  # noqa: PLR2004
        assert np.all(support_out == supports[pc])
    else:
        assert np.all(kind == 2)  # noqa: PLR2004


def test_kernel_defers_row_after_support_miss() -> None:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    n = 3
    pc = np.zeros(n, np.int32)
    pd = np.arange(n, dtype=np.int32)
    out = np.empty(n)
    kind = np.empty(n, np.uint8)
    policies = np.zeros((n, 2), np.uint64)
    seeds = np.full((1, 2), (1 << 60) - 1, np.uint64)
    args = (
        pc,
        pd,
        np.tile(-S, (n, 1)),
        np.full((n, 1), -F),
        np.arange(60, dtype=np.int32)[None, :],
        np.zeros(1, np.int32),
        np.ones(1),
        False,
        1e-6,
        out,
        kind,
        seeds,
        policies,
    )
    assert sweep_batch(*args, stop_on_support_miss=True) == 3  # noqa: PLR2004
    assert np.array_equal(kind, [255, 254, 254])
    assert np.isnan(out).all()
    seeds[0] = [P, Q]
    assert sweep_batch(*args, stop_on_support_miss=True) == 0
    assert np.all(kind == 4)  # noqa: PLR2004
    assert np.max(np.abs(out - VALUE)) < 1e-9  # noqa: PLR2004


def test_paired_isolated_edges_move_in_opposite_directions() -> None:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    s = np.array(
        [
            -0.4026671677760904,
            -0.3979960627586634,
            -0.3927380866464558,
            -0.38796416833988606,
            -0.3831366054706515,
            -0.3783119299518909,
            -0.3734572377052099,
            -0.3676171318207585,
            -0.36278131968743343,
            -0.35790261663936157,
            -0.35281713229130485,
            -0.34788195053701454,
            -0.34844287017916925,
            -0.3433751252503391,
            -0.33789479973162684,
            -0.33277089815970234,
            -0.327552768099293,
            -0.3220826666813272,
            -0.31666186017030373,
            -0.3111993579252109,
            -0.3057042620654687,
            -0.30042788954730754,
            -0.3375764096216484,
            -0.3315649066967039,
            -0.32638188330358975,
            -0.32113306497675836,
            -0.31589901523638064,
            -0.3105279449808905,
            -0.3051827889678185,
            -0.29961118942194703,
            -0.2942310934543996,
            -0.2887891692158323,
            -0.2832073538085286,
            -0.2776241853120468,
            -0.2719893517885459,
            -0.26633305271138497,
            -0.26064268128379425,
            -0.2543819204709524,
            -0.24869933364195948,
            -0.24296433930829275,
            -0.23713741785742787,
            -0.23132694097086132,
            -0.2252380110841241,
            -0.21932540932331485,
            -0.213130166910518,
            -0.20715583260261583,
            -0.20109987242097038,
            -0.19488659433670147,
            -0.18868081201901848,
            -0.18243767804049366,
            -0.17615768757219252,
            -0.16985780388874677,
            -0.16216707795660473,
            -0.1560280363296488,
            -0.14983094055555018,
            -0.14350194546940354,
            -0.13723990970217614,
            -0.13079020559018828,
            -0.12434683661148044,
            -0.11768983102715774,
        ]
    )
    f = 0.25825181643253137
    seeds = np.array([[1152850863123914751, 1152850863128080383]], np.uint64)
    expected = -0.17866045816637277
    result = solve_supported(s, f, False, *map(int, seeds[0]))  # noqa: FBT003
    assert result is not None
    assert result.stage.kind == 5  # noqa: PLR2004
    assert abs(result.stage.value - expected) < 1e-8  # noqa: PLR2004
    out = np.empty(1)
    kind = np.empty(1, np.uint8)
    masks = np.zeros((1, 2), np.uint64)
    failed = sweep_batch(
        np.zeros(1, np.int32),
        np.zeros(1, np.int32),
        -s[None, :],
        np.array([[-f]]),
        np.arange(60, dtype=np.int32)[None, :],
        np.zeros(1, np.int32),
        np.ones(1),
        False,  # noqa: FBT003
        1e-6,
        out,
        kind,
        seeds,
        masks,
    )
    assert failed == 0
    assert kind[0] == 5  # noqa: PLR2004
    assert abs(out[0] - expected) < 1e-8  # noqa: PLR2004


def test_paired_hole_intervals_shift_one_action() -> None:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    s = np.array(
        [
            -0.35443151091336567,
            -0.3495385377182094,
            -0.3442039416613912,
            -0.3392275677743438,
            -0.3342054563604621,
            -0.3291819864852916,
            -0.324135476554298,
            -0.3183628554448803,
            -0.31332746229769,
            -0.30825510278973683,
            -0.30302401622376585,
            -0.29790255109574065,
            -0.2923764167924844,
            -0.28716480407531875,
            -0.281665357795775,
            -0.2764051542602377,
            -0.27106979567536726,
            -0.26555236877164307,
            -0.2600605216604688,
            -0.2545310551934664,
            -0.24897076241624222,
            -0.2437416370635117,
            -0.29111796631885567,
            -0.28476778931926416,
            -0.27939274991776764,
            -0.2739645103492518,
            -0.26853903384671485,
            -0.2630130455284063,
            -0.2574956837286908,
            -0.2518201556055816,
            -0.2462657833943129,
            -0.24066236214512923,
            -0.23495692484711755,
            -0.2292441961435014,
            -0.2234897568239272,
            -0.21771385191218395,
            -0.21190706236032458,
            -0.20573527484002602,
            -0.19992565342696395,
            -0.19407538955691817,
            -0.18815541118559814,
            -0.18223907444594561,
            -0.17613931212467626,
            -0.1701440524031323,
            -0.16396826979953272,
            -0.15792172265497217,
            -0.15181444586963172,
            -0.1456064018849137,
            -0.13939574469760113,
            -0.1331565188540435,
            -0.12688717004088534,
            -0.12060026481410879,
            -0.11340305136257517,
            -0.10721037322073801,
            -0.10098621886329184,
            -0.09467044350016385,
            -0.08838935206979687,
            -0.08198603501497365,
            -0.07557998799546839,
            -0.06904133912400101,
        ]
    )
    f = 0.2693233637785153
    seeds = np.array([[1152921230785970175, 1152921230794326015]], np.uint64)
    expected = -0.1321459733379427
    result = solve_supported(s, f, False, *map(int, seeds[0]))  # noqa: FBT003
    assert result is not None
    assert result.stage.kind == 5  # noqa: PLR2004
    assert abs(result.stage.value - expected) < 1e-8  # noqa: PLR2004
    out = np.empty(1)
    kind = np.empty(1, np.uint8)
    masks = np.zeros((1, 2), np.uint64)
    failed = sweep_batch(
        np.zeros(1, np.int32),
        np.zeros(1, np.int32),
        -s[None, :],
        np.array([[-f]]),
        np.arange(60, dtype=np.int32)[None, :],
        np.zeros(1, np.int32),
        np.ones(1),
        False,  # noqa: FBT003
        1e-6,
        out,
        kind,
        seeds,
        masks,
    )
    assert failed == 0
    assert kind[0] == 5  # noqa: PLR2004
    assert abs(out[0] - expected) < 1e-8  # noqa: PLR2004


def test_support_kernel_keeps_checker_seeds_separate() -> None:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    n = 1000
    pc = np.arange(n, dtype=np.int32) % 2
    pd = np.arange(n, dtype=np.int32)
    seeds = np.array([[P, Q], [(1 << 60) - 1, (1 << 60) - 1]], np.uint64)
    out = np.empty(n)
    kind = np.empty(n, np.uint8)
    masks = np.zeros((n, 2), np.uint64)
    failures = sweep_batch(
        pc,
        pd,
        np.tile(-S, (n, 1)),
        np.full((n, 1), -F),
        np.tile(np.arange(60, dtype=np.int32), (2, 1)),
        np.zeros(2, np.int32),
        np.ones(2),
        False,  # noqa: FBT003
        1e-6,
        out,
        kind,
        seeds,
        masks,
    )
    assert failures == n // 2
    assert np.all(kind[::2] == 4)  # noqa: PLR2004
    assert np.max(np.abs(out[::2] - VALUE)) < 1e-9  # noqa: PLR2004
    assert np.all(kind[1::2] == 255)  # noqa: PLR2004
    assert np.isnan(out[1::2]).all()


def test_support_kernel_rejects_invalid_masks_and_deferral_order() -> None:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    pc = np.array([1, 0], np.int32)
    pd = np.array([0, 1], np.int32)
    seeds = np.array([[P, Q], [P, Q]], np.uint64)
    masks = np.zeros((2, 2), np.uint64)
    args = (
        pc,
        pd,
        np.tile(-S, (2, 1)),
        np.full((2, 1), -F),
        np.tile(np.arange(60, dtype=np.int32), (2, 1)),
        np.zeros(2, np.int32),
        np.ones(2),
        False,
        1e-6,
        np.empty(2),
        np.empty(2, np.uint8),
    )
    with pytest.raises(ValueError, match="supplied together"):
        sweep_batch(*args, supports=seeds)
    with pytest.raises(ValueError, match="shape"):
        sweep_batch(*args, supports=seeds[:1], support_out=masks)
    bad = seeds.copy()
    bad[0, 0] = 1 << 60
    with pytest.raises(ValueError, match="bit out of range"):
        sweep_batch(*args, supports=bad, support_out=masks)
    with pytest.raises(ValueError, match="Checker-row order"):
        sweep_batch(*args, supports=seeds, support_out=masks, stop_on_support_miss=True)


def test_warm_highs_returns_support_on_request() -> None:
    from stl.leap import WarmLP  # noqa: PLC0415

    s = np.linspace(0.7, -0.4, 60)
    f = -0.8
    result = WarmLP(False).solve(s, f, support=True)  # noqa: FBT003
    assert result.support is not None
    recovered = solve_supported(s, f, False, *result.support, edges=False)  # noqa: FBT003
    assert recovered is not None
    assert abs(recovered.stage.value - result.value) < 1e-9  # noqa: PLR2004


# ──────────────────────────────────────────────────────────────────────
# We certify the recurrence crash basis, kink seeds, and in-kernel residue path.
# ──────────────────────────────────────────────────────────────────────


def shifted_stages(
    drop: float = 0.04, f: float = 0.55, n: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a 60-second window over one kink, as consecutive Checker ST does."""
    sigma = -0.35 + np.cumsum(np.linspace(0.004, 0.008, 100))
    sigma[70:] -= drop
    return np.array([sigma[j : j + 60] for j in range(11, 11 + n)]), np.full(n, f)


def kinked_stages(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = []
    f = []
    for _ in range(n):
        base = np.cumsum(rng.uniform(0.002, 0.01, 60)) - rng.uniform(0.2, 0.5)
        base[int(rng.integers(1, 60)) :] -= rng.uniform(0.005, 0.08)
        s.append(base)
        f.append(min(1.0, base[0] + rng.uniform(0.2, 1.0)))
    return np.array(s), np.array(f)


def kernel(
    s: np.ndarray,
    f: np.ndarray,
    window: bool,  # noqa: FBT001
    **options: Any,  # noqa: ANN401
) -> tuple[np.ndarray, np.ndarray, int]:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    n = len(f)
    out = np.empty(n)
    kind = np.empty(n, np.uint8)
    failed = sweep_batch(
        np.zeros(n, np.int32),
        np.arange(n, dtype=np.int32),
        np.ascontiguousarray(-s),
        np.ascontiguousarray(-f[:, None]),
        np.arange(60, dtype=np.int32)[None, :],
        np.zeros(1, np.int32),
        np.ones(1),
        window,
        1e-6,
        out,
        kind,
        **options,
    )
    return out, kind, failed


@pytest.mark.parametrize("window", [False, True])
def test_crash_packing_matches_cold_and_lp(window: bool) -> None:  # noqa: FBT001
    from stl.leap_kernel import solve_packing  # noqa: PLC0415

    s, f = kinked_stages(300, 17)
    s = np.vstack([S, *shifted_stages()[0], s])
    f = np.r_[F, shifted_stages()[1], f]
    results = {}
    for crash in (False, True):
        out = np.empty(len(f))
        kind = np.empty(len(f), np.uint8)
        assert (
            solve_packing(np.ascontiguousarray(s), f, window, out, kind, crash=crash)
            == 0
        )
        assert np.all(kind == 3)  # noqa: PLR2004
        results[crash] = out
    assert np.max(np.abs(results[True] - results[False])) <= 1e-6  # noqa: PLR2004
    assert abs(results[True][0] - (max(VALUE, F) if window else VALUE)) < 1e-9  # noqa: PLR2004
    for i in range(0, len(f), 7):
        assert abs(results[True][i] - solve_lp(s[i], f[i], window).value) <= 1e-8  # noqa: PLR2004


def test_crash_packing_keeps_unsupported_and_nonfinite_contract() -> None:
    from stl.leap_kernel import solve_packing  # noqa: PLC0415

    out = np.empty(2)
    kind = np.empty(2, np.uint8)
    s = np.ones((2, 60))
    f = np.array([0.0, 1.0 + 1e-13])
    assert solve_packing(s, f, False, out, kind, crash=True) == 2  # noqa: FBT003, PLR2004
    assert np.all(kind == 255)  # noqa: PLR2004
    assert np.isnan(out).all()
    with pytest.raises(ValueError, match="finite"):
        solve_packing(
            np.full((1, 60), np.nan),
            np.ones(1),
            False,  # noqa: FBT003
            out[:1],
            kind[:1],
            crash=True,
        )


def test_kink_index_matches_python_authority() -> None:
    from stl import leap_kernel  # noqa: PLC0415

    rng = np.random.default_rng(5)
    cases = [np.linspace(-0.5, 0.5, 60), np.zeros(60), S, *shifted_stages()[0]]
    tie = np.linspace(0, 0.3, 60)
    tie[10:] -= 0.02
    tie[40:] -= 0.02
    cases.append(tie)
    cases += [rng.normal(size=60).cumsum() * 0.01 for _ in range(300)]
    for s in cases:
        assert leap_kernel.kink_index(np.ascontiguousarray(s)) == kink_index(s)
    assert kink_index(tie) == 10  # noqa: PLR2004
    assert kink_index(np.zeros(60)) is None
    with pytest.raises(ValueError, match="60"):
        leap_kernel.kink_index(np.zeros(59))


def random_mask(rng: np.random.Generator) -> int:
    mask = (1 << 60) - 1
    for _ in range(int(rng.integers(0, 4))):
        start = int(rng.integers(0, 60))
        end = min(59, start + int(rng.integers(0, 15)))
        for x in range(start, end + 1):
            mask &= ~(1 << x)
    return mask


def test_kink_seed_matches_python_authority() -> None:
    from stl import leap_kernel  # noqa: PLC0415

    rng = np.random.default_rng(8)
    for _ in range(3000):
        p, q = random_mask(rng), random_mask(rng)
        old, new = (
            None if rng.random() < 0.1 else int(rng.integers(1, 60))  # noqa: PLR2004
            for _ in range(2)
        )
        assert leap_kernel.kink_seed(p, q, old, new) == kink_seed(p, q, old, new)
    # A Dropper hole at k*=45 moves right; the mirrored hole at 59-k* moves left.
    p = ((1 << 60) - 1) & ~(0b111 << 12) & ~(0b111 << 43)
    q = ((1 << 60) - 1) & ~(0b111 << 12) & ~(0b111 << 42)
    moved_p, moved_q = kink_seed(p, q, 45, 46)
    assert moved_p == ((1 << 60) - 1) & ~(0b111 << 11) & ~(0b111 << 44)
    assert moved_q == ((1 << 60) - 1) & ~(0b111 << 11) & ~(0b111 << 43)
    assert kink_seed(p, q, 45, None) == (p, q)
    assert kink_seed(p, q, 45, 45) == (p, q)
    with pytest.raises(ValueError, match="kink"):
        leap_kernel.kink_seed(p, q, 0, 5)
    with pytest.raises(ValueError, match="kink"):
        leap_kernel.kink_seed(1 << 60, q, 5, 6)


def unsupported_residue(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """We find residue stages with f < s[0], which the packing LP does not accept."""
    rng = np.random.default_rng(seed)
    s = []
    f = []
    while len(f) < count:
        success = rng.uniform(-1, 1, 60)
        failure = float(rng.uniform(-1, 1))
        if (
            failure < success[0]
            and solve_stage(success, failure, False, allow_lp=False).kind == 255  # noqa: FBT003, PLR2004
        ):
            s.append(success)
            f.append(failure)
    return np.array(s), np.array(f)


@pytest.mark.parametrize("crash", [False, True])
@pytest.mark.parametrize("attempts", [0, 1, 4])
@pytest.mark.parametrize("window", [False, True])
def test_kernel_native_residue_matches_lp(
    crash: bool,  # noqa: FBT001
    attempts: int,
    window: bool,  # noqa: FBT001
) -> None:
    s, f = kinked_stages(200, 23)
    sequence = shifted_stages()
    hard = unsupported_residue(3, 29)
    s = np.vstack([S, *sequence[0], s, hard[0]])
    f = np.r_[F, sequence[1], f, hard[1]]
    plain, plain_kind, plain_failed = kernel(s, f, window)
    out, kind, failed = kernel(
        s, f, window, native=True, crash=crash, kink_attempts=attempts
    )
    residue = plain_kind == 255  # noqa: PLR2004
    unsupported = f < s[:, 0]
    assert plain_failed == np.count_nonzero(residue)
    # With f > s[0], v60 <= f, so the lifted pure bounds settle every window stage.
    assert np.count_nonzero(residue & ~unsupported) == (
        0 if window else plain_failed - 3
    )
    assert failed == np.count_nonzero(residue & unsupported)
    assert np.all(kind[residue & unsupported] == 255)  # noqa: PLR2004
    assert np.isnan(out[residue & unsupported]).all()
    solved = residue & ~unsupported
    assert set(np.unique(kind[solved])) <= {3, 4, 5}
    assert np.array_equal(kind[~residue], plain_kind[~residue])
    assert np.array_equal(out[~residue], plain[~residue])
    if attempts == 0:
        assert not np.isin(kind, [4, 5]).any()
    elif not window:
        assert np.count_nonzero(kind == 4) >= 10  # noqa: PLR2004
    for i in np.flatnonzero(solved):
        assert abs(out[i] - solve_lp(s[i], f[i], window).value) <= 1e-8  # noqa: PLR2004


def test_kernel_native_seeds_follow_the_shifted_kink() -> None:
    s, f = shifted_stages()
    out, kind, failed = kernel(s, f, False, native=True, crash=True, kink_attempts=1)  # noqa: FBT003
    assert failed == 0
    assert kind[0] == 3  # noqa: PLR2004
    assert np.count_nonzero(kind == 4) >= 20  # noqa: PLR2004
    for i in range(len(f)):
        assert abs(out[i] - solve_stage(s[i], f[i], False).value) <= 1e-8  # noqa: FBT003, PLR2004


def test_kernel_native_seeds_restart_in_each_worker_chunk() -> None:
    s, f = shifted_stages()
    s = np.tile(s, (80, 1))
    f = np.tile(f, 80)
    out, kind, failed = kernel(s, f, False, native=True, crash=True, kink_attempts=1)  # noqa: FBT003
    assert failed == 0
    assert np.all(np.isin(kind, [3, 4]))
    # Each 1024-class chunk starts without a seed, so its first residue class uses
    # the LP.
    assert np.all(kind[[0, 1024, 2048]] == 3)  # noqa: PLR2004
    assert np.max(np.abs(out - np.tile(out[:30], 80))) <= 1e-6  # noqa: PLR2004


def test_packing_reports_certified_basis_masks() -> None:
    from stl.leap_kernel import solve_packing  # noqa: PLC0415

    s = np.vstack([S, np.ones(60)])
    f = np.array([F, 0.0])
    out = np.empty(2)
    kind = np.empty(2, np.uint8)
    basis = np.full((2, 2), 7, np.uint64)
    assert solve_packing(s, f, False, out, kind, crash=True, support_out=basis) == 1  # noqa: FBT003
    p, q = map(int, basis[0])
    assert p
    assert p.bit_count() == q.bit_count()
    assert np.all(basis[1] == 0)
    with pytest.raises(ValueError, match="support shape"):
        solve_packing(s, f, False, out, kind, support_out=np.zeros((1, 2), np.uint64))  # noqa: FBT003


def test_kernel_edge_move_limit_matches_python_authority() -> None:
    from stl.leap_kernel import solve_packing  # noqa: PLC0415

    rng = np.random.default_rng(250)
    t = S + rng.normal(0, 3e-3, 60)
    ft = F + rng.normal(0, 3e-3)
    out = np.empty(1)
    kind = np.empty(1, np.uint8)
    basis = np.zeros((1, 2), np.uint64)
    assert (
        solve_packing(
            np.ascontiguousarray(S[None]),
            np.array([F]),
            False,  # noqa: FBT003
            out,
            kind,
            crash=True,
            support_out=basis,
        )
        == 0
    )
    # The kernel seeds the second stage with the first stage's crash basis.
    guess = kink_seed(int(basis[0, 0]), int(basis[0, 1]), kink_index(S), kink_index(t))
    reference = solve_supported(t, ft, False, *guess)  # noqa: FBT003
    assert reference is not None
    assert reference.stage.kind == 5  # noqa: PLR2004
    assert reference.attempts > 1
    stages = np.vstack([S, t])
    fs = np.array([F, ft])
    out, kind, failed = kernel(
        stages,
        fs,
        False,  # noqa: FBT003
        native=True,
        crash=True,
        kink_attempts=reference.attempts,
    )
    assert failed == 0
    assert kind.tolist() == [3, 5]
    assert abs(out[1] - reference.stage.value) < 1e-12  # noqa: PLR2004
    out, kind, failed = kernel(
        stages,
        fs,
        False,  # noqa: FBT003
        native=True,
        crash=True,
        kink_attempts=reference.attempts - 1,
    )
    assert failed == 0
    assert kind.tolist() == [3, 3]
    assert abs(out[1] - solve_lp(t, ft, False).value) <= 1e-8  # noqa: FBT003, PLR2004


def test_kernel_native_options_are_validated() -> None:
    s, f = shifted_stages(n=2)
    with pytest.raises(ValueError, match="native residue mode"):
        kernel(s, f, False, crash=True)  # noqa: FBT003
    with pytest.raises(ValueError, match="native residue mode"):
        kernel(s, f, False, kink_attempts=1)  # noqa: FBT003
    seeds = np.zeros((1, 2), np.uint64)
    masks = np.zeros((2, 2), np.uint64)
    with pytest.raises(ValueError, match="excludes support seeds"):
        kernel(s, f, False, native=True, supports=seeds, support_out=masks)  # noqa: FBT003


def test_support_oracle_caps_attempts() -> None:
    wrong = None
    for i in range(59):
        if ((P >> i) ^ (P >> (i + 1))) & 1 and ((Q >> (58 - i)) ^ (Q >> (59 - i))) & 1:
            wrong = (P ^ (3 << i), Q ^ (3 << (58 - i)))
            if solve_supported(S, F, False, *wrong, edges=False) is None:  # noqa: FBT003
                break
    assert wrong is not None
    result = solve_supported(S, F, False, *wrong)  # noqa: FBT003
    assert result is not None
    assert result.attempts > 1
    assert (
        solve_supported(S, F, False, *wrong, max_attempts=result.attempts - 1) is None  # noqa: FBT003
    )
    capped = solve_supported(S, F, False, *wrong, max_attempts=result.attempts)  # noqa: FBT003
    assert capped is not None
    assert capped.stage.kind == 5  # noqa: PLR2004
    assert solve_supported(S, F, False, P, Q, max_attempts=0) is None  # noqa: FBT003


def test_native_fallback_routes_only_h_keys_to_the_kernel() -> None:
    from stl.leap import NativeFallback  # noqa: PLC0415

    fallback = NativeFallback(1, crash=True, kernel_native=True, kink_attempts=1)
    assert fallback.kernel_options(("H2", 40)) == {
        "native": True,
        "crash": True,
        "kink_attempts": 1,
    }
    assert fallback.kernel_options(("REV", 2461)) == {}
    assert NativeFallback(1, crash=True).kernel_options(("H1", 40)) == {}
    with pytest.raises(ValueError, match="kernel residue mode"):
        NativeFallback(1, kink_attempts=1)


def test_sweep_counts_kernel_residue_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stl import leap as b  # noqa: PLC0415

    b.profiles()  # profiles() reads b.N, so we fill its cache before N shrinks.
    monkeypatch.setattr(b, "N", 8)
    bits = np.zeros((8, 2), np.uint8)
    bits[0, 0] = 0b1111
    options = {"native": True, "crash": True, "kink_attempts": 1}

    def kernel_stub(*args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        assert kwargs == options
        out, kind = args[-2:]
        out[:] = [0.1, 0.2, 0.3, np.nan]
        kind[:] = [3, 4, 5, 255]
        return 1

    monkeypatch.setattr(b, "sweep_batch", kernel_stub)

    class Native:
        native_solves = 0
        native_seconds = 0.0
        highs_seconds = 0.0

        def kernel_options(self, key: tuple[str, int]) -> dict:  # noqa: ARG002
            return options

        def solve(self, *args: Any) -> tuple[np.ndarray, int, int]:  # noqa: ANN401
            n = len(args[-3])
            self.native_solves += n
            return np.full(n, 0.4), 0, 0

    store = b.TableStore(tmp_path)
    monkeypatch.setattr(store, "load", lambda key: np.zeros((8, 17012)))  # noqa: ARG005
    record = b.sweep_key(
        ("H1", 40),
        bits,
        store,
        np.zeros((8, 17012)),
        min_clock=720,
        full=True,
        fallback=cast("Fallback", Native()),
    )
    assert record["failures"] == 4  # noqa: PLR2004
    assert record["states"] == 4  # noqa: PLR2004
    assert record["kernel_native_solves"] == 1
    assert record["support_solves"] == 2  # noqa: PLR2004
    assert record["native_solves"] == 1
    assert record["lp_solves"] == 0


def test_sweep_kernel_residue_matches_python_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stl import leap as b  # noqa: PLC0415
    from stl.leap import N, NativeFallback, profiles  # noqa: PLC0415

    p = profiles()
    key = ("H1", 40)
    m = 40
    dest = b.child_key(key, int(p.st[0]) + 60)
    succ = np.zeros((N, N + 1))
    succ[:, -1] = -1.0
    fail = np.zeros((N, 184))
    fail[:, -1] = -1.0
    s, f = kinked_stages(m, 41)
    succ[np.arange(m)[:, None], p.succ[0][None, :]] = -s
    fail[np.arange(m), p.idx0[p.fail[0]]] = (1 - p.rev[0] - f) / p.rev[0]
    bits = np.zeros((N, (N + 7) // 8), np.uint8)
    for pd in range(m):
        bits[0, pd // 8] |= 1 << (pd % 8)
    store = b.TableStore(tmp_path)
    children = {b.child_key(key): succ, dest: fail}
    monkeypatch.setattr(store, "load", lambda child: children[child])
    tables = {}

    def create(k: tuple[str, int]) -> np.ndarray:  # noqa: ARG001
        tables[len(tables)] = np.lib.format.open_memmap(
            tmp_path / f"out{len(tables)}.npy", mode="w+", shape=(N, N + 1)
        )
        tables[len(tables) - 1][:] = np.nan
        return tables[len(tables) - 1]

    monkeypatch.setattr(store, "create", create)
    records = []
    for options in ({}, {"crash": True, "kernel_native": True, "kink_attempts": 1}):
        with NativeFallback(1, **options) as fallback:
            records.append(
                b.sweep_key(
                    key,
                    bits,
                    store,
                    np.zeros((N, N + 1)),
                    min_clock=720,
                    full=True,
                    fallback=fallback,
                )
            )
    first, second = (tables[i][0, :m] for i in range(2))
    assert np.isfinite(first).all()
    assert np.max(np.abs(first - second)) <= 1e-6  # noqa: PLR2004
    assert records[0]["failures"] == records[1]["failures"] > 0
    assert (
        records[1]["kernel_native_solves"] + records[1]["support_solves"]
        == records[1]["failures"]
    )
    for pd in range(0, m, 5):
        assert abs(second[pd] - solve_lp(s[pd], f[pd], False).value) <= 1e-8  # noqa: FBT003, PLR2004


# ──────────────────────────────────────────────────────────────────────
# We certify the production packing fallback before resuming the sweep.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 17, 2049])
@pytest.mark.parametrize("window", [False, True])
def test_native_binding_matches_lp(n: int, window: bool) -> None:  # noqa: FBT001
    from stl.leap_kernel import solve_packing  # noqa: PLC0415

    scales = np.linspace(0.7, 1.0, n)
    s = np.ascontiguousarray(scales[:, None] * S)
    f = scales * F
    out = np.empty(n)
    kind = np.empty(n, np.uint8)
    assert solve_packing(s, f, window, out, kind) == 0
    assert np.all(kind == 3)  # noqa: PLR2004
    expected = scales * (F if window else VALUE)
    assert np.max(np.abs(out - expected)) < 1e-9  # noqa: PLR2004


def test_native_binding_rejects_bad_inputs_and_reports_unsupported() -> None:
    from stl.leap_kernel import solve_packing  # noqa: PLC0415

    out = np.empty(1)
    kind = np.empty(1, np.uint8)
    with pytest.raises(ValueError, match="finite"):
        solve_packing(np.full((1, 60), np.nan), np.ones(1), False, out, kind)  # noqa: FBT003
    with pytest.raises(ValueError, match="shape"):
        solve_packing(np.zeros((1, 59)), np.ones(1), False, out, kind)  # noqa: FBT003
    assert solve_packing(np.ones((1, 60)), np.zeros(1), False, out, kind) == 1  # noqa: FBT003
    assert kind[0] == 255  # noqa: PLR2004
    assert np.isnan(out[0])


def test_native_fallback_sends_only_misses_to_highs(tmp_path: Path) -> None:
    from stl.leap import N, NativeFallback, profiles  # noqa: PLC0415

    p = profiles()
    n = 2
    succ = np.lib.format.open_memmap(
        tmp_path / "succ.npy", mode="w+", dtype=float, shape=(n, N + 1)
    )
    fail = np.lib.format.open_memmap(
        tmp_path / "fail.npy", mode="w+", dtype=float, shape=(n, N + 1)
    )
    succ[:] = fail[:] = 0.0
    succ[:, -1] = fail[:, -1] = -1.0
    succ[0, p.succ[0]] = -S
    fail[0, p.fail[0]] = (1 - p.rev[0] - F) / p.rev[0]
    succ[1, p.succ[0]] = -0.2
    fail[1, p.fail[0]] = (1 - p.rev[0] + 0.1) / p.rev[0]
    succ.flush()
    fail.flush()
    with NativeFallback(workers=1) as fallback:
        out, lps, cached = fallback.solve(
            ("H1", 51),
            ("H1", 54),
            succ,
            fail,
            np.zeros(n, np.int32),
            np.arange(n, dtype=np.int32),
            False,  # noqa: FBT003
        )
        assert lps == 1
        assert cached == 0
        assert fallback.native_solves == 1
        assert abs(out[0] - VALUE) < 1e-9  # noqa: PLR2004
        assert abs(out[1] - solve_lp(np.full(60, 0.2), -0.1, False).value) < 1e-9  # noqa: FBT003, PLR2004


def test_native_fallback_reuses_certified_rev_values(tmp_path: Path) -> None:
    from stl.leap import N, NativeFallback, profiles  # noqa: PLC0415

    p = profiles()
    succ = np.lib.format.open_memmap(
        tmp_path / "succ.npy", mode="w+", dtype=float, shape=(1, N + 1)
    )
    fail = np.lib.format.open_memmap(
        tmp_path / "fail.npy", mode="w+", dtype=float, shape=(1, 184)
    )
    succ[:] = fail[:] = 0.0
    succ[:, -1] = fail[:, -1] = -1.0
    succ[0, p.succ[0]] = -S
    fail[0, p.idx0[p.fail[0]]] = (1 - p.rev[0] - F) / p.rev[0]
    succ.flush()
    fail.flush()
    with NativeFallback(workers=1) as fallback:
        args = (
            ("H1", 51),
            ("REV", 3400),
            succ,
            fail,
            np.zeros(1, np.int32),
            np.zeros(1, np.int32),
            False,
        )
        first, lps, cached = fallback.solve(*args)
        second, lps, cached = fallback.solve(*args)
        assert cached == 1
        assert lps == 0
        assert fallback.native_solves == 1
        assert np.array_equal(first, second)


def test_sweep_records_native_and_highs_work_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stl import leap as b  # noqa: PLC0415

    b.profiles()  # profiles() reads b.N, so we fill its cache before N shrinks.
    monkeypatch.setattr(b, "N", 8)
    bits = np.zeros((8, 23), np.uint8)
    bits[0, 0] = 7

    def reject(*args: Any) -> int:  # noqa: ANN401
        args[-2][:] = np.nan
        args[-1][:] = 255
        return len(args[0])

    monkeypatch.setattr(b, "sweep_batch", reject)

    class Native:
        native_solves = 0
        native_seconds = 0.0
        highs_seconds = 0.0

        def solve(self, *args: Any) -> tuple[np.ndarray, int, int]:  # noqa: ANN401
            n = len(args[-3])
            self.native_solves += n
            self.native_seconds += 0.1
            return np.zeros(n), 0, 0

    record = b.sweep_key(
        ("REV", 3600),
        bits,
        b.TableStore(tmp_path),
        np.zeros((8, 17012)),
        min_clock=720,
        full=True,
        fallback=cast("Fallback", Native()),
    )
    assert record["native_solves"] == record["failures"] == 3  # noqa: PLR2004
    assert record["lp_solves"] == 0
    assert record["lp_seconds"] == 0.0
    assert record["native_seconds"] == 0.1  # noqa: PLR2004


# ──────────────────────────────────────────────────────────────────────
# Golden fixture
# ──────────────────────────────────────────────────────────────────────

# We hold the numba leap kernel to the outputs of the retired Rust kernel.
#
# `data/leap_kernel_golden.npz` stores seeded stage families and child tables,
# with the values, kinds, failure counts, and basis masks that `stl_solver_rs`
# returned for them. The commit that added the fixture holds
# `make_leap_golden.py`, which captured it. The fixture covers square and window
# stages under the rungs alone and under the production residue options that
# `leap.RESIDUE` held at the capture, the packing LP from the slack and
# crash bases, and the warm and cold packing runs of the leap benchmarks. We run
# the production cases with the captured options, so a later change of
# `RESIDUE` keeps this parity check. A build reads its child tables as read-only
# memmaps, and we hold those to the fixture too. We require equal kinds,
# equal failure counts, equal basis masks, and equal values, with NaN in the
# same places. Equal means equal as numbers, so a signed zero matches either
# sign: `llvm.minnum` and `llvm.maxnum` may return either zero of a tie, the
# fixture came from an arm64 Mac, and CI runs on x86-64. A gate on the certified
# values would hide a change of rung or basis.

FIXTURE = Path(__file__).with_name("leap_kernel_golden.npz")


@pytest.fixture(scope="module")
def golden() -> dict:
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}


def production(golden: dict) -> dict:
    return json.loads(str(golden["meta"]))["production"]


def inputs(golden: dict, label: str) -> dict:
    if label == "tables":
        return {
            name: golden[f"table_{name}"]
            for name in (
                "pcs",
                "pds",
                "succ_table",
                "fail_table",
                "succ_col",
                "fail_col",
                "rev",
            )
        }
    order = golden["stage_order"]
    return {
        "pcs": np.zeros(len(order), np.int32),
        "pds": order,
        "succ_table": np.ascontiguousarray(-golden["stage_s"]),
        "fail_table": np.ascontiguousarray(-golden["stage_f"][:, None]),
        "succ_col": np.arange(60, dtype=np.int32)[None, :],
        "fail_col": np.zeros(1, np.int32),
        "rev": np.ones(1),
    }


def test_fixture_covers_every_kind(golden: dict) -> None:
    meta = json.loads(str(golden["meta"]))
    assert meta["engine"] == "stl_solver_rs"
    kinds = set()
    for name, counts in meta["coverage"].items():
        if name.startswith("sweep_"):
            kinds |= {int(k) for k in counts}
    assert kinds == {0, 1, 2, 3, 4, 255}
    assert FIXTURE.stat().st_size < 500_000  # noqa: PLR2004


def test_fixture_holds_the_live_production_options(golden: dict) -> None:
    from stl.leap import RESIDUE, NativeFallback  # noqa: PLC0415

    if NativeFallback(**RESIDUE).kernel_options(("H1", 44)) != production(golden):
        pytest.skip(
            "production options changed since the Rust capture; "
            "the fixture still checks the kernel under the captured options"
        )


@pytest.mark.parametrize("mode", ["rungs", "production"])
@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("label", ["stages", "tables"])
def test_sweep_matches_golden(
    golden: dict,
    label: str,
    window: bool,  # noqa: FBT001
    mode: str,
) -> None:
    from stl.leap_kernel import sweep_batch  # noqa: PLC0415

    name = f"sweep_{label}_{'window' if window else 'square'}_{mode}"
    args = inputs(golden, label)
    n = len(args["pcs"])
    out = np.full(n, -7.0)
    kind = np.full(n, 77, np.uint8)
    failed = sweep_batch(
        args["pcs"],
        args["pds"],
        args["succ_table"],
        args["fail_table"],
        args["succ_col"],
        args["fail_col"],
        args["rev"],
        window,
        1e-6,
        out,
        kind,
        **({} if mode == "rungs" else production(golden)),
    )
    assert failed == int(golden[f"{name}_failed"])
    assert np.array_equal(kind, golden[f"{name}_kind"])
    assert np.array_equal(out, golden[f"{name}_value"], equal_nan=True)


@pytest.mark.parametrize("mapped", [(True, True), (True, False), (False, True)])
def test_memmapped_tables_share_the_compiled_kernel(
    golden: dict, tmp_path: Path, mapped: tuple[bool, bool]
) -> None:
    from stl import leap_kernel  # noqa: PLC0415

    name = "sweep_tables_square_production"
    args = inputs(golden, "tables")

    def sweep(
        succ_table: np.ndarray, fail_table: np.ndarray
    ) -> tuple[int, np.ndarray, np.ndarray]:
        n = len(args["pcs"])
        out = np.full(n, -7.0)
        kind = np.full(n, 77, np.uint8)
        failed = leap_kernel.sweep_batch(
            args["pcs"],
            args["pds"],
            succ_table,
            fail_table,
            args["succ_col"],
            args["fail_col"],
            args["rev"],
            False,  # noqa: FBT003
            1e-6,
            out,
            kind,
            **production(golden),
        )
        return failed, out, kind

    sweep(args["succ_table"], args["fail_table"])
    compiled = set(leap_kernel._sweep_parallel.signatures)  # noqa: SLF001
    tables = []
    for table, memmap in zip(("succ_table", "fail_table"), mapped, strict=False):
        if memmap:
            np.save(tmp_path / f"{table}.npy", args[table])
            tables.append(np.load(tmp_path / f"{table}.npy", mmap_mode="r"))
        else:
            tables.append(args[table])
    failed, out, kind = sweep(*tables)
    assert set(leap_kernel._sweep_parallel.signatures) == compiled  # noqa: SLF001
    assert failed == int(golden[f"{name}_failed"])
    assert np.array_equal(kind, golden[f"{name}_kind"])
    assert np.array_equal(out, golden[f"{name}_value"], equal_nan=True)


@pytest.mark.parametrize("crash", [False, True])
@pytest.mark.parametrize("window", [False, True])
def test_packing_matches_golden(golden: dict, crash: bool, window: bool) -> None:  # noqa: FBT001
    from stl.leap_kernel import solve_packing  # noqa: PLC0415

    name = f"packing_{'crash' if crash else 'slack'}_{'window' if window else 'square'}"
    s = golden["stage_s"]
    f = golden["stage_f"]
    n = len(f)
    out = np.full(n, -7.0)
    kind = np.full(n, 77, np.uint8)
    support = np.full((n, 2), 7, np.uint64)
    failed = solve_packing(s, f, window, out, kind, crash=crash, support_out=support)
    assert failed == int(golden[f"{name}_failed"])
    assert np.array_equal(kind, golden[f"{name}_kind"])
    assert np.array_equal(support, golden[f"{name}_support"])
    assert np.array_equal(out, golden[f"{name}_value"], equal_nan=True)


@pytest.mark.parametrize("warm", [False, True])
def test_packing_runs_match_golden(golden: dict, warm: bool) -> None:  # noqa: FBT001
    from stl.leap_kernel import solve_packing_runs  # noqa: PLC0415

    name = f"runs_{'warm' if warm else 'cold'}"
    result = solve_packing_runs(
        golden["stage_s"], golden["stage_f"], golden["stage_groups"], warm
    )
    assert np.count_nonzero(~np.isfinite(result[:, 0])) == int(golden[f"{name}_failed"])
    assert np.array_equal(result, golden[f"{name}_result"], equal_nan=True)
