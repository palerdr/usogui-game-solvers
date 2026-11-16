# Copyright 2026 The STL authors
"""Shared fixtures and fakes of the test suite.

- ``hal``, ``baku``, ``referee``, ``game`` and ``game_at_r1`` are engine
  fixtures.
- ``StageTable`` stands in for ``stl.dth.Table.stage`` with a fixed
  certified stage, and ``certified_stage`` builds that stage.
- ``make_decision`` builds one canonical decision, and ``make_reveal`` builds
  one public half-round reveal. ``make_forecast`` asks an opponent model for
  one forecast.
- ``RecordingHal`` is a Hal stand-in that records each consultation, and
  ``make_session`` builds a ``PlaySession`` around it.
- ``StubHal`` is a deterministic Hal that never opens the DTH table.
  ``make_client`` builds a local browser app around it, and ``play_out``
  drives one game of that app to game over.
- ``MemoryStore`` holds the hosted session records in memory with the
  compare-and-set of the Redis store, and ``begin`` opens the first decision
  of a hosted game.

- ``fma`` and ``solve_stage`` provide the scalar leap reference for
  numerical parity tests.

Test modules import the fakes and references as ``tests.conftest``.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, cast

from numba import njit
from numba.core import types
from numba.extending import intrinsic

from stl.leap import GAP, StageResult, solve_lp, stage_matrix

if TYPE_CHECKING:
    from llvmlite.ir import IRBuilder, Value
from typing import TypedDict, Unpack

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stl.dth import Stage
from stl.game import (
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)
from stl.hal import (
    ACTION_COUNT,
    CanonicalDecision,
    PerfectHalForecast,
    PerfectHalOpponentModel,
    PublicDecisionState,
    PublicHalfRound,
    PublicPlayerState,
)
from stl.server.app import SessionConfig, create_app
from stl.server.session import PlaySession


@pytest.fixture
def hal() -> Player:
    return Player(name="Hal", physicality=PHYSICALITY_HAL)


@pytest.fixture
def baku() -> Player:
    return Player(name="Baku", physicality=PHYSICALITY_BAKU)


@pytest.fixture
def referee() -> Referee:
    return Referee()


@pytest.fixture
def game(hal: Player, baku: Player, referee: Referee) -> Game:
    """Game with Hal dropping first, seeded for determinism."""
    g = Game(player1=hal, player2=baku, referee=referee, first_dropper=hal)
    g.seed(42)
    return g


@pytest.fixture
def game_at_r1(game: Game) -> Game:
    """Game with clock set to 720 (8:12 AM) — manga R1 start."""
    game.game_clock = 720.0
    return game


def certified_stage(matrix: np.ndarray | None = None) -> Stage:
    resolved = (
        np.eye(ACTION_COUNT, dtype=np.float64)
        if matrix is None
        else np.asarray(matrix, dtype=np.float64)
    )
    uniform = np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float64)
    return Stage(
        state=(0, 60, 0, 60),
        value=1.0 / ACTION_COUNT,
        matrix=resolved,
        drop=uniform.copy(),
        check=uniform.copy(),
        gap=0.0,
    )


class StageTable:
    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self.matrix = matrix
        self.states: list[tuple[int, int, int, int]] = []

    def stage(self, state: tuple[int, int, int, int]) -> Stage:
        a, b, c, d = state
        normalized = (int(a), int(b), int(c), int(d))
        self.states.append(normalized)
        stage = certified_stage(self.matrix)
        return Stage(
            state=normalized,
            value=stage.value,
            matrix=stage.matrix,
            drop=stage.drop,
            check=stage.check,
            gap=stage.gap,
        )


def make_decision(
    *,
    role: str = "dropper",
    actor_name: str = "Hal",
    legal_seconds: tuple[int, ...] = tuple(range(1, 61)),
    checker_cylinder: float = 12.0,
    checker_ttd: float = 60.0,
    dropper_cylinder: float = 24.0,
    dropper_ttd: float = 120.0,
) -> CanonicalDecision:
    return CanonicalDecision(
        role=role,
        actor_name=actor_name,
        turn_duration=60,
        legal_seconds=legal_seconds,
        checker_cylinder_seconds=checker_cylinder,
        checker_ttd_seconds=checker_ttd,
        dropper_cylinder_seconds=dropper_cylinder,
        dropper_ttd_seconds=dropper_ttd,
    )


def make_reveal(
    *,
    self_role: str = "dropper",
    self_name: str = "Hal",
    opponent_name: str = "Baku",
    self_action: int = 3,
    opponent_action: int = 7,
    game_index: int = 0,
    half_round_index: int = 0,
    game_over: bool = False,
) -> PublicHalfRound:
    if self_role == "dropper":
        dropper_name, checker_name = self_name, opponent_name
        drop_time, check_time = self_action, opponent_action
    elif self_role == "checker":
        dropper_name, checker_name = opponent_name, self_name
        drop_time, check_time = opponent_action, self_action
    else:
        msg = "self_role must be dropper or checker"
        raise ValueError(msg)
    return PublicHalfRound(
        game_index=game_index,
        half_round_index=half_round_index,
        pre_decision_state=PublicDecisionState(
            game_clock_seconds=720.0,
            round_index=1,
            half_index=1,
            turn_duration=60,
            players=(
                PublicPlayerState(self_name, 24.0, 120.0),
                PublicPlayerState(opponent_name, 12.0, 60.0),
            ),
        ),
        dropper_name=dropper_name,
        checker_name=checker_name,
        drop_time=drop_time,
        check_time=check_time,
        outcome="check_success",
        game_over=game_over,
        winner_name=self_name if game_over else None,
    )


def make_forecast(
    model: PerfectHalOpponentModel,
    role: str,
    *,
    game_index: int = 0,
    decision_index: int = 0,
) -> PerfectHalForecast:
    return model.predict(
        role,
        state_regime=(0, 1, 0, 2),
        game_index=game_index,
        game_decision_index=decision_index,
    )


class RecordingHal:
    """Hal stand-in that records exactly when it is consulted."""

    def __init__(self, second: int = 30) -> None:
        self.second = second
        self.calls: list[str] = []
        self.provider = object()

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        del game
        self.calls.append(role)
        return min(self.second, turn_duration)


def make_session(
    *,
    hal_agent: RecordingHal | StubHal | None = None,
    seed: int | None = 41,
    start_clock: int = OPENING_START_CLOCK,
    max_half_rounds: int | None = None,
    human_display_name: str = "Baku",
) -> PlaySession:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    human = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=random.Random(seed),
    )
    game.game_clock = start_clock
    return PlaySession(
        game=game,
        hal_agent=hal_agent if hal_agent is not None else RecordingHal(),
        hal=hal,
        human=human,
        human_display_name=human_display_name,
        game_seed=seed,
        start_clock=start_clock,
        max_half_rounds=max_half_rounds,
    )


class StubHal:
    """Deterministic Hal. Avoids memory-mapping the 2.4 GB DTH artifact."""

    def __init__(self, second: int = 37) -> None:
        self.second = second
        self.provider = object()

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        del game, role
        return min(self.second, turn_duration)


class ClientConfig(TypedDict, total=False):
    human_name: str
    start_clock: int
    max_half_rounds: int | None


def make_client(**config: Unpack[ClientConfig]) -> TestClient:
    app = create_app(
        hal_factory=StubHal,
        config=SessionConfig(seed=41, **config),
        webclient_dist=None,
    )
    return TestClient(app)


def play_out(client: TestClient, snapshot: dict) -> dict:
    """Drive one game from the rules screen to game over with the first legal second."""

    snapshot = client.post(
        "/api/session/begin", json={"sequence": snapshot["sequence"]}
    ).json()
    while snapshot["phase"] == "awaiting_action":
        snapshot = client.post(
            "/api/session/action",
            json={
                "sequence": snapshot["sequence"],
                "second": snapshot["legal_seconds"][0],
            },
        ).json()
        snapshot = client.post(
            "/api/session/ack", json={"sequence": snapshot["sequence"]}
        ).json()
    assert snapshot["phase"] == "game_over"
    return snapshot


class MemoryStore:
    def __init__(self) -> None:
        self.rows = {}

    async def get(self, key: str) -> str | None:
        return self.rows.get(key)

    async def compare_set(self, key: str, old: str | None, new: str) -> bool:
        if self.rows.get(key) != old:
            return False
        self.rows[key] = new
        return True


def begin(client: TestClient) -> dict:
    state = client.get("/api/session").json()
    response = client.post("/api/session/begin", json={"sequence": state["sequence"]})
    assert response.status_code == 200
    return response.json()


# Scalar reference for the frozen numerical kernel.
@intrinsic
def _fma(
    _typingctx: object, a: types.Type, b: types.Type, c: types.Type
) -> tuple | None:
    """We return a*b+c with one rounding."""
    if not (a == types.float64 and b == types.float64 and c == types.float64):
        return None

    def codegen(
        _context: object,
        builder: IRBuilder,
        _signature: object,
        args: tuple[Value, Value, Value],
    ) -> Value:
        return cast("Value", builder.fma(*args))

    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True)
def _fused(a: float, b: float, c: float) -> float:
    return _fma(a, b, c)  # pyright: ignore[reportCallIssue] -- intrinsic removes typing context.


def fma(a: float, b: float, c: float) -> float:
    """We return a*b+c with one rounding and raise the errors of math.fma."""
    x, y, z = float(a), float(b), float(c)
    r = _fused(x, y, z)
    if math.isnan(r):
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            message = "invalid operation in fma"
            raise ValueError(message)
    elif math.isinf(r) and math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
        message = "overflow in fma"
        raise OverflowError(message)
    return r


CERT_RADIUS = 1e-10


def solve_stage(
    s: np.ndarray | list[float],
    f: float,
    window: bool,
    *,
    allow_lp: bool = True,
) -> StageResult:
    s = np.asarray(s, dtype=np.float64)
    if s.shape != (60,) or not np.isfinite(s).all() or not math.isfinite(f):
        message = "stage needs 60 finite successes and a finite failure"
        raise ValueError(message)
    f = float(f)
    s = list(map(float, s))
    lo, hi = min(s), max(s)
    lower, upper = max(lo, min(f, s[0])), min(hi, max(f, s[0]))
    if window:
        lower, upper = max(lower, f), max(upper, f)
    kind = 0
    if upper - lower <= GAP:
        value = (lower + upper) / 2
        if window and f >= (max(lo, min(f, s[0])) + min(hi, max(f, s[0]))) / 2:
            kind = 2
        return StageResult(value, kind, lower, upper)
    d = s[0] - f
    if abs(d) >= 1e-12:
        b = [0.0] + [(s[k - 1] - s[k]) / d for k in range(1, 60)]
        r = [1.0]
        for k in range(1, 60):
            acc = 0.0
            try:
                for j in range(k):
                    acc = fma(b[k - j], r[j], acc)
            except (OverflowError, ValueError):
                acc = math.nan
            r.append(acc)
            if not math.isfinite(acc):
                break
        q = [max(x, 0.0) if math.isfinite(x) else math.nan for x in reversed(r)]
        total = sum(q)
        if math.isfinite(total) and total > 0:
            if min(r) >= 0 and lo >= -2 and hi <= 2 and abs(f) <= 2 and abs(d) >= GAP:
                value = f + d / total
                lower, upper = value - CERT_RADIUS, value + CERT_RADIUS
            else:
                q = np.array(q) / total
                payoffs = stage_matrix(s, f, False) @ q
                lower, upper = float(payoffs.min()), float(payoffs.max())
                value = (lower + upper) / 2
            if window:
                lower, upper = max(lower, f), max(upper, f)
                kind = 2 if f >= value else 1
                value = max(value, f)
            else:
                kind = 1
            if math.isfinite(upper - lower) and upper - lower <= GAP:
                return StageResult(value, kind, lower, upper)
    if allow_lp:
        return solve_lp(s, f, window)
    return StageResult(math.nan, 255, -math.inf, math.inf)
