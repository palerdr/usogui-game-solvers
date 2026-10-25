"""Contract tests for the browser API, its seat-scoped snapshot and the types.ts mirror.

The load-bearing one is :func:`test_no_unrevealed_action_reaches_the_client`.
Everything else guards the sequence protocol, legality delegation and schema
drift.
"""

from __future__ import annotations

import json
import re
import types
from enum import Enum
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from stl.game import (
    LS_WINDOW_START,
    OPENING_START_CLOCK,
    TOTAL_TTD_MAX,
    HalfRoundResult,
)
from stl.server.app import (
    Leaderboard,
    LeaderboardEntry,
    OutcomeView,
    PlayerView,
    SeriesConfig,
    SessionConfig,
    Snapshot,
    create_app,
    snapshot_from_session,
)
from stl.server.session import Phase
from tests.conftest import (
    StubHal as _StubHal,
    make_client as _client,
    make_session as _session,
    play_out as _play_out,
)


_SNAPSHOT_FIELDS = {
    "sequence",
    "phase",
    "game_index",
    "human_name",
    "clock_display",
    "clock_seconds",
    "round",
    "half",
    "turn_duration",
    "leap_window",
    "dropper_name",
    "checker_name",
    "human_role",
    "legal_seconds",
    "players",
    "cylinder_max",
    "ttd_max",
    "half_rounds",
    "last_outcome",
    "winner_name",
    "winner_is_human",
    "stopped",
}
_PLAYER_FIELDS = {
    "name",
    "character",
    "role",
    "cylinder_seconds",
    "ttd_seconds",
    "deaths",
    "is_human",
}
_OUTCOME_FIELDS = {
    "round",
    "half",
    "dropper",
    "checker",
    "drop_time",
    "check_time",
    "result",
    "st_gained",
    "death_duration",
    "survived",
    "survival_probability",
    "game_over",
    "session_ending",
    "winner_name",
}


def _walk(payload) -> list:
    """Every scalar anywhere in a JSON payload, at any depth."""

    if isinstance(payload, dict):
        out = []
        for key, value in payload.items():
            out.append(key)
            out.extend(_walk(value))
        return out
    if isinstance(payload, list):
        return [item for value in payload for item in _walk(value)]
    return [payload]


def test_no_unrevealed_action_reaches_the_client() -> None:
    """While the human is deciding, no action exists to leak.

    Hal's second is chosen inside ``PlaySession.submit``, after the human has
    committed, so a snapshot taken during AWAITING_ACTION cannot carry it. This
    walks the whole payload rather than checking named fields, so a future
    nested addition cannot reintroduce the leak unnoticed.
    """

    client = _client()
    snapshot = client.get("/api/session").json()
    begun = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()

    assert begun["phase"] == "awaiting_action"
    assert begun["last_outcome"] is None
    assert set(begun) == _SNAPSHOT_FIELDS
    assert all(set(player) == _PLAYER_FIELDS for player in begun["players"])
    keys = _walk(begun)
    assert "drop_time" not in keys
    assert "check_time" not in keys

    revealed = client.post(
        "/api/session/action",
        json={"sequence": begun["sequence"], "second": begun["legal_seconds"][0]},
    ).json()
    assert revealed["phase"] == "awaiting_ack"
    assert revealed["last_outcome"]["drop_time"] >= 1
    assert revealed["last_outcome"]["check_time"] >= 1
    assert set(revealed["last_outcome"]) == _OUTCOME_FIELDS
    # The same walk finds both keys once they are public, so the assertions
    # above are detecting absence rather than passing vacuously.
    revealed_keys = _walk(revealed)
    assert "drop_time" in revealed_keys
    assert "check_time" in revealed_keys


def test_the_reveal_is_dropped_again_once_acknowledged() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    snapshot = client.post(
        "/api/session/action",
        json={"sequence": snapshot["sequence"], "second": snapshot["legal_seconds"][0]},
    ).json()
    snapshot = client.post("/api/session/ack", json={"sequence": snapshot["sequence"]}).json()
    if snapshot["phase"] == "awaiting_action":
        assert snapshot["last_outcome"] is None
        assert "drop_time" not in _walk(snapshot)


def test_legal_seconds_are_empty_until_the_human_is_on_the_clock() -> None:
    client = _client()
    assert client.get("/api/session").json()["legal_seconds"] == []


def test_a_stale_sequence_is_rejected() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    client.post("/api/session/begin", json={"sequence": snapshot["sequence"]})
    replayed = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]})
    assert replayed.status_code == 409


def test_session_replacement_is_sequenced_and_cannot_abandon_live_play() -> None:
    client = _client()
    initial = client.get("/api/session").json()
    assert client.post("/api/session", json={}).status_code == 422

    begun = client.post(
        "/api/session/begin", json={"sequence": initial["sequence"]}
    ).json()
    assert (
        client.post("/api/session", json={"sequence": initial["sequence"]}).status_code
        == 409
    )
    assert (
        client.post("/api/session", json={"sequence": begun["sequence"]}).status_code
        == 409
    )


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_restart_clears_each_phase_and_rejects_previous_moves(steps) -> None:
    client = _client(max_half_rounds=1)
    initial = client.get("/api/session").json()
    state = initial
    for route, extra in [
        ("begin", {}), ("action", {"second": 60}), ("ack", {}),
    ][:steps]:
        state = client.post(
            f"/api/session/{route}", json={"sequence": state["sequence"], **extra}
        ).json()
    response = client.post(
        "/api/session/restart", json={"sequence": state["sequence"]}
    )
    assert response.status_code == 200
    fresh = response.json()
    assert fresh == {**initial, "sequence": state["sequence"] + 1}
    history = client.get("/api/transcript").json()
    assert history["games"] == []
    assert history["current_game"]["public_history"] == []
    assert not any(history["tally"].values())
    assert history["current_game"]["seed"] != 41
    assert history["base_seed"] == history["current_game"]["seed"]
    for route in ("action", "restart"):
        assert client.post(
            f"/api/session/{route}",
            json={"sequence": state["sequence"], "second": 1} if route == "action"
            else {"sequence": state["sequence"]},
        ).status_code == 409

    finished = _play_out(client, fresh)
    next_game = client.post(
        "/api/session", json={"sequence": finished["sequence"]}
    ).json()
    assert next_game["game_index"] == 1
    assert client.get("/api/transcript").json()["current_game"]["seed"] == history["base_seed"] + 1


def test_session_replacement_keeps_a_monotonic_sequence_and_display_only_name() -> None:
    client = _client(start_clock=LS_WINDOW_START)
    initial = client.get("/api/session").json()
    label = "Alice <the challenger>"
    replaced = client.post(
        "/api/session",
        json={"sequence": initial["sequence"], "human_name": label},
    ).json()

    assert replaced["sequence"] > initial["sequence"]
    assert replaced["human_name"] == label
    assert {player["name"] for player in replaced["players"]} == {"Hal", label}
    human = next(player for player in replaced["players"] if player["is_human"])
    opponent = next(player for player in replaced["players"] if not player["is_human"])
    assert human["character"] == "baku"
    assert human["role"] == replaced["human_role"]
    assert opponent["character"] == "hal"
    assert opponent["role"] != replaced["human_role"]
    assert replaced["winner_is_human"] is None
    assert client.get("/api/rules").json()["human_name"] == label


def test_hal_is_a_reserved_human_display_name() -> None:
    client = _client()
    sequence = client.get("/api/session").json()["sequence"]
    response = client.post(
        "/api/session", json={"sequence": sequence, "human_name": " hAl "}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        ("/api/session/begin", {"sequence": True}),
        ("/api/session/begin", {"sequence": 0.0}),
        ("/api/session/action", {"sequence": 0, "second": True}),
        ("/api/session/action", {"sequence": 0, "second": 2.0}),
        ("/api/session", {"sequence": 0, "start_clock": 720.0}),
        ("/api/session", {"sequence": 0, "max_half_rounds": False}),
    ],
)
def test_mutation_requests_reject_coercible_non_integer_values(endpoint, body) -> None:
    assert _client().post(endpoint, json=body).status_code == 422


def test_terminal_replacement_finishes_then_resets_provider_once() -> None:
    class _LifecycleProvider:
        def __init__(self) -> None:
            self.resets = 0
            self.outcomes = []

        def reset_game(self) -> None:
            self.resets += 1

        def end_game(self, outcome) -> None:
            self.outcomes.append(outcome)

    provider = _LifecycleProvider()
    hal = _StubHal()
    hal.provider = provider
    app = create_app(
        hal_factory=lambda: hal,
        config=SessionConfig(seed=41, max_half_rounds=1),
        webclient_dist=None,
    )
    client = TestClient(app)
    snapshot = client.get("/api/session").json()
    snapshot = client.post(
        "/api/session/begin", json={"sequence": snapshot["sequence"]}
    ).json()
    snapshot = client.post(
        "/api/session/action",
        json={"sequence": snapshot["sequence"], "second": snapshot["legal_seconds"][0]},
    ).json()
    assert snapshot["last_outcome"]["game_over"] is False
    assert snapshot["last_outcome"]["session_ending"] is True
    snapshot = client.post(
        "/api/session/ack", json={"sequence": snapshot["sequence"]}
    ).json()
    assert snapshot["phase"] == "game_over"
    assert provider.resets == 1
    assert len(provider.outcomes) == 1

    replacement = client.post(
        "/api/session", json={"sequence": snapshot["sequence"]}
    ).json()
    assert replacement["phase"] == "rules"
    assert replacement["sequence"] > snapshot["sequence"]
    assert provider.resets == 2
    assert len(provider.outcomes) == 1


def test_an_illegal_second_is_refused_without_advancing_the_game() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    refused = client.post(
        "/api/session/action", json={"sequence": snapshot["sequence"], "second": 99}
    )
    assert refused.status_code == 422
    assert client.get("/api/session").json()["sequence"] == snapshot["sequence"]


def test_only_a_human_dropper_is_offered_the_leap_second() -> None:
    """The 61 asymmetry is decided server-side and never in the client."""

    client = _client(start_clock=LS_WINDOW_START)
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    assert snapshot["turn_duration"] == 61
    if snapshot["human_role"] == "dropper":
        assert snapshot["legal_seconds"][-1] == 61
    else:
        assert snapshot["legal_seconds"][-1] == 60


def test_a_full_game_reaches_a_terminal_phase() -> None:
    client = _client(max_half_rounds=6)
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    while snapshot["phase"] == "awaiting_action":
        snapshot = client.post(
            "/api/session/action",
            json={"sequence": snapshot["sequence"], "second": snapshot["legal_seconds"][0]},
        ).json()
        snapshot = client.post(
            "/api/session/ack", json={"sequence": snapshot["sequence"]}
        ).json()
    assert snapshot["phase"] == "game_over"
    assert snapshot["half_rounds"] >= 1


def test_rules_text_is_served_for_the_opening_screen() -> None:
    payload = _client().get("/api/rules").json()
    assert payload["human_name"] == "Baku"
    assert len(payload["lines"]) > 0


@pytest.mark.parametrize("phase_endpoint", ["ack", "action"])
def test_transitions_are_refused_from_the_rules_phase(phase_endpoint: str) -> None:
    client = _client()
    sequence = client.get("/api/session").json()["sequence"]
    body = {"sequence": sequence}
    if phase_endpoint == "action":
        body["second"] = 30
    assert client.post(f"/api/session/{phase_endpoint}", json=body).status_code == 409


def test_rules_carry_the_public_hal_label_and_turn_contract() -> None:
    app = create_app(
        hal_factory=_StubHal,
        config=SessionConfig(seed=41),
        series=SeriesConfig(hal_agent="dth", public_hal_label="Opponent A"),
        webclient_dist=None,
    )
    payload = TestClient(app).get("/api/rules").json()
    assert payload["hal_label"] == "Opponent A"
    assert _client().get("/api/rules").json()["hal_label"] == "dth"


def test_the_reveal_names_its_own_round_and_half() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    assert (snapshot["round"], snapshot["half"]) == (1, 1)
    revealed = client.post(
        "/api/session/action",
        json={"sequence": snapshot["sequence"], "second": snapshot["legal_seconds"][0]},
    ).json()
    assert (revealed["last_outcome"]["round"], revealed["last_outcome"]["half"]) == (1, 1)


def test_a_series_keeps_hal_and_seeds_each_game_like_the_cli() -> None:
    """Game N of a browser series is the CLI's game N: seed + N, one Hal."""

    class _LifecycleProvider:
        def __init__(self) -> None:
            self.resets = 0
            self.outcomes = []

        def reset_game(self) -> None:
            self.resets += 1

        def end_game(self, outcome) -> None:
            self.outcomes.append(outcome)

        def match_summary(self) -> str:
            return f"stub: {len(self.outcomes)} games"

    provider = _LifecycleProvider()
    hal = _StubHal()
    hal.provider = provider
    app = create_app(
        hal_factory=lambda: hal,
        config=SessionConfig(seed=41, max_half_rounds=2),
        series=SeriesConfig(hal_agent="dth"),
        webclient_dist=None,
    )
    client = TestClient(app)

    first = client.get("/api/session").json()
    assert first["game_index"] == 0
    first = _play_out(client, first)
    written = client.get("/api/transcript").json()
    assert written["schema_version"] == "arena-public-play-session-v1"
    assert written["base_seed"] == 41
    assert [game["seed"] for game in written["games"]] == [41]
    assert written["hal_summary"] == "stub: 1 games"

    second = client.post("/api/session", json={"sequence": first["sequence"]}).json()
    assert second["game_index"] == 1
    assert second["phase"] == "rules"
    assert provider.resets == 2
    served = client.get("/api/transcript").json()
    assert served["current_game"]["game_index"] == 1
    assert served["current_game"]["seed"] == 42
    assert served["current_game"]["public_history"] == []
    assert served["tally"]["stopped"] == 1

    second = _play_out(client, second)
    written = client.get("/api/transcript").json()
    assert [game["seed"] for game in written["games"]] == [41, 42]
    assert [game["game_index"] for game in written["games"]] == [0, 1]
    assert len(provider.outcomes) == 2

    # Replacing an unstarted game does not consume a game index or a seed.
    third = client.post("/api/session", json={"sequence": second["sequence"]}).json()
    replaced = client.post("/api/session", json={"sequence": third["sequence"]}).json()
    assert replaced["game_index"] == 2
    assert client.get("/api/transcript").json()["current_game"]["seed"] == 43
    assert len(client.get("/api/transcript").json()["games"]) == 2


def test_transcript_never_carries_an_unrevealed_action() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    payload = client.get("/api/transcript").json()
    assert payload["current_game"]["public_history"] == []
    assert "drop_second" not in _walk(payload)
    assert "check_second" not in _walk(payload)

    client.post(
        "/api/session/action",
        json={"sequence": snapshot["sequence"], "second": snapshot["legal_seconds"][0]},
    )
    revealed = client.get("/api/transcript").json()
    assert len(revealed["current_game"]["public_history"]) == 1
    assert "drop_second" in _walk(revealed)


def test_hal_details_are_withheld_when_concealed() -> None:
    class _TalkativeProvider:
        def match_summary(self) -> str:
            return "dth complete: 0 moves"

        def experiment_diagnostics(self) -> dict:
            return {"secret": 1}

    def _factory(conceal: bool) -> TestClient:
        hal = _StubHal()
        hal.provider = _TalkativeProvider()
        app = create_app(
            hal_factory=lambda: hal,
            config=SessionConfig(seed=41),
            series=SeriesConfig(conceal_hal_details=conceal),
            webclient_dist=None,
        )
        return TestClient(app)

    open_payload = _factory(False).get("/api/transcript").json()
    assert open_payload["hal_summary"] == "dth complete: 0 moves"
    assert open_payload["hal_diagnostics"] == {"secret": 1}
    concealed = _factory(True).get("/api/transcript").json()
    assert "hal_summary" not in concealed
    assert "hal_diagnostics" not in concealed


def test_new_session_accepts_a_start_clock_and_explicit_seed() -> None:
    client = _client()
    initial = client.get("/api/session").json()
    replaced = client.post(
        "/api/session",
        json={"sequence": initial["sequence"], "start_clock": LS_WINDOW_START, "seed": 7},
    ).json()
    assert replaced["clock_seconds"] == LS_WINDOW_START
    assert replaced["leap_window"] is True
    assert client.get("/api/transcript").json()["current_game"]["seed"] == 7
    back = client.post(
        "/api/session",
        json={"sequence": replaced["sequence"], "start_clock": OPENING_START_CLOCK},
    ).json()
    assert back["leap_window"] is False


def test_the_built_client_is_served_from_the_root(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<title>built client</title>", encoding="utf-8")
    (dist / "app.js").write_text("console.log(1)", encoding="utf-8")
    app = create_app(hal_factory=_StubHal, config=SessionConfig(seed=41), webclient_dist=dist)
    client = TestClient(app)
    page = client.get("/")
    assert "built client" in page.text
    assert page.headers["cache-control"] == "no-cache"
    assert client.get("/app.js").status_code == 200
    assert "cache-control" not in client.get("/app.js").headers
    # The API still wins over the static mount.
    assert client.get("/api/session").json()["phase"] == "rules"


def test_a_missing_build_is_explained_rather_than_a_404(tmp_path) -> None:
    app = create_app(
        hal_factory=_StubHal, config=SessionConfig(seed=41), webclient_dist=tmp_path / "none"
    )
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "npm --prefix web run build" in response.text


@pytest.mark.parametrize(
    ("half", "human_second", "fatal_player", "winner_is_human"),
    [
        (1, 1, "human", False),
        (2, 60, "hal", True),
    ],
)
def test_snapshot_carries_authoritative_winner_seat(
    half: int,
    human_second: int,
    fatal_player: str,
    winner_is_human: bool,
) -> None:
    session = _session(human_display_name="A display label")
    session.game.current_half = half
    player = session.human if fatal_player == "human" else session.hal
    player.ttd = TOTAL_TTD_MAX
    session.begin()
    session.submit(human_second)

    snapshot = snapshot_from_session(session)
    assert snapshot.last_outcome is not None
    assert snapshot.last_outcome.game_over is True
    assert snapshot.last_outcome.session_ending is True
    assert snapshot.winner_is_human is winner_is_human


_TYPES = Path(__file__).resolve().parents[2] / "web" / "src" / "types.ts"


def _interface_fields(source: str, name: str) -> dict[str, tuple[str, bool]]:
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"web/src/types.ts has no interface {name}")
    fields: dict[str, tuple[str, bool]] = {}
    for field, optional, annotation in re.findall(
        r"^\s*(\w+)(\?)?:\s*([^;]+);", match.group(1), re.MULTILINE
    ):
        fields[field] = (" ".join(annotation.split()), not bool(optional))
    return fields


def _typescript_type(annotation: object) -> str:
    origin = get_origin(annotation)
    if origin is list:
        return f"{_typescript_type(get_args(annotation)[0])}[]"
    if origin in (types.UnionType, Union):
        return " | ".join(_typescript_type(item) for item in get_args(annotation))
    if origin is Literal:
        return " | ".join(json.dumps(item) for item in get_args(annotation))
    if annotation is type(None):
        return "null"
    if annotation in (int, float):
        return "number"
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is Phase:
        return "Phase"
    if annotation is HalfRoundResult:
        return "OutcomeResult"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        raise AssertionError(f"TypeScript alias missing for enum {annotation.__name__}")
    raise AssertionError(f"unsupported Python wire annotation {annotation!r}")


def _model_fields(model: type[BaseModel]) -> dict[str, tuple[str, bool]]:
    return {
        name: (_typescript_type(field.annotation), field.is_required())
        for name, field in model.model_fields.items()
    }


def _string_union(source: str, name: str) -> set[str]:
    match = re.search(rf"export type {name}\s*=\s*(.*?);", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"web/src/types.ts has no type alias {name}")
    return set(re.findall(r'"([^"]+)"', match.group(1)))


@pytest.mark.parametrize(
    ("model", "interface"),
    [
        (Snapshot, "Snapshot"),
        (PlayerView, "PlayerView"),
        (OutcomeView, "OutcomeView"),
        (Leaderboard, "Leaderboard"),
        (LeaderboardEntry, "LeaderboardEntry"),
    ],
)
def test_typescript_mirrors_the_python_schema(model, interface: str) -> None:
    source = _TYPES.read_text(encoding="utf-8")
    assert _model_fields(model) == _interface_fields(source, interface)


def test_typescript_enum_aliases_match_python_values() -> None:
    source = _TYPES.read_text(encoding="utf-8")
    assert _string_union(source, "Phase") == {item.value for item in Phase}
    assert _string_union(source, "OutcomeResult") == {
        item.value for item in HalfRoundResult
    }
