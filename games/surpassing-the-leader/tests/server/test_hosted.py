# Copyright 2026 The STL authors
"""Exercise player isolation and commits across stateless HTTP workers.

The second part plays translated Hal through the hosted app: its opponent
memory survives game boundaries and worker replacement, and the production
entry serves it.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx
import pytest
from fastapi.testclient import TestClient

from stl.game import LS_WINDOW_START, Game
from stl.hal import TranslatedHal
from stl.server import hosted as production
from stl.server.app import SeriesConfig, SessionConfig, create_app, is_offensive
from stl.server.hosted import (
    COOKIE,
    PLAYER_COOKIE,
    OpponentMemory,
    create_hosted_app,
    game_row,
    life_left,
)
from stl.server.session import PolicyDrivenAgent
from tests.conftest import MemoryStore, begin
from tests.conftest import StageTable as _StageTable

if TYPE_CHECKING:
    from fastapi import FastAPI


class Hal:
    provider = object()

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        return 37


class MemoryLedger:
    """The database functions' contract in memory.

    A closed game is final, a row keeps its owner and cannot shrink, a name
    needs a recorded game, and a standing is the player's best win.
    """

    def __init__(self) -> None:
        self.games, self.names, self.writes, self.clock = {}, {}, 0, 0

    def _store(self, key: tuple[str, int], row: dict) -> None:
        self.clock += 1
        self.games[key] = {**row, "updated": self.clock}

    async def record_game(self, row: dict) -> None:
        self.writes += 1
        key = (row["series_id"], row["game_index"])
        held = self.games.get(key)
        if held is None or (
            held["status"] == "active"
            and held["player_id"] == row["player_id"]
            and row["half_rounds"] >= held["half_rounds"]
        ):
            self._store(key, row)

    async def abandon_series(self, series_id: str) -> None:
        for key, row in list(self.games.items()):
            if key[0] == series_id and row["status"] == "active":
                self._store(key, {**row, "status": "abandoned"})

    async def set_player_name(self, player_id: str, name: str) -> str:
        if not any(row["player_id"] == player_id for row in self.games.values()):
            return "no_game"
        self.names[player_id] = name
        return "ok"

    async def leaderboard(self, player_id: str) -> dict:
        best = {}
        for row in sorted(self.games.values(), key=lambda row: row["updated"]):
            if row["status"] == "finished" and row["human_won"]:
                held = best.get(row["player_id"])
                if held is None or row["score"] > held["score"]:
                    best[row["player_id"]] = row
        won = list(best.values())
        ranked = sorted(
            (row for row in won if row["player_id"] in self.names),
            key=lambda row: -row["score"],
        )
        mine = [row for row in won if row["player_id"] == player_id]
        return {
            "entries": [
                {
                    "rank": rank,
                    "name": self.names[row["player_id"]],
                    "score": row["score"],
                    "half_rounds": row["half_rounds"],
                    "is_you": row["player_id"] == player_id,
                    # A careless ledger must not leak these through the route.
                    "player_id": row["player_id"],
                    "game_seed": row["game_seed"],
                }
                for rank, row in enumerate(ranked[:10], 1)
            ],
            "your_rank": next(
                (
                    rank
                    for rank, row in enumerate(ranked, 1)
                    if row["player_id"] == player_id
                ),
                None,
            ),
            "your_name": self.names.get(player_id),
            "your_score": mine[0]["score"] if mine else None,
        }


def hosted(
    store: MemoryStore, version: str = "test", ledger: MemoryLedger | None = None
) -> FastAPI:
    return create_hosted_app(
        store,
        lambda game_seed, policy_seed, sequence_start=0: create_app(
            hal_factory=Hal,
            config=SessionConfig(seed=game_seed),
            webclient_dist=None,
            sequence_start=sequence_start,
        ),
        version=version,
        secure_cookie=False,
        ledger=ledger,
    )


def test_players_are_isolated_and_a_fresh_worker_recovers_the_game() -> None:
    store = MemoryStore()
    first, other = TestClient(hosted(store)), TestClient(hosted(store))
    state = begin(first)
    response = first.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 5}
    )
    assert response.status_code == 200
    resolved = response.json()
    assert resolved["last_outcome"] is not None
    assert other.get("/api/session").json()["phase"] == "rules"
    replacement = TestClient(hosted(store))
    replacement.cookies.update(first.cookies)
    assert replacement.get("/api/session").json() == resolved
    assert first.cookies[COOKIE] != other.cookies[COOKIE]
    transcript = replacement.get("/api/transcript").json()

    def check_seeds(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"seed", "game_seed", "base_seed"}:
                    assert child is None
                check_seeds(child)
        elif isinstance(value, list):
            for child in value:
                check_seeds(child)

    check_seeds(transcript)


def test_conflicting_commits_reveal_only_the_winner() -> None:
    async def exercise() -> None:
        store = MemoryStore()
        transport = httpx.ASGITransport(app=hosted(store))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            state = (await client.get("/api/session")).json()
            state = (
                await client.post(
                    "/api/session/begin", json={"sequence": state["sequence"]}
                )
            ).json()
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/api/session/action",
                        json={"sequence": state["sequence"], "second": second},
                    )
                    for second in (1, 60)
                ]
            )
            assert sorted(r.status_code for r in responses) == [200, 409]
            loser = next(r for r in responses if r.status_code == 409)
            assert set(loser.json()) == {"detail"}
            winner = next(r for r in responses if r.status_code == 200)
            assert (await client.get("/api/session")).json() == winner.json()
            assert len(json.loads(next(iter(store.rows.values())))["events"]) == 2

    asyncio.run(exercise())


def test_restart_survives_worker_recovery_and_keeps_other_players_isolated() -> None:
    store = MemoryStore()
    client, other = TestClient(hosted(store)), TestClient(hosted(store))
    other_state = begin(other)
    state = begin(client)
    state = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    ).json()
    response = client.post("/api/session/restart", json={"sequence": state["sequence"]})
    assert response.status_code == 200
    fresh = response.json()
    assert fresh["phase"] == "rules"
    assert fresh["sequence"] > state["sequence"]
    assert fresh["last_outcome"] is None
    assert fresh["half_rounds"] == 0
    # The restart opens a new record: no commands to replay, and the sequence
    # continues, so a request from before the restart stays stale.
    records = [json.loads(row) for row in store.rows.values()]
    restarted = [row for row in records if row["events"] == []]
    assert len(restarted) == 1
    assert restarted[0]["sequence_start"] == fresh["sequence"]
    stale = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    )
    assert stale.status_code == 409
    assert other.get("/api/session").json() == other_state
    replacement = TestClient(hosted(store))
    replacement.cookies.update(client.cookies)
    assert replacement.get("/api/session").json() == fresh
    history = replacement.get("/api/transcript").json()
    assert history["current_game"]["public_history"] == []
    assert history["base_seed"] is None
    assert not any(history["tally"].values())
    state = begin(replacement)
    assert (
        replacement.post(
            "/api/session/action", json={"sequence": state["sequence"], "second": 60}
        ).status_code
        == 200
    )


def test_initial_reads_only_issue_one_cookie_and_reject_client_seeds() -> None:
    client = TestClient(hosted(MemoryStore()))
    for route in ("rules", "transcript"):
        assert "set-cookie" not in client.get(f"/api/{route}").headers
    response = client.get("/api/session")
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    response = client.post("/api/session", json={"sequence": 0, "seed": 12})
    assert response.status_code == 422
    response = client.post(
        "/api/session/begin",
        json={"sequence": 0},
        headers={"origin": "https://other.example"},
    )
    assert response.status_code == 403


def test_expired_sessions_require_reload_and_failed_storage_reveals_nothing() -> None:
    store = MemoryStore()
    client = TestClient(hosted(store))
    begin(client)
    replacement = TestClient(hosted(store, version="new-code"))
    replacement.cookies.update(client.cookies)
    assert (
        replacement.post(
            "/api/session/action", json={"sequence": 1, "second": 1}
        ).status_code
        == 409
    )
    assert replacement.get("/api/session").json()["phase"] == "rules"
    assert replacement.cookies[COOKIE] != client.cookies[COOKIE]

    async def fail(key: str, old: str | None, new: str) -> bool:
        del key, old, new
        msg = "storage failed"
        raise RuntimeError(msg)

    store.compare_set = fail
    response = replacement.post("/api/session/begin", json={"sequence": 0})
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}


def win(client: TestClient) -> dict:
    """Check and drop on 60 against a Hal who plays 37: Hal alone takes doses."""
    state = begin(client)
    while state["phase"] != "game_over":
        route, body = "ack", {"sequence": state["sequence"]}
        if state["phase"] == "awaiting_action":
            route, body = "action", {**body, "second": 60}
        response = client.post(f"/api/session/{route}", json=body)
        assert response.status_code == 200
        state = response.json()
    assert state["winner_is_human"] is True
    return state


def test_each_resolved_half_round_reaches_the_ledger_and_a_win_scores_life_left() -> (
    None
):
    ledger = MemoryLedger()
    client = TestClient(hosted(MemoryStore(), ledger=ledger))
    final = win(client)
    (row,) = ledger.games.values()
    human = next(player for player in final["players"] if player["is_human"])
    assert row["status"] == "finished"
    assert row["human_won"] is True
    assert row["score"] == 300 - human["ttd_seconds"] - human["cylinder_seconds"]
    assert row["score"] == life_left(final)
    assert row["half_rounds"] == final["half_rounds"] == len(row["public_history"])
    # One write for each half-round, and one repeat at the closing acknowledgement.
    assert ledger.writes == final["half_rounds"] + 1
    assert row["game_seed"].isdigit()
    assert row["policy_seed"].isdigit()
    assert len(row["player_id"]) == 64
    assert row["player_id"] != client.cookies[PLAYER_COOKIE]

    nameless = client.get("/api/leaderboard").json()
    assert nameless["entries"] == []
    assert nameless["your_score"] == row["score"]
    assert client.post("/api/leaderboard/name", json={"name": "Hal"}).status_code == 422
    assert (
        client.post("/api/leaderboard/name", json={"name": "x" * 17}).status_code == 422
    )
    refused = client.post(
        "/api/leaderboard/name",
        json={"name": "Baku"},
        headers={"origin": "https://other.example"},
    )
    assert refused.status_code == 403
    board = client.post("/api/leaderboard/name", json={"name": " Baku "}).json()
    assert board["your_rank"] == 1
    assert board["your_name"] == "Baku"
    assert board["entries"] == [
        {
            "rank": 1,
            "name": "Baku",
            "score": row["score"],
            "half_rounds": final["half_rounds"],
            "is_you": True,
        }
    ]
    stranger = TestClient(hosted(MemoryStore(), ledger=ledger))
    assert stranger.get("/api/leaderboard").json()["entries"][0]["is_you"] is False


def test_a_restart_abandons_the_open_game_and_a_lost_race_writes_nothing() -> None:
    ledger = MemoryLedger()
    client = TestClient(hosted(MemoryStore(), ledger=ledger))
    state = begin(client)
    state = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    ).json()
    assert [row["status"] for row in ledger.games.values()] == ["active"]
    client.post("/api/session/restart", json={"sequence": state["sequence"]})
    assert [row["status"] for row in ledger.games.values()] == ["abandoned"]

    class SlowStore(MemoryStore):
        """Yield on each read, so both requests replay the same record."""

        async def get(self, key: str) -> str | None:
            await asyncio.sleep(0)
            return await super().get(key)

    async def race() -> None:
        raced = MemoryLedger()
        transport = httpx.ASGITransport(app=hosted(SlowStore(), ledger=raced))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as player:
            state = (await player.get("/api/session")).json()
            state = (
                await player.post(
                    "/api/session/begin", json={"sequence": state["sequence"]}
                )
            ).json()
            responses = await asyncio.gather(
                *[
                    player.post(
                        "/api/session/action",
                        json={"sequence": state["sequence"], "second": second},
                    )
                    for second in (1, 60)
                ]
            )
        # Both requests passed the inner referee; the compare-and-set chose one.
        loser = next(r for r in responses if r.status_code == 409)
        assert loser.json() == {"detail": "Stale sequence. Reload the current turn."}
        assert raced.writes == 1

    asyncio.run(race())


def test_failed_ledger_cannot_hide_reveal_and_absence_means_no_board() -> None:
    ledger = MemoryLedger()

    async def fail(row: dict) -> None:
        msg = "ledger failed"
        raise RuntimeError(msg)

    ledger.record_game = fail
    client = TestClient(hosted(MemoryStore(), ledger=ledger))
    state = begin(client)
    response = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    )
    assert response.status_code == 200
    assert response.json()["last_outcome"] is not None

    bare = TestClient(hosted(MemoryStore()))
    assert bare.get("/api/leaderboard").status_code == 404
    assert bare.post("/api/leaderboard/name", json={"name": "Baku"}).status_code == 404


def test_a_game_the_human_did_not_win_has_no_score() -> None:
    snapshot = {
        "winner_is_human": False,
        "ttd_max": 300.0,
        "players": [{"is_human": True, "ttd_seconds": 0.0, "cylinder_seconds": 0.0}],
    }
    assert life_left(snapshot) is None
    assert life_left({**snapshot, "winner_is_human": None}) is None
    unresolved = {"last_outcome": None}
    assert (
        game_row(
            player_id="p",
            series_id="s",
            policy_seed=1,
            version="v",
            snapshot=unresolved,
            transcript={"current_game": {"public_history": []}},
        )
        is None
    )


def final_score(ledger: MemoryLedger) -> float:
    (score,) = [
        row["score"] for row in ledger.games.values() if row["status"] == "finished"
    ]
    return score


def test_a_later_abandoned_game_keeps_the_standing_and_a_name_needs_a_game() -> None:
    ledger = MemoryLedger()
    client = TestClient(hosted(MemoryStore(), ledger=ledger))
    client.get("/api/session")
    early = client.post("/api/leaderboard/name", json={"name": "Baku"})
    assert early.status_code == 409
    assert ledger.names == {}
    final = win(client)
    assert (
        client.post("/api/leaderboard/name", json={"name": "Baku"}).json()["your_rank"]
        == 1
    )
    state = client.post("/api/session", json={"sequence": final["sequence"]}).json()
    state = client.post(
        "/api/session/begin", json={"sequence": state["sequence"]}
    ).json()
    state = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    ).json()
    # The next game opened its own record and row; an open game risks nothing yet.
    assert len(ledger.games) == 2
    assert len({series for series, _ in ledger.games}) == 2
    assert client.get("/api/leaderboard").json()["your_rank"] == 1
    client.post("/api/session/restart", json={"sequence": state["sequence"]})
    board = client.get("/api/leaderboard").json()
    abandoned = [row for row in ledger.games.values() if row["status"] == "abandoned"]
    assert len(abandoned) == 1
    assert board["your_rank"] == 1
    assert board["your_score"] == final_score(ledger)


def test_the_closing_acknowledgement_repeats_a_failed_final_write() -> None:
    ledger = MemoryLedger()
    record = ledger.record_game
    failed = []

    async def fail_the_first_closing_write(row: dict) -> None:
        if row["status"] == "finished" and not failed:
            failed.append(row)
            msg = "ledger failed"
            raise RuntimeError(msg)
        await record(row)

    ledger.record_game = fail_the_first_closing_write
    client = TestClient(hosted(MemoryStore(), ledger=ledger))
    final = win(client)
    (row,) = ledger.games.values()
    assert failed
    assert row["status"] == "finished"
    assert row["half_rounds"] == final["half_rounds"]


def test_a_row_that_cannot_be_built_does_not_hide_a_committed_reveal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(**_: object) -> None:
        msg = "current_game"
        raise KeyError(msg)

    monkeypatch.setattr("stl.server.hosted.game_row", broken)
    client = TestClient(hosted(MemoryStore(), ledger=MemoryLedger()))
    state = begin(client)
    response = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    )
    assert response.status_code == 200
    assert response.json()["last_outcome"] is not None


def test_a_leaderboard_name_shows_as_the_text_it_holds() -> None:
    ledger = MemoryLedger()
    client = TestClient(hosted(MemoryStore(), ledger=ledger))
    win(client)
    for name in (
        "\u202eGNIHTYNA",  # right-to-left override
        "\u200b" * 16,  # zero-width spaces
        "A" + "\u0301" * 15,  # stacked combining marks
        "---",  # no letter or digit
        "line\u2028break",
        "\u3164" * 16,  # Hangul fillers are letters with no glyph
        "\uff28\uff21\uff2c",  # fullwidth HAL
    ):
        response = client.post("/api/leaderboard/name", json={"name": name})
        assert response.status_code == 422, name
    assert ledger.names == {}
    accepted = client.post("/api/leaderboard/name", json={"name": "Jose\u0301 \u7345"})
    assert accepted.json()["your_name"] == "Jos\u00e9 \u7345"


def test_a_capped_session_is_recorded_as_stopped() -> None:
    player = {"cylinder_seconds": 0.0, "ttd_seconds": 0.0, "deaths": 0}
    row = game_row(
        player_id="p",
        series_id="s",
        policy_seed=1,
        version="v",
        snapshot={
            "last_outcome": {"game_over": False, "session_ending": True},
            "players": [{**player, "is_human": True}, {**player, "is_human": False}],
            "winner_is_human": None,
            "half_rounds": 1,
            "pure_dth": False,
        },
        transcript={
            "hal_agent": "dth",
            "current_game": {
                "game_index": 0,
                "seed": 7,
                "start_clock": 720,
                "public_history": [{}],
            },
        },
    )
    assert row is not None
    assert row["status"] == "stopped"
    assert row["human_won"] is None
    assert row["score"] is None


def test_the_name_filter_refuses_slurs_and_passes_ordinary_words() -> None:
    # The terms stay in rot13 here for the reason they do in the filter.
    def plain(term: str) -> str:
        return codecs.decode(term, "rot13")

    slur = plain("avttre")
    for name in (
        slur,
        slur.upper(),
        " ".join(slur),
        "Baku " + " ".join(slur),
        "n1" + slur[2:],
        "n\u0456" + slur[2:],  # Cyrillic i
        slur.replace("e", "eee"),
        "xX_" + plain("avttn") + "_Xx",
        plain("snttbg") + "123",
        plain("s4tt0g"),
        plain("fcvpf"),
        plain("pbbba"),
        "the " + plain("xvxr"),
        plain("uvgyre") + "fan",
        "KKK",
        "K.K.K.",
        plain("anmv") + " 88",
    ):
        assert is_offensive(name), codecs.encode(name, "rot13")
    for name in (
        "Baku",
        "Madarame",
        "Raccoon",
        "Spice",
        "Niger",
        "Nigeria",
        "Nigeriens",
        "Knight",
        "Grape",
        "Pakistan",
        "Dykstra",
        "Scunthorpe",
        "Cocoon",
        "Snigger",
        "Sniggering",
        "Jos\u00e9 \u7345",
        "Damn",
        # Letters that would spell a term only across a word boundary.
        "Dana Zimmer",
        "Anna Zimmer",
        "Dan Aziz",
        "Ben Azir",
        # Names that hold a whole-word term inside them, or collapse onto one.
        "Benazir",
        "Ignazio",
        "Nazir",
        "Malcolm K",
        "K Smith",
        "Con Murphy",
        "Gok Wan",
        "Heb",
        "J K",
        "Al Bo",
    ):
        assert not is_offensive(name), name

    ledger = MemoryLedger()
    client = TestClient(hosted(MemoryStore(), ledger=ledger))
    win(client)
    refused = client.post("/api/leaderboard/name", json={"name": slur})
    assert refused.status_code == 422
    assert ledger.names == {}


def test_a_name_post_succeeds_only_on_a_known_answer() -> None:
    for answer, status in (("ok", 200), ("too_soon", 429), ("None", 503), ("new", 503)):
        ledger = MemoryLedger()

        async def answered(player_id: str, name: str, answer: str = answer) -> str:
            return answer

        ledger.set_player_name = answered
        client = TestClient(hosted(MemoryStore(), ledger=ledger))
        client.get("/api/session")
        response = client.post("/api/leaderboard/name", json={"name": "Baku"})
        assert response.status_code == status, answer


def test_the_board_shows_ten_entries_even_if_the_ledger_returns_more() -> None:
    ledger = MemoryLedger()

    async def twelve(player_id: str) -> dict:
        return {
            "entries": [
                {
                    "rank": rank,
                    "name": f"P{rank}",
                    "score": 300 - rank,
                    "half_rounds": 4,
                    "is_you": False,
                }
                for rank in range(1, 13)
            ],
            "your_rank": 12,
            "your_name": "P12",
            "your_score": 288,
        }

    ledger.leaderboard = twelve
    board = (
        TestClient(hosted(MemoryStore(), ledger=ledger)).get("/api/leaderboard").json()
    )
    assert [entry["rank"] for entry in board["entries"]] == list(range(1, 11))
    assert board["your_rank"] == 12


def test_the_next_game_opens_a_fresh_record_so_replay_follows_one_game() -> None:
    store = MemoryStore()
    client = TestClient(hosted(store))
    final = win(client)
    (before,) = [json.loads(row) for row in store.rows.values()]
    # Hal can die at the first revival roll, so a won game can hold five commands.
    assert len(before["events"]) >= 5
    fresh = client.post("/api/session", json={"sequence": final["sequence"]}).json()
    (after,) = [json.loads(row) for row in store.rows.values()]
    assert after["events"] == []
    assert after["series_id"] != before["series_id"]
    assert after["game_seed"] != before["game_seed"]
    assert after["sequence_start"] == fresh["sequence"] == final["sequence"] + 1
    assert fresh["phase"] == "rules"
    assert fresh["half_rounds"] == 0
    # A request from the finished game stays stale, and a new worker recovers the
    # new one.
    stale = client.post("/api/session", json={"sequence": final["sequence"]})
    assert stale.status_code == 409
    replacement = TestClient(hosted(store))
    replacement.cookies.update(client.cookies)
    state = begin(replacement)
    assert (
        replacement.post(
            "/api/session/action", json={"sequence": state["sequence"], "second": 60}
        ).status_code
        == 200
    )


class SeededHal:
    """A Hal whose seconds depend on its seed and on how often it was asked."""

    provider = object()

    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed)

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        return self.random.randint(1, 60)


def counted(store: MemoryStore, built: list[int]) -> FastAPI:
    def factory(game_seed: int, policy_seed: int, sequence_start: int = 0) -> FastAPI:
        built.append(game_seed)
        return create_app(
            hal_factory=lambda: SeededHal(policy_seed),
            config=SessionConfig(seed=game_seed),
            webclient_dist=None,
            sequence_start=sequence_start,
        )

    return create_hosted_app(store, factory, version="test", secure_cookie=False)


def play(clients: list[TestClient], moves: random.Random) -> dict:
    """Play to the end, passing each request to the next client in turn."""
    state = begin(clients[0])
    for turn in range(2000):
        if state["phase"] == "game_over":
            return state
        client = clients[turn % len(clients)]
        # Reads between moves must leave the held game as a replay would find it.
        assert client.get("/api/session").json() == state
        client.get("/api/transcript")
        route, body = "ack", {"sequence": state["sequence"]}
        if state["phase"] == "awaiting_action":
            route, body = "action", {**body, "second": moves.randint(1, 60)}
        response = client.post(f"/api/session/{route}", json=body)
        assert response.status_code == 200
        state = response.json()
    msg = "the game did not end"
    raise AssertionError(msg)


def test_one_process_builds_a_game_once_and_a_new_process_replays_the_same_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The hosted app draws each session's game and policy seeds at random, and a
    # short game records fewer than five events. Seeded draws fix the game; with
    # seed 7 it records 19 events. The seeds still pass through the store.
    monkeypatch.setattr(
        "stl.server.hosted.secrets.randbits",
        random.Random(7).getrandbits,
    )
    store, built = MemoryStore(), []
    client = TestClient(counted(store, built))
    final = play([client], random.Random(7))
    # The session read built the one game; every later request played on it.
    assert len(built) == 1
    assert len(json.loads(next(iter(store.rows.values())))["events"]) >= 5
    replacement = TestClient(counted(store, []))
    replacement.cookies.update(client.cookies)
    assert replacement.get("/api/session").json() == final


def test_stale_worker_catches_up_before_answering() -> None:
    store, built = MemoryStore(), ([], [])
    first, second = (TestClient(counted(store, games)) for games in built)
    first.get("/api/session")
    second.cookies.update(first.cookies)
    # Requests alternate between two processes, so each held game is stale
    # at every request. A stale answer would break the sequence or the reads.
    final = play([first, second], random.Random(11))
    # Each process built the game once and then played only the commands it lacked.
    assert [len(games) for games in built] == [1, 1]
    replacement = TestClient(counted(store, []))
    replacement.cookies.update(first.cookies)
    assert replacement.get("/api/session").json() == final


def test_a_lost_race_and_a_refused_command_drop_the_held_game() -> None:
    async def exercise() -> None:
        store, built = MemoryStore(), []
        transport = httpx.ASGITransport(app=counted(store, built))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            state = (await client.get("/api/session")).json()
            body = {"sequence": state["sequence"]}
            state = (await client.post("/api/session/begin", json=body)).json()
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/api/session/action",
                        json={"sequence": state["sequence"], "second": second},
                    )
                    for second in (1, 60)
                ]
            )
            assert sorted(r.status_code for r in responses) == [200, 409]
            winner = next(r for r in responses if r.status_code == 200).json()
            assert (await client.get("/api/session")).json() == winner
            # A refused command costs one rebuild and changes nothing.
            before = len(built)
            refused = await client.post(
                "/api/session/action", json={"sequence": 0, "second": 5}
            )
            assert refused.status_code != 200
            assert (await client.get("/api/session")).json() == winner
            assert len(built) == before + 1
            fresh = httpx.ASGITransport(app=counted(store, []))
            async with httpx.AsyncClient(
                transport=fresh, base_url="http://test", cookies=client.cookies
            ) as other:
                assert (await other.get("/api/session")).json() == winner

    asyncio.run(exercise())


def test_the_warm_up_read_builds_one_game_for_every_caller() -> None:
    built = []
    app = counted(MemoryStore(), built)
    answers = [TestClient(app).get("/api/rules") for _ in range(3)]
    assert [answer.status_code for answer in answers] == [200, 200, 200]
    assert answers[0].json() == answers[2].json()
    assert len(built) == 1
    assert all(COOKIE not in answer.cookies for answer in answers)


# Translated Hal in the hosted app.


def translated_hal() -> TranslatedHal:
    return TranslatedHal(_StageTable())


def translated_hosted(
    store: MemoryStore, *, start_clock: int = 720, max_half_rounds: int = 6
) -> FastAPI:
    def factory(game_seed: int, policy_seed: int, sequence_start: int) -> FastAPI:
        return create_app(
            hal_factory=lambda: PolicyDrivenAgent(translated_hal(), seed=policy_seed),
            config=SessionConfig(
                seed=game_seed, start_clock=start_clock, max_half_rounds=max_half_rounds
            ),
            series=SeriesConfig(conceal_hal_details=True),
            webclient_dist=None,
            sequence_start=sequence_start,
        )

    return create_hosted_app(
        store,
        factory,
        version="translated-test",
        secure_cookie=False,
        memory=OpponentMemory(),
        policy_label="translated-hal-v1",
    )


def move(client: TestClient, state: dict, action: int = 12) -> dict:
    response = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": action}
    )
    assert response.status_code == 200, response.text
    return response.json()


def observations(store: MemoryStore) -> int:
    record = json.loads(next(iter(store.rows.values())))
    p = translated_hal()
    p.restore_state(record["opponent_memory"])
    return p.opponent_model.total_observations


@pytest.mark.parametrize("boundary", ["restart", "next_game"])
def test_memory_survives_game_boundaries_and_worker_replacement(boundary: str) -> None:
    store = MemoryStore()
    client = TestClient(translated_hosted(store, max_half_rounds=1))
    revealed = move(client, begin(client))
    if boundary == "restart":
        path, state = "/api/session/restart", revealed
    else:
        state = client.post(
            "/api/session/ack", json={"sequence": revealed["sequence"]}
        ).json()
        assert state["phase"] == "game_over"
        path = "/api/session"
    response = client.post(path, json={"sequence": state["sequence"]})
    assert response.status_code == 200, response.text
    assert observations(store) == 1
    recovered = TestClient(translated_hosted(store, max_half_rounds=1))
    recovered.cookies.update(client.cookies)
    assert recovered.get("/api/session").json() == response.json()
    revealed = move(recovered, begin(recovered))
    restarted = recovered.post(
        "/api/session/restart", json={"sequence": revealed["sequence"]}
    )
    assert restarted.status_code == 200
    assert observations(store) == 2
    stranger = TestClient(translated_hosted(store))
    assert stranger.get("/api/session").json()["sequence"] == 0
    records = [json.loads(r) for r in store.rows.values()]
    assert sum("opponent_memory" in r for r in records) == 1
    transcript = recovered.get("/api/transcript").text
    assert "offsets" not in transcript
    assert "opponent_memory" not in transcript


def test_cold_replay_and_warm_worker_produce_the_same_next_reveal() -> None:
    store = MemoryStore()
    warm = TestClient(translated_hosted(store))
    state = move(warm, begin(warm))
    warm.post("/api/session/restart", json={"sequence": state["sequence"]})
    state = move(warm, begin(warm), 17)
    state = warm.post("/api/session/ack", json={"sequence": state["sequence"]}).json()
    fork_store = MemoryStore()
    fork_store.rows = dict(store.rows)
    cold = TestClient(translated_hosted(fork_store))
    cold.cookies.update(warm.cookies)
    assert cold.get("/api/session").json() == state
    assert move(cold, state, 21) == move(warm, state, 21)
    assert fork_store.rows == store.rows


def test_failed_commit_does_not_learn_or_reveal_a_speculative_move() -> None:
    class RejectStore(MemoryStore):
        reject = False

        async def compare_set(self, key: str, old: str | None, new: str) -> bool:
            return False if self.reject else await super().compare_set(key, old, new)

    store = RejectStore()
    client = TestClient(translated_hosted(store))
    state = begin(client)
    before = dict(store.rows)
    store.reject = True
    refused = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 8}
    )
    assert refused.status_code == 409
    assert "last_outcome" not in refused.text
    assert store.rows == before
    store.reject = False
    revealed = move(client, state, 8)
    client.post("/api/session/restart", json={"sequence": revealed["sequence"]})
    assert observations(store) == 1


def test_canonical_leap_61_resolves_and_does_not_enter_the_60_action_model() -> None:
    store = MemoryStore()
    client = TestClient(translated_hosted(store, start_clock=LS_WINDOW_START - 60))
    state = begin(client)
    first = move(client, state, 60)
    state = client.post("/api/session/ack", json={"sequence": first["sequence"]}).json()
    assert state["human_role"] == "dropper"
    assert state["legal_seconds"][-1] == 61
    revealed = move(client, state, 61)
    assert revealed["last_outcome"] is not None
    client.post("/api/session/restart", json={"sequence": revealed["sequence"]})
    assert observations(store) == 1


def test_corrupt_durable_memory_fails_closed() -> None:
    store = MemoryStore()
    client = TestClient(translated_hosted(store))
    state = move(client, begin(client))
    client.post("/api/session/restart", json={"sequence": state["sequence"]})
    key = next(iter(store.rows))
    record = json.loads(store.rows[key])
    record["opponent_memory"] = "invalid"
    store.rows[key] = json.dumps(record)
    fresh = TestClient(translated_hosted(store))
    fresh.cookies.update(client.cookies)
    response = fresh.get("/api/session")
    assert response.status_code == 503
    assert "last_outcome" not in response.text


def test_production_serves_translated_hal_and_checkpoints_its_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _StageTable()
    monkeypatch.setattr(
        production,
        "Table",
        SimpleNamespace(open=lambda *args, **kwargs: table),
    )
    store = MemoryStore()
    monkeypatch.setattr(production, "RedisSessionStore", lambda *args: store)
    monkeypatch.setenv("KV_REST_API_URL", "https://example.invalid")
    monkeypatch.setenv("KV_REST_API_TOKEN", "test")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    with TestClient(
        production.create_production_app(Path("unused"), "test"),
        base_url="https://testserver",
    ) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["policy"] == "translated-hal-v1"
        state = move(client, begin(client))
        response = client.post(
            "/api/session/restart", json={"sequence": state["sequence"]}
        )
        assert response.status_code == 200
        record = json.loads(next(iter(store.rows.values())))
        assert "opponent_memory" in record
