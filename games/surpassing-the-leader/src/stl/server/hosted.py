# Copyright 2026 The STL authors
"""The hosted game on Vercel: cookies, the Redis command log and the ledger.

``create_production_app`` is the Vercel entry, and this is the only module
that reads environment variables. Each player gets a secure cookie and one
Redis record per game: the game's private seeds, the sequence its session
starts at, and the commands accepted since. A worker replays those commands
through the local app of ``stl.server.app`` and commits each mutation with a
compare-and-set before it returns a reveal. Translated Hal's opponent evidence
rides in the same record as a checkpoint taken at each game boundary.

The game ledger is the permanent copy: one row per game in Supabase,
rewritten after each resolved half-round, so a closed tab still leaves every
move behind. Rows hold only what an outcome screen has already shown, plus the
server's private seeds. The seeds never reach a browser; they let an offline
job replay a game exactly. ``GET /api/leaderboard`` ranks each player's best
win by the winner's seconds of life left.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from stl.dth import Table
from stl.hal import TranslatedHal
from stl.server.app import (
    PlayerNameRequest,
    SeriesConfig,
    SessionConfig,
    create_app,
    top_standings,
)
from stl.server.session import PolicyDrivenAgent

# The deployment bundle imports this entry point by its generated module name.
__all__ = ["create_production_app"]

MAX_REQUEST_BYTES = 4096
HTTP_OK = 200
COOKIE = "stl_session"
TTL_SECONDS = 7 * 24 * 60 * 60
# The session cookie dies with each code version. The player cookie outlives
# deployments, so a standing on the leaderboard follows the same browser.
PLAYER_COOKIE = "stl_player"
PLAYER_TTL_SECONDS = 365 * 24 * 60 * 60
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_log = logging.getLogger(__name__)
# Latency diagnosis: one id and one boot time per process.
_INSTANCE = secrets.token_hex(4)
_BOOTED = time.monotonic()


def _process_age() -> str:
    """Seconds since the kernel started this process, where Linux reports it."""
    try:
        with Path("/proc/self/stat").open() as file:
            ticks = int(file.read().rsplit(")", 1)[1].split()[19])
        with Path("/proc/uptime").open() as file:
            uptime = float(file.read().split()[0])
        return f"{uptime - ticks / os.sysconf('SC_CLK_TCK'):.1f}"
    except (OSError, ValueError, IndexError):
        return "-"


_AGE_AT_IMPORT = _process_age()
# "yes" when this module loaded from shipped bytecode and not from source.
_BYTECODE = (
    "yes" if __spec__ and __spec__.cached and Path(__spec__.cached).exists() else "no"
)
_READ_PATHS = {"/api/rules", "/api/session", "/api/transcript"}
# Games one process keeps between requests. The store stays the authority.
HELD_GAMES = 256
_WRITE_PATHS = {
    "/api/session",
    "/api/session/begin",
    "/api/session/action",
    "/api/session/ack",
    "/api/session/restart",
}


class SessionStore(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def compare_set(self, key: str, old: str | None, new: str) -> bool: ...


class SessionMemory(Protocol):
    def dump(self, app: FastAPI) -> str: ...
    def restore(self, app: FastAPI, memory: str) -> None: ...


def _public(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: None if key in {"seed", "game_seed", "base_seed"} else _public(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_public(value) for value in payload]
    return payload


def _cross_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    return bool(origin) and origin.rstrip("/") != str(request.base_url).rstrip("/")


def _player_id(request: Request) -> str | None:
    token = request.cookies.get(PLAYER_COOKIE, "")
    if _TOKEN.fullmatch(token) is None:
        return None
    return hashlib.sha256(token.encode()).hexdigest()


async def _ledger_write(write: Callable[[], Awaitable[None]]) -> None:
    """A ledger failure must not hide a reveal the session store has committed.

    The caller passes a function, so building the row fails here as well.
    """
    try:
        await write()
    except Exception:
        _log.exception("game ledger write failed")


def create_hosted_app(
    store: SessionStore,
    factory: Callable[[int, int, int], FastAPI],
    *,
    version: str,
    secure_cookie: bool = True,
    ledger: GameLedger | None = None,
    memory: SessionMemory | None = None,
    policy_label: str = "certified-dth",
) -> FastAPI:
    """Replay accepted commands with private seeds; persist before revealing.

    A record is one game: its seeds, the sequence its session starts at, and
    the commands accepted since. A restart and a next game each replace the
    record with a fresh one and an empty command list, so replay cost follows
    the current game. An optional memory adapter checkpoints opponent evidence
    at each game boundary. A new worker restores that evidence before replay.
    The checkpoint and accepted commands share one compare-and-set record.

    A process keeps the game it last served beside the stored text that game
    matches. A request whose stored text still matches plays on that game and
    skips the rebuild. A held game that another process has moved past plays
    the commands it lacks. A held game from another record is rebuilt, so a
    process cannot answer from a stale game.
    """
    # key -> stored text, game, series, and the commands that game has played.
    held_games: OrderedDict[str, tuple[str, FastAPI, str, list]] = OrderedDict()
    unplayed: list[FastAPI] = []
    app = FastAPI(
        title="Surpassing The Leader", docs_url=None, redoc_url=None, openapi_url=None
    )

    served = 0

    @app.middleware("http")
    async def diagnose(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Latency diagnosis: an answer names its process, that process's age
        # and request count, the handler's time, and the held-game result.
        nonlocal served
        served += 1
        started = time.monotonic()
        response = await call_next(request)
        response.headers["x-stl-diagnosis"] = (
            f"instance={_INSTANCE} up={started - _BOOTED:.1f} n={served} "
            f"ms={(time.monotonic() - started) * 1000:.0f} "
            f"held={getattr(request.state, 'held', '-')} "
            f"age={_process_age()} age_at_import={_AGE_AT_IMPORT} "
            f"bytecode={_BYTECODE}"
        )
        return response

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "policy": policy_label, "version": version}

    async def board(player_id: str | None) -> JSONResponse:
        # An unknown player reads the same public standings with no row marked.
        try:
            payload = await cast("GameLedger", ledger).leaderboard(player_id or "")
            body = top_standings(payload).model_dump()
        except (httpx.HTTPError, RuntimeError, ValueError):
            return JSONResponse(
                {"detail": "The leaderboard is unavailable."},
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.get("/api/leaderboard")
    async def read_leaderboard(request: Request) -> JSONResponse:
        if ledger is None:
            return JSONResponse({"detail": "Unknown route"}, status_code=404)
        return await board(_player_id(request))

    @app.post("/api/leaderboard/name")
    async def post_name(request: Request) -> JSONResponse:
        if ledger is None:
            return JSONResponse({"detail": "Unknown route"}, status_code=404)
        if _cross_origin(request):
            return JSONResponse(
                {"detail": "Cross-origin action refused"}, status_code=403
            )
        data = await request.body()
        if len(data) > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "Request is too large"}, status_code=413)
        try:
            name = PlayerNameRequest.model_validate_json(data).name
        except ValidationError as error:
            # The player reads this line, so drop pydantic's "Value error, " label.
            reason = error.errors()[0]["msg"].removeprefix("Value error, ")
            return JSONResponse(
                {"detail": reason[:1].upper() + reason[1:] + "."}, status_code=422
            )
        player_id = _player_id(request)
        if player_id is None:
            return JSONResponse(
                {"detail": "Session expired. Reload the page."}, status_code=409
            )
        try:
            posted = await ledger.set_player_name(player_id, name)
        except (httpx.HTTPError, RuntimeError, ValueError):
            return JSONResponse(
                {"detail": "The leaderboard is unavailable."}, status_code=503
            )
        if posted == "no_game":
            return JSONResponse(
                {"detail": "Play a game before you post a name."}, status_code=409
            )
        if posted == "too_soon":
            return JSONResponse(
                {"detail": "Wait a minute before you change the name."},
                status_code=429,
            )
        if posted != "ok":
            # An answer this server does not know is no proof the name was set.
            return JSONResponse(
                {"detail": "The leaderboard is unavailable."}, status_code=503
            )
        return await board(player_id)

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    async def route(request: Request, path: str) -> JSONResponse:
        route_path = f"/api/{path}"
        command: dict = {}
        mutation = request.method == "POST"
        if route_path not in (_WRITE_PATHS if mutation else _READ_PATHS):
            return JSONResponse({"detail": "Unknown route"}, status_code=404)
        if mutation:
            if _cross_origin(request):
                return JSONResponse(
                    {"detail": "Cross-origin action refused"}, status_code=403
                )
            data = await request.body()
            if len(data) > MAX_REQUEST_BYTES:
                return JSONResponse({"detail": "Request is too large"}, status_code=413)
            try:
                command = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                return JSONResponse({"detail": "Invalid JSON"}, status_code=422)
            if not isinstance(command, dict):
                return JSONResponse({"detail": "Expected an object"}, status_code=422)
            if any(
                key in command for key in ("seed", "start_clock", "max_half_rounds")
            ):
                return JSONResponse(
                    {"detail": "Game settings are server-controlled"}, status_code=422
                )
        token = request.cookies.get(COOKIE, "")
        valid_token = _TOKEN.fullmatch(token) is not None
        player_id = _player_id(request)
        player_token = None
        key = (
            "stl:v1:" + hashlib.sha256(token.encode()).hexdigest()
            if valid_token
            else None
        )
        try:
            raw = await store.get(key) if key else None
            record = json.loads(raw) if raw else None
            new_cookie = False
            if record is not None and record["version"] != version:
                record, raw, key = None, None, None
            if record is None:
                if mutation:
                    return JSONResponse(
                        {"detail": "Session expired. Reload the page."}, status_code=409
                    )
                record = {
                    "version": version,
                    "series_id": secrets.token_hex(16),
                    "game_seed": secrets.randbits(128),
                    "policy_seed": secrets.randbits(128),
                    "sequence_start": 0,
                    "events": [],
                }
                # The client's initial reads run in parallel. Only this route issues a
                # cookie.
                if route_path == "/api/session":
                    token = secrets.token_urlsafe(32)
                    key = "stl:v1:" + hashlib.sha256(token.encode()).hexdigest()
                    raw = json.dumps(record, separators=(",", ":"))
                    if not await store.compare_set(key, None, raw):
                        message = "session creation collision"
                        raise RuntimeError(message)
                    new_cookie = True
            # The page makes one session read, so one load mints one player.
            # Two tabs opened together can mint two; the browser keeps the last.
            if player_id is None and not mutation and route_path == "/api/session":
                player_token = secrets.token_urlsafe(32)
            series_id = record["series_id"]
            history = None
            # The held game leaves the table while this request uses it, so a
            # concurrent request for the same player rebuilds its own copy.
            held = held_games.pop(key, None) if key else None
            played = len(held[3]) if held else 0
            if held is not None and held[0] == raw:
                request.state.held = "hit"
                inner, pending = held[1], []
            elif (
                held is not None
                and held[2] == series_id
                and record["events"][:played] == held[3]
            ):
                request.state.held = "behind"
                inner, pending = held[1], record["events"][played:]
            elif raw is None and route_path == "/api/rules":
                # The client's warm-up read carries no cookie. The rules of an
                # unplayed game are the same for every seed, so one game answers.
                request.state.held = "rules"
                if not unplayed:
                    unplayed.append(factory(0, 0, 0))
                inner, pending = unplayed[0], []
            else:
                request.state.held = "miss"
                inner = factory(
                    record["game_seed"],
                    record["policy_seed"],
                    record.get("sequence_start", 0),
                )
                if memory is not None and record.get("opponent_memory") is not None:
                    memory.restore(inner, record["opponent_memory"])
                pending = record["events"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=inner), base_url="http://arena"
            ) as client:
                for event in pending:
                    replay = await client.post(event["path"], json=event["body"])
                    if replay.status_code != HTTP_OK:
                        message = "stored session could not be replayed"
                        raise RuntimeError(message)
                response = await client.request(
                    request.method, route_path, json=command if mutation else None
                )
                snapshot = response.json() if response.status_code == HTTP_OK else None
                # A resolved half-round changes the ledger row. The closing
                # acknowledgement repeats the last write in case it failed.
                if (
                    ledger is not None
                    and player_id is not None
                    and snapshot is not None
                    and (
                        route_path == "/api/session/action"
                        or (
                            route_path == "/api/session/ack"
                            and snapshot["phase"] == "game_over"
                        )
                    )
                ):
                    transcript = await client.get("/api/transcript")
                    if transcript.status_code == HTTP_OK:
                        history = transcript.json()
            restarted = snapshot is not None and route_path == "/api/session/restart"
            next_game = (
                snapshot is not None and mutation and route_path == "/api/session"
            )
            if restarted or next_game:
                checkpoint = memory.dump(inner) if memory is not None else None
                # The old record validated this request. Open the new game as
                # its own record: new seeds, so a reload cannot rehearse Hal's
                # samples, and the next sequence number, so a request from
                # before the change stays stale.
                record = {
                    "version": version,
                    "series_id": secrets.token_hex(16),
                    "game_seed": secrets.randbits(128),
                    "policy_seed": secrets.randbits(128),
                    "sequence_start": int(response.json()["sequence"]),
                    "events": [],
                }
                if checkpoint is not None:
                    record["opponent_memory"] = checkpoint
                fresh = factory(
                    record["game_seed"], record["policy_seed"], record["sequence_start"]
                )
                if checkpoint is not None and memory is not None:
                    memory.restore(fresh, checkpoint)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=fresh), base_url="http://arena"
                ) as client:
                    response = await client.get("/api/session")
                if response.status_code != HTTP_OK:
                    message = "fresh game could not be opened"
                    raise RuntimeError(message)
                inner = fresh
            elif mutation and response.status_code == HTTP_OK:
                record["events"].append({"path": route_path, "body": command})
            if mutation and response.status_code == HTTP_OK:
                updated = json.dumps(record, separators=(",", ":"))
                if not await store.compare_set(cast("str", key), raw, updated):
                    # The losing request must not reveal its speculative Hal action.
                    # Its game played a move the store refused, so it is dropped.
                    return JSONResponse(
                        {"detail": "Stale sequence. Reload the current turn."},
                        status_code=409,
                        headers={"Cache-Control": "no-store"},
                    )
                raw = updated
            # A refused command leaves the game unheld; the next request rebuilds.
            if key and raw and (not mutation or response.status_code == HTTP_OK):
                held_games[key] = (
                    raw,
                    inner,
                    record["series_id"],
                    list(record["events"]),
                )
                while len(held_games) > HELD_GAMES:
                    held_games.popitem(last=False)
            # The ledger follows the committed record, so a losing request
            # writes nothing. A restart closes the old series' open game.
            if ledger is not None and restarted:
                await _ledger_write(
                    lambda: cast("GameLedger", ledger).abandon_series(series_id)
                )
            elif history is not None:

                async def record_row() -> None:
                    row = game_row(
                        player_id=cast("str", player_id),
                        series_id=series_id,
                        policy_seed=record["policy_seed"],
                        version=version,
                        snapshot=cast("dict", snapshot),
                        transcript=history,
                    )
                    if row is not None:
                        await cast("GameLedger", ledger).record_game(row)

                await _ledger_write(record_row)
            result = JSONResponse(
                _public(response.json()),
                status_code=response.status_code,
                headers={"Cache-Control": "no-store"},
            )
            if new_cookie:
                result.set_cookie(
                    COOKIE,
                    token,
                    max_age=TTL_SECONDS,
                    secure=secure_cookie,
                    httponly=True,
                    samesite="strict",
                    path="/",
                )
            if player_token is not None:
                result.set_cookie(
                    PLAYER_COOKIE,
                    player_token,
                    max_age=PLAYER_TTL_SECONDS,
                    secure=secure_cookie,
                    httponly=True,
                    samesite="strict",
                    path="/",
                )
            return result
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError):
            return JSONResponse(
                {"detail": "Game storage is unavailable. Please retry."},
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )

    return app


class RedisSessionStore:
    """Use an atomic Redis compare-and-set to serialize competing mutations."""

    def __init__(self, url: str, token: str) -> None:
        if not url.startswith("https://") or not token:
            message = "session storage requires an HTTPS URL and token"
            raise ValueError(message)
        self.url, self.token = url, token
        self._client: httpx.AsyncClient | None = None

    async def command(self, command: list) -> object:
        # One client per worker keeps the TLS connection to the store open, so
        # each command costs a round trip and not a fresh handshake.
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        try:
            response = await self._client.post(
                self.url,
                json=command,
                headers={"Authorization": f"Bearer {self.token}"},
            )
        except (httpx.TransportError, RuntimeError):
            # A suspended worker can wake with a dead connection or a closed
            # event loop; open a fresh client once and retry.
            self._client = httpx.AsyncClient(timeout=15)
            response = await self._client.post(
                self.url,
                json=command,
                headers={"Authorization": f"Bearer {self.token}"},
            )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            message = "session storage command failed"
            raise RuntimeError(message)
        return body["result"]

    async def get(self, key: str) -> str | None:
        return cast("str | None", await self.command(["GET", key]))

    async def compare_set(self, key: str, old: str | None, new: str) -> bool:
        script = """
local current = redis.call('GET', KEYS[1])
if (ARGV[1] == 'missing' and current) or
   (ARGV[1] == 'present' and current ~= ARGV[2]) then return 0 end
redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[4])
return 1
"""
        result = await self.command(
            [
                "EVAL",
                script,
                1,
                key,
                "missing" if old is None else "present",
                old or "",
                new,
                TTL_SECONDS,
            ]
        )
        return result == 1


class GameLedger(Protocol):
    async def record_game(self, row: dict[str, object]) -> None: ...
    async def abandon_series(self, series_id: str) -> None: ...
    async def set_player_name(self, player_id: str, name: str) -> str:
        """Return ``ok``, ``no_game`` for a player with no recorded game, or
        ``too_soon`` for a rename inside the hold period."""
        ...

    async def leaderboard(self, player_id: str) -> dict[str, object]: ...


class SupabaseLedger:
    """Call the ledger's Postgres functions through Supabase's REST gateway."""

    def __init__(self, url: str, key: str) -> None:
        if not url.startswith("https://") or not key:
            message = "the game ledger requires an HTTPS URL and a secret key"
            raise ValueError(message)
        self.url, self.key = url.rstrip("/"), key
        self._client: httpx.AsyncClient | None = None

    async def _rpc(self, function: str, arguments: dict[str, object]) -> object:
        # One client per worker keeps the TLS connection open, as the session
        # store does; a suspended worker gets one fresh client and one retry.
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5)
        endpoint = f"{self.url}/rest/v1/rpc/{function}"
        headers = {"apikey": self.key}
        try:
            response = await self._client.post(
                endpoint, json=arguments, headers=headers
            )
        except (httpx.TransportError, RuntimeError):
            self._client = httpx.AsyncClient(timeout=5)
            response = await self._client.post(
                endpoint, json=arguments, headers=headers
            )
        response.raise_for_status()
        return response.json() if response.content else None

    async def record_game(self, row: dict[str, object]) -> None:
        await self._rpc("record_game", {"p": row})

    async def abandon_series(self, series_id: str) -> None:
        await self._rpc("abandon_series", {"p_series": series_id})

    async def set_player_name(self, player_id: str, name: str) -> str:
        return str(
            await self._rpc("set_player_name", {"p_player": player_id, "p_name": name})
        )

    async def leaderboard(self, player_id: str) -> dict[str, object]:
        return cast(
            "dict[str, object]", await self._rpc("leaderboard", {"p_player": player_id})
        )


def life_left(snapshot: dict) -> float | None:
    """The leaderboard score: a winning human's distance from permanent death.

    Capacity less accrued death time less the squandered time still waiting in
    the cylinder. A game the human did not win has no score.
    """

    if not snapshot["winner_is_human"]:
        return None
    human = next(player for player in snapshot["players"] if player["is_human"])
    return max(
        0.0,
        snapshot["ttd_max"] - human["ttd_seconds"] - human["cylinder_seconds"],
    )


def game_row(
    *,
    player_id: str,
    series_id: str,
    policy_seed: int,
    version: str,
    snapshot: dict,
    transcript: dict,
) -> dict[str, object] | None:
    """One ledger row from a post-reveal snapshot and the inner transcript."""

    outcome = snapshot["last_outcome"]
    game = transcript["current_game"]
    if outcome is None or not game["public_history"]:
        return None
    if outcome["game_over"]:
        status = "finished"
    elif outcome["session_ending"]:
        status = "stopped"
    else:
        status = "active"
    finished = status == "finished"
    human = next(player for player in snapshot["players"] if player["is_human"])
    hal = next(player for player in snapshot["players"] if not player["is_human"])
    return {
        "player_id": player_id,
        "series_id": series_id,
        "game_index": game["game_index"],
        "status": status,
        "human_won": bool(snapshot["winner_is_human"]) if finished else None,
        "score": life_left(snapshot) if finished else None,
        "half_rounds": snapshot["half_rounds"],
        "human_cylinder_seconds": human["cylinder_seconds"],
        "human_ttd_seconds": human["ttd_seconds"],
        "human_deaths": human["deaths"],
        "hal_cylinder_seconds": hal["cylinder_seconds"],
        "hal_ttd_seconds": hal["ttd_seconds"],
        "hal_deaths": hal["deaths"],
        "start_clock": game["start_clock"],
        # The ledger keeps this column from the retired pure-DTH variant.
        "pure_dth": False,
        "hal_agent": transcript["hal_agent"],
        "code_version": version,
        # The seeds are 128-bit; text keeps them exact in JSON and Postgres.
        "game_seed": str(game["seed"]),
        "policy_seed": str(policy_seed),
        "public_history": game["public_history"],
    }


class OpponentMemory:
    """Checkpoint the app's translated Hal evidence, which hal.py serializes."""

    def dump(self, app: FastAPI) -> str:
        return app.state.hal_provider.dump_state()

    def restore(self, app: FastAPI, memory: str) -> None:
        app.state.hal_provider.restore_state(memory)


def create_production_app(artifact: Path, version: str) -> FastAPI:
    """The Vercel entry: translated Hal over the bundled table.

    ``stl.server.deploy`` rechecks 1,200 classes of the table and assigns
    ``version`` before it writes the bundle. A cold start checks only the
    shape and the six anchors: rehashing 2.3 GB cost about ten seconds before
    the first answer.
    """
    # httpx logs one INFO line for each call, the in-process game calls
    # included. A move wrote up to forty lines to Vercel's log pipe.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    table = Table.open(artifact)

    def factory(game_seed: int, policy_seed: int, sequence_start: int = 0) -> FastAPI:
        return create_app(
            hal_factory=lambda: PolicyDrivenAgent(
                TranslatedHal(table), seed=policy_seed
            ),
            config=SessionConfig(seed=game_seed),
            series=SeriesConfig(hal_agent="translated-v1", conceal_hal_details=True),
            webclient_dist=None,
            sequence_start=sequence_start,
        )

    url = os.environ.get("KV_REST_API_URL") or os.environ.get(
        "UPSTASH_REDIS_REST_URL", ""
    )
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get(
        "UPSTASH_REDIS_REST_TOKEN", ""
    )
    # The ledger is optional: without its two settings the game plays as
    # before, records nothing, and serves no leaderboard.
    ledger_url = os.environ.get("SUPABASE_URL", "")
    ledger_key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get(
        "SUPABASE_SERVICE_ROLE_KEY", ""
    )
    ledger = None
    if ledger_url and ledger_key:
        try:
            ledger = SupabaseLedger(ledger_url, ledger_key)
        except ValueError:
            # A bad ledger setting costs the leaderboard and never the game.
            logging.getLogger(__name__).exception("game ledger is misconfigured")
    return create_hosted_app(
        RedisSessionStore(url, token),
        factory,
        version=version,
        ledger=ledger,
        memory=OpponentMemory(),
        policy_label="translated-hal-v1",
    )
