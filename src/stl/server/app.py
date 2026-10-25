"""The browser game's HTTP routes, wire schema and leaderboard name filter.

The server holds exactly one live session, because it is a local single-player
surface, but that session sits inside one repeated-opponent series: Hal is built
once and keeps its opponent model across games, each finished game is appended
to a public transcript in the ``arena-public-play-session-v1`` shape, and each
new game receives a fresh canonical referee whose seed is the base seed plus the
game index. The STL engine remains the only referee; this module sequences
requests and serializes state.

Every mutating endpoint carries the sequence number the client last saw, so a
double-submitted action or a stale second tab is rejected rather than silently
replayed.

``snapshot_from_session`` is the single seat-scoped serializer. Everything the
browser is allowed to know passes through it and nothing else. The rule: an
unrevealed action never leaves the process. Hal's second does not exist during
``Phase.AWAITING_ACTION`` (see ``stl.server.session.PlaySession.submit``), and
``last_outcome`` is populated only once the half-round has resolved.

``is_offensive`` is a basic slur filter for names that every player reads on
the leaderboard. It refuses slurs and a few hate terms; ordinary profanity
passes. It is a word list, so it stops casual abuse and no determined writer.
The terms are stored in rot13 so that no slur shows in the source, a diff, or a
code search.
"""

from __future__ import annotations

import codecs
import random
import re
import threading
import unicodedata
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from stl.game import (
    CYLINDER_MAX,
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    TOTAL_TTD_MAX,
    TURN_DURATION_LEAP,
    Game,
    HalfRoundResult,
    Player,
    Referee,
)
from stl.hal import reset_provider_game
from stl.server.session import (
    CANONICAL_HAL_NAME,
    CANONICAL_HUMAN_NAME,
    Phase,
    PlaySession,
    SessionPhaseError,
    validate_human_display_name,
)

HalFactory = Callable[[], object]

# The player-facing rules that the opening screen shows.
RULES = (
    "GOAL — Be the last player alive.",
    "",
    "EACH HALF-ROUND",
    "Dropper secretly chooses a handkerchief-drop second from 1..60.",
    "Checker independently chooses a check second from 1..60 (1 is immediate).",
    "Second 0 and passing are illegal; check succeeds when check >= drop.",
    "ST means Squandered Time.",
    "A success adds ST = check - drop + 1 to the Checker's vial.",
    "",
    "FAILED CHECK, DEATH, AND REVIVAL",
    "A failed check injects q = current vial ST + 60 seconds.",
    "q >= 300 or TTD + q > 300 is fatal; equality is eligible when q < 300.",
    "TTD means Total Time Dead; every revived dose is added to it.",
    "Revival chance falls as vial ST and prior TTD rise.",
    "A revival clears vial ST.",
    "Roles swap after every half-round that both players survive.",
)

# Refused wherever the letters occur inside a word.
_ANYWHERE = (
    "avttre", "avttn", "avtyrg", "snttbg", "jrgonpx", "gbjryurnq",
    "enturnq", "puvatpubat", "wvtnobb", "cbepuzbaxrl", "mvccreurnq",
    "furznyr", "ergneq", "uvgyre",
)
# Refused as a whole word only, because ordinary words and names contain them
# ("raccoon", "spice", "Pakistan", "grape", "Benazir").
_WHOLE_WORD = (
    "fcvp", "pbba", "puvax", "tbbx", "cnxv", "snt", "xvxr", "xlxr", "urro",
    "genaal", "genaavr", "ornare", "xxx", "anmv", "arbanmv", "encr",
    "encvfg", "encvat",
)
# Ordinary words that hold a refused term.
_ALLOWED = re.compile(r"snigger(s|ed|ing)?")
# Words this short may be one word spelled with gaps ("k k k", "n i g ...").
_SPELLED_OUT = 3

# Digits and symbols that stand in for letters, then Cyrillic and Greek
# letters that look like Latin ones.
_LOOKALIKES = str.maketrans(
    "013457@$!|" "\u0430\u0435\u043e\u0456\u0441\u0440\u0445\u0443\u043a\u0455\u0458\u0433"
    "\u03bf\u03b1\u03b9\u03b5\u03ba\u03c1\u03c4\u03c5\u03bd",
    "oieastasii" "aeoicpxyksjr" "oaiekptuv",
)


def _stretched(term: str) -> str:
    """A pattern for the term with any letter held longer: ``coon`` matches
    ``cooon`` and never ``con``, so a doubled letter stays part of the word."""

    return "".join(
        f"{letter}{{{len(list(run))},}}" for letter, run in groupby(term)
    )


_TERMS = [codecs.decode(term, "rot13") for term in _ANYWHERE]
_WORDS = [codecs.decode(term, "rot13") for term in _WHOLE_WORD]
_INSIDE = re.compile("|".join(_stretched(term) for term in _TERMS))
_ENTIRE = re.compile("(?:" + "|".join(_stretched(term) for term in _WORDS) + ")s?")


def _refused(word: str) -> bool:
    if _ALLOWED.fullmatch(word):
        return False
    return bool(_ENTIRE.fullmatch(word) or _INSIDE.search(word))


def is_offensive(name: str) -> bool:
    """Whether a display name holds a refused term."""

    folded = unicodedata.normalize("NFKD", name).casefold().translate(_LOOKALIKES)
    words = [word for word in re.split(r"[^a-z]+", folded) if word]
    if any(_refused(word) for word in words):
        return True
    # Join each run of short words, so a term spelled with gaps reads as one
    # word. Long words stay apart: "Dana Zimmer" must not read as one string.
    for short, run in groupby(words, key=lambda word: len(word) <= _SPELLED_OUT):
        if short and _refused("".join(run)):
            return True
    return False


# Phases in which the reveal is public and may be serialized.
_REVEALED = (Phase.AWAITING_ACK, Phase.GAME_OVER)


class PlayerView(BaseModel):
    """One display row with server-owned seat, character, and current role."""

    name: str
    character: Literal["hal", "baku"]
    role: Literal["dropper", "checker"]
    cylinder_seconds: float
    ttd_seconds: float
    deaths: int
    is_human: bool


class OutcomeView(BaseModel):
    """A reveal with distinct engine-game-over and session-ending signals.

    ``round`` and ``half`` are the resolved half-round's own, not the next
    half's: the engine has already advanced by the time the reveal is shown.
    """

    round: int
    half: int
    dropper: str
    checker: str
    drop_time: int
    check_time: int
    result: HalfRoundResult
    st_gained: float
    death_duration: float
    survived: bool | None
    survival_probability: float | None
    game_over: bool
    session_ending: bool
    winner_name: str | None


class Snapshot(BaseModel):
    sequence: int
    phase: Phase
    game_index: int
    human_name: str
    clock_display: str
    clock_seconds: float
    round: int
    half: int
    turn_duration: int
    leap_window: bool
    dropper_name: str
    checker_name: str
    human_role: Literal["dropper", "checker"]
    legal_seconds: list[int]
    players: list[PlayerView]
    cylinder_max: float
    ttd_max: float
    half_rounds: int
    last_outcome: OutcomeView | None
    winner_name: str | None
    winner_is_human: bool | None
    stopped: bool


class ActionRequest(BaseModel):
    sequence: Annotated[int, Field(strict=True, ge=0)]
    second: Annotated[int, Field(strict=True, ge=1, le=61)]


class SequencedRequest(BaseModel):
    sequence: Annotated[int, Field(strict=True, ge=0)]


class NewSessionRequest(BaseModel):
    sequence: Annotated[int, Field(strict=True, ge=0)]
    human_name: str | None = None
    seed: Annotated[int, Field(strict=True)] | None = None
    start_clock: Annotated[int, Field(strict=True, ge=0)] | None = None
    max_half_rounds: Annotated[int, Field(strict=True, ge=1)] | None = None

    @field_validator("human_name")
    @classmethod
    def _valid_human_name(cls, value: str | None) -> str | None:
        return None if value is None else validate_human_display_name(value)


LEADERBOARD_NAME_LENGTH = 16
LEADERBOARD_SIZE = 10
# Control, format (bidi and zero-width), surrogate, private-use, unassigned,
# and line or paragraph separators.
_HIDDEN_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}
# Blank glyphs that Unicode files as letters or symbols: the Hangul fillers
# and the empty braille cell.
_BLANK_GLYPHS = {"\u115f", "\u1160", "\u3164", "\uffa0", "\u2800"}


class LeaderboardEntry(BaseModel):
    """One standing: a player's best win."""

    rank: int
    name: str
    score: float
    half_rounds: int
    is_you: bool


class Leaderboard(BaseModel):
    """The top standings plus the requesting player's own.

    Player identifiers and seeds stay in the ledger. ``your_score`` is the player's
    best win, if one exists; ``your_rank`` also needs a posted name.
    """

    entries: list[LeaderboardEntry]
    your_rank: int | None
    your_name: str | None
    your_score: float | None


def top_standings(payload: dict[str, object]) -> Leaderboard:
    """The board a browser may see: the ledger's answer, cut to the top ten."""

    board = Leaderboard.model_validate(payload)
    board.entries = board.entries[:LEADERBOARD_SIZE]
    return board


class PlayerNameRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        # Every player reads this name, so it must show as the text it holds:
        # no direction overrides, no invisible characters, no stacked marks.
        name = validate_human_display_name(unicodedata.normalize("NFC", value))
        if len(name) > LEADERBOARD_NAME_LENGTH:
            raise ValueError(
                f"a leaderboard name has at most {LEADERBOARD_NAME_LENGTH} characters"
            )
        categories = [unicodedata.category(character) for character in name]
        if any(category in _HIDDEN_CATEGORIES for category in categories) or any(
            character in _BLANK_GLYPHS for character in name
        ):
            raise ValueError("a leaderboard name must use visible characters")
        # Fullwidth and styled letters fold to plain ones, so they cannot spell Hal.
        if unicodedata.normalize("NFKC", name).casefold() == CANONICAL_HAL_NAME.casefold():
            raise ValueError(f"{CANONICAL_HAL_NAME!r} is reserved for the opponent")
        if not any(category[0] in "LN" for category in categories):
            raise ValueError("a leaderboard name needs a letter or a digit")
        marks = 0
        for category in categories:
            marks = marks + 1 if category[0] == "M" else 0
            if marks > 2:
                raise ValueError("a leaderboard name has too many combining marks")
        if is_offensive(name):
            raise ValueError("choose another leaderboard name")
        return name


def _outcome_view(session: PlaySession) -> OutcomeView | None:
    record = session.last_record
    if record is None or session.phase not in _REVEALED:
        return None
    return OutcomeView(
        round=int(record.round_num + 1),
        half=int(record.half),
        dropper=session.display_canonical_name(record.dropper),
        checker=session.display_canonical_name(record.checker),
        drop_time=int(record.drop_time),
        check_time=int(record.check_time),
        result=record.result,
        st_gained=float(record.st_gained),
        death_duration=float(record.death_duration),
        survived=record.survived,
        survival_probability=record.survival_probability,
        game_over=bool(session.game.game_over),
        session_ending=session.terminal,
        winner_name=session.display_canonical_name(session.winner_name),
    )


def snapshot_from_session(session: PlaySession) -> Snapshot:
    """Serialize everything the browser may see, and nothing more."""

    game = session.game
    dropper, checker = session.roles()
    turn_duration = session.turn_duration()
    # Legality is the engine's call and is only meaningful while the human is
    # on the clock; sending it otherwise would invite the client to act early.
    legal = (
        list(session.legal_actions()) if session.phase is Phase.AWAITING_ACTION else []
    )
    return Snapshot(
        sequence=session.sequence,
        phase=session.phase,
        game_index=session.game_index,
        human_name=session.human_display_name,
        clock_display=game.format_game_clock(),
        clock_seconds=float(game.game_clock),
        round=int(game.round_num + 1),
        half=int(game.current_half),
        turn_duration=int(turn_duration),
        # The leap turn is in effect only when the engine lengthened the turn.
        leap_window=turn_duration == TURN_DURATION_LEAP,
        dropper_name=session.display_name(dropper),
        checker_name=session.display_name(checker),
        human_role=session.human_role(),
        legal_seconds=legal,
        players=[
            PlayerView(
                name=session.display_name(player),
                character="baku" if player is session.human else "hal",
                role="dropper" if player is dropper else "checker",
                cylinder_seconds=float(player.cylinder),
                ttd_seconds=float(player.ttd),
                deaths=int(player.deaths),
                is_human=player is session.human,
            )
            for player in (game.player1, game.player2)
        ],
        cylinder_max=float(CYLINDER_MAX),
        ttd_max=float(TOTAL_TTD_MAX),
        half_rounds=session.half_rounds,
        last_outcome=_outcome_view(session),
        winner_name=session.display_canonical_name(session.winner_name),
        winner_is_human=(
            None if game.winner is None else game.winner is session.human
        ),
        stopped=session.stopped,
    )


TRANSCRIPT_SCHEMA = "arena-public-play-session-v1"

# The built browser client. ``npm --prefix web run build``
# writes it; when it exists the Python server serves it alone, with no Vite
# process, and when it does not the root route explains how to get it.
DEFAULT_WEBCLIENT_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

_NO_CLIENT_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Surpassing The Leader</title>
<body style="background:#000;color:#f2f2f2;font:14px/1.6 ui-monospace,monospace;padding:32px">
<h1 style="font-size:15px;letter-spacing:.22em">SURPASSING THE LEADER</h1>
<p>The API is running, but the browser client has not been built.</p>
<p>Either build it once and reload this page:</p>
<pre>npm --prefix web install
npm --prefix web run build</pre>
<p>or run the development client, which proxies to this server:</p>
<pre>npm --prefix web run dev</pre>
</body>
"""


@dataclass
class SessionConfig:
    """Per-game settings."""

    human_name: str = CANONICAL_HUMAN_NAME
    seed: int | None = None
    start_clock: int = OPENING_START_CLOCK
    max_half_rounds: int | None = None

    def __post_init__(self) -> None:
        self.human_name = validate_human_display_name(self.human_name)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if isinstance(self.start_clock, bool) or not isinstance(self.start_clock, int):
            raise TypeError("start_clock must be an integer")
        if self.start_clock < 0:
            raise ValueError("start_clock must be nonnegative")
        if self.max_half_rounds is not None:
            if isinstance(self.max_half_rounds, bool) or not isinstance(
                self.max_half_rounds, int
            ):
                raise TypeError("max_half_rounds must be an integer or None")
            if self.max_half_rounds <= 0:
                raise ValueError("max_half_rounds must be positive")


@dataclass
class SeriesConfig:
    """Session-wide presentation: Hal's label and whether its details stay private."""

    hal_agent: str = "dth"
    public_hal_label: str | None = None
    conceal_hal_details: bool = False

    @property
    def hal_label(self) -> str:
        return self.public_hal_label or self.hal_agent


def _new_session(
    hal_agent: object,
    config: SessionConfig,
    *,
    game_index: int = 0,
    game_seed: int | None = None,
    sequence_start: int = 0,
    reset_provider: bool = True,
) -> PlaySession:
    hal = Player(name=CANONICAL_HAL_NAME, physicality=PHYSICALITY_HAL)
    human = Player(name=CANONICAL_HUMAN_NAME, physicality=PHYSICALITY_BAKU)
    game = Game(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=random.Random(game_seed),
    )
    game.game_clock = config.start_clock
    if reset_provider:
        reset_provider_game(getattr(hal_agent, "provider", None))
    return PlaySession(
        game=game,
        hal_agent=hal_agent,
        hal=hal,
        human=human,
        human_display_name=config.human_name,
        game_index=game_index,
        game_seed=game_seed,
        start_clock=config.start_clock,
        max_half_rounds=config.max_half_rounds,
        sequence_start=sequence_start,
    )


def _game_seed(base_seed: int | None, game_index: int) -> int | None:
    """The per-game seed rule: base seed plus game index."""

    return None if base_seed is None else base_seed + game_index


def create_app(
    *,
    hal_factory: HalFactory,
    config: SessionConfig | None = None,
    series: SeriesConfig | None = None,
    webclient_dist: Path | None = DEFAULT_WEBCLIENT_DIST,
    sequence_start: int = 0,
) -> FastAPI:
    """Build the app. ``hal_factory`` is called once, never inside a request.

    ``sequence_start`` numbers the first session. The hosted server opens a
    fresh app after a restart and continues the player's sequence there, so a
    request from before the restart stays stale.

    Provider construction memory-maps a multi-gigabyte table, which does not
    belong on a request path, so it happens here at startup.
    """

    app = FastAPI(title="Surpassing The Leader")

    base_config = config or SessionConfig()
    series_config = series or SeriesConfig()
    hal_agent = hal_factory()
    # The hosted adapter checkpoints provider memory at game boundaries.
    app.state.hal_provider = getattr(hal_agent, "provider", None)
    series_seed = base_config.seed
    lock = threading.Lock()
    state: dict[str, object] = {
        "session": _new_session(
            hal_agent,
            base_config,
            game_seed=_game_seed(base_config.seed, 0),
            sequence_start=sequence_start,
        ),
        "game_index": 0,
        "recorded": False,
    }
    transcript: dict[str, object] = {
        "schema_version": TRANSCRIPT_SCHEMA,
        "hal_agent": series_config.hal_agent,
        "public_hal_label": series_config.public_hal_label,
        "human_name": base_config.human_name,
        "base_seed": base_config.seed,
        "start_clock": base_config.start_clock,
        "games": [],
    }
    tally = {"human_wins": 0, "hal_wins": 0, "no_winner": 0, "stopped": 0}

    def _session() -> PlaySession:
        session = state["session"]
        assert isinstance(session, PlaySession)
        return session

    def _game_index() -> int:
        index = state["game_index"]
        assert isinstance(index, int)
        return index

    def _require(sequence: int, session: PlaySession) -> None:
        if sequence != session.sequence:
            raise HTTPException(
                status_code=409,
                detail=f"stale sequence {sequence}; session is at {session.sequence}",
            )

    def _hal_details() -> dict[str, object]:
        details: dict[str, object] = {}
        match_summary = getattr(getattr(hal_agent, "provider", None), "match_summary", None)
        if callable(match_summary):
            details["hal_summary"] = match_summary()
        diagnostics = getattr(
            getattr(hal_agent, "provider", None), "experiment_diagnostics", None
        )
        if callable(diagnostics):
            details["hal_diagnostics"] = diagnostics()
        return details

    def _record_finished_game(session: PlaySession) -> None:
        """Append a terminal game to the series once."""

        if state["recorded"]:
            return
        record = session.finish()
        games = transcript["games"]
        assert isinstance(games, list)
        games.append(record)
        if session.stopped:
            tally["stopped"] += 1
        elif session.game.winner is None:
            tally["no_winner"] += 1
        elif session.game.winner is session.human:
            tally["human_wins"] += 1
        else:
            tally["hal_wins"] += 1
        state["recorded"] = True

    @app.get("/api/rules")
    def rules() -> dict[str, object]:
        return {
            "human_name": _session().human_display_name,
            "hal_label": series_config.hal_label,
            "lines": list(RULES),
        }

    @app.get("/api/session", response_model=Snapshot)
    def read_session() -> Snapshot:
        with lock:
            return snapshot_from_session(_session())

    @app.post("/api/session", response_model=Snapshot)
    def new_session(request: NewSessionRequest) -> Snapshot:
        with lock:
            previous = _session()
            _require(request.sequence, previous)
            if previous.phase not in (Phase.RULES, Phase.GAME_OVER):
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot replace an active session in phase {previous.phase.value}",
                )
            finished = previous.phase is Phase.GAME_OVER
            if finished:
                _record_finished_game(previous)
                state["game_index"] = _game_index() + 1
            game_index = _game_index()
            config = SessionConfig(
                human_name=(
                    request.human_name
                    if request.human_name is not None
                    else previous.human_display_name
                ),
                seed=series_seed,
                start_clock=(
                    request.start_clock
                    if request.start_clock is not None
                    else base_config.start_clock
                ),
                max_half_rounds=(
                    request.max_half_rounds
                    if request.max_half_rounds is not None
                    else base_config.max_half_rounds
                ),
            )
            game_seed = (
                request.seed
                if request.seed is not None
                else _game_seed(series_seed, game_index)
            )
            state["session"] = _new_session(
                hal_agent,
                config,
                game_index=game_index,
                game_seed=game_seed,
                sequence_start=previous.sequence + 1,
                reset_provider=finished,
            )
            state["recorded"] = False
            transcript["human_name"] = config.human_name
            return snapshot_from_session(_session())

    @app.post("/api/session/restart", response_model=Snapshot)
    def restart(request: SequencedRequest) -> Snapshot:
        nonlocal series_seed
        with lock:
            previous = _session()
            _require(request.sequence, previous)
            sequence = previous.sequence + 1
            series_seed = _game_seed(base_config.seed, sequence)
            state["session"] = _new_session(
                hal_agent, base_config, game_seed=series_seed, sequence_start=sequence
            )
            state["game_index"] = 0
            state["recorded"] = False
            transcript["games"] = []
            transcript["human_name"] = base_config.human_name
            transcript["base_seed"] = series_seed
            tally.update({key: 0 for key in tally})
            return snapshot_from_session(_session())

    @app.post("/api/session/begin", response_model=Snapshot)
    def begin(request: SequencedRequest) -> Snapshot:
        with lock:
            session = _session()
            _require(request.sequence, session)
            try:
                session.begin()
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if session.phase is Phase.GAME_OVER:
                _record_finished_game(session)
            return snapshot_from_session(session)

    @app.post("/api/session/action", response_model=Snapshot)
    def act(request: ActionRequest) -> Snapshot:
        with lock:
            session = _session()
            _require(request.sequence, session)
            if session.phase is not Phase.AWAITING_ACTION:
                raise HTTPException(status_code=409, detail=f"phase is {session.phase.value}")
            if request.second not in session.legal_actions():
                raise HTTPException(
                    status_code=422,
                    detail=f"{request.second} is not a legal second this turn",
                )
            try:
                session.submit(request.second)
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return snapshot_from_session(session)

    @app.post("/api/session/ack", response_model=Snapshot)
    def acknowledge(request: SequencedRequest) -> Snapshot:
        with lock:
            session = _session()
            _require(request.sequence, session)
            try:
                session.acknowledge()
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if session.phase is Phase.GAME_OVER:
                _record_finished_game(session)
            return snapshot_from_session(session)

    @app.get("/api/transcript")
    def read_transcript() -> dict[str, object]:
        """The public series so far: finished games plus the live game's reveals.

        Everything here has already been revealed on an outcome screen. The
        live game's history contains only resolved half-rounds, so a request
        made while the human is deciding cannot see an unrevealed second.
        """

        with lock:
            session = _session()
            payload: dict[str, object] = {
                **transcript,
                "tally": dict(tally),
                "current_game": {
                    "game_index": session.game_index,
                    "seed": session.game_seed,
                    "start_clock": session.start_clock,
                    "phase": session.phase.value,
                    "half_rounds": session.half_rounds,
                    "public_history": list(session.public_history),
                },
            }
            if not series_config.conceal_hal_details:
                payload.update(_hal_details())
            return payload

    if webclient_dist is not None and (webclient_dist / "index.html").is_file():

        @app.middleware("http")
        async def _revalidate_page(request, call_next):
            # Vite names its bundles by content hash, so they may be cached
            # freely; the page that references them must be re-checked on
            # every load or a rebuild stays invisible until a hard refresh.
            response = await call_next(request)
            if request.url.path in ("/", "/index.html"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        # Registered last so the API routes above take precedence.
        app.mount("/", StaticFiles(directory=webclient_dist, html=True), name="webclient")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def no_client() -> str:
            return _NO_CLIENT_PAGE

    return app
