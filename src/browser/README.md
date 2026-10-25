# Browser App Instructions

`src/browser/` holds the deploy notes in
[`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md). The game's TypeScript client
lives in `web/`, its design brief in `docs/GAMEPLAY.md`, and its Python side in
`src/stl/server/`. Build the client once, then run the local server:

```powershell
npm --prefix web run build
uv run python -m stl serve
```

## Boundaries

- `stl.server` imports `stl.hal`, `stl.dth` and `stl.game`. No other module
  imports it.
- The hosted entry `stl.server.hosted` loads no numba and no training stack.
  `tests/server/test_deploy.py` checks this in a fresh interpreter, from a
  copy of the sources as the bundle holds them.
- `python -m stl serve` builds Hal with `stl.hal.make_hal`: `--hal translated`
  (the default) or `--hal exact` (the certified DTH equilibrium).
- `stl/server/app.py` holds the player-facing rules text. The client's
  character frames are static files under `web/public/art/`.
- `tests/conftest.py` holds the fakes that more than one test shares.

## Server invariants

`stl/server/app.py` serves the TypeScript client in `web/`. It builds
its provider once at startup, because provider construction memory-maps a
multi-gigabyte artifact, which must not happen on a request path.
`snapshot_from_session` in the same module is the only serializer that faces
the browser, so the seat-scoping rule has exactly one place to be enforced and
one place to be tested. `stl.server.session.PlaySession` samples Hal's action
inside `submit`, after the human's second is accepted; `src/stl/README.md`
states that hidden-information guarantee.

The browser server is one repeated-opponent series: one Hal is retained across
games, game `N` is seeded with the base seed plus `N`, and every finished game
is appended to a public transcript in the `arena-public-play-session-v1`
shape. `GET /api/transcript` serves that transcript plus the live game's
resolved half-rounds. When `web/dist/` has been built, the Python server
serves it at `/`, so one process is the whole game.

The web session's `human_name` is a presentation label only; it never
replaces Baku's rule-bearing identity. `Hal` is reserved as a display label.
Browser session replacement is a sequenced mutation, is allowed only before
play or after a terminal acknowledgement, and advances the sequence across the
replacement. Browser snapshots carry server-owned character, role, and
winner-seat fields, so the client never infers identity from presentation
labels.

The browser requests a sequenced restart on page load. This abandons the active
game and clears the visible series before returning the title page. Ordinary
session reads still recover state for stale-request handling and worker replay.

## Hosted sessions

The Vercel entry `stl.server.hosted.create_production_app` serves translated
Hal. It isolates players with secure cookies and Redis command logs, and
`RedisSessionStore` holds the Upstash store and its atomic compare-and-set.
The entry replays accepted commands through the same local HTTP app and
commits each mutation before returning a reveal. A process keeps the game it
last served and rebuilds it from the command log only when Redis shows that
the game moved on elsewhere. `SupabaseLedger` writes the game's public history
to Supabase, and `GET /api/leaderboard` ranks each player's best win by the
winner's seconds of life left. A restart and a next game each replace the
record with fresh seeds, an empty command list, and the next sequence number,
so replay covers only the current game. Translated Hal's opponent evidence is
stored as a private checkpoint with each game record and restored before
command replay, so reload and next-game requests retain it. Separate cookies
keep separate models; workers share the immutable table. The local server
keeps one repeated-opponent series.

## Deployment

[`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md) holds the certified table
build, the local launch commands, and the Vercel pipeline.

- `uv run --isolated --python 3.13 python -m stl deploy` builds the bundle in
  the ignored `artifacts/vercel/`, runs `vercel build`, adds 3.13 bytecode,
  and deploys a preview.
- The bundle holds every `src/stl` Python file under `runtime/src/` and the
  DTH table at `runtime/artifacts/dth/V.npy`. `stl.server.deploy.prepare`
  rechecks 1,200 classes and computes the hosted code version from the
  sha256 of the copied Python files and of `V.npy`, and writes it into the
  bundle's `app.py`.
- A change to any `src/stl` Python file ends every live hosted session at the
  next deploy.
- The bundle takes the pins of its packages from the root `uv.lock` and
  copies the ignored Vercel project link from `.vercel/project.json`.

## Working in this subtree

Run `uv run python -m pytest tests/server -q`. The client is not covered by
`pytest`; run `npm --prefix web test`,
`npm --prefix web run typecheck`, and
`npm --prefix web run build` after a client change. The
typecheck needs `npm --prefix web install` once.
