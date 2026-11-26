# Surpassing the Leader

## What it does

A browser game against Hal, a complete pure-DTH solver, and the certified
public leap-game values behind the papers. The rules come from *Usogui* by
Toshio Sako; revival probabilities are explicit modeling choices.

| Home | Reason it exists |
| --- | --- |
| `src/stl/game.py` | The live referee and action legality |
| `src/stl/dth.py`, `dth_kernel.py` | Read/certify a DTH stage; build the complete table with numba |
| `src/stl/leap.py`, `leap_kernel.py` | Build, audit, and read the public leap table; frozen numerical kernel |
| `src/stl/hal.py` | Exact equilibrium and the selected Translated opponent model |
| `src/stl/server/` | Sessions, HTTP routes, hosted persistence, and deployment |
| `src/stl/__main__.py` | The four commands: `dth`, `leap`, `serve`, `deploy` |
| `web/` | Browser UI, tests, and the ten committed character frames |
| `src/ocaml/`, `src/cpp/` | Independent reference solvers; OCaml also keeps the CS 3110 terminal game |
| [`src/bend/`](src/bend/README.md) | Bend solver for the complete pure-DTH table |
| `src/c/` | Retired C recurrence source, retained as native work; no live caller |
| `src/lean/` | Mathematical proofs and `CLAIMS.md`; source citations refer to `938fb25` |
| `tests/` | Tests mirror the Python modules; golden data pins numerical and Hal behavior |
| `docs/` | [Rules](docs/RULES.md), [DTH](docs/DTH.md), [leap](docs/LEAP.md), [Hal](docs/HAL.md), [gameplay](docs/GAMEPLAY.md), papers, and source evidence |
| `artifacts/` (ignored) | DTH and leap tables, native reference tables, and deployment bundle |

The OCaml game originated in CS 3110 Spring 2026 with James Cenawood (JCC463),
Orion Hoch (OGH6), Emily Wei (EJW235), and Max Masleyev (MAM759). Its terminal
UI follows David Kaloper Meršinjak's `notty` (ISC).

## How to run

From `games/surpassing-the-leader`, use Python 3.12 and uv:

```sh
uv sync --locked
uv run python -m stl --help
uv run python -m stl dth
npm --prefix web ci
npm --prefix web run build
uv run python -m stl serve
```

`dth` writes `artifacts/dth/V.npy` (2.16 GiB, SHA-256 `e43977b7...`).
`serve` defaults to Translated Hal at `127.0.0.1:8000`; flags are
`--hal translated|exact`, `--host`, `--port`, and `--seed`.
`npm --prefix web run dev` runs the client development server.

The full leap build writes about 82 GB and takes hours:

```sh
uv run python -m stl leap --help
uv run python -m stl leap
```

`dth --output PATH` and `leap --dth PATH --output DIRECTORY` select artifacts.
A leap build requires an empty output directory. An interrupted build must
restart in a new empty directory. `open_leap()` reads completed results and
checks the DTH reference values; `LeapTable.stage()` checks stage certificates.

Independent native programs:

```sh
opam exec --switch=stl-dth-ocaml -- dune exec --root src/ocaml dth-play
cmake -S src/cpp --preset release
cmake --build src/cpp/build/release
src/cpp/build/release/dth-solve-tablebase --fresh --output artifacts/cpp --threads 4
```

The C++ solver requires an absent or empty output directory for `--fresh`;
use `--resume` for an interrupted build or `--verify-only` for a completed one.
The OCaml full table build has not been completed:

```sh
opam exec --switch=stl-dth-ocaml -- dune exec --root src/ocaml dth-solve-tablebase -- artifacts/ocaml/value.bin
```

Build all papers
from `docs/paper/`; the five included figure files are committed:

```sh
cd docs/paper
tectonic dth_exact_solution.tex
tectonic stl.tex
tectonic pm_hal.tex
```

Deployment requires owner approval. From the root, with the existing Vercel
project link at `.vercel/project.json`, create a preview:

```sh
uv run --isolated --python 3.13 python -m stl deploy
```

`deploy --artifact PATH` selects the DTH table. It verifies the table, builds
`web/`, copies Python and the table into ignored `artifacts/vercel/`, runs
Vercel build, adds Python 3.13 bytecode, and uploads a prebuilt archive.
Check `/api/health`, one full game, separate browser sessions, competing
submissions, and the leaderboard on the preview. After those checks pass,
rebuild the prepared bundle for production and add bytecode again:

```sh
npx vercel pull --cwd artifacts/vercel --environment=production --yes
npx vercel build --cwd artifacts/vercel --prod --yes
uv run --isolated --locked --python 3.13 python -c 'from stl.server.deploy import BUNDLE, add_bytecode; print(add_bytecode(BUNDLE))'
npx vercel deploy --cwd artifacts/vercel --prebuilt --archive=tgz --prod --skip-domain
```

Repeat the checks on the staged production URL. In the Vercel dashboard,
open that deployment's menu and select **Promote**. [Staged production
promotion preserves the build; preview promotion rebuilds it](https://vercel.com/docs/deployments/promoting-a-deployment).
The staged path retains the bytecode added after the local build.

Hosted configuration: `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` and Fluid Compute
for the large table bundle; Redis through `KV_REST_API_URL` and
`KV_REST_API_TOKEN` (or `UPSTASH_REDIS_REST_URL` and
`UPSTASH_REDIS_REST_TOKEN`); optional leaderboard through `SUPABASE_URL` and
`SUPABASE_SECRET_KEY` (or `SUPABASE_SERVICE_ROLE_KEY`). Secrets stay in hosted
settings. `stl.server.hosted` is the only environment reader. Each deployment gets a new session version and ends old
sessions once the replacement deploy serves them.

## How to test

```sh
uv run pytest -n 8
uv run --isolated --locked --python 3.13 pytest tests/server -n 0
npm --prefix web run typecheck
npm --prefix web test
npm --prefix web run build
uv run ruff check
uv run ruff format --check
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml @fmt
ctest --test-dir src/cpp/build/release --output-on-failure
(cd src/lean && lake exe cache get && lake build)
```

The default pytest suite includes slow tests. `-m "not slow"` is only an
iteration shortcut. Tests requiring the real artifacts skip if those files
are absent; release verification must include them. After numerical edits,
rebuild DTH into a separate path and compare its complete SHA-256, run the
leap golden fixture, and recertify all 16 Figure 3 records. Never weaken an
assertion checking surviving behavior. `leap_kernel.py` is byte-frozen.

## Design decisions

One flat Python package follows the domain names. The server is a subpackage
because sessions, routes, persistence, and deployment form one larger domain.
DTH's scalar reader is separate from its numba sweep so hosted requests load
neither numba nor training libraries. No Python package imports a native peer.

Hal and the browser use the small DTH artifact. Exact Hal certifies pure DTH,
and Translated Hal exploits a public-history forecast; neither claims full
STL optimality or loads the 82 GB leap table. The leap table solves the public
clock-aware formulation independently. [RULES.md](docs/RULES.md) distinguishes
retired L0, pure DTH L1, and public leap L2.

Hosted requests read Redis as authority, replay private seeds and accepted
commands, and compare-and-set before revealing an action. Per-process held
games only cache that state. Opponent memory is serialized between games.
The game works without the optional Supabase leaderboard.

The frozen leap kernel's historical doc paths and Lean's source-line citations
read against `938fb25`. [LEAP.md](docs/LEAP.md) and [DTH.md](docs/DTH.md) replace
the old guides. The original figure scripts and retired experiments live in
git history; committed figures plus numerical tests keep the papers buildable.

## Status

The simplification is local on main; production still serves the prior bundle.
No push, deployment, or hosted settings change is part of the cleanup.
Translated Hal is the selected production opponent; the other studies and
failed gates are summarized in [HAL.md](docs/HAL.md).

Paper source claims have been preserved for owner review. In
`docs/paper/dth_exact_solution.tex`, the 8.01-second complete-build timing and
"Python with a C kernel" statement describe an older implementation. The
following shortcut description also needs checking against the current numba
DTH sweep. `docs/paper/pm_hal.tex` still says PM plays local pure-DTH games,
although that provider is now retired. Its historical record path is pinned
to `938fb25`. No mathematical or timing claim was revised during relocation.
