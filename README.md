# Surpassing The Leader Solver Monorepo

This repository contains deliberately separate game-solving projects:

| Project | Purpose | Rules |
| --- | --- | --- |
| `src/stl/` | Canonical leap-aware referee compatibility layer and formulation shell; no complete STL solver yet | L2 public game; only Baku as Dropper may use second 61 |
| `src/dth/` | Completed pure Drop the Handkerchief solve and its audit | Exact 289,374,121-class quotient tablebase; literal seconds 1..60 |
| `src/dth_compact/` | The paper's solver: one numba file that builds the complete 289,374,121-class table in about 49 s | Pure DTH; literal seconds 1..60 |
| `src/dth_ocaml/` | Hand-written exact OCaml reference; `play/` keeps the CS 3110 STL engine and terminal UI | Pure DTH with literal seconds 1..60 and the repository-wide frozen revival model |
| `src/dth_cpp/` | In-progress native exact DTH implementation | Pure DTH; build order and root integration remain subtree-owned while work is active |
| `src/arena/` | Game library: session, provider adapters, exact and translated Hal | Canonical STL referee with pluggable policy-provider adapters |
| `src/browser/` | Browser game app: `python -m browser`, the TypeScript client, and the hosted Vercel deployment | Canonical STL, or pure DTH with `--pure-dth`, through `src/arena/` |

Repository-wide game contracts live in [`docs/`](docs/). The mathematical
paper, a certified exact solution of the complete DTH game, lives at
[`paper/`](paper/). Primary game evidence lives in
[`docs/game-sources/`](docs/game-sources/), with the manga reference images in
[`docs/game-sources/reference-art/`](docs/game-sources/reference-art/), and
cited research lives in [`docs/papers/`](docs/papers/). The OCaml project
records its authorship in [`src/dth_ocaml/AUTHORS.md`](src/dth_ocaml/AUTHORS.md).

## Setup

Use Python 3.12 or 3.13 and `uv`. The browser app is its own uv project in
`src/browser/`:

```powershell
uv sync --dev
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
uv run --project src/dth_compact pytest src/dth_compact/tests -q
uv run --project src/browser python -m pytest src/browser/tests -q
npm --prefix src/browser/webclient run typecheck
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
```

## Commands

```powershell
# The certified STL leap table
uv run python -m stl.solver.leap_build --full --min-clock 720

# Pure DTH complete solution and its audit
uv run python -m dth --help

# Complete exact DTH quotient tablebase
uv run python -m dth complete
uv run python -m dth complete-audit

# The paper's compact solver: its own uv project
uv run --project src/dth_compact src/dth_compact/main.py
uv run --project src/dth_compact pytest src/dth_compact/tests -q

# Browser game on 127.0.0.1:8000, once the client is built; exact Hal is the default
npm --prefix src/browser/webclient run build
uv run --project src/browser python -m browser
uv run --project src/browser python -m browser --hal-agent translated
```

Each project owns its `config/`, `docs/`, tests, checkpoints, and outputs.
Generated artifacts are gitignored and must not be mixed between projects or
across incompatible schema versions.

## Exact DTH paper

The paper is available as both the rendered
[`paper/dth_exact_solution.pdf`](paper/dth_exact_solution.pdf) and its
[`paper/dth_exact_solution.tex`](paper/dth_exact_solution.tex) source. It gives
a certified exact solution of the complete finite DTH game, including the root
value and equilibrium strategies, and describes the one-file solver in
[`src/dth_compact/`](src/dth_compact/). Reproduce its result from the
repository root in about a minute:

```powershell
uv run --project src/dth_compact src/dth_compact/main.py
uv run --project src/dth_compact pytest src/dth_compact/tests -q
```

With the compact solver's table built (`src/dth_compact/artifacts/V.npy`), the
figures and the PDF can be rebuilt:

```powershell
uv run --project src/dth_compact python paper/generate_figure_data.py
uv run --with matplotlib --with seaborn python paper/make_figures.py
Push-Location paper
tectonic dth_exact_solution.tex --synctex --keep-logs
Pop-Location
```

## Canonical contracts

- [`docs/ACTION_TIMING.md`](docs/ACTION_TIMING.md) owns literal action seconds,
  successful-check timing, and leap-second action legality.
- [`docs/CANONICAL_EXTENSIVE_FORM.md`](docs/CANONICAL_EXTENSIVE_FORM.md) owns the
  full-game state, transition, death, revival, clock, and terminal boundaries.
- [`docs/FOUNDATIONS.md`](docs/FOUNDATIONS.md) owns the shared zero-sum matrix
  and exact-solver foundations without repeating game rules.
- [`docs/game-sources/EVIDENCE.md`](docs/game-sources/EVIDENCE.md) records
  the documentary basis for frozen rule choices.

## Repository policy

Root guidance applies everywhere. Read the nearest binding subtree `README.md` before
working inside a project. Folder-specific context belongs beside that folder,
not in the repository root. Root Markdown is limited to this overview and the
global agent contract; project Markdown stays under its owning `src/` subtree.
