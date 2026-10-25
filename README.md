# Surpassing The Leader Solver Monorepo

This repository contains deliberately separate game-solving projects:

| Project | Purpose | Rules |
| --- | --- | --- |
| `src/stl/` | Canonical leap-aware referee, the certified leap table, the DTH solver `stl.dth` with its numba sweep `stl.dth_kernel`, which builds the complete 289,374,121-class table, Hal in `stl.hal`, and the browser game's server in `stl.server`. `python -m stl` runs the four commands `dth`, `leap`, `serve` and `deploy` | L2 public game; only Baku as Dropper may use second 61. DTH: literal seconds 1..60 |
| `src/dth_ocaml/` | Hand-written exact OCaml reference; `play/` keeps the CS 3110 STL engine and terminal UI | Pure DTH with literal seconds 1..60 and the repository-wide frozen revival model |
| `src/dth_cpp/` | In-progress native exact DTH implementation | Pure DTH; build order and root integration remain subtree-owned while work is active |
| `src/browser/` | The browser game's TypeScript client and its deploy notes | Canonical STL through `stl.server` |

Repository-wide game contracts live in [`docs/`](docs/). The mathematical
paper, a certified exact solution of the complete DTH game, lives at
[`paper/`](paper/). Primary game evidence lives in
[`docs/game-sources/`](docs/game-sources/), with the manga reference images in
[`docs/game-sources/reference-art/`](docs/game-sources/reference-art/), and
cited research lives in [`docs/papers/`](docs/papers/). The OCaml project
records its authorship in [`src/dth_ocaml/AUTHORS.md`](src/dth_ocaml/AUTHORS.md).

## Setup

Use Python 3.12 or 3.13 and `uv`. One root `pyproject.toml` and one
`uv.lock` cover every Python package, the browser server included:

```powershell
uv sync --locked
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
npm --prefix web run typecheck
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
```

## Commands

```powershell
# The certified STL leap table: artifacts/leap
uv run python -m stl leap

# The certified DTH table: artifacts/dth/V.npy, sha256 e43977b7...
uv run python -m stl dth

# Browser game on 127.0.0.1:8000, once the client is built; translated Hal is the default
npm --prefix web run build
uv run python -m stl serve
uv run python -m stl serve --hal exact
```

Each project owns its `config/`, `docs/`, tests, checkpoints, and outputs.
Generated artifacts are gitignored and must not be mixed between projects or
across incompatible schema versions.

## Exact DTH paper

The paper is available as both the rendered
[`paper/dth_exact_solution.pdf`](paper/dth_exact_solution.pdf) and its
[`paper/dth_exact_solution.tex`](paper/dth_exact_solution.tex) source. It gives
a certified exact solution of the complete finite DTH game, including the root
value and equilibrium strategies, and describes the solver in
[`src/stl/dth.py`](src/stl/dth.py) and
[`src/stl/dth_kernel.py`](src/stl/dth_kernel.py). Reproduce its result from
the repository root in about two minutes:

```powershell
uv run python -m stl dth
uv run python -m pytest tests/test_dth.py tests/test_dth_kernel.py -q
```

With the table built (`artifacts/dth/V.npy`), the figures and the PDF can be
rebuilt:

```powershell
uv run python paper/generate_figure_data.py
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
