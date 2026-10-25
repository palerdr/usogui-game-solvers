# Agent Instructions

## Communication

- Keep progress updates concise and factual.
- Treat pre-existing uncommitted changes as user work.

## Repository boundaries

- `docs/` owns repository-wide canonical game and solver contracts.
- `paper/` owns the project's mathematical paper. Re-render and visually
  inspect its PDF after changing the TeX. Its figure scripts read the STL
  leap artifact through `stl.reader` and the DTH table through `stl.dth`,
  and import no other project module.
- `docs/papers/` owns primary evidence and cited literature.
- `src/stl/` and `src/dth_ocaml/` are peer projects. They must not import one another. `src/arena/` is the game library
  that the browser app uses: it may consume their public interfaces, but they
  must not import it or one another.
- `src/formal/` is a Lean project that machine-checks the solvers' mathematics.
  It imports no other project, and no project imports it.
- `src/browser/` is the browser game app, `python -m browser`: the FastAPI
  server, the TypeScript client in `src/browser/webclient/`, and the Vercel
  deploy pipeline in `src/browser/deploy/`. It may import `arena`,
  `stl.game`, and `stl.dth`, and no project imports it. The client renders
  state and collects input; it never derives game rules and never receives an
  unrevealed action.
- Each project owns its configs, docs, tests, checkpoints, and outputs.
- Generated data must remain gitignored. The character frames under
  `src/browser/webclient/public/art/` are source art, not generated data, and
  are tracked.

Read the nearest nested `README.md` only when working in that subtree. Do not
place subsystem status, plans, or invariants in the repository root.

`CLAUDE.md` at the repository root imports every subtree `README.md` so Claude
Code loads them all as instruction files. Adding a subtree means adding its
`README.md` and an import line there.

## Layers

The table lists the import rules of each project. No test enforces them.

| Layer | Projects | May import |
| --- | --- | --- |
| Peers | `stl`, `dth_ocaml`, `dth_cpp` | No other project |
| Proofs | `formal` | Nothing |
| Library | `arena` | `stl.game`, `stl.reader`, `stl.dth` |
| App | `browser` | `arena`, `stl.game`, `stl.dth` |
| Consumer | `paper` | `stl.reader`, `stl.dth` |

- No project imports `browser`, and `arena` imports no `browser` module.
- `paper` owns no importable module. `stl.reader` is the one read path into a
  completed leap artifact. Code outside `stl` reads leap tables through it,
  not through the `stl.solver.leap_*` builder modules.
- Inside `arena`, `contracts.py`, `agent.py`, `session.py`, and
  `variants.py` import `stl.game` and each other alone. `stl.dth` enters
  through the adapters and `arena/policies/`.
- No project imports `torch`, `gymnasium`, `stable_baselines3`, or
  `sb3_contrib`. `arena/policies/__init__.py` imports nothing.
- The `file:line` citations in `src/formal` read against commit `938fb25`.

## Fingerprints and locks

- `python -m stl.dth_kernel` writes the ignored `artifacts/dth/V.npy`
  (sha256 `e43977b7...`), which the leap manifest binds by hash.
  `stl.dth.Table.open` fails closed on its shape and the six anchors. No
  lock enters a DTH digest.
- `stl.dth` imports numpy and scipy only, so the hosted bundle needs no
  numba. The numba sweep lives in `stl.dth_kernel`.
- `stl.solver.leap_build.builder_hash()` hashes the builder sources, the
  numba kernel `leap_kernel.py` among them, with the root `pyproject.toml` and
  `uv.lock`, and a resume refuses a changed hash. Do not start or resume a
  leap build across an edit to the root `pyproject.toml` or `uv.lock`.
- The builder hash reads `src/stl/solver/leap_*.py` by glob. Name a new STL
  builder module `src/stl/solver/leap_*.py`, and put new non-builder STL code
  outside `src/stl/solver/`.
- The STL leap kernel `src/stl/solver/leap_kernel.py` is numba code. Its
  first call compiles it into the ignored `src/stl/solver/__pycache__/`;
  `NUMBA_CACHE_DIR` moves that cache.
- One root `pyproject.toml` and one `uv.lock` cover every Python package,
  the browser server included. The deploy bundle takes its package pins from
  that lock. A dependency change therefore also changes the leap builder hash.

## Frozen global rules

- Actions are literal seconds beginning at 1; action 0 is illegal.
- A successful check uses inclusive elapsed time: `ST = check - drop + 1`.
- Normal action sets are 1..60.
- The injected dose is vial contents plus a fixed 60-second penalty, `q = s + 60`.
- Capacity is 300 seconds; `t + q > 300` is fatal and `t + q == 300` stays
  revival-eligible when `q < 300`.
- One revival-probability surface is frozen for the whole repository in
  `docs/REVIVAL_MODEL.md`. No project may carry its own revival constants or
  reintroduce an explicit CPR count or per-player physicality as a revival-odds
  input at any rung.
- In STL's leap window only Baku as Dropper may choose 61; Checker remains
  capped at 60. Both players know the leap rule from game initialization.
- DTH does not inherit STL-only leap or information-state mechanics.
- The `dth_ocaml` solver library in `lib/solver/`, its tests, and
  `bin/solve_tablebase.ml` are an independent OCaml implementation of **pure
  DTH**, not STL: actions are literal seconds 1..60 and it has no leap window.
  It exists as a hand-written reference for the exact solver and is held to
  the same frozen rules and the same 1e-6 saddle-gap gate as its Python peer.
  `src/dth_ocaml/play/` is a second dune package that holds the CS 3110 STL
  engine and its terminal UI. The solver library does not link it.
- `stl.dth` and `stl.dth_kernel` are the paper's solver of **pure DTH**, held
  to the 1e-6 saddle-gap gate, and `stl.dth.Table` is the arena's one DTH
  table. `src/c/fast_kernel.c` is the retired C recurrence kernel; no code
  calls it.
- `docs/FORMULATION_LADDER.md` fixes which games are claimed at all. Work that
  does not sit on a rung is not a supported claim.

Any rules change must update canonical docs, evidence citations, schemas, and
tests together. Do not weaken solver firewalls, gates, tolerances, or artifact
validation to make a change pass.

## Validation

```powershell
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
npm --prefix src/browser/webclient run typecheck
opam exec --switch=stl-dth-ocaml -- dune build --root src/dth_ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/dth_ocaml
lake -d src/formal build
```

The typecheck needs `npm --prefix src/browser/webclient install` once. The browser
client is not covered by `pytest`, so skipping it leaves the front end
unchecked.

`pytest` runs distributed at `-n 8`. Add `-n 0` for a readable serial
traceback when something fails. The hosted runtime runs on Python 3.13, so CI
also runs `uv run --locked --python 3.13 python -m pytest src/browser/tests -q -n 0`.

Everything runs by default, including the `slow` tests that build real complete
artifacts end to end — they are the only coverage of resume and backend
parity, and under xdist they cost about 19 seconds. Use `-m "not slow"` for a
tighter loop when iterating on unrelated code, never as the check before a
merge.

After code changes, run `graphify update .`. Use `graphify query`, `path`, or
`explain` for architecture questions when `graphify-out/graph.json` exists.
