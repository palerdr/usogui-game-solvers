# Agent Instructions

## Communication

- Keep progress updates concise and factual.
- Treat pre-existing uncommitted changes as user work.

## Repository boundaries

- `docs/` owns repository-wide canonical game and solver contracts.
- `paper/` owns the project's mathematical paper. Re-render and visually
  inspect its PDF after changing the TeX. Its figure scripts read the STL
  leap artifact through `stl.leap` and the DTH table through `stl.dth`,
  and import no other project module.
- `docs/papers/` owns primary evidence and cited literature.
- `src/stl/`, `src/ocaml/` and `src/cpp/` are peer projects. They must not import one another.
- `src/lean/` is a Lean project that machine-checks the solvers' mathematics.
  It imports no other project, and no project imports it.
- `src/stl/server/` is the browser game's Python side: the session, the
  FastAPI routes, the hosted Vercel entry and the deploy pipeline. `python -m
  stl serve` runs it locally. `web/` holds the TypeScript client. The client renders state and collects input; it never derives game
  rules and never receives an unrevealed action.
- Each project owns its configs, docs, tests, checkpoints, and outputs.
- Generated data must remain gitignored. The character frames under
  `web/public/art/` are source art, not generated data, and
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
| Peers | `stl`, `ocaml`, `cpp` | No other project |
| Proofs | `lean` | Nothing |
| Consumer | `paper` | `stl.leap`, `stl.dth` |

- Inside `stl`, `hal.py` imports `stl.dth` and `stl.game`, and `server/`
  imports `hal`, `dth` and `game`. No module below `server/` imports it.
- `paper` owns no importable module. `stl.leap.open_leap` is the one read
  path into a completed leap artifact.
- No module imports `torch`, `gymnasium`, `stable_baselines3`, or
  `sb3_contrib`. The hosted entry `stl.server.hosted` loads no numba;
  `tests/server/test_deploy.py` checks both in a fresh interpreter.
- The `file:line` citations in `src/lean` read against commit `938fb25`.

## Fingerprints and locks

- `python -m stl dth` writes the ignored `artifacts/dth/V.npy`
  (sha256 `e43977b7...`), which the leap manifest binds by hash.
  `stl.dth.Table.open` fails closed on its shape and the six anchors. No
  lock enters a DTH digest.
- `stl.dth` imports numpy and scipy only, so the hosted bundle needs no
  numba. The numba sweep lives in `stl.dth_kernel`.
- `stl.leap.builder_hash()` hashes `src/stl/game.py`, `leap.py` and
  `leap_kernel.py` by package-relative name, and a resume refuses a changed
  hash. Do not start or resume a leap build across an edit to those three
  files.
- `python -m stl leap` writes the ignored `artifacts/leap` (82 GB), whose
  manifest binds the sha256 of `artifacts/dth/V.npy`.
- The STL leap kernel `src/stl/leap_kernel.py` is numba code. Its first call
  compiles it into the ignored `src/stl/__pycache__/`; `NUMBA_CACHE_DIR`
  moves that cache. `tests/leap_kernel_golden.npz` holds it to the retired
  Rust kernel's outputs.
- One root `pyproject.toml` and one `uv.lock` cover every Python package,
  the browser server included. The deploy bundle takes its package pins from
  that lock. `python -m stl deploy` computes the hosted code version from the
  sha256 of every `src/stl` Python file and of `V.npy`, so any change to
  them ends live hosted sessions once at the next deploy. A leap resume
  refuses a changed Python, numpy, numba or HiGHS version through its
  checkpoint identity.

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
- The `src/ocaml` solver library in `lib/solver/`, its tests, and
  `bin/solve_tablebase.ml` are an independent OCaml implementation of **pure
  DTH**, not STL: actions are literal seconds 1..60 and it has no leap window.
  It exists as a hand-written reference for the exact solver and is held to
  the same frozen rules and the same 1e-6 saddle-gap gate as its Python peer.
  `src/ocaml/play/` holds the CS 3110 STL engine and its terminal UI in the
  same dune package, `dth`. The solver library does not link it.
- `stl.dth` and `stl.dth_kernel` are the paper's solver of **pure DTH**, held
  to the 1e-6 saddle-gap gate, and `stl.dth.Table` is Hal's one DTH
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
npm --prefix web run typecheck
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune runtest --root src/ocaml
opam exec --switch=stl-dth-ocaml -- dune build --root src/ocaml @fmt
cmake -S src/cpp --preset release
cmake --build src/cpp/build/release
ctest --test-dir src/cpp/build/release --output-on-failure
lake -d src/lean exe cache get
lake -d src/lean build
```

The typecheck needs `npm --prefix web install` once. The browser
client is not covered by `pytest`, so skipping it leaves the front end
unchecked.

`pytest` runs distributed at `-n 8`. Add `-n 0` for a readable serial
traceback when something fails. The hosted runtime runs on Python 3.13, so CI
also runs `uv run --locked --python 3.13 python -m pytest tests/server -q -n 0`.

Everything runs by default, including the `slow` tests that build real complete
artifacts end to end — they are the only coverage of resume and backend
parity, and under xdist they cost about 19 seconds. Use `-m "not slow"` for a
tighter loop when iterating on unrelated code, never as the check before a
merge.

After code changes, run `graphify update .`. Use `graphify query`, `path`, or
`explain` for architecture questions when `graphify-out/graph.json` exists.
