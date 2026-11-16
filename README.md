# Usogui solvers

## What it does

Solvers for three games from Toshio Sako's *Usogui*: Surpassing the Leader, Tower of Karma and
Air Poker. Each Rust crate in `crates/` holds one solver and its certificate, ported from the
Python reference and checked against it. `games/` holds the earlier work on each game with its
history.

| Path | Languages | Content |
| --- | --- | --- |
| `crates/` | Rust | The solvers (next table) |
| `games/surpassing-the-leader/` | Python, TypeScript, Lean, OCaml, C++, C | The referee, Hal and the game server; the browser game; the Lean proofs; the OCaml, C++ and C ports of the DTH solver; the papers |
| `games/tower-of-karma/` | Python | The Tower of Karma reference solver |
| `games/air-poker/` | Python | The Air Poker engine and reference solver |

Each directory in `games/` has its own README, environment and tests.

| Crate | Game | What it solves |
| --- | --- | --- |
| `stl-dth` | Surpassing the Leader, pure Drop the Handkerchief | The complete value table: 289,374,121 state classes, each a 60 x 60 matrix game certified to a saddle gap of 1e-6 |
| `stl-leap` | Surpassing the Leader, public leap game | The clock-aware game on top of the DTH table: 2,243 keys, 9.45 billion reachable states |
| `tok` | Tower of Karma (D_K) | The bead-guessing game by CFR+ or a sequence-form LP on public-tree arrays, certified by best responses through the rules engine |
| `ap` | Air Poker | The public-state robust backward induction: an exact round-1 kernel, a sampled kernel with time-uniform confidence boxes, and a certified root interval |

Headline results of the Rust builds on an Apple M5 Pro (15 cores):

| Solver | Root value | Reduction | Wall time, Rust vs Python |
| --- | --- | --- | --- |
| `stl-dth` | 0.0898500728 (saddle gap 8.3e-17) | 5.27 billion raw states to 289,374,121 classes (18.2x) | full build 41 s vs 57 s (numba) |
| `stl-leap` | 0.037175 (Hal wins 51.86%) | 35.5 billion states to 9.45 billion reachable (26.6%) | per key 1.3-2x faster; full audit 67 s vs 147 s |
| `tok` | D_3 in [-0.0590926489, -0.0590917258] | 66,430,111 tree nodes to 3,175,048 information sets | certificate 2-3 s vs 76 s; CFR+ iteration 57 ms vs 0.47 s |
| `ap` | Team 0 in [-0.98425, 0.30460] after one allocation round | 5.2e10 hand pairs to 21,904,800 kernel entries | exact kernel 2.3 s vs 46 s; solve 9 min vs 73 min |

## How to run

Use Rust 1.97 (pinned in `rust-toolchain.toml`) and cmake, which the `highs` crate needs to
build HiGHS. Artifacts go to the ignored `artifacts/` directory.

```sh
cargo run --release -p stl-dth -- build
cargo run --release -p stl-dth -- verify
cargo run --release -p stl-dth -- stage 0 0 0 0
```

`build` writes `artifacts/dth/V.npy` (2.3 GB, SHA-256 `e43977b7...29b2`). The leap solver reads
that table:

```sh
cargo run --release -p stl-leap -- build --dth artifacts/dth/V.npy --output artifacts/leap
cargo run --release -p stl-leap -- audit --artifact artifacts/leap --dth artifacts/dth/V.npy
```

A full leap build writes about 82 GB into an empty directory. `resweep` and `read` check stored
keys and single values.

```sh
cargo run --release -p tok -- solve-cfr 3
cargo run --release -p tok -- certify 3 cfr
cargo run --release -p ap -- exact
cargo run --release -p ap -- allocate
```

Every binary prints its flags with `--help`.

## How to test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace --release -- --ignored
(cd games/surpassing-the-leader && uv run pytest -n 8)
(cd games/tower-of-karma && uv run pytest)
(cd games/air-poker && uv run python -B -m unittest discover -s src/test)
```

The ignored tests read large artifacts: `artifacts/dth/V.npy` from `stl-dth build`,
`artifacts/leap` (a full leap build), `artifacts/tok/cfr_3_p{0,1}.tbl` (see
`crates/tok/tests/data/README.md`), and a 15-20 minute AP allocation re-run.

## Design decisions

One Cargo workspace owns the lints (`clippy::all` and `unsafe_code = "deny"`),
the dependency versions and the toolchain. `CLAUDE.md` states the coding rules.

Each crate proves parity with its Python reference through golden fixtures in
`crates/<crate>/tests/data/`, whose `README.md` names the source commit and the generator
command. The references are stl `55e342e`, TOK `f185b24` and AP `cadd581`. Each commit is in
this repository's history, imported under `games/`.

- `stl-dth` writes a table byte-identical to the numba build. It follows the numba kernel's
  operation order and its `min`/`max` tie rule, so signed zeros match.
- `stl-leap` matches the kernel golden fixture bit for bit and re-sweeps six stored keys bit for
  bit against a fresh Python sweep. HiGHS stages differ by at most 4.9e-11, because the `highs`
  crate has no warm-basis API and each fallback stage solves cold. It reads the DTH table
  through `stl_dth::Table`.
- `tok` rounds 10-action dot products and sums in numpy's order. CFR+ meets exact regret ties,
  and rounding decides them; natural order drifted from Python by up to 0.72 at K = 2.
- `ap` reproduces the sampler streams, the DP counts and the exact kernel bit for bit. Its
  confidence bounds use `libm`'s `lgamma` (std's is unstable) and differ from Python's Apple
  `lgamma` by at most 4.8e-12 at n = 3e8.

The port dropped code that the solvers do not need: the stl referee, Hal and server; the leap
warm-start, edge-move and support-seed paths that the full build never enables; TOK's object
tree, which duplicated the array path; and AP's sensitivity study and tiny exact test game.
Those parts stay in `games/`.

Dependencies: `clap`, `highs`, `memmap2`, `rayon`, `serde`, `serde_json`, `libm` (ap), and
`sha2` for tests.

## Status

All four solvers are ported and pass their parity tests. The repository is local and has not
been pushed.

- No full leap rebuild has run in Rust. The audit and the key re-sweeps cover it.
- The TOK D_3 LP (about 31 million rows) has not been solved in Python or Rust.
- AP self-play draws mixed strategies from its own counter-based stream, so the iteration-1
  reach differs from Python (408 vs 399 sequences). The root interval does not depend on it.
