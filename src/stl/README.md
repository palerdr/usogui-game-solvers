# STL Project Instructions

This subtree contains the STL referee and the L2 leap builder. We require the full artifact audit before claiming an L2 solve. We do
not provide a learned model, opponent model, or gameplay agent. Git history is the archive for
the removed experiments; do not restore them under `legacy`, `old`, or
version-suffixed names.

## Retained surfaces

- `engine/` is retained only as the compatibility interface currently consumed
  by `src/arena/`. It is a behavioral reference for already-frozen rules, not
  the canonical full STL formulation and not a foundation to extend casually.
- `tests/engine/` protects that compatibility interface.
- `solver/leap_profiles.py` owns the STL quotient, the turn duration, and
  the child clocks.
  `solver/leap_oracle.py` certifies scalar stages; `solver/leap_build.py` owns
  reachability and the full sweep checkpoint.
  `solver/leap_kernel.py` is the numba leap kernel that the sweep calls.
  `solver/leap_oracle.py` and `solver/leap_support.py` remain its scalar
  authorities, and `tests/test_leap_golden.py` holds it to the outputs of the
  retired Rust kernel.
- `reader.py` is the public read path into a completed leap artifact.
  Other projects and the paper's
  figure scripts read leap tables through it, not through `solver/`.
  `open_leap()` checks the manifest schema, the complete flag, and the SHA-256
  of the DTH tail before it returns a table. `verify="files"` also re-hashes
  each file that the manifest lists. `LeapTable.stage()` re-certifies one
  stored value with HiGHS and raises when the Bellman residual or the saddle
  gap exceeds 1e-6. Default paths resolve from this package, or from the
  absolute paths in `STL_LEAP_ARTIFACT` and `STL_LEAP_DTH`. The module opens
  the artifact read-only. It sits outside `solver/`, so the builder hash does
  not cover it; keep it there. `tests/test_reader.py` reproduces the paper's
  16 H1 records and skips when `outputs/leap-full-native/` is absent.
- Repository-wide canonical rules and formulation contracts belong in root
  `docs/`. Generated experiment data remains gitignored and STL-owned.

Read `docs/GAME_AND_SOLVER.md` for STL's full-game formulation and intended
solver architecture. It specializes the repository-wide contracts without
redefining their shared mechanics.

## Rebuild rule

Freeze the full game's state, observations, actions, transition/chance model,
and utility in the canonical documentation before adding implementation code.
Then add the smallest implementation that realizes that contract. Do not add
solver, learning, or play modules merely to preserve an earlier experiment.

Run `uv run python -m pytest src/stl/tests -q` for the retained surface.

## Leap build

The leap kernel is `solver/leap_kernel.py`. numba compiles it on first use,
in 14 to 35 seconds, and caches the machine code in the ignored
`solver/__pycache__/`. The kernel passes its inputs read-only, so writable
arrays and the memmapped child tables of a build share one compiled version.
A change to the kernel source or to the numba version in `uv.lock`
invalidates that cache. Set `NUMBA_CACHE_DIR` to move it, for example to a
disk that a container keeps.

For the full public game, run
`uv run python -m stl.solver.leap_build --full --min-clock 720 --output <dir>`.
This command sweeps through the clock-720 opening. Resume uses the same
command and requires matching source, DTH, and checkpoint file hashes.
The builder commits packed-file hashes before removing each cold dense table.
You can read the opening value and LP-certified opening strategies in
`report.json`. The audit checks packed membership and finite values for all
reachable states, then checks sampled Bellman equations with HiGHS. It records
the audit sample count and residuals in `audit.json` and the final manifest.
For fallback stages, workers first solve the triangular packing LP in the
numba kernel and check both saddle bounds on the original matrix. Python sends rejected
stages to `highspy`. Those workers reuse HiGHS bases and cache certified revival
values only when the child tables, profiles, and window flag match. The report
counts kernel failures, new LP solves, and cache hits as separate quantities.
The scalar LP retries with HiGHS' interior-point method if the first LP attempt
fails its status or saddle-gap check. If both methods fail, it retries IPM after
subtracting a common payoff constant. It checks each candidate against the
original matrix at the same 1e-6 gate.
The scalar retry sequence uses the failure payoff, then the first success
payoff, as common offsets if earlier attempts fail.

The full sweep solves H1 and H2 residue inside the kernel, as
`leap_build.RESIDUE` selects. Each 1024-class chunk moves the certified
support of its preceding residue class with the success-payoff kink and tries
it once. On a miss, the chunk runs the packing LP from the recurrence crash
basis. REV keys keep the Python fallback and its revival cache, and their
packing LP also starts from the crash basis. Every stored value still passes
the full 1e-6 matrix gate; [`LEAP_CERTIFICATE.md`](docs/LEAP_CERTIFICATE.md)
holds the equations. Records count kernel LP solves as `kernel_native_solves`
and support hits as `support_solves`, and `failures` still counts all
residue. The kernel time then includes the residue LPs, so compare residue
paths on sweep time. A change to `RESIDUE` changes the builder hash, so a
checkpoint cannot resume across it.

The packed bitmap excludes the implicit WIN column. Values follow its set
bits in row-major order. Each dense table keeps NaN in unreachable cells and
-1 in its last column. The manifest records file hashes and marks this segment
as incomplete. A complete L2 claim still requires the full audit.

The slow leap tests require `src/dth_compact/artifacts/V.npy`. Run
`uv run python -m pytest src/stl/tests -q`; its first run compiles the kernel
in each worker. The full reachability test checks the measured state totals
without performing a value sweep.

## Numba kernel

On 2026-10-23 we replaced the Rust `stl_solver_rs` leap kernel with the numba
port in `solver/leap_kernel.py`. The port keeps the Rust operation order. It
writes `f64::mul_add`, `f64::min`, and `f64::max` as `llvm.fma`,
`llvm.minnum`, and `llvm.maxnum`, and it orders pivots by the IEEE total
order, as `f64::total_cmp` did. It keeps the 1024-class chunks, the 16-lane
batches, and the per-chunk packing tableau and kink seed.

Before the retirement we compared this kernel with `stl_solver_rs` on
113,175,708 rows in 725 cases: square and window stages under 13 option sets,
all 10 residue samples under `outputs/leap-residue/` in Dropper-major and
shuffled order, seeded stage families, random child tables, the support-seed
API, the packing LP, and the kink helpers. Values, kinds, failure counts, and
basis masks agreed bit for bit. Every sample value stayed within 4.97e-7 of
its certified value, inside the 1e-6 gate. The engines differed only on
malformed arguments: the kernel accepts an `int` for a `bool` and a `float`
for an integer, it rejects Fortran-ordered `supports`, `support_out`, and `s`
arrays, which Rust read in memory order, and its error messages differ.

After the port moved here, we ran it and the last `stl_solver_rs` build on
the H1_44 and H2_42 samples on AC power with 15 threads. Both engines wrote
the same bits in four workloads: the rungs alone, the production options of
`RESIDUE` in Dropper-major and shuffled order, and the window path. With the
production options, H1_44 took 1.08 s in numba against 1.26 s in Rust, and
H2_42 took 4.31 s against 7.64 s. An earlier run of the port on lower power
took 2.13 s against 2.70 s and 3.66 s against 6.74 s. The window path, where
the leap row wins every class, took 0.08 s against 0.04 s. A cold cache adds
a compile of 14 seconds on AC power, and up to 35 seconds on battery, to the
first calls.

`tests/data/leap_kernel_golden.npz` keeps 1,940 seeded stage classes and
1,100 random-table classes with the Rust outputs: square and window stages,
the rungs alone and the production options that `RESIDUE` held at the
capture, the packing LP from the slack and crash bases, and the warm and cold
packing runs of the benchmarks. `tests/test_leap_golden.py` requires equal
kinds, failure counts, masks, and values. It runs the production cases with
the captured options, so a change of `RESIDUE` keeps the check. It also reads
the child tables as read-only memmaps. The commit that added the fixture
holds `tests/make_leap_golden.py`, which captured it.
