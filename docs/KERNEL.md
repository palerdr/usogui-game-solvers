# Kernel sampler for rounds 2 to 5

## What it produces

For every public chance prefix `h` (numbers and result classes of the rounds
played so far) and every next number pair `(a, b)`, one entry:

```text
counts[h, a, b] : int64[5]     # W0, W0c, W1, W1c, N; T is unreachable
n[h, a, b]      : int64        # samples that reached h and played (a, b)
```

The robust backward induction in `GOALS.md` reads an entry as a confidence set
for the kernel `K(r | h, a, b)`, never as a point estimate. The sampler's only
promises are that every sample is an exact draw from the modeled chance
process and that the confidence sets hold simultaneously with probability at
least `1 - delta` under the stated recipe. Everything about value error is the
solver's job.

## Three structural facts the design rests on

1. **A fold hides a class; it does not change the chance process.** The
   kernel after a hidden round is the mixture of the kernels after each
   revealed class, weighted by the kernel of that round. Samples for a hidden
   prefix are therefore the union of the samples for its revealed prefixes,
   and the counts add. Only revealed prefixes are sampled:

   | Round of the entry | Revealed prefixes | Next pairs | Entries |
   |---|---|---|---|
   | 2 | 125 | 16 | 2,000 |
   | 3 | 10,000 | 9 | 90,000 |
   | 4 | 450,000 | 4 | 1,800,000 |
   | 5 | 9,000,000 | 1 | 9,000,000 |
   | total | | | 10,892,000 |

   Hidden-class prefixes bring the total the solver reads to 21,904,800, all
   pooled from the same counts. `N` has positive mass only where both targets
   are impossible, so most `N` prefixes before round 5 have zero reach and
   stay empty.

2. **One trajectory samples every prefix on its path at its chance reach.**
   Simulating five rounds along a fixed number sequence `s` and recording the
   class of each round adds one sample to the round-2, 3, 4 and 5 entries of
   the prefixes it passes through. A prefix with chance reach `rho` under `s`
   receives `N_s * rho` samples in expectation. That is the allocation the
   value bound wants within a sequence, with no rejection and no targeting.

3. **Across sequences the right weights are equilibrium reach, which is
   unknown until the solve.** An unsampled entry carries the trivial
   confidence set, the whole simplex, which is valid. The solve is therefore
   valid from zero samples and tightens wherever samples land. Allocation
   across number sequences is iterated: solve, read the reach of each
   sequence under both robust equilibria, allocate the next batch
   proportionally with a uniform floor, resample, solve again.

## Sampling one hand

Uniform over `L(D, n)` by count-and-unrank over card values, not cards. For
value `v = 1..13` the deck holds `a_v` cards (0..4). Completion counts

```text
C[v][k][s] = sum over j = 0..min(a_v, k) of binom(a_v, j) * C[v+1][k-j][s - j*v]
C[14][0][0] = 1, otherwise C[14][.][.] = 0
```

give `C[1][5][n] = |L(D, n)|`. Unranking draws an integer `u` in `[0, C[1][5][n])`
and walks `v = 1..13`, choosing `j` by the cumulative counts, then chooses
which `j` of the `a_v` available suits uniformly. The table is 13 x 6 x 65
integers, built once per `(D, n)`, about 2e4 operations. Every legal mask has
probability exactly `1 / |L(D, n)|`, and a zero count is a no-rank.

A round draws both hands against the same pre-round deck, resamples a double
no-rank while a resolving outcome exists, settles `N` when none does, and
removes the exact union. This is `sample_round` in `src/sensitivity/sampler.py`
with the filter replaced by the DP; the classification does not change.

## Compute path

| Stage | Where | Throughput target | Why |
|---|---|---|---|
| Reference | NumPy filter sampler, exists | 1.4e4 trajectories/s on 14 cores | parity oracle |
| CPU | the DP sampler in a numba hot loop (numba 0.68.1 supports the pinned Python 3.14) | at least 1e5 trajectories/s on 14 cores | overnight budgets of 1e10 |
| GPU | the same DP as a CUDA kernel, one thread per trajectory, counts accumulated with atomic adds into a dense array | at least 2e6 trajectories/s on the laptop 4070 | budgets of 3e10 in hours |

The dense count array is indexed by number-sequence id (14,400) times the
base-5 code of the revealed classes so far; through round 4 that is 9,000,000
rows of five `int64`, 360 MB, which fits device memory. Nothing per trajectory
is stored.

Pseudo-random numbers come from a counter-based generator keyed by
`(run seed, batch, trajectory index)`, so a batch is reproducible on either
device and independent across batches.

## Parity gates

Each gate runs on the reference sampler, the CPU path, and the GPU path.

1. **Counts.** `C[1][5][n]` equals `len(generate_legal_hands(n, D))` for every
   canonical target on the full deck and on 200 seeded depleted decks, and is
   zero exactly when the enumeration is empty.
2. **Bijection.** On decks small enough to enumerate, unranking every integer
   in `[0, count)` yields every legal mask exactly once.
3. **Round 1 against the exact kernel.** 1e8 full-deck trajectories per
   number pair; all 100 class probabilities within 4 standard errors of
   `docs/results/kernel-round1.json`.
4. **Later rounds against the exact counter.** For 50 seeded depleted decks,
   the conditional class frequencies agree with `exact_kernel_counts` on that
   deck within 4 standard errors.
5. **Settlement parity.** Sampled rounds replayed through the engine produce
   the same `RoundRecord` class, including one-sided no-rank and `N`.
6. **Device agreement.** CPU and GPU class frequencies for the same batch
   specification agree within sampling error, and both pass gates 1 to 5.

## Confidence sets

Per entry and class, a time-uniform confidence sequence from the counts alone,
so the sets stay valid under the adaptive allocation of fact 3 at any sample
size. Each class indicator is a Bernoulli variable, so its likelihood is known
exactly and the Beta(1/2, 1/2) mixture likelihood ratio is a mean-one
martingale (Robbins' mixture method); Ville's inequality makes

```text
C_n = { p : p^k (1 - p)^(n - k) >= alpha * B(k + 1/2, n - k + 1/2) / pi }
```

cover `p` at every `n` simultaneously with probability at least `1 - alpha`.
This replaces the empirical-Bernstein form named in the first draft: for an
indicator it is exact, has no slack constants, and shrinks near 0 and 1 as
`(log(1/alpha) + log(n)/2) / n`, the behavior the design wanted, since most
entries sit near 0 or 1 (63 and 8 almost always win). The per-entry level is
`delta / E`, with `E` the 21,904,800 entries the solver reads, by union bound,
and each of the five reachable classes takes a fifth of it; the tie class is
pinned to zero because ties are impossible. Pooled hidden-prefix entries
receive their own sets from their pooled counts.

The solver turns the five one-dimensional sets into one confidence set on the
simplex, the box intersected with the simplex; `tighten` applies the implied
bounds `p_i >= 1 - sum of the other upper ends`. The adversarial and favorable
kernel choices inside that set are closed-form: sort the children by value
and push mass toward the worst or best classes within the box.

## Budget

Within a number sequence, the round-`k` contribution to the half-width of the
root value is bounded by `c * sqrt(P_k / N_s)`, where `P_k` is the number of
positive-reach prefixes at round `k` and `c` collects the spread of values
and the confidence constant. The round-5 term dominates with `P_5` at most
625 per sequence. Reaching 0.01 for the sequences that carry equilibrium
reach needs about 6e8 trajectories each. If the two robust equilibria put
their mass on about 50 sequences, that is 3e10 trajectories, four hours on
the GPU target and three days on the CPU target. The uniform floor over all
14,400 sequences at 1e6 each adds 1.4e10. Both figures are planning numbers;
the solve reports the achieved width.

## Order of work

| Slice | Deliverable | Gate |
|---|---|---|
| K1 | DP count and unrank, CPU hot loop | gates 1, 2; throughput |
| K2 | trajectory sampler and dense count accumulation | gates 3, 4, 5 |
| K3 | GPU port | gate 6 |
| K4 | confidence sets and pooled hidden entries | coverage on 19-card decks against the exact counter over 1,000 seeds |
| K5 | adaptive allocation loop with the robust solver | width falls with budget; seeds reproduce |

K1 to K4 run on this machine. K3 needs the laptop and its CUDA toolchain.

## K1 result (2026-10-29)

`src/kernel/dp.py` holds `fill_value_counts`, `fill_table`, `count` and
`unrank` as numba functions with caller-owned workspaces, and
`src/test/test_kernel_dp.py` holds gates 1 and 2: counts equal
`generate_legal_hands` on the full deck, on 200 seeded decks of 5 to 42 cards
and for every target 0..69 on eleven small decks; unranking every integer in
`[0, count)` yields every legal hand exactly once on 21 decks of 8 to 19 cards
for every target 6..64. `uv run python -B -m src.kernel.dp` builds one table
and unranks ten hands per deck over 200,000 seeded decks of 12 to 52 cards:

| Threads | Tables/s | Unranks/s | Time per table and ten unranks |
|---|---|---|---|
| 1 | 2.5e5 | 2.5e6 | 4.0 us |
| 15 | 2.8e6 | 2.8e7 | 0.36 us |

A trajectory needs five tables and ten unranks plus retries, so the DP alone
supports about 5e5 trajectories/s on this machine, five times the K2 target.
Allocating the table inside the parallel loop limited scaling to 3.2x on 15
threads; one workspace per thread restored 11x.

## K2 result (2026-10-29)

`src/kernel/sampler.py` plays trajectories in a numba parallel loop:
`play_round` builds the deck's table, draws each hand by `bounded` (top bits
of a 64-bit word with rejection, so exactly uniform) and `unrank`, classifies
with `evaluate5` (a port of the engine evaluator, equal on every canonical
legal hand) and removes the union. The generator is splitmix64 on a state
keyed by `(seed, batch, index)`. `sample_sequence` returns the dense
`781 x 6` count array of one number sequence, one row per revealed prefix of
each round; `sample_rounds` counts single rounds on any deck.

`src/test/test_kernel_sampler.py` holds the gates at unit scale, each with a
fixed seed: gate 3 on 4e5 rounds per pair and on the round-1 row of 2e5
trajectories; gate 4 on 50 seeded decks of 14 to 32 cards at 1e5 rounds each
against `exact_kernel_counts`; gate 5 on 204 sampled rounds replayed through
the engine, class and next deck equal, all five reachable classes seen. The
dense counts are consistent across rounds (each child row sums to its parent
cell) and reproduce under the same seed and batch.

Gate 3 at full scale, `uv run python -B -m src.kernel.sampler round1`:

| Rounds per pair | Cells | Max abs z | Mean abs z | Wall |
|---|---|---|---|---|
| 1e8 | 150 | 2.33 | 0.51 | 1,006 s on 15 threads |

The run is `docs/results/kernel-round1-parity.json`. Throughput of
`uv run python -B -m src.kernel.sampler bench`, five-round trajectories:

| Threads | Trajectories/s |
|---|---|
| 1 | 6.3e4 |
| 15 | 7.2e5 |

That is seven times the CPU target, so the 3e10-trajectory budget is twelve
hours on this machine without the GPU.

## K4 result (2026-10-29)

`src/kernel/confidence.py` holds `interval` (bisection on the mixture
confidence sequence above, endpoints rounded outward), vectorized `intervals`,
`boxes` and `tighten`. `src/kernel/table.py` holds `Sequences` (the 14,400
number sequences and the index of every number prefix: 25, 400, 3,600,
14,400, 14,400) and `KernelCounts`, which adds a sequence's `781 x 6` sampler
output into the entries of its number prefixes and pools hidden positions by
one summation per position over base-6 class codes; `entries` is 21,904,800,
as counted by hand in the first draft.

`src/test/test_kernel_confidence.py` holds the gate: 1,000 seeded 19-card
decks with a random canonical pair, 2,000 single rounds each, the five
reachable class intervals at level `0.05 / 5` checked against
`exact_kernel_counts` at every sample size from 1 to 2,000. Seeds with any
violation: 16 of 1,000, against an allowance of 50; mean final width 0.031. The
table test adds three sampled sequences sharing two rounds and checks the
aggregation and the pooled rows against sums of revealed rows.

Boxes for all 18,662,400 round-5 entries at `delta = 1e-3` take 1.6 s on 15
threads and 0.9 GB per bound array.
