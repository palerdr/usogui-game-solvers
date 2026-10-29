# Air Poker Solve Targets

## Headline

One certified root value for Team 0 (Baku) in `canonical-uniform-v1`, with the
equilibrium strategy, the public-game state count against the raw count, peak
memory, and wall time. The value is exact in round 1 and a certified interval
afterward, because the hidden deck posterior cannot be summed exactly.

## Reformulation

`InformationState` in `src/engine/game.py` holds own resources, the opponent's
sealed-BIOS count, both revealed numbers, the betting state and the complete
round history. The opponent's unused metal cards follow from the fixed root
plus history. Residual air is a public constant at the canonical root. After
both numbers are revealed, the two lower players therefore hold the same
information set. The only private moment is the simultaneous number choice,
which closes at reveal.

Consequences:

- The game is a public-information stochastic game. Equilibrium follows from
  backward induction over public states.
- Each round is a matrix game over numbers (at most 5x5, solved by LP), a
  chance draw whose lower-visible content is one of five result classes (win 0,
  win 0 with calamity, win 1, win 1 with calamity, tie), then a
  perfect-information betting street. A fold hides the class.
- Betting needs no mixing. Number choice does.
- The hidden hands enter only through the result kernel: the probability of
  each result class at the next chance node, given the public chance prefix
  (numbers and result classes so far) and the next number pair.

This supersedes the MCCFR production path in `SOLVER.md` for
`canonical-uniform-v1`. MCCFR remains the tool for models that give the lower
players private information, such as `private-root-v1`.

## Targets

| Output | Target | Certificate |
|---|---|---|
| Root value for Team 0 | interval of width at most 0.02 at confidence 99.9%; stretch 0.01 | robust-game backward induction: nature picks adversarially inside per-entry kernel confidence rectangles for the lower bound and favorably for the upper bound |
| Round-1 kernel | exact | enumeration of 52,341,125,760 hand pairs over the 25 number pairs; suit orbits cut the work by 24 |
| Equilibrium strategy | mixed number distribution at every public round-start state, deterministic betting policy | saddle gap at most 1e-9 at every matrix stage; the exact best response reports NashConv 0 under the point kernel |
| Engine agreement | 10^7 self-play games on independent seeds land inside the interval | the reformulation never leaves engine transitions |
| Result sensitivity | reach-weighted total variation between result-conditioned and number-only kernels, per round | measured 2026-10-29, see below; prefixes cannot be collapsed |
| Resources | wall time and peak memory on the build machine, kernel table bytes, trajectories sampled | manifest |

## Measured reductions

All counts come from the engine's own `legal_actions` and `apply`.

| Reduction | Before | After |
|---|---|---|
| Hidden hands leave the strategy state | 5.2e10 hand pairs at the first chance layer alone | 21,904,825 kernel entries of five probabilities |
| Betting history to (contributions, actor, check flag) | 5,315,068 tree nodes in round 1 | 290 DAG states per round-1 configuration |
| Chip and first-bettor configurations | 47 x 47 x 2 possible | 1,458 reachable at round 5 |
| Public game, results in the key | | 6,903,045,650 round-start states; 510,273,828,900 betting DAG nodes |
| Public game, numbers only | | 44,960,050 round-start states; 2,852,337,700 betting DAG nodes |
| Suit symmetry S4 | | factor 24 on every kernel computation |

Depth-first order over number sequences keeps live memory near the kernel
table, about 1 GB. No value table is stored.

## Not symmetries

- Round order is not exchangeable. The upper policy is uniform over legal hands
  of the depleted deck, and that normalizer depends on the removed cards, so a
  permuted round sequence has a different posterior.
- Poker rank does not decompose over cards. No dynamic program replaces
  enumeration for winner probabilities. The sum constraint alone decomposes,
  which is why the DP count in `SOLVER.md` works for legal-hand counts only.

## Measured result sensitivity (2026-10-29)

`src/sensitivity/measure.py`, exact uniform sampler, results in
`docs/results/sensitivity-full.json` (60 random number sequences, 100,000
trajectories each) and `sensitivity-round2.json` (all 400 round-1 x round-2
number combinations, 50,000 each). The null column is the same statistic under
independence with the same sample sizes.

| Round | Mean weighted TV | Null | Max over sequences |
|---|---|---|---|
| 2 (400 combinations) | 0.016 | 0.005 | 0.125 |
| 3 | 0.027 | 0.005 | 0.097 |
| 4 | 0.045 | 0.008 | 0.119 |
| 5 | 0.090 | 0.012 | 0.284 |

Past results move the kernel by far more than the interval target, so the
number-only collapse is rejected. The kernel stays keyed by numbers and result
classes: 21.9M entries, 6.9e9 public round-start states.

Two further facts came out of the same run:

- Ties are impossible. Equal poker rank forces an equal rank multiset and so an
  equal sum, and the ten metal cards are distinct. Four result classes remain.
- Round 5 can reach a deck on which both remaining targets are impossible.
  MODEL.md M10 calls zero resolving mass a model error and the engine raises
  there. Rate across random sequences: mean 2.8% of trajectories, maximum
  45% for the sequence (39,47),(45,8),(36,15),(26,44),(25,63). The extreme
  targets 8, 15 and 63 consume the aces, deuces and kings early, and 63 needs
  three kings. A strategy can steer into this state. Ruled 2026-10-29 (M10):
  the round ends with no judgement, antes return, play continues. The sampler
  records it as result class `N`.

## Exact round-1 kernel (2026-10-29)

`src/kernel/exact.py` enumerates every pair in `L(D,n0) x L(D,n1)` on the full
deck: 52,341,125,760 hand pairs over the 25 number pairs, 46 s wall on 14
cores, integer counts in `docs/results/kernel-round1.json`. Parity:
`src/test/test_exact_kernel.py` matches the engine's own `iter_chance_outcomes`
on 19-card decks including one-sided and two-sided no-rank, and all 100
sampled class probabilities from the sensitivity run (800,000 samples per
pair) fall within 2.9 standard errors of the exact values.

Probability that Team 0 wins the judgement, with or without calamity:

| Baku \ Lalo | 8 | 15 | 44 | 47 | 63 |
|---|---|---|---|---|---|
| 25 | 0.0155 | 0.1713 | 0.5085 | 0.4484 | 0.0006 |
| 26 | 0.0039 | 0.1442 | 0.4695 | 0.4096 | 0.0001 |
| 36 | 0.0024 | 0.1066 | 0.3693 | 0.3035 | 0.0001 |
| 39 | 0.0025 | 0.1132 | 0.3898 | 0.3130 | 0.0001 |
| 45 | 0.0140 | 0.1640 | 0.5509 | 0.4319 | 0.0002 |

Probability of calamity (the loser pays its contribution again):

| Baku \ Lalo | 8 | 15 | 44 | 47 | 63 |
|---|---|---|---|---|---|
| 25 | 0.686 | 0.604 | 0.322 | 0.292 | 0.128 |
| 26 | 0.659 | 0.586 | 0.331 | 0.304 | 0.151 |
| 36 | 0.374 | 0.393 | 0.419 | 0.423 | 0.431 |
| 39 | 0.291 | 0.332 | 0.445 | 0.457 | 0.523 |
| 45 | 0.138 | 0.210 | 0.497 | 0.525 | 0.699 |

Lalo's extreme numbers are near-certain wins: 8 forces three or four aces, 63
forces three or four kings. Baku's numbers beat only 44 and 47, and only about
half the time. Calamity runs from 13% to 70% of judgements.

## The risk

Sampling 21.9M result-keyed entries to half-width 0.005 each needs about 1e13
trajectories. The CPU sampler reaches 14,000 five-round trajectories per
second on 14 cores. Samples must be allocated by reach, and the sampler must
move to a GPU or a compiled kernel.

## Order of work

1. Result-sensitivity measurement with an exact uniform sampler.
2. Exact round-1 kernel.
3. Sampled kernels for rounds 2 to 5 with rigorous per-entry intervals,
   samples allocated by reach.
4. Robust backward induction over public states.
5. Certification: saddle gaps, best response, engine self-play.
6. Report in the form of this file's headline.
