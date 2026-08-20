# Air Poker Solver Contract

## Claim

We solve the game identified in `MODEL.md`, not an informal version of Air
Poker. The canonical tree is too large for exhaustive equilibrium computation.
The defensible architecture is therefore:

```text
exact engine
→ exact reduced-game oracle
→ exact-distribution chance sampling
→ external-sampling MCCFR
→ independent evaluation and response diagnostics
```

“Exact” is used only with a named axis:

| Axis | Meaning |
|---|---|
| mechanics-exact | Every transition is delegated to the locked engine. |
| distribution-exact | Chance samples come from the modeled distribution without hand abstraction. |
| traversal-exact | Every history is enumerated; permitted only on reduced games. |
| equilibrium-exact | A certified equilibrium or bound is computed; not currently available for the canonical game. |

The production result will initially be a trained average strategy with measured
error, not a claimed exact Nash equilibrium.

## Game conditions

CFR guarantees require a finite, two-player, zero-sum game with perfect recall.
The locked model is finite and zero-sum by construction. Perfect recall is a
tested obligation:

```text
same player + same InformationState
→ same ordered legal actions
→ same sequence of that player's prior information/action pairs
```

Any violation is an engine/model defect. Splitting or merging information sets
inside the solver is prohibited.

## Strategy and value

A behavioral strategy is a distribution `sigma_i(I)` over the ordered actions
at every information set `I` owned by player `i`. For profile `sigma`, terminal
value is

\[
v_i(\sigma)=\sum_{z\in Z}\pi^\sigma(z)u_i(z),
\qquad v_0(\sigma)+v_1(\sigma)=0.
\]

Reach factors as

\[
\pi^\sigma(h)=\pi_c(h)\pi_0^\sigma(h)\pi_1^\sigma(h).
\]

Policy lookup is keyed only by `(player, InformationState)`. Full state may be
used to traverse nature and apply transitions, never to select a lower action.

## Exact oracle

The oracle runs on the existing tiny Air Poker fixture and on production-engine
fixtures with deterministic or small upper supports. It has four independent
parts.

### Exact profile evaluation

- terminal: return engine utility;
- chance: probability-weight every complete `HandPair` child;
- player: strategy-weight every legal action using the information key;
- missing profile entry: use a declared uniform fallback and report it.

All probability vectors must be finite, nonnegative, correctly sized, and sum to
one within a fixed tolerance.

### Information-set-consistent best response

For fixed `sigma_-i`, the responder selects one action for an entire information
set after aggregating all compatible histories with opponent-and-chance reach.
A responder that chooses separately from hidden full states is clairvoyant and
invalid.

Report

\[
BR_i(\sigma_{-i})=\max_{\sigma_i}u_i(\sigma_i,\sigma_{-i}),
\]

\[
\operatorname{NashConv}(\sigma)
=BR_0(\sigma_1)+BR_1(\sigma_0),
\qquad
\operatorname{exploitability}=\operatorname{NashConv}/2.
\]

These equalities use the engine's zero-sum returns. Exact best response is a
reduced-game metric until canonical feasibility is demonstrated.

### Vanilla CFR

For updating player `i`, the counterfactual regret increment at information set
`I` is

\[
\Delta R_i(I,a)=v_i(\sigma_{I\to a},I)-v_i(\sigma,I),
\]

weighted by chance and opponent reach, not player `i`'s own reach. Contributions
from every history in `I` are summed before the table is updated.

Regret matching uses

\[
\sigma_i(I,a)=
\begin{cases}
R_i^+(I,a)/\sum_b R_i^+(I,b) & \sum_bR_i^+(I,b)>0,\\
1/|A(I)| & \text{otherwise.}
\end{cases}
\]

Average strategies are accumulated with the acting player's own reach. The
average strategy, not the final iterate, is the default reported policy.

### CFR+

The exact reference follows alternating vector-form CFR+:

\[
R_i^{t+1}(I,a)=\max\{R_i^t(I,a)+\Delta R_i^t(I,a),0\}.
\]

Rules:

- update Team 0, then Team 1;
- hold regrets fixed during a complete player pass;
- aggregate a complete information-set delta before clipping;
- use regret-matching+;
- use delayed linear average weight `max(t-delay,0)`;
- fix traversal order for deterministic regression tests.

The current `src/exact/cfr_plus.py` is a candidate reference implementation. It
is not accepted until an independent evaluator and best response show decreasing
exploitability on hand-solvable fixtures.

## Why canonical exhaustive traversal is rejected

On the full deck, individual legal-hand supports range from 28 to 124,156 for
the canonical targets. A first-round joint chance node contains roughly 1.7
million to 8.8 billion exact pairs before any betting branch. Later hidden deck
states multiply the number of histories and information-set members.

Consequences:

- never materialize the canonical joint chance support;
- never build the full canonical tree;
- do not gate production on canonical exact CFR, best response, or sequence-form LP;
- retain exhaustive traversal only as a reduced-game oracle.

## Exact uniform-hand sampling

For available cards ordered as `c_i` with Air Poker values `w_i`, compute integer
completion counts

\[
C(i,k,s)=C(i+1,k,s)+C(i+1,k-1,s-w_i),
\]

with `C(i,0,0)=1` and impossible states zero. Then walk the cards once. At each
step, include `c_i` with probability

\[
\frac{C(i+1,k-1,s-w_i)}{C(i,k,s)}.
\]

This count-and-unrank procedure samples every exact legal mask with probability
`1/C(0,5,n)` without enumerating the output set. Integer counts avoid probability
drift; only the final random choice uses floating point or integer-range sampling.

Required properties:

- count equals `len(generate_legal_hands(n,D))` on enumerable decks;
- every sample contains five distinct available cards with sum `n`;
- impossible targets return no-rank;
- seeded draws are reproducible;
- suit-distinct masks receive equal mass;
- the algorithm does not cache unbounded per-deck hand lists.

For the baseline, sample both marginals against the same pre-round deck and
return the complete pair. The normal-node modeled probability is
`1/(count_0*count_1)` when both supports are nonempty. Retry conditioning follows
`MODEL.md`; zero resolving mass raises a model error.

## Production algorithm: external-sampling MCCFR

One iteration contains an alternating traversal for each updating player.

At a traversal for player `i`:

1. sample each chance node from the exact modeled kernel;
2. at opponent nodes, sample from the opponent's current regret-matched policy;
3. reuse one sampled action if that opponent information set reappears in the
   same traversal;
4. at player `i` nodes, enumerate every legal action;
5. update only player `i`'s ordinary cumulative counterfactual regrets;
6. accumulate a documented sampled estimator of the average strategy.

Version 1 samples from the target distribution itself. There is no alternate
proposal and no importance weighting. Regret clipping experiments must not be
called CFR+ unless a separate sampled-CFR+ derivation and validation are supplied.

External sampling is selected first because it preserves exact lower action
sets while eliminating the canonical chance cross-product and opponent branch.
If updating-player branching is later proven too expensive, outcome sampling or
a documented action abstraction may be evaluated as a new algorithm—not silently
substituted.

## Tables

Each player owns

```python
dict[InformationState, InfoSetEntry]

InfoSetEntry:
    actions: tuple[PlayerAction, ...]
    regret_sum: float64[action_count]
    strategy_sum: float64[action_count]
    visits: int
```

Invariants:

- one information key has one owner and one ordered action tuple;
- all values are finite;
- cumulative sampled regrets may be signed; exact CFR+ regrets are nonnegative;
- unvisited and all-nonpositive entries use uniform regret matching;
- serialization includes an information/action codec version;
- tables from different model fingerprints never merge.

## Evaluation

Training and evaluation use independent seed streams.

| Context | Allowed metric |
|---|---|
| Reduced exhaustive fixture | exact value, best responses, NashConv, exploitability |
| Canonical fixed profile | Monte Carlo estimated value and event rates with confidence intervals |
| Trained response against a frozen canonical profile | empirical response gain or exploitability lower bound |
| Certified canonical response procedure | exploitability bound, with proof conditions stated |

Self-play value near zero, small regret mass, or stable action frequencies alone
do not establish equilibrium.

Canonical reports include:

- model and solver fingerprints;
- iterations, traversals, wall time, and nodes touched;
- information-set count and table bytes by player;
- chance samples, retry rate, folds, showdowns, ties, calamities, and terminal causes;
- mean utility with interval and effective sample count;
- frozen-response gains for both players;
- convergence traces on held-out evaluation seeds.

## Acceptance gates

### Oracle gate

- exact profile values match hand calculations;
- best response is constant within each responder information set;
- NashConv is zero at a known solved fixture and positive for a weak profile;
- vanilla CFR and CFR+ approach the same value;
- average-profile exploitability decreases with training.

### Sampler gate

- DP counts equal exhaustive counts on reduced and selected full-deck targets;
- sampled frequencies pass predeclared statistical tolerances;
- exact and sampled fixed-profile values agree within confidence intervals;
- retry and overlap are sampled only through complete hand pairs.

### MCCFR gate

- fixed seeds reproduce table digests and metric traces;
- reduced-game MCCFR approaches the exact oracle;
- no policy access reads hidden engine state;
- a canonical traversal completes without joint chance enumeration;
- profiling separates sampler, traversal, table lookup, and serialization cost.

### Reporting gate

- evaluation does not mutate training state;
- checkpoint/resume matches uninterrupted seeded training;
- uncertainty from sampling is separated from uncertainty in the behavioral model;
- claims use only the metric labels permitted above.

## Prohibited shortcuts

- keying a lower policy by `GameState`;
- independently sampling winner, poker category, overlap, calamity, or deck removal;
- treating hidden upper outcomes as public observations;
- changing legal action order after a table is created;
- truncating raises without a new model ID;
- declaring convergence from self-play value alone;
- using neural approximation before tabular bottlenecks are measured;
- sharing tables across roots, upper models, observation schemas, or utilities.

## Mathematical references

- Zinkevich et al., *Regret Minimization in Games with Incomplete Information*,
  `papers/NIPS-2007-regret-minimization-in-games-with-incomplete-information-Paper.pdf`.
- Tammelin, *Solving Large Imperfect Information Games Using CFR+*,
  `papers/sswp947.pdf`.
- Lanctot et al., *Monte Carlo Sampling for Regret Minimization in Extensive
  Games*, `papers/NIPS-2009-monte-carlo-sampling-for-regret-minimization-in-extensive-games-Paper.pdf`.

The papers justify algorithmic claims under their assumptions. They do not
justify Air Poker rules or the upper behavioral model.
