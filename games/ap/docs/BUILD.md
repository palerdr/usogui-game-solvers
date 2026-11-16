# Air Poker Build Order

## Delivery rule

Build one narrow slice at a time. A slice ends with executable evidence and an
acceptance gate. Later slices may not compensate for an earlier failed gate.

```text
B0 repository
→ B1 model closure
→ B2 traversal boundary
→ B3 exact evaluation
→ B4 exact response
→ B5 exact regret solvers
→ B6 exact hand sampler
→ B7 sampled chance
→ B8 MCCFR
→ B9 canonical runner
→ B10 evaluation
→ B11 scale decisions
→ B12 behavioral models
```

## Current checkpoint

| Area | State |
|---|---|
| Poker mathematics | Implemented; exhaustive five-card evaluator coverage. |
| Game engine | Implemented; immutable transitions and information projection. |
| Tiny Air Poker | Implemented as a small exhaustive fixture. |
| CFR+ candidate | Implemented in `src/exact`; not independently certified. |
| Best response/exploitability | Missing. |
| Exact hand sampler | Missing. |
| Production MCCFR | Missing. |
| Canonical trained strategy | Missing. |

## Target layout

```text
src/
├── engine/
│   ├── game.py
│   └── poker_math.py
├── exact/
│   ├── tiny_air_poker.py
│   └── cfr_plus.py
├── solver/
│   ├── interface.py
│   ├── tables.py
│   ├── evaluate.py
│   ├── best_response.py
│   ├── cfr.py
│   ├── chance.py
│   ├── mccfr.py
│   ├── checkpoint.py
│   └── run.py
└── test/
    ├── test_poker_math.py
    ├── test_game.py
    ├── test_exact_cfr_plus.py
    ├── test_solver_interface.py
    ├── test_solver_evaluate.py
    ├── test_solver_best_response.py
    ├── test_solver_cfr.py
    ├── test_chance_sampler.py
    ├── test_mccfr.py
    └── test_checkpoint.py
```

Files are created only in the slice that needs them. Do not scaffold empty
modules.

---

## B0 — Repository baseline

### B0.1 Package rename

**Goal:** make `src` the single Python package root.

**Work**

- move the complete legacy source tree to `src/`, including uncommitted exact work;
- rewrite imports to the `src.*` package;
- rewrite test, lint, and documentation paths;
- preserve package `__init__.py` files;
- reject remaining live references to the old path.

**Test**

```powershell
rg -n "from c[o]de\.|c[o]de/test|c[o]de/engine|c[o]de/solver" README.md docs src
.\.venv\Scripts\python.exe -B -m unittest discover -s src/test -v
ruff check src
```

**Gate:** imports, all tests, and lint pass from the repository root.

### B0.2 Remove obsolete toy package

**Goal:** remove the standalone `kuhn/` implementation and its bytecode.

**Work**

- delete only the resolved workspace directory `kuhn/`;
- remove active documentation and test dependencies on it;
- retain small analytic tests only when embedded in solver test modules and
  directly useful to a contract.

**Gate:** the directory is absent and no active build stage depends on it.

### B0.3 Documentation authority

**Goal:** make the four control documents discoverable.

**Work**

- link `EVIDENCE.md`, `MODEL.md`, `SOLVER.md`, and `BUILD.md` from `README.md`;
- keep `ENGINE.md` as the low-level executable mechanics contract;
- keep `Air-Poker-Summary.md` and `docs/papers/` as supporting sources;
- ensure assumption claims point from evidence to model, never in reverse.

**Gate:** a new contributor can identify source facts, selected assumptions,
mathematical claims, and implementation order without reading source code.

---

## B1 — Model closure

### B1.1 Root manifest

**Goal:** serialize the exact `canonical-uniform-v1` root.

**Files:** `src/solver/interface.py`, `src/test/test_solver_interface.py`.

**Work**

- record both ordered metal-card tuples;
- record residual air `(300,300)`;
- record start-after-open-and-ante semantics;
- record upper policy IDs;
- compute a stable model fingerprint from canonical JSON;
- fail on unknown or omitted fields.

**Test**

- repeated construction produces the same bytes and hash;
- changing one card, one air second, policy ID, utility rule, or schema version
  changes the hash;
- tuple ordering is preserved;
- dirty source state is recorded separately from the model hash.

**Gate:** every future table and artifact can name the game it belongs to.

### B1.2 Reachability and terminal closure

**Goal:** prove that test fixtures never fall out of the model.

**Work**

- traverse deterministic and small-support engine fixtures exhaustively;
- assert each state is exactly terminal, chance, or player;
- assert every nonterminal node has a successor;
- assert terminal returns are finite, zero-sum, and in `{-1,0,1}`;
- audit inability-to-pay, Round-5, tie, fold, showdown, calamity, and no-rank paths;
- explicitly test retry zero-resolving-mass rejection.

**Gate:** no fixture relies on an implicit fallback transition.

### B1.3 Information and perfect recall

**Goal:** certify the lower information partition on exhaustive fixtures.

**Work**

- retain each player's own sequence of `(InformationState, action)` pairs;
- map every information key to actor, ordered actions, and own-memory signature;
- compare histories with different hidden hands and decks but equal observations;
- inject sentinel hidden values and verify they never enter a policy key;
- verify Team 1 cannot observe Team 0's current choice before reveal.

**Gate:** every repeated key has one owner, one action tuple, and one own-memory
signature.

**Out of scope:** changing the observation schema to make a solver easier.

---

## B2 — Traversal boundary

### B2.1 Node adapter

**Goal:** give algorithms one thin engine-owned interface.

**Files:** `src/solver/interface.py`, `src/test/test_solver_interface.py`.

**Interface**

```python
root() -> State
kind(state) -> TERMINAL | CHANCE | PLAYER
actor(state) -> Team | None
player_actions(state) -> tuple[Action, ...]
chance_outcomes(state) -> Iterable[(HandPair, probability)]
next_state(state, action) -> State
information_key(state, team) -> InformationState
returns(state) -> tuple[float, float]
```

**Work**

- delegate directly to public engine functions;
- assert phase preconditions without repairing states;
- preserve engine action order exactly;
- expose counters outside `GameState`;
- support injected roots and upper policies for tests.

**Test**

- every node kind maps correctly;
- every legal action round-trips through `apply`;
- illegal actions propagate engine errors;
- exact chance probabilities preserve engine normalization;
- counters do not affect hashes, observations, or transitions.

**Gate:** solver algorithms can run without importing engine internals.

### B2.2 Traversal audit counters

**Goal:** measure fixture shape before optimization.

**Counters**

- histories and terminals;
- nodes by kind and actor;
- information sets by player;
- legal actions by phase;
- chance support size;
- maximum depth;
- repeated histories and information keys.

**Gate:** fixture counts are deterministic snapshots reviewed with every rule
change.

---

## B3 — Exact profile evaluation

### B3.1 Strategy table contract

**Goal:** centralize policy validation and fallback behavior.

**Files:** `src/solver/tables.py`, `src/test/test_solver_evaluate.py`.

**Work**

- define `InfoSetEntry` with immutable ordered actions;
- store float64 regret and strategy arrays;
- validate shape, finiteness, nonnegativity where required, and normalization;
- implement uniform fallback for unvisited entries;
- report fallback counts instead of hiding them;
- prohibit action reordering on repeated lookup.

**Gate:** malformed profiles fail at creation or lookup, not midway through a run.

### B3.2 Exhaustive evaluator

**Goal:** compute the exact value of a fixed profile on enumerable games.

**Files:** `src/solver/evaluate.py`, `src/test/test_solver_evaluate.py`.

**Work**

- probability-weight complete chance children;
- strategy-weight player children;
- use compensated summation where large supports make accumulation error visible;
- verify zero-sum value at the root;
- expose node counters and fallback counts;
- keep evaluation read-only with respect to strategy tables.

**Fixtures**

- a one-decision hand-calculated game inside the test module;
- default tiny Air Poker;
- production engine with deterministic hands;
- production engine with a two-outcome hidden hand-pair policy;
- tie, fold, overlap/calamity, no-rank, and retry terminals.

**Gate:** evaluator outputs match direct hand calculations and repeated runs are
bitwise stable under fixed traversal order.

---

## B4 — Exact best response

### B4.1 Responder information aggregation

**Goal:** prevent clairvoyant full-state response.

**Files:** `src/solver/best_response.py`, `src/test/test_solver_best_response.py`.

**Work**

- freeze the opponent profile;
- propagate opponent-and-chance reach to compatible responder histories;
- aggregate continuation values by responder information set;
- choose one action per information set;
- return the response policy, value, and per-information-set action values;
- define deterministic tie-breaking by engine action order.

**Adversarial tests**

- construct two hidden histories in one information set with different
  state-wise optimal actions;
- show the legal aggregate response chooses one shared action;
- show a deliberately clairvoyant helper obtains a strictly higher, invalid value;
- assert hidden state never appears in response policy keys.

**Gate:** best response is information-set consistent and improves against a
deliberately weak profile.

### B4.2 NashConv report

**Goal:** make exact reduced-game equilibrium quality independently measurable.

**Work**

- compute both best-response values;
- compute NashConv and exploitability using `SOLVER.md` definitions;
- include evaluator and response node counts;
- reject non-zero-sum utilities;
- serialize the report independently of training state.

**Gate:** a hand-solved equilibrium reports zero within tolerance and a weak
profile reports positive NashConv.

---

## B5 — Exact regret solvers

### B5.1 Certify the current CFR+ candidate

**Goal:** decide whether `src/exact/cfr_plus.py` is a valid reference.

**Work**

- compare short iterations to hand-computed regret and average-strategy updates;
- permute hidden-history enumeration and compare results within tolerance;
- verify information-set deltas are aggregated before clipping;
- verify Team 0 then Team 1 alternating passes;
- verify delayed linear averaging weights;
- plot or record exploitability at logarithmic iteration checkpoints.

**Gate:** average-profile exploitability decreases on tiny Air Poker and the
implementation agrees with hand-computed iterations.

### B5.2 Add vanilla CFR

**Goal:** create an algorithmically independent convergence reference.

**Files:** `src/solver/cfr.py`, `src/test/test_solver_cfr.py`.

**Work**

- implement ordinary cumulative regret;
- implement simultaneous or explicitly documented alternating updates;
- use reach-weighted average strategy;
- share only table validation and traversal interface with CFR+;
- do not share update code whose defect could make both agree falsely.

**Gate:** vanilla CFR and CFR+ approach the same reduced-game value and both show
declining independently computed exploitability.

### B5.3 Move accepted generic code

**Goal:** separate reusable algorithms from fixtures.

**Work**

- keep tiny Air Poker under `src/exact` as a fixture;
- move accepted generic CFR+ implementation to `src/solver/cfr.py` or a dedicated
  nonempty solver module;
- leave a compatibility import only if it has a removal date;
- update tests and documentation.

**Gate:** `src/exact` describes fixtures; `src/solver` owns accepted algorithms.

---

## B6 — Exact legal-hand count and sampling

### B6.1 DP count

**Goal:** count legal exact masks without emitting them.

**Files:** `src/engine/poker_math.py`, `src/test/test_poker_math.py`.

**Work**

- implement suffix state `count(index, cards_needed, sum_needed)`;
- use Python integers for exact counts;
- validate target and deck through existing public validators;
- return zero for impossible states;
- expose a stable public count function;
- benchmark full-deck canonical targets.

**Test matrix**

- every target on decks small enough for brute force;
- all canonical targets on the full deck;
- depleted decks with missing suits and ranks;
- targets 6 and 64;
- fewer than five available cards;
- equality with `len(generate_legal_hands(target,deck))` wherever enumeration is
  accepted.

**Gate:** counts are exact and substantially cheaper in memory than enumeration.

### B6.2 Uniform unranking

**Goal:** sample one legal exact mask uniformly from DP counts.

**Work**

- draw an integer rank uniformly from `[0,count)`;
- walk include/exclude branches using exact completion counts;
- return a five-bit mask or no-rank;
- accept an injected RNG;
- keep sample complexity bounded by deck size and DP table size;
- avoid building a tuple of legal hands.

**Tests**

- legality for thousands of seeded samples;
- exact boundary unranking for rank `0` and `count-1`;
- one-to-one coverage of every mask on tiny decks by iterating all integer ranks;
- deterministic sequence for a fixed seed;
- frequency checks with predeclared tolerance on small supports.

**Gate:** exhaustive unranking covers every legal mask exactly once.

### B6.3 Cache policy

**Goal:** prevent hidden memory blowups.

**Work**

- benchmark transient DP versus bounded cache;
- key only by `(deck_mask,target,algorithm_version)`;
- set and test an explicit maximum cache size;
- expose hit, miss, eviction, bytes, and build-time metrics;
- never cache full canonical hand lists in production.

**Gate:** cache bounds hold during a representative multi-round sample run.

---

## B7 — Sampled chance boundary

### B7.1 Engine-owned `HandPair` sampler

**Goal:** sample the same distribution that exact enumeration defines.

**Files:** `src/solver/chance.py`, `src/test/test_chance_sampler.py`.

**API**

```python
sample_chance_outcome(state, upper_policies, rng) -> (HandPair, probability)
```

**Work**

- validate phase, policy tuple, target availability, and probabilities;
- sample both policies against the same pre-round deck;
- preserve overlap;
- return the complete pair and true modeled probability;
- condition retries exactly;
- reject zero resolving mass;
- verify the returned pair is accepted by engine `apply`.

**Gate:** sampled chance never reconstructs settlement or deck logic.

### B7.2 Exact-versus-sampled agreement

**Goal:** validate semantics independently of MCCFR.

**Work**

- enumerate small supports exactly;
- sample them with multiple fixed seeds;
- compare pair frequencies, overlap rate, winner rate, and expected fixed-profile
  value;
- use confidence intervals and predeclared statistical thresholds;
- store seeds and sample counts in failure output.

**Gate:** all exact expectations fall within the declared sampled tolerances.

### B7.3 Canonical smoke path

**Goal:** prove a complete canonical trajectory avoids joint enumeration.

**Work**

- use random legal lower policies only for the smoke test;
- complete or explicitly terminate five rounds;
- log DP builds, pair draws, retry, union removal, and terminal cause;
- assert no call materializes a Cartesian product of upper hands.

**Gate:** a fixed-seed canonical game completes reproducibly within measured
resource bounds.

---

## B8 — External-sampling MCCFR

### B8.1 Traversal core

**Goal:** implement one updating-player traversal.

**Files:** `src/solver/mccfr.py`, `src/test/test_mccfr.py`.

**Work**

- sample chance with B7;
- sample opponent actions from current regret matching;
- memoize opponent action by information key for the traversal;
- enumerate updating-player actions;
- propagate sampled counterfactual values;
- aggregate regret deltas by information set;
- apply deltas after the traversal;
- keep RNG state outside game state.

**Gate:** one traversal touches all updater actions and exactly one sampled branch
at every chance/opponent decision encountered.

### B8.2 Average-strategy estimator

**Goal:** define the policy that will be reported.

**Work**

- implement the documented external-sampling average estimator;
- record visits and total estimator weight;
- normalize only when reading/exporting a profile;
- use uniform fallback for zero mass;
- test against exact average updates on small fixtures.

**Gate:** sampled average strategy converges toward the exact reference on reduced
games.

### B8.3 Determinism and isolation

**Goal:** make stochastic bugs reproducible.

**Work**

- derive independent RNG streams for Team 0 traversal, Team 1 traversal,
  evaluation, and diagnostics;
- record seed derivation in the manifest;
- forbid module-global RNG;
- hash tables at fixed checkpoints;
- verify evaluation cannot change training tables or RNG streams.

**Gate:** identical seed, model, code, and iteration count produce identical
digests.

### B8.4 Reduced-game convergence

**Goal:** validate MCCFR against the exact oracle.

**Work**

- train at logarithmic checkpoints;
- evaluate average profiles with exact NashConv;
- repeat across declared seeds;
- report median and spread rather than one favorable trajectory;
- compare estimated values to exact values.

**Gate:** exploitability trends downward and fixed-profile values agree with the
oracle within declared tolerances.

---

## B9 — Canonical training runner

### B9.1 Run configuration

**Goal:** make a run fully reconstructible.

**Files:** `src/solver/run.py`, `src/solver/checkpoint.py`.

**Configuration**

- model ID and fingerprint;
- algorithm and schema version;
- iteration/traversal budget;
- all seeds and derivation rule;
- averaging rule;
- numeric dtype;
- cache limits;
- checkpoint/evaluation cadence;
- output directory chosen by the caller.

**Gate:** configuration validates before allocating solver tables.

### B9.2 Manifest

**Goal:** bind outputs to inputs.

**Manifest fields**

- UTC start/end and elapsed monotonic time;
- repository commit and dirty diff digest;
- Python/platform/dependency versions;
- document, engine, model, and solver hashes;
- root inputs and upper policy fingerprint;
- seeds, budgets, cache settings, and dtype;
- final counters and artifact hashes.

**Gate:** every artifact references exactly one immutable manifest.

### B9.3 Checkpoint/resume

**Goal:** resume without changing the stochastic trajectory.

**Work**

- serialize information/action codecs, arrays, counters, iteration, and RNG states;
- write to a temporary file, flush, then atomically replace the target;
- validate checksum and model fingerprint before load;
- reject partial, foreign, or newer-schema checkpoints;
- keep at least one prior valid checkpoint until replacement succeeds.

**Gate:** interrupted-plus-resumed training matches uninterrupted table and metric
digests.

### B9.4 Bounded canonical pilot

**Goal:** collect scale evidence, not claim convergence.

**Work**

- run a small fixed traversal budget;
- record time by sampler/traversal/table/checkpoint;
- record information-set growth and bytes;
- record max updater branching and recursion depth;
- run a separate fixed-profile evaluation;
- label output `pilot`.

**Gate:** no hidden joint enumeration, memory remains bounded, and the next scale
decision is supported by measurements.

---

## B10 — Canonical evaluation

### B10.1 Monte Carlo profile evaluator

**Goal:** estimate value without touching training state.

**Files:** `src/solver/evaluate.py` plus dedicated tests.

**Work**

- freeze and copy the average profile;
- use independent chance and action seeds;
- batch terminal samples;
- report mean, variance, standard error, interval method, and sample count;
- stratify terminal outcomes by round and cause;
- verify utility symmetry by evaluating both player views.

**Gate:** repeated batches have calibrated coverage on reduced games with known
exact value.

### B10.2 Frozen-opponent response training

**Goal:** find empirical weaknesses without mislabeling them exact exploitability.

**Work**

- freeze one side of the evaluated profile;
- train only the responding side from empty tables;
- evaluate response gain on held-out seeds;
- swap sides;
- repeat across declared response seeds and budgets;
- report the maximum validated gain as a lower bound, not a certificate.

**Gate:** response training is isolated from the original profile and reports
uncertainty.

### B10.3 Strategy report

**Goal:** produce a decision-useful artifact.

**Report**

- model and solver identity;
- value interval;
- empirical response gains;
- policy probabilities at high-reach information sets;
- fold, call, raise-size, and number-choice frequencies;
- event rates for overlap, calamity, no-rank, retry, and terminal cause;
- table coverage and uniform fallbacks;
- convergence and resource plots or machine-readable series;
- explicit limitations and prohibited claims.

**Gate:** the report distinguishes model uncertainty, sampling uncertainty, and
response-search incompleteness.

---

## B11 — Scale decision

This slice begins only after the canonical pilot.

### B11.1 Profile bottlenecks

**Goal:** identify the dominant cost with evidence.

**Required measurements**

- DP construction and sampling time;
- engine transition time;
- updating-player subtree expansion;
- information-key hashing and table lookup;
- table allocation and resident bytes;
- checkpoint time and size;
- evaluation variance.

**Gate:** one dominant bottleneck is named with a reproducible benchmark.

### B11.2 Choose one response

| Bottleneck | First permitted response |
|---|---|
| DP rebuild time | bounded count-table cache or incremental rank-count state |
| Updating-player branching | benchmark outcome sampling as a separately validated algorithm |
| Raise action explosion | propose a new action-abstraction model ID and quantify omitted actions |
| Information table memory | compact codecs and packed arrays before function approximation |
| Evaluation variance | stratification or common-random-number comparisons with valid intervals |
| Checkpoint size | stable binary array format and compression after correctness tests |

Only one response is introduced per experiment. Re-run reduced oracle agreement
and canonical pilot after each change.

**Out of scope by default:** neural policies, distributed writers, GPUs, and
lossy hand abstraction.

---

## B12 — Behavioral model research

This slice starts only after `canonical-uniform-v1` trains and evaluates end to
end.

### B12.1 Upper policy feature contract

**Goal:** make nonuniform construction falsifiable.

**Work**

- define an `UpperObservation` that excludes forbidden full state;
- define normalized features for hand strength, future flexibility, and overlap
  risk only when the upper player can know them;
- separate structural no-rank, accidental failure, and deliberate no-rank;
- publish parameters and exact probability normalization;
- verify beta zero reproduces the uniform exact-mask policy.

**Gate:** identical upper observations produce identical distributions regardless
of hidden engine truth.

### B12.2 Persistent type model

**Goal:** represent stable upper skill without per-round parameter resampling.

**Work**

- draw a discrete type pair at root from a published common prior;
- retain types across all rounds;
- keep types hidden from lower information states;
- verify a degenerate prior reproduces the fixed-type model;
- assign a new model ID and fingerprint.

**Gate:** posterior effects arise only through observable history and reach, not
through leaked type fields.

### B12.3 Cross-model matrix

**Goal:** separate numerical convergence from behavioral robustness.

For strategy trained under model `i` and evaluated under model `j`, report

\[
M_{ij}=V_j(\sigma_i).
\]

Also report response gain under each evaluation model. Compare value, policy
distance at high-reach information sets, and terminal-event shifts.

**Gate:** no strategy is called robust from within-model convergence alone.

---

## Global verification

Run before every accepted slice:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s src/test -v
ruff check src
git diff --check
rg -n "from c[o]de\.|c[o]de/test|c[o]de/engine|c[o]de/solver|[Kk]uhn/" README.md docs/ENGINE.md docs/EVIDENCE.md docs/MODEL.md docs/SOLVER.md src
```

Long stochastic or exhaustive tests must be labeled and runnable separately, but
the default suite must retain deterministic semantic coverage.

## Global done definition

The first solver milestone is complete only when:

- the four control documents and executable engine agree;
- exact reduced evaluation, response, CFR, and CFR+ cross-check one another;
- exact-distribution sampling agrees with enumeration;
- reduced MCCFR converges toward the exact oracle;
- a canonical run completes, checkpoints, resumes, and evaluates independently;
- every artifact carries a model and solver fingerprint;
- reports use qualified claims from `SOLVER.md`;
- model uncertainty is evaluated separately from solver error.
