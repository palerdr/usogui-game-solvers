# Air Poker Solver Plan

**Status:** scoped implementation contract  
**Mechanical authority:** [`ENGINE.md`](./ENGINE.md) and `code/engine/game.py`  
**Current repository state:** the engine and poker math are implemented and tested; `code/solver/` contains no solver yet  
**Version 1 target:** the canonical `initial_state()` game with two lower strategic players and the engine's uniform upper policies

This document is intentionally an implementation plan, not a catalogue of every plausible solver or behavioral model. It keeps the parts of the previous plan that are required by the engine and the cited regret-minimization papers, moves scale-dependent claims behind measurements, and defers speculative upper-player modeling.

---

## 1. Audit result

The previous plan had the right mechanical boundary and literature, but it described a mature solver stack before the repository had its first traversal. It also treated exact traversal as a practical reference for the canonical game without measuring the engine's chance support.

The revised plan makes these decisions:

| Decision | Result |
|---|---|
| Engine authority | Retained. The solver never reimplements betting, settlement, deck consumption, retry conditioning, observations, or returns. |
| In-memory representation | Simplified. Use the engine's hashable `GameState`, `InformationState`, `MetalCard`, and `BettingAction` directly. Add codecs only when persistence is implemented. |
| Exact algorithms | Scoped to known small games and reduced-support Air Poker fixtures. They are correctness oracles, not a promise to enumerate the canonical tree. |
| Canonical training | External-sampling MCCFR is the scaling path, after its estimator and chance sampler agree with exact traversal on reduced fixtures. |
| CFR+ | Retained as an exact reduced-game reference following Tammelin's alternating, vector-form algorithm. It is not mixed casually with sampled CFR. |
| Exploitability | Exact only where an information-set-consistent best response is exactly enumerable. Canonical approximate response gains must not be labeled exact exploitability. |
| Upper behavior models | Logit choice, failures, latent types, model grids, and sensitivity studies are deferred until the uniform baseline works. |
| Infrastructure | Parallel writes, neural policies, RL wrappers, abstractions, elaborate checkpoint layouts, and experiment grids are out of Version 1. |

### 1.1 Findings corrected by this audit

1. A fixed canonical root is a game conditional on that exact deal and residual-air input. Merely omitting opponent cards or air from `InformationState` does not create a prior over those initial values. If they are meant to be privately uncertain, a later model must define a root distribution over valid calls to `initial_state(...)`.
2. The canonical game still has imperfect information because Team 1 acts without Team 0's current choice and later decisions merge histories with different hidden upper hands and deck states.
3. The present engine exposes exact chance enumeration but no sampled chance operation. Canonical MCCFR therefore has an explicit API/performance prerequisite; the solver must not duplicate the engine's retry kernel to work around it.
4. External sampling draws a deterministic choice for each sampled opponent information set during one traversal. If that information set is encountered again, the sampled action must be reused. Sampling independently per full history is not the algorithm in Lanctot et al. [3].
5. Exact canonical best response and exploitability are not established as feasible. The first canonical reports must distinguish exact reduced-game metrics, Monte Carlo value estimates, and empirical response gains.

---

## 2. Authority and solved-game boundary

The precedence order is:

1. [`ENGINE.md`](./ENGINE.md) and `code/engine/game.py` for mechanics and information;
2. `code/test/` for executable regression behavior;
3. [`Air-Poker-Summary.md`](./Air-Poker-Summary.md) for narrative context only;
4. the cited papers for solver algorithms, not Air Poker rules;
5. this document for implementation sequence.

If the narrative description and the engine abstraction differ, the engine wins. For example, the narrative discusses uncertainty about calamity after folding, while the locked engine settles a fold without calamity. A solver flag may not change that ruling.

### 2.1 Version 1 game

Version 1 fixes all of the following for one run:

```text
root state             initial_state()
metal-card sets        TEAM_0_METAL_CARDS and TEAM_1_METAL_CARDS
residual air           (300, 300)
strategic players      Team.ZERO and Team.ONE
upper policies         (uniform_upper_policy, uniform_upper_policy)
chance event           one complete HandPair
lower observation      information_state(state, acting_team)
terminal utility       terminal_returns(state)
```

This is a conditional solution of the engine's canonical starting state. It is not a solution over an unknown deal, uncertain initial residual air, manga-only timing, physiology, communication, Witch Pain, or deliberate invalid construction.

Any run with different `initial_state(...)` arguments is a separately configured game and must not share results or checkpoints with the Version 1 baseline.

The uniform upper policy means uniform over exact legal five-card masks, not poker categories or rank patterns. Both policies are evaluated against the same pre-round deck. The engine alone applies their `HandPair`, removes the union of cards, determines poker results and calamity, and handles retry conditioning.

### 2.2 Non-negotiable solver rules

- Call `current_actor`, `legal_actions`, `iter_chance_outcomes`, `apply`, `information_state`, and `terminal_returns`; do not reproduce their logic.
- Use `InformationState` only for a lower policy lookup. Never key a policy by `GameState`.
- Never add `round.hands`, `available_deck_mask`, opponent residual air, opponent unplayed card values, RNG state, or a future chance result to an information key.
- Sample or enumerate the complete correlated `HandPair`. Never sample winner, overlap, calamity, or deck removal independently.
- Keep the upper model fixed during a solve. Refitting it creates a different game and requires a new run.
- Preserve engine action order. A table's vector index is the index in the exact tuple returned by `legal_actions(state)`.
- Treat a zero resolving probability on retry as a model error, exactly as the engine does.

---

## 3. Feasibility boundary measured from the current code

`generate_legal_hands` was measured on the full deck for the ten canonical targets:

| Team | Exact-hand support sizes |
|---|---|
| Team 0 | 25: 61,892; 26: 70,892; 36: 124,156; 39: 112,172; 45: 61,892 |
| Team 1 | 8: 92; 15: 5,536; 44: 70,892; 47: 44,892; 63: 28 |

Because initial upper policies factor and both see the full deck, one Round-1 chance node contains the Cartesian product of the selected supports. Across the 25 canonical metal-card pairs, that is **1,732,976 to 8,801,667,152 exact `HandPair` outcomes** before any betting branch or later round is considered.

Consequences:

- Do not build or cache the full canonical tree.
- Do not use canonical exact CFR, CFR+, exact profile evaluation, or exact best response as early acceptance gates.
- Exact enumeration remains mandatory on deterministic and small-support fixtures because it is the oracle for sampled code.
- Before canonical MCCFR, provide an exact-distribution chance sampler that does not walk the joint Cartesian product.
- Benchmark legal-hand generation and sampling before setting iteration or runtime targets.

These are scope constraints, not permission to approximate mechanics or information.

---

## 4. Minimal software boundary

The first implementation should be small:

```text
code/solver/
├── interface.py       # thin engine adapter and node classification
├── tables.py          # information-set entries and strategy profiles
├── exact.py           # exact evaluator and exact best response for small games
├── cfr.py             # vanilla CFR and exact CFR+ for small games
├── chance.py          # sampled chance boundary, added only in Phase 4
├── mccfr.py           # external-sampling MCCFR, added only in Phase 4
└── run.py             # manifests/checkpoints, added only in Phase 5

code/test/
├── test_solver_interface.py
├── test_solver_exact.py
├── test_solver_cfr.py
└── test_solver_mccfr.py
```

Do not create empty future modules. Add each file when its phase starts.

### 4.1 Thin Air Poker adapter

The algorithm-facing object may use this shape:

```python
class NodeKind(Enum):
    TERMINAL = auto()
    CHANCE = auto()
    PLAYER = auto()


class AirPokerTraversal:
    def root(self) -> GameState: ...
    def kind(self, state: GameState) -> NodeKind: ...
    def actor(self, state: GameState) -> Team | None: ...
    def player_actions(
        self, state: GameState
    ) -> tuple[MetalCard | BettingAction, ...]: ...
    def chance_outcomes(
        self, state: GameState
    ) -> Iterator[tuple[HandPair, float]]: ...
    def next_state(
        self, state: GameState, action: MetalCard | BettingAction | HandPair
    ) -> GameState: ...
    def information_key(self, state: GameState, team: Team) -> InformationState: ...
    def returns(self, state: GameState) -> tuple[float, float]: ...
```

Its implementation is delegation:

| Adapter operation | Engine source |
|---|---|
| `root` | `initial_state()` |
| terminal/chance/player classification | `state.phase` |
| `actor` | `current_actor(state)` |
| player actions | `legal_actions(state)` |
| exact chance | `iter_chance_outcomes(state, upper_policies)` |
| transition | `apply(state, action)` |
| information key | `information_state(state, team)` |
| returns | `terminal_returns(state)` |

The adapter may assert preconditions but may not repair or reinterpret a state. The CFR code can use the same small method set with test games such as Kuhn poker without turning this project into a general game framework.

### 4.2 In-memory table contract

Use engine values directly until checkpointing requires a codec:

```python
PlayerAction = MetalCard | BettingAction


@dataclass(slots=True)
class InfoSetEntry:
    actions: tuple[PlayerAction, ...]
    regret_sum: np.ndarray       # float64, shape (len(actions),)
    strategy_sum: np.ndarray     # float64, same shape
```

Maintain one `dict[InformationState, InfoSetEntry]` per team. On every lookup:

```text
same InformationState => same acting Team => same ordered action tuple
```

Reject non-finite, negative, wrong-length, or non-normalized strategy vectors. If all positive regrets are zero, regret matching returns the uniform distribution. An unvisited average-strategy entry also falls back to uniform and is reported as unvisited.

### 4.3 Information and perfect-recall contract

During exact fixture traversal, retain each team's own sequence of prior `(InformationState, action)` pairs. Assert:

```text
same team and InformationState => same own prior information/action sequence
```

This is the repository's executable perfect-recall audit. It also catches action-set instability and accidental use of hidden state. The standard CFR guarantees in Zinkevich et al. require finite, two-player, zero-sum, perfect-recall play [1].

### 4.4 Exact evaluation contract

For a fixed behavior profile:

- terminal node: return `terminal_returns(state)`;
- chance node: probability-weight every child from `iter_chance_outcomes`;
- player node: strategy-weight every legal action using the acting player's `InformationState`;
- never choose an action separately for two histories in the same information set.

The exact evaluator is required for small games and reduced-support Air Poker fixtures only. It is also the oracle for Monte Carlo value estimates.

### 4.5 Regret contracts

For updating player `i`, the immediate contribution at a history in information set `I` is:

\[
\pi_{-i}(h)\left(v_i(h,a)-v_i(h)\right),
\]

where `pi_-i` includes chance and the opponent, but not player `i`. Contributions from every history in one information set are summed before its cumulative regret is changed.

Vanilla CFR uses ordinary cumulative regrets and reach-weighted average strategies as defined by Zinkevich et al. [1]. CFR+ follows Tammelin [2]:

- vector-form alternating player passes;
- regret-matching+;
- `R <- max(R + delta, 0)` after the complete information-set delta is known;
- delayed linear average weight `max(iteration - delay, 0)`;
- fixed pass order `Team.ZERO`, then `Team.ONE`;
- as in Tammelin's Algorithm 1, the pass updating player `i` accumulates the non-updating player's average strategy at that player's information sets using that player's own reach and the iteration weight.

Do not call a sampled regret-clipping experiment “CFR+.” The first external-sampling implementation uses ordinary regret matching.

### 4.6 Best-response and reporting contract

An exact best responder chooses one action for an entire responding-player information set, after aggregating continuation values across all compatible histories with opponent-and-chance reach weights. A greedy full-state responder is an illegal clairvoyant policy. `BR_i(sigma_-i)` below means player `i`'s maximum engine return against the fixed opposing strategy.

For profile `sigma`, report both conventions where exact response is available:

\[
\operatorname{NashConv}(\sigma)=BR_0(\sigma_1)+BR_1(\sigma_0),
\qquad
\operatorname{exploitability}=\operatorname{NashConv}/2.
\]

The report must state that utilities are the engine's `-1/0/+1` match returns.

---

## 5. Phased implementation

Each phase ends at its acceptance gate. Later work does not begin merely because earlier code “runs.”

### Phase 0 - Lock and measure the baseline

Deliverables:

- record the exact `initial_state()` inputs and uniform upper-policy pair;
- keep the existing engine regression command green;
- add a small script or test that records legal-hand support and chance-pair counts;
- benchmark `generate_legal_hands` for the canonical targets;
- write a short decision record for the sampled-chance API described in Phase 4.

Acceptance gate:

- all existing engine tests pass;
- the run scope says “canonical fixed root,” not “unknown private deal”;
- no milestone assumes full canonical enumeration.

Current baseline at this audit: **48 engine and poker-math tests pass**.

### Phase 1 - Adapter and information audit

Deliverables:

- `AirPokerTraversal` delegating to the seven public engine operations;
- deterministic upper-policy fixtures with one or a few hand-pair outcomes;
- node, actor, transition, probability, action-order, information-key, and perfect-recall tests;
- counters for histories, information sets, actions, chance support, and maximum depth on each fixture.

Acceptance gate:

- every engine action round-trips through the adapter unchanged;
- every repeated information key has one team, one ordered action tuple, and one own-memory signature;
- hidden hand/deck changes that leave `InformationState` equal use the same policy entry;
- chance probabilities accepted by the adapter use the engine's normalization semantics and tolerance.

### Phase 2 - Exact reduced-game oracle

Deliverables:

- normalized `StrategyProfile` and uniform fallback;
- exact profile evaluator;
- exact information-set-consistent best response;
- NashConv/exploitability reporting;
- known small-game tests plus reduced-support Air Poker tests.

Minimum fixtures:

- matching pennies or another hand-computable zero-sum game;
- Kuhn poker for imperfect information and known equilibrium behavior;
- Air Poker with deterministic hands;
- Air Poker with a two-outcome hidden `HandPair` distribution;
- no-rank/retry, overlap, fold, and terminal-tie paths already supported by engine fixtures.

Acceptance gate:

- exact values match hand calculations or known game values;
- a best response never varies within one information set;
- a best response improves against a deliberately weak profile;
- exact chance expectation matches direct enumeration on every Air Poker fixture.

### Phase 3 - Exact CFR and CFR+ on reduced games

Deliverables:

- vanilla CFR with reach-weighted average strategy;
- exact alternating CFR+ following the contract in section 4.5;
- deterministic convergence and traversal-order tests;
- exploitability curves from the independent Phase 2 evaluator.

Acceptance gate:

- vanilla CFR and CFR+ approach the same reduced-game value;
- average-profile exploitability decreases on Kuhn poker and the reduced Air Poker fixture;
- CFR+ regrets remain nonnegative and are clipped once per complete information-set update;
- changing hidden-history enumeration order does not change a deterministic iteration result within tolerance;
- average-strategy weights match a hand-computed short run.

No canonical full-tree run is part of this gate.

### Phase 4 - Exact-distribution sampling and canonical MCCFR

This phase first closes the current engine API gap.

The required sampled-chance semantic contract is:

```python
sample_chance_outcome(
    state: GameState,
    upper_policies: tuple[UpperPolicy, UpperPolicy],
    rng: random.Random,
) -> tuple[HandPair, float]
```

Whether this becomes a public engine function or a public engine-owned chance-kernel object is decided in Phase 0. Either form must:

- use the same policy validation as `iter_chance_outcomes`;
- sample both policies against the same pre-round deck;
- return the complete `HandPair` and its modeled probability;
- reproduce the retry distribution conditional on not drawing double no-rank;
- raise on zero resolving mass;
- produce only actions accepted by `apply`;
- match exact frequencies on small supports.

The first semantic implementation may materialize small policy supports. If benchmarks show that canonical uniform sampling is too slow, add an exact uniform-hand sampler in `poker_math.py` using dynamic-programming counts and combinatorial unranking. Its count must equal `len(generate_legal_hands(target, deck))`, and its samples must be uniform over exact masks. This optimization changes no card or game representation.

Then implement external-sampling MCCFR as specified by Lanctot et al. [3]:

- enumerate every action of the updating lower player;
- sample chance from the modeled distribution;
- sample the opponent's action from the current policy;
- memoize one sampled action per opponent information set for that traversal;
- use ordinary sampled cumulative regrets and the external-sampling estimator;
- document and test the sampled average-strategy estimator;
- use modeled-distribution sampling only; no alternate proposal or importance weighting in Version 1.

Acceptance gate:

- sampled chance frequencies and fixed-profile values agree with exact reduced fixtures within statistical tolerance;
- external-sampling MCCFR trends toward the Phase 3 solution on the same fixtures;
- repeated seeds reproduce trajectories;
- a canonical trajectory completes without joint chance enumeration or hidden-state leakage;
- profiling identifies legal-hand sampling, traversal, and table growth separately.

### Phase 5 - Canonical runs, evaluation, and persistence

Deliverables:

- a seeded canonical training runner using the fixed Version 1 game;
- an independent Monte Carlo evaluator for fixed-profile value and terminal event rates;
- confidence intervals for sampled values;
- frozen-opponent response training as a diagnostic;
- checkpoint/resume only after table identities and action ordering are stable;
- a run manifest containing engine/docs/code hashes, initial-state inputs, upper-policy fingerprint, algorithm, hyperparameters, seed, iteration count, and codec version.

Metric labels are strict:

| Metric | Allowed label |
|---|---|
| Exact reduced-game best responses | `NashConv`, `exploitability` |
| Canonical Monte Carlo profile value | `estimated value` with confidence interval |
| Finite trained response against a frozen profile | `empirical response gain` or `exploitability lower bound` |
| A proven/certified canonical response bound | `canonical exploitability bound` |

Do not describe a canonical profile as an approximate Nash equilibrium solely because training regret or self-play value appears stable.

Acceptance gate:

- checkpoint/resume reproduces an uninterrupted seeded run;
- evaluation uses a separate seed stream and does not mutate training tables;
- reports separate solver sampling error from model choice;
- the result is called a trained canonical profile unless an exploitability certificate is actually available.

### Phase 6 - Optional behavioral-model research

Only after the uniform baseline passes Phase 5 may the project add:

- fixed logit choice over exact legal hands;
- accidental or deliberate no-rank models;
- persistent latent upper types;
- alternate initial-deal priors;
- cross-model sensitivity studies.

The papers constrain the terminology and boundaries:

- McKelvey and Palfrey [4] justify logit response inside an extensive-form equilibrium model. A standalone fixed upper softmax is a **logit choice policy**, not automatically an AQRE.
- Harsanyi [5] justifies a root chance draw over persistent hidden types under an explicit prior. A type must remain fixed for the match and must not enter a lower `InformationState`.
- Every added upper model must expose exact and sampled distributions derived from one implementation and must be frozen for a solve.
- A prior mixture optimizes expected value under that prior; it is not worst-case robustness.

No behavioral parameter grid is locked by this plan because the game description and current code do not identify those parameters.

---

## 6. Definition of done for Version 1

Version 1 is complete when:

1. the existing engine contract remains green and no solver module duplicates a mechanical rule;
2. information-state and perfect-recall audits pass on every exact fixture;
3. exact evaluation, best response, vanilla CFR, and CFR+ agree on reduced games;
4. the sampled chance kernel is statistically equivalent to exact engine chance on reduced supports;
5. external-sampling MCCFR agrees with the exact reduced oracle within stated statistical tolerance;
6. a seeded canonical run trains, evaluates, checkpoints, and resumes without enumerating a joint upper-hand Cartesian product;
7. the final report uses only metrics its evaluation method supports.

Anything beyond this list is a new phase, not an implicit Version 1 requirement.

---

## 7. Literature anchors

The local papers are implementation references. [`literature_analysis.md`](./papers/literature_analysis.md) is useful background, but any earlier modeling proposal in it yields to the locked engine and this scoped plan.

[1] Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. **“Regret Minimization in Games with Incomplete Information.”** NeurIPS 2007. [Local PDF](./papers/NIPS-2007-regret-minimization-in-games-with-incomplete-information-Paper.pdf). Used for perfect recall, counterfactual reach/regret, regret matching, reach-weighted average strategy, and the low-regret equilibrium connection.

[2] Oskari Tammelin. **“Solving Large Imperfect Information Games Using CFR+.”** 2014. [Local PDF](./papers/1407.5042v1.pdf). Used for vector-form alternating CFR+, regret-matching+, and delayed linear averaging.

[3] Marc Lanctot, Kevin Waugh, Martin Zinkevich, and Michael Bowling. **“Monte Carlo Sampling for Regret Minimization in Extensive Games.”** NeurIPS 2009. [Local PDF](./papers/NIPS-2009-monte-carlo-sampling-for-regret-minimization-in-extensive-games-Paper.pdf). Used for the sampled counterfactual-regret framework and external sampling of chance and opponent actions.

[4] Richard D. McKelvey and Thomas R. Palfrey. **“Quantal Response Equilibria for Extensive Form Games.”** [Local PDF](./papers/sswp947.pdf). Deferred; used only to constrain later logit/AQRE terminology.

[5] John C. Harsanyi. **“Games with Incomplete Information Played by ‘Bayesian’ Players, Part I: The Basic Model.”** 1967. [Local PDF](./papers/Harasyani.pdf). Deferred; used only for a later explicit root prior over persistent hidden types.
