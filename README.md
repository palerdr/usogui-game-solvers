# Air Poker

A compact Python game engine and solver foundation for Air Poker from Toshio
Sako's *Usogui*.

The project models the two lower players as the strategic players. Their allied
upper players are fixed stochastic chance policies. The authoritative mechanical
and information-state contract is [ENGINE.md](docs/ENGINE.md); implementation
choices elsewhere must preserve it.

## Current foundation

- `code/engine/poker_math.py` provides 52-bit `CardMask` sets, Air Poker rank
  sums, legal five-card generation, and exact five-card poker evaluation.
- `code/engine/game.py` provides a 455-line immutable state machine for number
  choice, upper chance, betting, shared-deck depletion, calamity, antes, and
  terminal utility.
- `code/test/test_poker_math.py` exhaustively verifies all 2,598,960 five-card
  hands and all 7,462 distinct poker strengths.
- `code/test/test_game.py` verifies the rules and declared abstractions in
  `ENGINE.md`, including information boundaries and complete five-round play.

Canonical states and actions are frozen, hashable values. `Card`, `CardMask`, and
`MetalCard` remain integer primitives; NumPy is unnecessary in ground-truth game
state but remains appropriate for solver tables and batch operations.

## Run the tests

The project targets Python 3.14 and uses the standard-library test runner:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s code/test -v
```

Run static checks with:

```powershell
ruff check code/engine code/test
```

The current suite contains 48 tests and completes in about 16 seconds on the
development machine. Most of that time is the intentional exhaustive evaluator
test.

## Solver-facing API

The lower-player solver should depend on these operations from
`code.engine.game`:

```python
initial_state(...)
current_actor(state)
legal_actions(state)
iter_chance_outcomes(state, upper_policies)
apply(state, action)
information_state(state, team)
terminal_returns(state)
```

`GameState` contains hidden truth. A solver player must choose using only its
`InformationState`, never by inspecting the complete state directly.

## Explicit modeling assumptions

The engine deliberately does not present every interpretation as manga canon:

- upper decisions use fixed policies, with uniform legal hands as the baseline;
- upper hands commit before lower betting and remain hidden from lower players;
- stationary double-no-rank retries are conditioned on eventual resolution;
- a fold settles without calamity;
- continuous breathing is omitted and Round 5 uses
  `sealed_bios * 300 + remaining_air_seconds`.

The chapter evidence and exact rationale for each assumption live in
[ENGINE.md](docs/ENGINE.md).

## Solver readiness

The mechanical and information-state foundation is ready for a lower-player
solver prototype, but not for naive exact traversal. The canonical Round-1 pair
`36` versus `15` has 687,327,616 exact upper-hand pairs; the largest canonical
full-deck pairing has roughly 8.8 billion.

Start with deterministic or reduced upper policies to validate a solver, then use
chance-sampled MCCFR or a documented hand abstraction. Solver algorithms and
approximations belong in `code/solver`; they must not change `ENGINE.md` mechanics.
