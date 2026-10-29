# Air Poker

A Python game engine and solver foundation for Air Poker from Toshio Sako's
*Usogui*.

The model treats the two lower players as strategic agents and the allied upper
players as fixed chance policies. It is a labeled mathematical model of the
depicted game, not a claim that every counterfactual is manga canon.

## Documentation

- [GOALS.md](docs/GOALS.md) — the solve targets, the public-information reformulation, and measured reductions.
- [EVIDENCE.md](docs/EVIDENCE.md) — source facts, derivations, and unresolved gaps.
- [MODEL.md](docs/MODEL.md) — the named and fingerprinted baseline assumptions.
- [SOLVER.md](docs/SOLVER.md) — equilibrium target, exact oracle, sampling, and MCCFR.
- [BUILD.md](docs/BUILD.md) — narrow implementation slices and acceptance gates.
- [ENGINE.md](docs/ENGINE.md) — executable mechanics and information-state contract.

`EVIDENCE → MODEL → ENGINE → SOLVER → BUILD` is the authority chain.

## Current foundation

- `src/engine/poker_math.py` implements 52-bit card sets, Air Poker sums, legal
  five-card generation, and exact poker evaluation.
- `src/engine/game.py` implements immutable number choice, upper chance, betting,
  shared-deck depletion, calamity, antes, and terminal utility.
- `src/test/test_poker_math.py` exhaustively verifies all 2,598,960 five-card
  hands and all 7,462 poker strengths.
- `src/exact/` contains a tiny exhaustive Air Poker fixture and a candidate
  alternating CFR+ reference.

The engine foundation is complete. The candidate CFR+ implementation still needs
an independent best response and exploitability oracle before production MCCFR
work begins.

## Verify

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s src/test -v
ruff check src
git diff --check
```

The current suite contains 65 tests. Its slowest test intentionally evaluates
every five-card hand.

## Solver boundary

Solvers use only the public operations in `src.engine.game`:

```python
initial_state(...)
current_actor(state)
legal_actions(state)
iter_chance_outcomes(state, upper_policies)
apply(state, action)
information_state(state, team)
terminal_returns(state)
```

`GameState` contains hidden truth. Lower policies use only `InformationState`.

The canonical upper chance tree cannot be traversed exactly: first-round joint
supports range from roughly 1.7 million to 8.8 billion hand pairs. The production
path is an exact reduced-game oracle followed by exact-distribution hand sampling
and external-sampling MCCFR. See [SOLVER.md](docs/SOLVER.md) for the permitted
claims and [BUILD.md](docs/BUILD.md) for the implementation order.
