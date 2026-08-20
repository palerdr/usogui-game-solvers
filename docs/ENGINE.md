# Air Poker Engine Contract

`src/engine/game.py` is the executable form of this document. This file is the
ground truth for the game represented by the solver; architecture built above it
may optimize the representation, but may not silently change these mechanics or
information boundaries.

## 1. Scope and evidence

The engine represents the two lower Air Poker players as the strategic players.
The two upper players are fixed stochastic policies that produce explicit chance
outcomes. This is a finite, two-player, zero-sum solver model of the depicted game,
not a claim that the four physical participants are one two-player game.

Rules are grounded in Toshio Sako's *Usogui*:

| Chapters | Engine rule |
|---|---|
| 429 | Each lower player receives 25 Air-BIOS; one BIOS contains about five minutes of air and is also a betting chip. |
| 430 | At most five poker rounds; antes rise from one through five; the selected metal cards are revealed before betting. |
| 431 | An opened BIOS cannot be wagered; betting order, action time, wager limits, and defeat for nonpayment are stated. |
| 436-440 | An upper player constructs five cards whose values sum to the selected metal card; named cards leave the shared virtual deck. |
| 444 | Reusing the same exact card in both hands creates calamity at judgement. |
| 448-456 | No-rank hands, a double-no-rank retry, Witch Pain, and the invalid Round 4 construction are shown. |
| 466-467 | The fifth poker round is followed by the physical endurance result; Lalo drowns and Baku wins. |

The core makes five explicit solver abstractions:

1. Upper decisions are chance, governed by caller-supplied fixed policies.
2. Both upper hands are resolved and committed before lower betting, but remain
   hidden from the lower players. This removes real-time interleaving.
3. Continuous breathing is not simulated. `remaining_air_seconds` is a private,
   caller-supplied value in `0..300`, does not change inside the core, and is used
   in the modeled total-air comparison after Round 5.
4. On a retry, stationary upper policies are conditioned on producing something
   other than another double no-rank. This exactly collapses repeated identical
   retries when eventual resolution has nonzero probability.
5. A fold settles without calamity because the hidden hands never reach judgement.
   The depicted match does not settle this counterfactual, so this is a visible
   engine ruling rather than a claimed manga rule.

Everything else--gauges, Morse code, speech, physiology, action delays, violence,
and Witch Pain's effect on later judgment--is outside this engine. Such features
belong in policies or an environment, not in the game state below.

## 2. Primitive representations

All state records are frozen, slotted dataclasses. Transitions return new states;
they never mutate an existing branch. Python integers are sufficient for this
small core; NumPy belongs in solver tables or batch calculations, not canonical
state.

```python
MetalCard = int       # the number printed on a physical metal card
Card = int            # a card id supplied by poker_math.py
CardMask = int        # any subset of the shared 52-card deck
Hand = CardMask | None  # None means no-rank
HandPair = tuple[Hand, Hand]
```

`CardMask` is intentionally not named `HandMask`: it can represent a hand, the
available deck, or any other card set.

```python
class Team(IntEnum):
    ZERO = 0
    ONE = 1

class Phase(Enum):
    NUMBER_CHOICE_0 = ...
    NUMBER_CHOICE_1 = ...
    UPPER_CHANCE = ...
    BETTING = ...
    TERMINAL = ...

class RoundAction(Enum):
    Check = ...
    Call = ...
    Fold = ...
    Raise = ...
```

There is no duplicate actor on `GameState`, nor a stored pot, ante, round number,
or terminal wrapper. The choice phase or `BettingState.actor` determines the
current actor; the other values are derived from the state that determines them.

## 3. Canonical state

```python
@dataclass(frozen=True, slots=True)
class LowerPlayerState:
    sealed_bios: int
    remaining_air_seconds: int
    remaining_metal_cards: tuple[MetalCard, ...]

@dataclass(frozen=True, slots=True)
class BettingAction:
    action: RoundAction
    extra: int = 0                 # positive only for Raise

@dataclass(frozen=True, slots=True)
class BettingState:
    first_bettor: Team
    actor: Team
    contributions: tuple[int, int]
    previous_action_was_check: bool = False
    actions: tuple[tuple[Team, BettingAction], ...] = ()

@dataclass(frozen=True, slots=True)
class RoundState:
    chosen_metal_cards: tuple[MetalCard | None, MetalCard | None]
    hands: HandPair | None         # None until upper chance resolves
    retry: bool
    betting: BettingState | None

@dataclass(frozen=True, slots=True)
class RoundRecord:
    chosen_metal_cards: tuple[MetalCard, MetalCard]
    first_bettor: Team
    actions: tuple[tuple[Team, BettingAction], ...]
    winner: Team | None            # None is an exact poker tie
    calamity: bool

@dataclass(frozen=True, slots=True)
class GameState:
    phase: Phase
    lower: tuple[LowerPlayerState, LowerPlayerState]
    available_deck_mask: CardMask
    round: RoundState
    history: tuple[RoundRecord, ...]
    winner: Team | None            # meaningful only in TERMINAL
```

Derived values are intentionally absent:

```text
round number = len(history) + 1
ante         = (1, 2, 3, 4, 5)[len(history)]
pot          = 2 * ante before betting exists; otherwise sum(contributions)
actor        = the choice phase or round.betting.actor
```

Before upper chance resolves, `round.betting` does not yet exist even though the
ante has been paid. During `NUMBER_CHOICE_0`, `NUMBER_CHOICE_1`, and
`UPPER_CHANCE`, the committed pot is therefore derived as `2 * ante`. When upper
chance resolves, `BettingState.contributions` begins as `(ante, ante)` and becomes
the source of the pot.

The canonical metal cards are:

```python
TEAM_0_METAL_CARDS = (25, 26, 36, 39, 45)
TEAM_1_METAL_CARDS = (8, 15, 44, 47, 63)
```

`initial_state` starts after each lower player has opened the first of 25 BIOS and
paid the Round 1 ante: 23 sealed BIOS remain, plus the supplied residual air in
the open BIOS. Each residual value must be an integer in `0..300`. Custom starting
metal-card sets must contain five cards per team, use values in `6..64`, and have
no duplicate value among all ten cards. Distinct values also make the Round-1
first-bettor comparison unambiguous.

## 4. Round transition

```text
pay ante
  -> Team 0 privately chooses one unused MetalCard
  -> Team 1 privately chooses one unused MetalCard
  -> both choices become public
  -> chance supplies both hidden upper hands against the same pre-round deck
  -> accepted cards are removed as one union
  -> lower betting
  -> pot and possible calamity settle
  -> next ante, or terminal comparison after Round 5
```

On Round 1, the team with the smaller revealed metal-card value bets first. On
later rounds, the previous round's first bettor alternates.

For target `n`, a valid hand contains exactly five currently available cards and
uses `A=1, 2=2, ..., K=13` with total `n`. `None` is no-rank, loses to any valid
hand, and consumes no cards. Both hands are checked against the same pre-round
deck, so exact-card overlap is legal. The next deck is:

```python
available_deck_mask & ~(hand_0 | hand_1)
```

The required baseline upper policy is uniform over every legal exact hand. If no
legal hand exists, it returns no-rank with probability one. A supplied policy must
return unique outcomes with finite, positive probabilities summing to one. On a
retry, the two policies must have nonzero probability of something other than
double no-rank. Calamity is never an independent random event; it is derived from
the sampled hand pair.

## 5. Betting and settlement

There is one betting street and four actions:

- `Check` is legal with nothing to call; two checks reach judgement.
- `Call` pays the outstanding amount and reaches judgement.
- `Fold` is legal only when facing a wager and concedes the pot without calamity.
- `Raise(extra)` first calls, then raises by a positive integer.

For acting team `i`:

```python
to_call = max(contributions) - contributions[i]
pot_after_call = sum(contributions) + to_call
max_extra = min(
    own_sealed_bios - to_call,
    opponent_sealed_bios,
    pot_after_call // 2,
)
```

The winner receives the pot. An exact poker tie returns each contribution. At
judgement, valid hands are compared by `poker_math.evaluate_mask`; a valid hand
beats no-rank.

If both judged hands are valid, overlap, and have a loser, calamity destroys an
additional number of the loser's sealed BIOS equal to that loser's total round
contribution, including ante. It is not transferred to the winner. Inability to
pay that penalty ends the model immediately in defeat. A fold does not trigger
calamity.

Before each later round, both players pay the next ante. Inability to pay ends in
defeat; if neither can pay, the model is a tie. After Round 5, modeled total air is:

```python
sealed_bios * 300 + remaining_air_seconds
```

The higher total wins and an exact model tie returns zero utility. This final
comparison is the explicit timeless abstraction from section 1, not a claim that
human oxygen consumption is deterministic.

## 6. Information state

`GameState` contains hidden truth. A lower solver acts only from
`information_state(state, team)`, which contains:

- its complete `LowerPlayerState`;
- the opponent's public sealed-BIOS count and number of unused metal cards;
- its own current choice and, once both choices close, both revealed choices;
- public betting state and completed `RoundRecord` history.

It never contains the opponent's exact residual air, unused metal-card values,
either upper hand, or the exact remaining deck. In particular, Team 1 cannot see
Team 0's choice while making its own.

Upper policies receive only the shared available deck, their own target, and the
opposing target on a retry. Initial policies therefore factor independently and
cannot inspect the opponent's current number or sampled hand.

## 7. Solver-facing engine

`game.py` exposes these public mechanical operations:

```python
initial_state(...) -> GameState
current_actor(state) -> Team | None
legal_actions(state) -> tuple[MetalCard | BettingAction, ...]
uniform_upper_policy(deck, target, opposing_target) -> outcomes
iter_chance_outcomes(state, upper_policies) -> Iterator[(HandPair, probability)]
apply(state, action) -> GameState
information_state(state, team) -> InformationState
terminal_returns(state) -> tuple[float, float]
```

`apply` is deterministic. At lower decisions it rejects actions outside
`legal_actions`. At upper chance it validates the selected pair mechanically--card
count, target sum, availability, and retry rules--but does not receive a policy
and therefore cannot verify policy support. A solver must choose chance outcomes
from `iter_chance_outcomes`. Terminal returns are `(1, -1)`, `(-1, 1)`, or
`(0, 0)`.

## 8. Non-negotiable invariants

- BIOS counts and betting contributions are never negative; residual air is in
  `0..300`.
- Exactly one unused metal card is removed per team per round.
- Both upper hands are validated against the same pre-round deck.
- A valid hand has five cards, is available, and sums to its target.
- Deck consumption is the union of valid hands; no-rank consumes nothing.
- The available-deck mask only loses cards and never gains them.
- Antes and wagers move between sealed stacks and the pot; calamity is the only
  core transition that destroys BIOS.
- Check is legal exactly at parity; call and fold exactly while behind.
- A raise is affordable, callable by the opponent, and no more than half-pot.
- Lower information never exposes hidden upper or opponent-private state.
- Lower decision phases have exactly one actor. `UPPER_CHANCE` and `TERMINAL` have
  no lower actor, and `TERMINAL` has no legal actions or transitions.
- All transitions are immutable and all engine actions and states are hashable.

This shape is deliberately small because every stored field either changes a
legal action, a transition, an observation, or terminal utility. Removing any of
them loses game information; adding derived facts duplicates it. That is the
foundation future solver architecture should trust.

## 9. Verification boundary

The standard-library tests in `src/test` are the executable regression contract:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s src/test -v
```

They exhaustively classify all 2,598,960 five-card hands, verify the 7,462 poker
strength classes, brute-force-check legal-hand generation on a reduced deck, and
exercise the state, information, betting, shared-deck, calamity, retry, ante, and
termination rules above.

Passing tests establishes that the implementation conforms to this document. It
does not independently prove the chapter interpretations. In particular, upper
policy choice, pre-betting hand commitment, retry conditioning, fold settlement,
and final modeled-air comparison remain the explicit abstractions in section 1.
