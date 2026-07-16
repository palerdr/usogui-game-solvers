"""Minimal immutable Air Poker engine for a two-lower-player solver."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from enum import Enum, IntEnum, auto
from math import isclose, isfinite
from numbers import Real

try:
    from .poker_math import (
        Card, CardMask, FULL_DECK_MASK, evaluate_mask, generate_legal_hands,
        mask_popcount, rank_sum, remove_cards,
    )
except ImportError:  # Allow direct execution from code/engine.
    from poker_math import (
        Card, CardMask, FULL_DECK_MASK, evaluate_mask, generate_legal_hands,
        mask_popcount, rank_sum, remove_cards,
    )

STARTING_BIOS = 25
BIO_SECONDS = 300
ANTES = (1, 2, 3, 4, 5)
TEAM_0_METAL_CARDS = (25, 26, 36, 39, 45)
TEAM_1_METAL_CARDS = (8, 15, 44, 47, 63)

# These aliases preserve the physical game vocabulary without wrapping integers.
MetalCard = int
Hand = CardMask | None
HandPair = tuple[Hand, Hand]

class Team(IntEnum):
    """One lower player and that player's allied upper player."""
    ZERO = 0
    ONE = 1

class Phase(Enum):
    """The five solver-visible stages of the game."""
    NUMBER_CHOICE_0 = auto()
    NUMBER_CHOICE_1 = auto()
    UPPER_CHANCE = auto()
    BETTING = auto()
    TERMINAL = auto()

class RoundAction(Enum):
    """A lower player's action on the single betting street."""
    Check = auto()
    Call = auto()
    Fold = auto()
    Raise = auto()

@dataclass(frozen=True, slots=True)
class LowerPlayerState:
    """A lower player's private resources and unused metal cards."""
    sealed_bios: int
    remaining_air_seconds: int
    remaining_metal_cards: tuple[MetalCard, ...]

@dataclass(frozen=True, slots=True)
class BettingAction:
    """A betting move; extra is positive only for a raise."""
    action: RoundAction
    extra: int = 0

@dataclass(frozen=True, slots=True)
class BettingState:
    """The public state of the current one-street betting sequence."""
    first_bettor: Team
    actor: Team
    contributions: tuple[int, int]
    previous_action_was_check: bool = False
    actions: tuple[tuple[Team, BettingAction], ...] = ()

@dataclass(frozen=True, slots=True)
class RoundState:
    """The current round's choices, hidden hands, retry flag, and betting."""
    chosen_metal_cards: tuple[MetalCard | None, MetalCard | None]
    hands: HandPair | None = None
    retry: bool = False
    betting: BettingState | None = None

@dataclass(frozen=True, slots=True)
class RoundRecord:
    """The public facts retained after a completed round."""
    chosen_metal_cards: tuple[MetalCard, MetalCard]
    first_bettor: Team
    actions: tuple[tuple[Team, BettingAction], ...]
    winner: Team | None
    calamity: bool

@dataclass(frozen=True, slots=True)
class GameState:
    """The complete hidden-truth state from which every transition is derived."""
    phase: Phase
    lower: tuple[LowerPlayerState, LowerPlayerState]
    available_deck_mask: CardMask
    round: RoundState
    history: tuple[RoundRecord, ...] = ()
    winner: Team | None = None

@dataclass(frozen=True, slots=True)
class InformationState:
    """Exactly what one lower solver player may know at a decision point."""
    team: Team
    phase: Phase
    own: LowerPlayerState
    opponent_sealed_bios: int
    opponent_metal_card_count: int
    visible_metal_cards: tuple[MetalCard | None, MetalCard | None]
    betting: BettingState | None
    history: tuple[RoundRecord, ...]

# A fixed upper policy maps its permitted observation to hand probabilities.
UpperPolicy = Callable[[CardMask, MetalCard, MetalCard | None], Iterable[tuple[Hand, float]]]

__all__ = (
    "Card", "CardMask", "MetalCard", "Hand", "HandPair", "STARTING_BIOS",
    "BIO_SECONDS", "ANTES", "TEAM_0_METAL_CARDS", "TEAM_1_METAL_CARDS", "Team",
    "Phase", "RoundAction", "LowerPlayerState", "BettingAction", "BettingState",
    "RoundState", "RoundRecord", "GameState", "InformationState", "UpperPolicy",
    "initial_state", "current_actor", "legal_actions", "uniform_upper_policy",
    "iter_chance_outcomes", "apply", "information_state", "terminal_returns",
)

def initial_state(
    team_0_metal_cards: Iterable[MetalCard] = TEAM_0_METAL_CARDS,
    team_1_metal_cards: Iterable[MetalCard] = TEAM_1_METAL_CARDS,
    remaining_air_seconds: tuple[int, int] = (BIO_SECONDS, BIO_SECONDS),
) -> GameState:
    """Create the state after both players open one BIOS and pay the first ante."""
    metal_cards = (
        _validate_metal_cards(team_0_metal_cards, "team_0_metal_cards"),
        _validate_metal_cards(team_1_metal_cards, "team_1_metal_cards"),
    )
    if len(set(metal_cards[0] + metal_cards[1])) != 10:
        raise ValueError("all ten metal cards must be distinct")
    if (type(remaining_air_seconds) is not tuple or len(remaining_air_seconds) != 2
            or any(type(value) is not int or not 0 <= value <= BIO_SECONDS
                   for value in remaining_air_seconds)):
        raise ValueError(f"remaining_air_seconds must contain two integers in 0..{BIO_SECONDS}")
    sealed = STARTING_BIOS - 1 - ANTES[0]
    lower = tuple(LowerPlayerState(sealed, remaining_air_seconds[i], metal_cards[i])
                  for i in range(2))
    return GameState(Phase.NUMBER_CHOICE_0, lower, FULL_DECK_MASK, RoundState((None, None)))

def current_actor(state: GameState) -> Team | None:
    """Derive the acting lower team, or None at chance and terminal phases."""
    if state.phase is Phase.NUMBER_CHOICE_0:
        return Team.ZERO
    if state.phase is Phase.NUMBER_CHOICE_1:
        return Team.ONE
    if state.phase is Phase.BETTING:
        if state.round.betting is None:
            raise ValueError("betting phase requires betting state")
        return state.round.betting.actor
    return None

def legal_actions(state: GameState) -> tuple[MetalCard | BettingAction, ...]:
    """Return every legal action for the current lower-player decision."""
    actor = current_actor(state)
    if state.phase in (Phase.NUMBER_CHOICE_0, Phase.NUMBER_CHOICE_1):
        return state.lower[int(actor)].remaining_metal_cards
    if state.phase is not Phase.BETTING:
        return ()
    betting = state.round.betting
    if betting is None:
        raise ValueError("betting phase requires betting state")
    i, j = int(betting.actor), int(other_team(betting.actor))
    to_call = max(betting.contributions) - betting.contributions[i]
    actions = ([BettingAction(RoundAction.Check)] if to_call == 0
               else [BettingAction(RoundAction.Fold)])
    if state.lower[i].sealed_bios >= to_call:
        if to_call:
            actions.insert(0, BettingAction(RoundAction.Call))
        max_extra = min(
            state.lower[i].sealed_bios - to_call,
            state.lower[j].sealed_bios,
            (sum(betting.contributions) + to_call) // 2,
        )
        actions.extend(BettingAction(RoundAction.Raise, n)
                       for n in range(1, max_extra + 1))
    return tuple(actions)

def uniform_upper_policy(
    available_deck_mask: CardMask, target: MetalCard, opposing_target: MetalCard | None,
) -> tuple[tuple[Hand, float], ...]:
    """Assign equal probability to every legal exact hand, or one to no-rank."""
    del opposing_target
    hands = generate_legal_hands(target, available_deck_mask)
    if not hands:
        return ((None, 1.0),)
    probability = 1.0 / len(hands)
    return tuple((hand, probability) for hand in hands)

def iter_chance_outcomes(
    state: GameState,
    upper_policies: tuple[UpperPolicy, UpperPolicy] = (uniform_upper_policy,
                                                       uniform_upper_policy),
) -> Iterator[tuple[HandPair, float]]:
    """Enumerate the joint upper-hand chance event for the current round."""
    if state.phase is not Phase.UPPER_CHANCE:
        raise ValueError("chance outcomes exist only during UPPER_CHANCE")
    if type(upper_policies) is not tuple or len(upper_policies) != 2:
        raise ValueError("upper_policies must contain exactly two policies")
    metal_0, metal_1 = state.round.chosen_metal_cards
    if metal_0 is None or metal_1 is None:
        raise ValueError("both metal cards must be chosen before upper chance")
    outcomes_0 = _policy_outcomes(upper_policies[0], state.available_deck_mask, metal_0,
                                  metal_1 if state.round.retry else None)
    outcomes_1 = _policy_outcomes(upper_policies[1], state.available_deck_mask, metal_1,
                                  metal_0 if state.round.retry else None)
    double_none = (sum(p for hand, p in outcomes_0 if hand is None)
                   * sum(p for hand, p in outcomes_1 if hand is None))
    resolving = 1.0 - double_none if state.round.retry else 1.0
    if resolving <= 1e-12:
        raise ValueError("retry policies must have a nonzero chance to resolve")
    for hand_0, probability_0 in outcomes_0:
        for hand_1, probability_1 in outcomes_1:
            if state.round.retry and hand_0 is None and hand_1 is None:
                continue
            yield (hand_0, hand_1), probability_0 * probability_1 / resolving

def apply(state: GameState, action: MetalCard | BettingAction | HandPair) -> GameState:
    """Apply one legal lower action or enumerated chance outcome immutably."""
    if state.phase is Phase.TERMINAL:
        raise ValueError("terminal states have no transitions")
    if state.phase in (Phase.NUMBER_CHOICE_0, Phase.NUMBER_CHOICE_1):
        return _choose_metal_card(state, action)
    if state.phase is Phase.UPPER_CHANCE:
        return _apply_upper_hands(state, action)
    if state.phase is Phase.BETTING:
        if not isinstance(action, BettingAction):
            raise TypeError("betting requires a BettingAction")
        return _apply_betting_action(state, action)
    raise ValueError("unknown game phase")

def information_state(state: GameState, team: Team) -> InformationState:
    """Project hidden truth into one lower player's solver information state."""
    team, opponent = _require_team(team), other_team(team)
    chosen = state.round.chosen_metal_cards
    revealed = state.phase not in (Phase.NUMBER_CHOICE_0, Phase.NUMBER_CHOICE_1)
    visible = (chosen[0] if revealed or team is Team.ZERO else None,
               chosen[1] if revealed or team is Team.ONE else None)
    return InformationState(
        team, state.phase, state.lower[int(team)], state.lower[int(opponent)].sealed_bios,
        len(state.lower[int(opponent)].remaining_metal_cards), visible,
        state.round.betting if state.phase is Phase.BETTING else None, state.history,
    )

def terminal_returns(state: GameState) -> tuple[float, float]:
    """Return zero-sum team utility for a terminal state."""
    if state.phase is not Phase.TERMINAL:
        raise ValueError("terminal returns require a terminal state")
    if state.winner is Team.ZERO:
        return (1.0, -1.0)
    if state.winner is Team.ONE:
        return (-1.0, 1.0)
    return (0.0, 0.0)

def other_team(team: Team) -> Team:
    """Return the opposing team."""
    return Team.ONE if _require_team(team) is Team.ZERO else Team.ZERO

def _require_team(team: Team) -> Team:
    if not isinstance(team, Team):
        raise TypeError("team must be a Team")
    return team

def _validate_metal_cards(cards: Iterable[MetalCard], name: str) -> tuple[MetalCard, ...]:
    try:
        values = tuple(cards)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    if len(values) != 5:
        raise ValueError(f"{name} must contain exactly five cards")
    if any(type(value) is not int or not 6 <= value <= 64 for value in values):
        raise ValueError(f"{name} values must be integers in range 6..64")
    if len(set(values)) != 5:
        raise ValueError(f"{name} cannot contain duplicates")
    return values

def _replace_team(values: tuple, team: Team, value) -> tuple:
    return (value, values[1]) if team is Team.ZERO else (values[0], value)

def _change_bios(lower: tuple[LowerPlayerState, LowerPlayerState], team: Team,
                 amount: int) -> tuple[LowerPlayerState, LowerPlayerState]:
    player = lower[int(team)]
    sealed = player.sealed_bios + amount
    if sealed < 0:
        raise ValueError("player cannot pay that many sealed BIOS")
    return _replace_team(lower, team, replace(player, sealed_bios=sealed))

def _choose_metal_card(state: GameState, action: object) -> GameState:
    actor = current_actor(state)
    if actor is None or type(action) is not int or action not in legal_actions(state):
        raise ValueError("metal-card choice is not legal in this state")
    player = state.lower[int(actor)]
    updated = replace(player, remaining_metal_cards=tuple(
        card for card in player.remaining_metal_cards if card != action))
    lower = _replace_team(state.lower, actor, updated)
    chosen = _replace_team(state.round.chosen_metal_cards, actor, action)
    phase = Phase.NUMBER_CHOICE_1 if actor is Team.ZERO else Phase.UPPER_CHANCE
    return replace(state, phase=phase, lower=lower,
                   round=replace(state.round, chosen_metal_cards=chosen))

def _validate_hand(hand: Hand, available_deck_mask: CardMask, target: MetalCard) -> None:
    if hand is None:
        return
    if type(hand) is not int:
        raise TypeError("upper hand must be a CardMask or None")
    if mask_popcount(hand) != 5:
        raise ValueError("upper hand must contain exactly five cards")
    if hand & ~available_deck_mask:
        raise ValueError("upper hand contains an unavailable card")
    if rank_sum(hand) != target:
        raise ValueError("upper hand does not sum to its metal card")

def _policy_outcomes(policy: UpperPolicy, available_deck_mask: CardMask,
                     target: MetalCard, opposing_target: MetalCard | None,
                     ) -> tuple[tuple[Hand, float], ...]:
    outcomes = tuple(policy(available_deck_mask, target, opposing_target))
    if not outcomes:
        raise ValueError("upper policy returned no outcomes")
    normalized, seen = [], set()
    for outcome in outcomes:
        if type(outcome) is not tuple or len(outcome) != 2:
            raise ValueError("upper policy outcomes must be (hand, probability) pairs")
        hand, raw_probability = outcome
        _validate_hand(hand, available_deck_mask, target)
        if hand in seen:
            raise ValueError("upper policy returned a duplicate hand")
        if isinstance(raw_probability, bool) or not isinstance(raw_probability, Real):
            raise TypeError("upper probabilities must be real numbers")
        probability = float(raw_probability)
        if not isfinite(probability) or probability <= 0.0:
            raise ValueError("upper probabilities must be finite and positive")
        normalized.append((hand, probability))
        seen.add(hand)
    if not isclose(sum(p for _, p in normalized), 1.0, abs_tol=1e-9):
        raise ValueError("upper policy probabilities must sum to one")
    return tuple(normalized)

def _apply_upper_hands(state: GameState, action: object) -> GameState:
    if type(action) is not tuple or len(action) != 2:
        raise TypeError("upper chance requires a two-hand tuple")
    metal_0, metal_1 = state.round.chosen_metal_cards
    if metal_0 is None or metal_1 is None:
        raise ValueError("upper chance requires two chosen metal cards")
    hands: HandPair = action
    _validate_hand(hands[0], state.available_deck_mask, metal_0)
    _validate_hand(hands[1], state.available_deck_mask, metal_1)
    if hands == (None, None):
        if state.round.retry:
            raise ValueError("retry chance outcomes exclude double no-rank")
        return replace(state, round=replace(state.round, retry=True))
    first = (other_team(state.history[-1].first_bettor) if state.history else
             Team.ZERO if metal_0 < metal_1 else Team.ONE)
    ante = ANTES[len(state.history)]
    betting = BettingState(first, first, (ante, ante))
    consumed = (hands[0] or 0) | (hands[1] or 0)
    return replace(state, phase=Phase.BETTING,
                   available_deck_mask=remove_cards(state.available_deck_mask, consumed),
                   round=replace(state.round, hands=hands, betting=betting))

def _apply_betting_action(state: GameState, action: BettingAction) -> GameState:
    if action not in legal_actions(state):
        raise ValueError("betting action is not legal in this state")
    betting = state.round.betting
    if betting is None:
        raise ValueError("betting phase requires betting state")
    actor, i = betting.actor, int(betting.actor)
    to_call = max(betting.contributions) - betting.contributions[i]
    actions = betting.actions + ((actor, action),)
    if action.action is RoundAction.Check:
        updated = replace(betting, actor=other_team(actor), previous_action_was_check=True,
                          actions=actions)
        successor = replace(state, round=replace(state.round, betting=updated))
        return (_settle_round(successor, _poker_winner(state.round.hands), True)
                if betting.previous_action_was_check else successor)
    if action.action is RoundAction.Fold:
        successor = replace(state, round=replace(
            state.round, betting=replace(betting, actions=actions)))
        return _settle_round(successor, other_team(actor), False)
    payment = to_call + action.extra
    lower = _change_bios(state.lower, actor, -payment)
    contributions = _replace_team(betting.contributions, actor,
                                  betting.contributions[i] + payment)
    updated = replace(betting, actor=other_team(actor), contributions=contributions,
                      previous_action_was_check=False, actions=actions)
    successor = replace(state, lower=lower,
                        round=replace(state.round, betting=updated))
    return (_settle_round(successor, _poker_winner(state.round.hands), True)
            if action.action is RoundAction.Call else successor)

def _poker_winner(hands: HandPair | None) -> Team | None:
    if hands is None or hands == (None, None):
        raise ValueError("judgement requires a resolved upper hand pair")
    hand_0, hand_1 = hands
    if hand_0 is None:
        return Team.ONE
    if hand_1 is None:
        return Team.ZERO
    score_0, score_1 = evaluate_mask(hand_0), evaluate_mask(hand_1)
    if score_0 == score_1:
        return None
    return Team.ZERO if score_0 > score_1 else Team.ONE

def _settle_round(state: GameState, winner: Team | None, showdown: bool) -> GameState:
    betting, hands = state.round.betting, state.round.hands
    if betting is None or hands is None:
        raise ValueError("round settlement requires hands and betting")
    lower = state.lower
    if winner is None:
        lower = _change_bios(lower, Team.ZERO, betting.contributions[0])
        lower = _change_bios(lower, Team.ONE, betting.contributions[1])
    else:
        lower = _change_bios(lower, winner, sum(betting.contributions))
    calamity = (showdown and winner is not None and hands[0] is not None
                and hands[1] is not None and bool(hands[0] & hands[1]))
    terminal_winner = None
    if calamity:
        loser, penalty = other_team(winner), betting.contributions[int(other_team(winner))]
        if lower[int(loser)].sealed_bios < penalty:
            lower = _replace_team(lower, loser, replace(lower[int(loser)], sealed_bios=0))
            terminal_winner = winner
        else:
            lower = _change_bios(lower, loser, -penalty)
    metal_0, metal_1 = state.round.chosen_metal_cards
    if metal_0 is None or metal_1 is None:
        raise ValueError("completed round requires two metal cards")
    record = RoundRecord((metal_0, metal_1), betting.first_bettor, betting.actions,
                         winner, calamity)
    history = state.history + (record,)
    settled = replace(state, lower=lower, history=history)
    if terminal_winner is not None:
        return replace(settled, phase=Phase.TERMINAL, winner=terminal_winner)
    if len(history) == len(ANTES):
        return _finish_by_air(settled)
    ante = ANTES[len(history)]
    can_pay = tuple(player.sealed_bios >= ante for player in lower)
    if not all(can_pay):
        next_winner = None if not any(can_pay) else Team.ZERO if can_pay[0] else Team.ONE
        return replace(settled, phase=Phase.TERMINAL, winner=next_winner)
    lower = _change_bios(lower, Team.ZERO, -ante)
    lower = _change_bios(lower, Team.ONE, -ante)
    return replace(settled, phase=Phase.NUMBER_CHOICE_0, lower=lower,
                   round=RoundState((None, None)), winner=None)

def _finish_by_air(state: GameState) -> GameState:
    totals = tuple(player.sealed_bios * BIO_SECONDS + player.remaining_air_seconds
                   for player in state.lower)
    winner = (None if totals[0] == totals[1] else
              Team.ZERO if totals[0] > totals[1] else Team.ONE)
    return replace(state, phase=Phase.TERMINAL, winner=winner)
