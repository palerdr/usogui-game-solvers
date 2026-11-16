"""A finite Air Poker fixture small enough for exact tree traversal.

This deliberately does not reimplement ``src.engine.game``.  The production
engine models five metal cards and full five-card construction, whose chance
tree is not enumerable.  This fixture preserves the solver-relevant sequence:
two lower players make hidden sequential number choices, a complete correlated
upper result is drawn and remains hidden, and they play a check/bet street.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Final


class Team(IntEnum):
    """The two strategic lower players."""

    ZERO = 0
    ONE = 1


class NodeKind(Enum):
    """The three node types consumed by the exact solver."""

    TERMINAL = auto()
    CHANCE = auto()
    PLAYER = auto()


class Action(Enum):
    """The fixed-limit betting actions available in the miniature game."""

    CHECK = auto()
    BET = auto()
    CALL = auto()
    FOLD = auto()


@dataclass(frozen=True, slots=True)
class UpperHandPair:
    """A hidden correlated upper result; never split it into marginal events."""

    label: str
    winner: Team | None


@dataclass(frozen=True, slots=True)
class TinyRoundRecord:
    """Public facts from a completed round, retained for perfect recall."""

    chosen: tuple[int, int]
    betting_history: tuple[Action, ...]
    winner: Team | None


UPPER_OUTCOMES: Final[tuple[tuple[UpperHandPair, float], ...]] = (
    (UpperHandPair("zero-made-hand", Team.ZERO), 0.40),
    (UpperHandPair("one-made-hand", Team.ONE), 0.40),
    (UpperHandPair("split-pot", None), 0.20),
)


@dataclass(frozen=True, slots=True)
class TinyState:
    """Complete state; policies must use :meth:`TinyAirPoker.information_key`."""

    phase: str
    remaining: tuple[tuple[int, ...], tuple[int, ...]]
    chosen: tuple[int | None, int | None] = (None, None)
    upper: UpperHandPair | None = None
    first_bettor: Team | None = None
    actor: Team | None = None
    betting_history: tuple[Action, ...] = ()
    points: tuple[int, int] = (0, 0)
    completed_rounds: int = 0
    history: tuple[TinyRoundRecord, ...] = ()


class TinyAirPoker:
    """Two-round (or configured three-round) Air Poker exact fixture."""

    def __init__(
        self,
        cards: tuple[tuple[int, ...], tuple[int, ...]] = ((10, 20), (11, 19)),
        upper_outcomes: tuple[tuple[UpperHandPair, float], ...] = UPPER_OUTCOMES,
    ) -> None:
        if (type(cards) is not tuple or len(cards) != 2
                or any(type(hand) is not tuple or len(hand) not in (2, 3)
                       for hand in cards)):
            raise ValueError("cards must contain two hands with two or three metal cards")
        if any(type(card) is not int for hand in cards for card in hand):
            raise TypeError("metal cards must be integers")
        if any(len(set(hand)) != len(hand) for hand in cards):
            raise ValueError("a player's metal cards must be distinct")
        if len(cards[0]) != len(cards[1]):
            raise ValueError("both players must have the same number of cards")
        if len(set(cards[0] + cards[1])) != len(cards[0]) + len(cards[1]):
            raise ValueError("metal cards must be unique across players")
        if not upper_outcomes:
            raise ValueError("upper_outcomes cannot be empty")
        probability = 0.0
        labels: set[str] = set()
        for outcome, weight in upper_outcomes:
            if not isinstance(outcome, UpperHandPair):
                raise TypeError("upper outcomes must contain UpperHandPair values")
            if outcome.label in labels:
                raise ValueError("upper outcome labels must be unique")
            if type(weight) not in (float, int) or isinstance(weight, bool) or weight <= 0:
                raise ValueError("upper outcome weights must be positive real numbers")
            labels.add(outcome.label)
            probability += float(weight)
        if abs(probability - 1.0) > 1e-12:
            raise ValueError("upper outcome weights must sum to one")
        self._cards = cards
        self._upper_outcomes = tuple((outcome, float(weight)) for outcome, weight in upper_outcomes)

    def root(self) -> TinyState:
        return TinyState("choose-0", self._cards, actor=Team.ZERO)

    def kind(self, state: TinyState) -> NodeKind:
        if state.phase == "terminal":
            return NodeKind.TERMINAL
        if state.phase == "chance":
            return NodeKind.CHANCE
        return NodeKind.PLAYER

    def actor(self, state: TinyState) -> Team | None:
        return state.actor

    def actions(self, state: TinyState) -> tuple[int | Action, ...]:
        if self.kind(state) is not NodeKind.PLAYER or state.actor is None:
            return ()
        if state.phase.startswith("choose"):
            return state.remaining[int(state.actor)]
        history = state.betting_history
        if history in ((), (Action.CHECK,)):
            return (Action.CHECK, Action.BET)
        if history in ((Action.BET,), (Action.CHECK, Action.BET)):
            return (Action.CALL, Action.FOLD)
        raise ValueError("non-terminal state has an invalid betting history")

    def chance_outcomes(self, state: TinyState) -> tuple[tuple[UpperHandPair, float], ...]:
        if self.kind(state) is not NodeKind.CHANCE:
            raise ValueError("chance outcomes are only available at a chance node")
        return self._upper_outcomes

    def next_state(self, state: TinyState, action: int | Action | UpperHandPair) -> TinyState:
        """Apply one legal action without revealing an upper outcome to either lower."""
        if self.kind(state) is NodeKind.TERMINAL:
            raise ValueError("terminal states have no actions")
        if self.kind(state) is NodeKind.CHANCE:
            if not isinstance(action, UpperHandPair) or action not in dict(self._upper_outcomes):
                raise ValueError("unknown upper-hand outcome")
            assert state.chosen[0] is not None and state.chosen[1] is not None
            first = Team.ZERO if state.chosen[0] < state.chosen[1] else Team.ONE
            return TinyState("bet", state.remaining, state.chosen, action, first, first,
                             points=state.points, completed_rounds=state.completed_rounds,
                             history=state.history)
        if action not in self.actions(state):
            raise ValueError("illegal player action")
        if state.phase == "choose-0":
            assert type(action) is int
            remaining = (tuple(card for card in state.remaining[0] if card != action),
                         state.remaining[1])
            return TinyState("choose-1", remaining, (action, None), actor=Team.ONE,
                             points=state.points, completed_rounds=state.completed_rounds,
                             history=state.history)
        if state.phase == "choose-1":
            assert type(action) is int
            remaining = (state.remaining[0],
                         tuple(card for card in state.remaining[1] if card != action))
            return TinyState("chance", remaining, (state.chosen[0], action),
                             points=state.points, completed_rounds=state.completed_rounds,
                             history=state.history)
        assert isinstance(action, Action)
        history = state.betting_history + (action,)
        if history in ((Action.CHECK,), (Action.BET,), (Action.CHECK, Action.BET)):
            return TinyState("bet", state.remaining, state.chosen, state.upper,
                             state.first_bettor, self._other(state.actor), history,
                             state.points, state.completed_rounds, state.history)
        if history in ((Action.CHECK, Action.CHECK),
                       (Action.BET, Action.CALL),
                       (Action.CHECK, Action.BET, Action.CALL)):
            assert state.upper is not None
            return self._settle(state, state.upper.winner, history)
        if history == (Action.BET, Action.FOLD):
            return self._settle(state, state.first_bettor, history)
        if history == (Action.CHECK, Action.BET, Action.FOLD):
            return self._settle(state, self._other(state.first_bettor), history)
        raise ValueError("invalid completed betting history")

    def information_key(self, state: TinyState, team: Team) -> tuple[object, ...]:
        """Return exactly the acting lower's remembered visible observations."""
        if state.actor is not team:
            raise ValueError("information keys are only requested for the acting team")
        own_remaining = state.remaining[int(team)]
        own_choice = state.chosen[int(team)]
        visible_choices = state.chosen
        if state.phase == "choose-1":
            visible_choices = (None, own_choice)
        return (team, state.phase, own_remaining, own_choice, visible_choices,
                state.first_bettor, state.betting_history, state.points,
                state.completed_rounds, state.history)

    @staticmethod
    def returns(state: TinyState) -> tuple[float, float]:
        if state.phase != "terminal":
            raise ValueError("returns are only available at terminal states")
        if state.points[0] == state.points[1]:
            return (0.0, 0.0)
        return (1.0, -1.0) if state.points[0] > state.points[1] else (-1.0, 1.0)

    @staticmethod
    def _other(team: Team | None) -> Team:
        if team is Team.ZERO:
            return Team.ONE
        if team is Team.ONE:
            return Team.ZERO
        raise ValueError("a betting state requires an actor")

    def _settle(self, state: TinyState, winner: Team | None,
                betting_history: tuple[Action, ...]) -> TinyState:
        points = list(state.points)
        if winner is not None:
            points[int(winner)] += 1
        completed = state.completed_rounds + 1
        assert state.chosen[0] is not None and state.chosen[1] is not None
        history = state.history + (TinyRoundRecord(
            (state.chosen[0], state.chosen[1]), betting_history, winner),)
        if completed == len(self._cards[0]):
            return TinyState("terminal", state.remaining, points=tuple(points),
                             completed_rounds=completed, history=history)
        return TinyState("choose-0", state.remaining, actor=Team.ZERO,
                         points=tuple(points), completed_rounds=completed, history=history)


__all__ = (
    "Action", "NodeKind", "Team", "TinyAirPoker", "TinyRoundRecord", "TinyState",
    "UpperHandPair", "UPPER_OUTCOMES",
)
