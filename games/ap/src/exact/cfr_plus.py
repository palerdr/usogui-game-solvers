"""Alternating full-tree CFR+ for finite, two-player, zero-sum games.

The implementation is intentionally small and exact: every chance outcome and
every lower action is traversed on every pass.  It is suitable for
``TinyAirPoker`` and similarly small fixtures only, never the canonical Air
Poker chance tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Generic, Hashable, Protocol, TypeVar

from .tiny_air_poker import NodeKind, Team


State = TypeVar("State")
PlayerAction = TypeVar("PlayerAction", bound=Hashable)
InfoKey = TypeVar("InfoKey", bound=Hashable)
ChanceAction = TypeVar("ChanceAction", bound=Hashable)


class ExactGame(Protocol[State, PlayerAction, InfoKey, ChanceAction]):
    """The minimal immutable game interface consumed by :class:`CFRPlus`."""

    def root(self) -> State: ...
    def kind(self, state: State) -> NodeKind: ...
    def actor(self, state: State) -> Team | None: ...
    def actions(self, state: State) -> tuple[PlayerAction, ...]: ...
    def chance_outcomes(self, state: State) -> tuple[tuple[ChanceAction, float], ...]: ...
    def next_state(self, state: State, action: PlayerAction | ChanceAction) -> State: ...
    def information_key(self, state: State, team: Team) -> InfoKey: ...
    def returns(self, state: State) -> tuple[float, float]: ...


@dataclass(slots=True)
class InfoSetEntry(Generic[PlayerAction]):
    """CFR+ regrets and reach-weighted average-policy mass for one information set."""

    actions: tuple[PlayerAction, ...]
    regret_sum: list[float]
    strategy_sum: list[float]

    @classmethod
    def create(cls, actions: tuple[PlayerAction, ...]) -> "InfoSetEntry[PlayerAction]":
        if not actions:
            raise ValueError("an information set requires at least one action")
        return cls(actions, [0.0] * len(actions), [0.0] * len(actions))

    def strategy(self) -> tuple[float, ...]:
        """Regret-matching+ with an exact uniform fallback."""
        total = sum(self.regret_sum)
        if total <= 0.0:
            return (1.0 / len(self.actions),) * len(self.actions)
        return tuple(regret / total for regret in self.regret_sum)

    def average_strategy(self) -> tuple[float, ...]:
        total = sum(self.strategy_sum)
        if total <= 0.0:
            return (1.0 / len(self.actions),) * len(self.actions)
        return tuple(weight / total for weight in self.strategy_sum)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """A compact immutable training summary."""

    iterations: int
    information_sets: tuple[int, int]


class CFRPlus(Generic[State, PlayerAction, InfoKey, ChanceAction]):
    """Exact alternating CFR+ with delayed linearly weighted averaging.

    Regret deltas are accumulated over every history in an information set, then
    clipped once with ``max(R + delta, 0)``.  Iterations update Team.ZERO then
    Team.ONE.  The pass updating one player accumulates the other player's
    own-reach weighted average strategy, producing one average update per
    player per iteration.
    """

    def __init__(self, game: ExactGame[State, PlayerAction, InfoKey, ChanceAction],
                 average_delay: int = 0) -> None:
        if type(average_delay) is not int or average_delay < 0:
            raise ValueError("average_delay must be a non-negative integer")
        self.game = game
        self.average_delay = average_delay
        self.tables: tuple[dict[InfoKey, InfoSetEntry[PlayerAction]],
                           dict[InfoKey, InfoSetEntry[PlayerAction]]] = ({}, {})
        self.iterations = 0

    def train(self, iterations: int) -> TrainingResult:
        """Run exactly ``iterations`` alternating CFR+ iterations."""
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        for _ in range(iterations):
            self.iterations += 1
            average_weight = float(max(self.iterations - self.average_delay, 0))
            for updating in (Team.ZERO, Team.ONE):
                deltas: dict[InfoKey, list[float]] = {}
                self._traverse(self.game.root(), updating, 1.0, 1.0, 1.0,
                               average_weight, deltas)
                for key, delta in deltas.items():
                    entry = self.tables[int(updating)][key]
                    entry.regret_sum[:] = [max(regret + change, 0.0)
                                           for regret, change in zip(entry.regret_sum, delta)]
        return TrainingResult(self.iterations,
                              (len(self.tables[0]), len(self.tables[1])))

    def average_profile(self) -> tuple[dict[InfoKey, tuple[float, ...]],
                                       dict[InfoKey, tuple[float, ...]]]:
        """Return normalized average strategies keyed solely by information state."""
        return tuple({key: entry.average_strategy() for key, entry in table.items()}
                     for table in self.tables)  # type: ignore[return-value]

    def current_profile(self) -> tuple[dict[InfoKey, tuple[float, ...]],
                                       dict[InfoKey, tuple[float, ...]]]:
        """Return current regret-matching+ strategies at every visited set."""
        return tuple({key: entry.strategy() for key, entry in table.items()}
                     for table in self.tables)  # type: ignore[return-value]

    def evaluate_average_profile(self) -> tuple[float, float]:
        """Exactly evaluate the current average strategy profile from the root."""
        return self._evaluate(self.game.root(), self.average_profile())

    def _entry(self, team: Team, key: InfoKey,
               actions: tuple[PlayerAction, ...]) -> InfoSetEntry[PlayerAction]:
        table = self.tables[int(team)]
        entry = table.get(key)
        if entry is None:
            entry = InfoSetEntry.create(actions)
            table[key] = entry
        elif entry.actions != actions:
            raise ValueError("the same information state exposed different ordered actions")
        return entry

    def _traverse(self, state: State, updating: Team, reach_zero: float,
                  reach_one: float, chance_reach: float, average_weight: float,
                  deltas: dict[InfoKey, list[float]]) -> tuple[float, float]:
        kind = self.game.kind(state)
        if kind is NodeKind.TERMINAL:
            return self.game.returns(state)
        if kind is NodeKind.CHANCE:
            value = [0.0, 0.0]
            for action, probability in self._chance_outcomes(state):
                child = self._traverse(self.game.next_state(state, action), updating,
                                       reach_zero, reach_one, chance_reach * probability,
                                       average_weight, deltas)
                value[0] += probability * child[0]
                value[1] += probability * child[1]
            return (value[0], value[1])

        actor = self.game.actor(state)
        if actor not in (Team.ZERO, Team.ONE):
            raise ValueError("a player node requires a lower-player actor")
        actions = self.game.actions(state)
        key = self.game.information_key(state, actor)
        entry = self._entry(actor, key, actions)
        strategy = entry.strategy()
        if actor is not updating and average_weight:
            own_reach = reach_zero if actor is Team.ZERO else reach_one
            for index, probability in enumerate(strategy):
                entry.strategy_sum[index] += average_weight * own_reach * probability

        children: list[tuple[float, float]] = []
        for index, action in enumerate(actions):
            child_reach_zero = reach_zero * strategy[index] if actor is Team.ZERO else reach_zero
            child_reach_one = reach_one * strategy[index] if actor is Team.ONE else reach_one
            children.append(self._traverse(self.game.next_state(state, action), updating,
                                           child_reach_zero, child_reach_one, chance_reach,
                                           average_weight, deltas))
        value = tuple(sum(strategy[index] * children[index][team]
                          for index in range(len(actions))) for team in range(2))
        if actor is updating:
            own_value = value[int(actor)]
            counterfactual_reach = chance_reach * (reach_one if actor is Team.ZERO else reach_zero)
            delta = deltas.setdefault(key, [0.0] * len(actions))
            for index, child in enumerate(children):
                delta[index] += counterfactual_reach * (child[int(actor)] - own_value)
        return value

    def _evaluate(self, state: State,
                  profile: tuple[dict[InfoKey, tuple[float, ...]],
                                 dict[InfoKey, tuple[float, ...]]]) -> tuple[float, float]:
        kind = self.game.kind(state)
        if kind is NodeKind.TERMINAL:
            return self.game.returns(state)
        if kind is NodeKind.CHANCE:
            value = [0.0, 0.0]
            for action, probability in self._chance_outcomes(state):
                child = self._evaluate(self.game.next_state(state, action), profile)
                value[0] += probability * child[0]
                value[1] += probability * child[1]
            return (value[0], value[1])
        actor = self.game.actor(state)
        if actor not in (Team.ZERO, Team.ONE):
            raise ValueError("a player node requires a lower-player actor")
        key = self.game.information_key(state, actor)
        entry = self._entry(actor, key, self.game.actions(state))
        strategy = profile[int(actor)].get(key, entry.average_strategy())
        value = [0.0, 0.0]
        for index, action in enumerate(entry.actions):
            child = self._evaluate(self.game.next_state(state, action), profile)
            value[0] += strategy[index] * child[0]
            value[1] += strategy[index] * child[1]
        return (value[0], value[1])

    def _chance_outcomes(self, state: State) -> tuple[tuple[ChanceAction, float], ...]:
        outcomes = self.game.chance_outcomes(state)
        if not outcomes or any(probability <= 0.0 for _, probability in outcomes):
            raise ValueError("chance probabilities must be positive")
        if not isclose(sum(probability for _, probability in outcomes), 1.0,
                         rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("chance probabilities must sum to one")
        return outcomes


__all__ = ("CFRPlus", "ExactGame", "InfoSetEntry", "TrainingResult")
