"""Self-play through the engine with the robust solver's strategies.

The solver keeps values and number strategies for prefixes of up to two
rounds; deeper decisions are recomputed on demand from the kernel boxes (a
round-4 subtree in under a second, a round-5 table in a tenth) and cached per
number prefix. Chance is drawn by the kernel sampler and applied through the
engine, so every transition of a self-play game is an engine transition.

``sequence_reach`` counts how often each of the 14,400 number sequences is
played when both lower players follow the lower-bound equilibrium and when
both follow the upper-bound one; the allocation loop samples in proportion.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from src.engine import game
from src.engine.game import ANTES, TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS
from src.kernel.sampler import sample_round_hands
from src.kernel.table import Sequences
from src.solver.robust import RobustSolver, betting_policy

WIN_DIGIT = {(game.Team.ZERO, False): 0, (game.Team.ZERO, True): 1, (game.Team.ONE, False): 2, (game.Team.ONE, True): 3}


def record_digit(record: game.RoundRecord) -> int:
    """Base-6 digit of a completed round as the lower players see it."""
    if record.actions and record.actions[-1][1].action is game.RoundAction.Fold:
        return 5
    if record.winner is None:
        return 4
    return WIN_DIGIT[(record.winner, record.calamity)]


def slot_action(slot: int, to_call: int) -> game.BettingAction:
    """The engine action of DAG action ``slot`` (check or fold, call, raises in order)."""
    if slot == 0:
        return game.BettingAction(game.RoundAction.Check if to_call == 0 else game.RoundAction.Fold)
    if to_call:
        return game.BettingAction(game.RoundAction.Call) if slot == 1 else game.BettingAction(game.RoundAction.Raise, slot - 1)
    return game.BettingAction(game.RoundAction.Raise, slot)


class Policy:
    """Engine actions from one robust equilibrium: ``use_hi`` picks the upper-bound game."""

    def __init__(self, solver: RobustSolver, use_hi: bool, cache: int = 32) -> None:
        self.solver = solver
        self.use_hi = use_hi
        self.cache = cache
        self.subtrees: OrderedDict[tuple, tuple] = OrderedDict()
        self.round_five: OrderedDict[tuple, tuple] = OrderedDict()

    def public(self, state: game.GameState) -> tuple[int, tuple, int]:
        """``(round, number prefix, class code)`` of the completed rounds."""
        prefix = tuple(record.chosen_metal_cards for record in state.history)
        code = 0
        for record in state.history:
            code = code * 6 + record_digit(record)
        return len(state.history), prefix, code

    def _cached(self, store: OrderedDict, key: tuple, compute):
        if key not in store:
            if len(store) >= self.cache:
                store.popitem(last=False)
            store[key] = compute()
        store.move_to_end(key)
        return store[key]

    def _subtree(self, prefix: tuple) -> tuple:
        solver = self.solver

        def compute():
            keep = solver.keep_depth
            solver.keep_depth = -1
            try:
                return solver._v(prefix, 3)
            finally:
                solver.keep_depth = keep

        return self._cached(self.subtrees, prefix, compute)

    def values(self, prefix: tuple, r: int) -> tuple[np.ndarray, np.ndarray]:
        """``V_r`` tables for a number prefix of length ``r`` (``r`` from 1 to 4)."""
        if r <= self.solver.keep_depth:
            return self.solver.kept_values[prefix]
        if r == 3:
            return self._subtree(prefix)[:2]
        return self._cached(self.round_five, prefix, lambda: self.solver._v(prefix, 4)[:2])

    def strategies(self, prefix: tuple, r: int) -> tuple:
        if r <= self.solver.keep_depth:
            return self.solver.kept_strategies[prefix]
        return self._subtree(prefix)[2]

    def number(self, state: game.GameState, rng: np.random.Generator) -> int:
        r, prefix, code = self.public(state)
        team = 0 if state.phase is game.Phase.NUMBER_CHOICE_0 else 1
        if r == ANTES.index(ANTES[-1]):
            return state.lower[team].remaining_metal_cards[0]
        rows, cols, chips, x_lo, y_lo, x_hi, y_hi = self.strategies(prefix, r)
        s0, s1 = (player.sealed_bios for player in state.lower)
        n = int(np.flatnonzero((chips[:, 0] == s0) & (chips[:, 1] == s1))[0])
        if team == 0:
            probabilities = (x_hi if self.use_hi else x_lo)[code, n]
            options = rows
        else:
            probabilities = (y_hi if self.use_hi else y_lo)[code, n]
            options = cols
        return int(options[rng.choice(len(options), p=probabilities / probabilities.sum())])

    def betting(self, state: game.GameState) -> game.BettingAction:
        r, prefix, code = self.public(state)
        pair = state.round.chosen_metal_cards
        full = prefix + (pair,)
        betting = state.round.betting
        ante = ANTES[r]
        s0 = state.lower[0].sealed_bios + betting.contributions[0] - ante
        s1 = state.lower[1].sealed_bios + betting.contributions[1] - ante
        index = self.solver.sequences.prefix_index[r][full]
        box_lo = self.solver.box_lo[r][index][code]
        box_hi = self.solver.box_hi[r][index][code]
        if r + 1 < len(ANTES):
            vnext_lo, vnext_hi = self.values(full, r + 1)
        else:
            vnext_lo = vnext_hi = np.zeros((1, 1, 1))
        actor = int(betting.actor)
        to_call = max(betting.contributions) - betting.contributions[actor]
        slot = betting_policy(r, int(betting.first_bettor), s0, s1, box_lo, box_hi, code, vnext_lo, vnext_hi,
                              self.use_hi, betting.contributions[0], betting.contributions[1],
                              int(betting.previous_action_was_check), *self.solver.arena.policy_arrays())
        return slot_action(int(slot), to_call)


def play_game(policies: tuple[Policy, Policy], seed: int, index: int) -> tuple[float, tuple]:
    """One game through the engine; return Team 0's utility and the number sequence played."""
    rng = np.random.default_rng([seed, index])
    state = game.initial_state()
    draw = 0
    while state.phase is not game.Phase.TERMINAL:
        if state.phase is game.Phase.NUMBER_CHOICE_0:
            state = game.apply(state, policies[0].number(state, rng))
        elif state.phase is game.Phase.NUMBER_CHOICE_1:
            state = game.apply(state, policies[1].number(state, rng))
        elif state.phase is game.Phase.UPPER_CHANCE:
            a, b = state.round.chosen_metal_cards
            _, _, hand_0, hand_1 = sample_round_hands(state.available_deck_mask, a, b, seed, index * 16 + draw)
            draw += 1
            state = game.apply(state, (hand_0, hand_1))
        else:
            actor = int(state.round.betting.actor)
            state = game.apply(state, policies[actor].betting(state))
    return game.terminal_returns(state)[0], tuple(record.chosen_metal_cards for record in state.history)


def sequence_reach(solver: RobustSolver, sequences: Sequences, games: int, seed: int) -> tuple[np.ndarray, dict]:
    """Reach per number sequence over ``games`` under each robust equilibrium, and the mean returns.

    A game that ends early (a bankruptcy) played a number prefix; its visit is
    shared equally among the sequences that extend that prefix.
    """
    reach = np.zeros(len(sequences))
    returns = {}
    for use_hi in (False, True):
        policy = Policy(solver, use_hi)
        total = 0.0
        for index in range(games):
            utility, played = play_game((policy, policy), seed + int(use_hi), index)
            total += utility
            key = tuple((int(a), int(b)) for a, b in played)
            members = sequences.prefix_of_sequence[len(key) - 1] == sequences.prefix_index[len(key) - 1][key]
            reach[members] += 1.0 / members.sum()
        returns["hi" if use_hi else "lo"] = total / games
    return reach, returns


__all__ = [
    "TEAM_0_METAL_CARDS",
    "TEAM_1_METAL_CARDS",
    "Policy",
    "play_game",
    "record_digit",
    "sequence_reach",
    "slot_action",
]
