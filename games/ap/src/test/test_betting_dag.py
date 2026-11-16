"""The betting DAG and settlement agree with the engine's legal actions and transitions."""

import unittest

import numpy as np

from src.engine import game
from src.engine.poker_math import FULL_DECK_MASK
from src.sensitivity.sampler import NOJ, W0, W0C, W1, W1C, HandTables
from src.solver.betting import BettingDag, evaluate, reachable_configurations, settle

TABLES = HandTables()
TEAM = (game.Team.ZERO, game.Team.ONE)


def hands_of_class(result: int) -> game.HandPair:
    """A full-deck legal hand pair for targets (25, 44) whose showdown gives ``result``."""
    masks_0, ranks_0 = TABLES.masks[25], TABLES.ranks[25]
    masks_1, ranks_1 = TABLES.masks[44], TABLES.ranks[44]
    for i in range(0, masks_0.size, 97):
        for j in range(0, masks_1.size, 89):
            overlap = bool(int(masks_0[i]) & int(masks_1[j]))
            win_0 = ranks_0[i] > ranks_1[j]
            if result == W0 and win_0 and not overlap or result == W0C and win_0 and overlap \
                    or result == W1 and not win_0 and not overlap or result == W1C and not win_0 and overlap:
                return int(masks_0[i]), int(masks_1[j])
    raise AssertionError(result)


def betting_state(round_index: int, sealed: tuple[int, int], first: int, contributions: tuple[int, int],
                  actor: int, checked: bool, hands: game.HandPair) -> game.GameState:
    """An engine state in the betting phase with the given public facts."""
    lower = (game.LowerPlayerState(sealed[0], 300, (26, 36, 39, 45)),
             game.LowerPlayerState(sealed[1], 300, (8, 15, 47, 63)))
    record = game.RoundRecord((25, 44), TEAM[1 - first], (), game.Team.ZERO, False)
    betting = game.BettingState(TEAM[first], TEAM[actor], contributions, checked)
    return game.GameState(game.Phase.BETTING, lower, FULL_DECK_MASK & ~(hands[0] | hands[1]),
                          game.RoundState((25, 44), hands, False, betting), (record,) * round_index)


def engine_value(state: game.GameState, dag: BettingDag, showdown_value, fold_value, memo=None) -> float:
    """Perfect-information value over the engine's own legal actions with table payoffs.

    ``memo`` keys the public betting facts so the walk visits each engine state
    once per distinct ``(contributions, actor, checked)``; legality and
    transitions still come from the engine at every visit.
    """
    betting = state.round.betting
    actor = int(betting.actor)
    key = (betting.contributions, actor, betting.previous_action_was_check)
    memo = {} if memo is None else memo
    if key in memo:
        return memo[key]
    showdown_index = {tuple(row): k for k, row in enumerate(dag.showdowns.tolist())}
    fold_index = {tuple(row): k for k, row in enumerate(dag.folds.tolist())}
    best = None
    for action in game.legal_actions(state):
        successor = game.apply(state, action)
        if len(successor.history) > len(state.history):
            c = list(betting.contributions)
            if action.action is game.RoundAction.Fold:
                v = fold_value[fold_index[(c[0], c[1], actor)]]
            else:
                if action.action is game.RoundAction.Call:
                    c[actor] = max(c)
                v = showdown_value[showdown_index[(c[0], c[1])]]
        else:
            v = engine_value(successor, dag, showdown_value, fold_value, memo)
        best = v if best is None else (max(best, v) if actor == 0 else min(best, v))
    memo[key] = best
    return best


class BettingDagTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reach = reachable_configurations()

    def test_reachable_configuration_counts_match_the_measured_reductions(self):
        self.assertEqual([len(r) for r in self.reach], [2, 134, 1066, 1394, 1458])
        self.assertEqual(len(BettingDag(23, 23, 1, 0)), 290)

    def test_dag_value_matches_engine_expectimax_on_sampled_configurations(self):
        rng = np.random.default_rng(3)
        hands = hands_of_class(W0)
        checked = 0
        for round_index in range(5):
            configs = sorted(self.reach[round_index])
            for k in rng.choice(len(configs), size=min(6, len(configs)), replace=False):
                s0, s1, first = configs[k]
                dag = BettingDag(s0, s1, game.ANTES[round_index], first)
                showdown_value = rng.uniform(-1, 1, size=len(dag.showdowns))
                fold_value = rng.uniform(-1, 1, size=len(dag.folds))
                expected = engine_value(betting_state(round_index, (s0, s1), first, (game.ANTES[round_index],) * 2,
                                                      first, False, hands), dag, showdown_value, fold_value)
                got = evaluate(dag.actor, dag.action_start, dag.action_target, showdown_value, fold_value,
                               np.empty(len(dag)))
                self.assertAlmostEqual(got, expected, places=12, msg=(round_index, s0, s1, first))
                checked += 1
        self.assertGreaterEqual(checked, 20)

    def test_settlement_matches_the_engine_for_every_class_and_fold(self):
        rng = np.random.default_rng(4)
        cases = 0
        for round_index in range(5):
            configs = sorted(self.reach[round_index])
            for k in rng.choice(len(configs), size=min(8, len(configs)), replace=False):
                s0, s1, first = configs[k]
                ante = game.ANTES[round_index]
                dag = BettingDag(s0, s1, ante, first)
                for c0, c1 in dag.showdowns[rng.choice(len(dag.showdowns), size=min(5, len(dag.showdowns)), replace=False)]:
                    sealed = (s0 - (c0 - ante), s1 - (c1 - ante))
                    for result in (W0, W0C, W1, W1C):
                        state = betting_state(round_index, sealed, first, (int(c0), int(c1)), 1 - first, True,
                                              hands_of_class(result))
                        self.compare(round_index, s0, s1, int(c0), int(c1), result,
                                     game.apply(state, game.BettingAction(game.RoundAction.Check)))
                        cases += 1
                for c0, c1, folder in dag.folds[rng.choice(len(dag.folds), size=min(3, len(dag.folds)), replace=False)]:
                    sealed = (s0 - (c0 - ante), s1 - (c1 - ante))
                    state = betting_state(round_index, sealed, first, (int(c0), int(c1)), int(folder), False,
                                          hands_of_class(W0))
                    self.compare(round_index, s0, s1, int(c0), int(c1), -1 - int(folder),
                                 game.apply(state, game.BettingAction(game.RoundAction.Fold)))
                    cases += 1
        self.assertGreater(cases, 100)

    def compare(self, round_index, s0, s1, c0, c1, result, successor) -> None:
        terminal, utility, t0, t1 = settle(round_index, s0, s1, c0, c1, result)
        context = (round_index, s0, s1, c0, c1, result)
        if successor.phase is game.Phase.TERMINAL:
            self.assertTrue(terminal, context)
            self.assertEqual(utility, game.terminal_returns(successor)[0], context)
        else:
            self.assertFalse(terminal, context)
            self.assertEqual((t0, t1), tuple(p.sealed_bios for p in successor.lower), context)
            self.assertIs(successor.phase, game.Phase.NUMBER_CHOICE_0)

    def test_no_judgement_returns_both_antes_through_the_engine(self):
        for round_index in (3, 4):
            ante = game.ANTES[round_index]
            s0, s1, first = sorted(self.reach[round_index])[7]
            lower = (game.LowerPlayerState(s0, 300, (26,)), game.LowerPlayerState(s1, 300, (8,)))
            record = game.RoundRecord((25, 44), TEAM[1 - first], (), game.Team.ZERO, False)
            state = game.GameState(game.Phase.UPPER_CHANCE, lower, 0, game.RoundState((36, 63), None, True),
                                   (record,) * round_index)
            self.compare(round_index, s0, s1, ante, ante, NOJ, game.apply(state, (None, None)))


if __name__ == "__main__":
    unittest.main()
