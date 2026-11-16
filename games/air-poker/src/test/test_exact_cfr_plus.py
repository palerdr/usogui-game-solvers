"""Regression tests for the exact two-card Air Poker CFR+ fixture."""

from __future__ import annotations

from unittest import TestCase

from src.exact import CFRPlus, TinyAirPoker
from src.exact.tiny_air_poker import Action, NodeKind, Team


class TinyAirPokerTests(TestCase):
    def test_three_card_configuration_is_supported(self) -> None:
        game = TinyAirPoker(cards=((10, 20, 30), (11, 19, 29)))
        self.assertEqual(game.actions(game.root()), (10, 20, 30))

    def test_second_choice_hides_first_choice_until_the_pair_is_complete(self) -> None:
        game = TinyAirPoker()
        root = game.root()
        after_zero = game.next_state(root, 10)
        self.assertEqual(game.actor(after_zero), Team.ONE)
        key = game.information_key(after_zero, Team.ONE)
        self.assertEqual(key[4], (None, None))
        chance = game.next_state(after_zero, 11)
        self.assertEqual(game.kind(chance), NodeKind.CHANCE)

    def test_upper_chance_is_small_complete_and_normalized(self) -> None:
        game = TinyAirPoker()
        state = game.next_state(game.next_state(game.root(), 10), 11)
        outcomes = game.chance_outcomes(state)
        self.assertEqual(len(outcomes), 3)
        self.assertAlmostEqual(sum(probability for _, probability in outcomes), 1.0)
        self.assertEqual(len({outcome.label for outcome, _ in outcomes}), 3)

    def test_betting_settles_a_two_round_match(self) -> None:
        game = TinyAirPoker()
        state = game.root()
        for choice in (10, 11):
            state = game.next_state(state, choice)
        outcome = game.chance_outcomes(state)[0][0]
        state = game.next_state(state, outcome)
        self.assertEqual(game.actions(state), (Action.CHECK, Action.BET))
        state = game.next_state(game.next_state(state, Action.BET), Action.FOLD)
        self.assertEqual(state.completed_rounds, 1)
        self.assertEqual(state.points, (1, 0))
        self.assertEqual(state.history[0].chosen, (10, 11))
        self.assertEqual(state.history[0].betting_history, (Action.BET, Action.FOLD))


class CFRPlusTests(TestCase):
    def test_training_is_deterministic_and_regrets_are_nonnegative(self) -> None:
        left = CFRPlus(TinyAirPoker())
        right = CFRPlus(TinyAirPoker())
        result = left.train(25)
        right.train(25)
        self.assertEqual(result.information_sets, (141, 141))
        self.assertEqual(left.average_profile(), right.average_profile())
        self.assertTrue(all(regret >= 0.0
                            for table in left.tables
                            for entry in table.values()
                            for regret in entry.regret_sum))

    def test_average_profile_is_normalized_and_zero_sum(self) -> None:
        solver = CFRPlus(TinyAirPoker(), average_delay=3)
        solver.train(100)
        for table in solver.average_profile():
            for strategy in table.values():
                self.assertAlmostEqual(sum(strategy), 1.0)
                self.assertTrue(all(probability >= 0.0 for probability in strategy))
        value = solver.evaluate_average_profile()
        self.assertAlmostEqual(sum(value), 0.0)
