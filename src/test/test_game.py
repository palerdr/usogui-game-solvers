"""Contract tests for the manga-grounded rules and declared solver abstractions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from unittest import TestCase

from src.engine.game import (
    ANTES,
    BIO_SECONDS,
    STARTING_BIOS,
    TEAM_0_METAL_CARDS,
    TEAM_1_METAL_CARDS,
    BettingAction,
    GameState,
    InformationState,
    Phase,
    RoundAction,
    Team,
    apply,
    current_actor,
    information_state,
    initial_state,
    iter_chance_outcomes,
    legal_actions,
    terminal_returns,
    uniform_upper_policy,
)
from src.engine.poker_math import (
    FULL_DECK_MASK,
    card_mask,
    generate_legal_hands,
    make_hand_mask,
    mask_popcount,
    parse_card,
    rank_sum,
)


def hand(*cards: str) -> int:
    return make_hand_mask(parse_card(card) for card in cards)


# Canonical Round-1 constructions: 36 is four eights plus four; 15 is A-2-3-4-5.
HAND_36 = hand("8c", "8d", "8h", "8s", "4d")
HAND_36_OVERLAP = hand("8c", "8d", "8h", "8s", "4c")
HAND_15 = hand("Ac", "2c", "3c", "4c", "5c")
HAND_25 = hand("3h", "4h", "5h", "6h", "7h")


def upper_chance_state() -> GameState:
    return apply(apply(initial_state(), 36), 15)


def betting_state(hand_0: int | None = HAND_36,
                  hand_1: int | None = HAND_15) -> GameState:
    return apply(upper_chance_state(), (hand_0, hand_1))


def check_check(state: GameState) -> GameState:
    state = apply(state, BettingAction(RoundAction.Check))
    return apply(state, BettingAction(RoundAction.Check))


class InitialStateAndRepresentationTests(TestCase):
    def test_chapters_429_to_431_initial_bios_air_and_first_ante(self) -> None:
        """One of 25 BIOS is open and the Round-1 ante is already committed."""
        state = initial_state()
        self.assertEqual(STARTING_BIOS, 25)
        self.assertEqual(BIO_SECONDS, 300)
        self.assertEqual(ANTES, (1, 2, 3, 4, 5))
        self.assertEqual(tuple(player.sealed_bios for player in state.lower), (23, 23))
        self.assertEqual(
            tuple(player.remaining_air_seconds for player in state.lower),
            (300, 300),
        )
        self.assertEqual(state.round.betting, None)

    def test_canonical_metal_cards_and_full_deck_are_loaded(self) -> None:
        state = initial_state()
        self.assertEqual(state.lower[0].remaining_metal_cards, TEAM_0_METAL_CARDS)
        self.assertEqual(state.lower[1].remaining_metal_cards, TEAM_1_METAL_CARDS)
        self.assertEqual(state.available_deck_mask, FULL_DECK_MASK)
        self.assertEqual(state.round.chosen_metal_cards, (None, None))
        self.assertEqual(state.history, ())
        self.assertIsNone(state.winner)

    def test_initial_input_rejects_invalid_resources_and_metal_cards(self) -> None:
        cases = (
            ((25, 25, 36, 39, 45), TEAM_1_METAL_CARDS, (300, 300)),
            ((5, 25, 36, 39, 45), TEAM_1_METAL_CARDS, (300, 300)),
            (TEAM_0_METAL_CARDS, (25, 15, 44, 47, 63), (300, 300)),
            (TEAM_0_METAL_CARDS[:-1], TEAM_1_METAL_CARDS, (300, 300)),
            (TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS, (-1, 300)),
            (TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS, (301, 300)),
        )
        for team_0, team_1, air in cases:
            with self.subTest(team_0=team_0, team_1=team_1, air=air):
                with self.assertRaises((TypeError, ValueError)):
                    initial_state(team_0, team_1, air)

    def test_states_and_actions_are_frozen_hashable_values(self) -> None:
        state = initial_state()
        action = BettingAction(RoundAction.Raise, 1)
        self.assertIsInstance(hash(state), int)
        self.assertIsInstance(hash(action), int)
        with self.assertRaises(FrozenInstanceError):
            state.phase = Phase.TERMINAL  # type: ignore[misc]

    def test_ground_truth_does_not_duplicate_derived_state(self) -> None:
        names = {field.name for field in fields(GameState)}
        self.assertEqual(
            names,
            {"phase", "lower", "available_deck_mask", "round", "history", "winner"},
        )
        self.assertNotIn("actor", names)
        self.assertNotIn("pot", names)
        self.assertNotIn("ante", names)
        self.assertNotIn("round_number", names)


class InformationBoundaryTests(TestCase):
    def test_team_one_cannot_observe_team_zeros_private_choice(self) -> None:
        initial = initial_state()
        chosen = apply(initial, 36)
        self.assertEqual(initial.round.chosen_metal_cards, (None, None))
        self.assertEqual(information_state(chosen, Team.ZERO).visible_metal_cards, (36, None))
        self.assertEqual(information_state(chosen, Team.ONE).visible_metal_cards, (None, None))
        self.assertEqual(current_actor(chosen), Team.ONE)

    def test_both_numbers_become_public_only_after_both_choices(self) -> None:
        state = upper_chance_state()
        for team in Team:
            with self.subTest(team=team):
                self.assertEqual(information_state(state, team).visible_metal_cards, (36, 15))
        self.assertIsNone(current_actor(state))
        self.assertEqual(legal_actions(state), ())

    def test_lower_information_omits_every_forbidden_private_fact(self) -> None:
        state = betting_state()
        observation = information_state(state, Team.ZERO)
        self.assertIsInstance(observation, InformationState)
        names = {field.name for field in fields(InformationState)}
        self.assertNotIn("available_deck_mask", names)
        self.assertNotIn("hands", names)
        self.assertNotIn("opponent_remaining_air_seconds", names)
        self.assertNotIn("opponent_remaining_metal_cards", names)
        self.assertEqual(observation.opponent_metal_card_count, 4)
        self.assertEqual(observation.visible_metal_cards, (36, 15))

    def test_own_number_is_removed_once_without_mutating_the_parent(self) -> None:
        parent = initial_state()
        child = apply(parent, 36)
        self.assertIn(36, parent.lower[0].remaining_metal_cards)
        self.assertNotIn(36, child.lower[0].remaining_metal_cards)
        self.assertEqual(len(child.lower[0].remaining_metal_cards), 4)
        with self.assertRaises(ValueError):
            apply(child, 36)


class UpperChanceAndSharedDeckTests(TestCase):
    def test_chapters_436_to_440_law_checks_count_sum_and_availability(self) -> None:
        state = upper_chance_state()
        self.assertEqual(mask_popcount(HAND_36), 5)
        self.assertEqual(rank_sum(HAND_36), 36)
        self.assertEqual(rank_sum(HAND_15), 15)
        with self.assertRaises(ValueError):
            apply(state, (HAND_15, HAND_15))  # Team 0's hand sums to the wrong target.
        four_cards = HAND_36 & ~card_mask(parse_card("4d"))
        with self.assertRaises(ValueError):
            apply(state, (four_cards, HAND_15))
        depleted = replace(state, available_deck_mask=state.available_deck_mask & ~card_mask(0))
        with self.assertRaises(ValueError):
            apply(depleted, (HAND_36, HAND_15))

    def test_both_hands_use_the_same_pre_round_deck_and_remove_the_union(self) -> None:
        state = apply(upper_chance_state(), (HAND_36_OVERLAP, HAND_15))
        self.assertEqual(mask_popcount(HAND_36_OVERLAP | HAND_15), 9)
        self.assertEqual(mask_popcount(state.available_deck_mask), 43)
        self.assertEqual(
            state.available_deck_mask,
            FULL_DECK_MASK & ~(HAND_36_OVERLAP | HAND_15),
        )

    def test_no_rank_loses_to_a_valid_hand_and_consumes_no_cards(self) -> None:
        state = apply(upper_chance_state(), (HAND_36, None))
        self.assertEqual(state.available_deck_mask, FULL_DECK_MASK & ~HAND_36)
        state = check_check(state)
        self.assertEqual(state.history[0].winner, Team.ZERO)
        self.assertFalse(state.history[0].calamity)

    def test_uniform_policy_is_exact_on_a_single_possible_hand(self) -> None:
        outcomes = uniform_upper_policy(HAND_15, 15, None)
        self.assertEqual(outcomes, ((HAND_15, 1.0),))
        self.assertEqual(uniform_upper_policy(HAND_15, 64, None), ((None, 1.0),))

    def test_initial_upper_policies_do_not_receive_the_opposing_target(self) -> None:
        seen: list[tuple[int, int | None]] = []

        def policy_0(deck: int, target: int, opposing: int | None):
            self.assertEqual(deck, FULL_DECK_MASK)
            seen.append((target, opposing))
            return ((HAND_36, 1.0),)

        def policy_1(deck: int, target: int, opposing: int | None):
            self.assertEqual(deck, FULL_DECK_MASK)
            seen.append((target, opposing))
            return ((HAND_15, 1.0),)

        outcomes = tuple(iter_chance_outcomes(upper_chance_state(), (policy_0, policy_1)))
        self.assertEqual(outcomes, (((HAND_36, HAND_15), 1.0),))
        self.assertEqual(seen, [(36, None), (15, None)])

    def test_chapters_448_to_456_double_no_rank_retries_with_both_targets_known(self) -> None:
        retry = apply(upper_chance_state(), (None, None))
        self.assertEqual(retry.phase, Phase.UPPER_CHANCE)
        self.assertTrue(retry.round.retry)
        self.assertEqual(retry.available_deck_mask, FULL_DECK_MASK)
        seen: list[tuple[int, int]] = []

        def policy_0(_deck: int, target: int, opposing: int | None):
            seen.append((target, opposing))
            return ((None, 0.25), (HAND_36, 0.75))

        def policy_1(_deck: int, target: int, opposing: int | None):
            seen.append((target, opposing))
            return ((None, 0.40), (HAND_15, 0.60))

        outcomes = tuple(iter_chance_outcomes(retry, (policy_0, policy_1)))
        self.assertEqual(seen, [(36, 15), (15, 36)])
        self.assertEqual(len(outcomes), 3)
        self.assertNotIn((None, None), tuple(outcome for outcome, _ in outcomes))
        self.assertAlmostEqual(sum(probability for _, probability in outcomes), 1.0)
        with self.assertRaises(ValueError):
            apply(retry, (None, None))

    def test_policy_distribution_validation_rejects_invalid_chance_models(self) -> None:
        state = upper_chance_state()

        def good_1(_deck: int, _target: int, _opposing: int | None):
            return ((HAND_15, 1.0),)

        invalid_policies = (
            lambda _d, _t, _o: (),
            lambda _d, _t, _o: ((HAND_36, 0.5),),
            lambda _d, _t, _o: ((HAND_36, 0.0), (None, 1.0)),
            lambda _d, _t, _o: ((HAND_36, 0.5), (HAND_36, 0.5)),
            lambda _d, _t, _o: ((HAND_15, 1.0),),
        )
        for policy in invalid_policies:
            with self.subTest(policy=policy), self.assertRaises((TypeError, ValueError)):
                tuple(iter_chance_outcomes(state, (policy, good_1)))


class BettingAndSettlementTests(TestCase):
    def test_round_one_first_bettor_is_the_lower_number(self) -> None:
        state = betting_state()
        self.assertEqual(state.round.betting.first_bettor, Team.ONE)
        self.assertEqual(current_actor(state), Team.ONE)

    def test_check_is_legal_at_parity_and_raise_is_capped_at_half_pot(self) -> None:
        state = betting_state()
        self.assertEqual(
            legal_actions(state),
            (BettingAction(RoundAction.Check), BettingAction(RoundAction.Raise, 1)),
        )
        state = apply(state, BettingAction(RoundAction.Raise, 1))
        actions = legal_actions(state)
        self.assertEqual(actions[:2], (
            BettingAction(RoundAction.Call), BettingAction(RoundAction.Fold),
        ))
        self.assertEqual(actions[-1], BettingAction(RoundAction.Raise, 2))
        self.assertNotIn(BettingAction(RoundAction.Check), actions)

    def test_canonical_round_one_raise_sequence_reaches_a_sixteen_bios_pot(self) -> None:
        state = betting_state()
        self.assertEqual(
            sum(player.sealed_bios for player in state.lower)
            + sum(state.round.betting.contributions),
            48,
        )
        for extra in (1, 2, 4):
            action = BettingAction(RoundAction.Raise, extra)
            self.assertIn(action, legal_actions(state))
            state = apply(state, action)
            self.assertEqual(
                sum(player.sealed_bios for player in state.lower)
                + sum(state.round.betting.contributions),
                48,
            )
        self.assertEqual(state.round.betting.contributions, (4, 8))
        self.assertEqual(sum(state.round.betting.contributions), 12)
        state = apply(state, BettingAction(RoundAction.Call))
        self.assertEqual(tuple(player.sealed_bios for player in state.lower), (14, 30))
        self.assertEqual(sum(player.sealed_bios for player in state.lower) + 2 * ANTES[1], 48)
        self.assertEqual([action.extra for _, action in state.history[0].actions], [1, 2, 4, 0])

    def test_round_two_first_bettor_alternates_and_canonical_pot_reaches_thirty(self) -> None:
        state = betting_state()
        for extra in (1, 2, 4):
            state = apply(state, BettingAction(RoundAction.Raise, extra))
        state = apply(state, BettingAction(RoundAction.Call))
        state = apply(apply(state, 25), 8)
        state = apply(state, (HAND_25, None))
        self.assertEqual(state.round.betting.first_bettor, Team.ZERO)
        for extra in (2, 1, 5, 1, 4):
            action = BettingAction(RoundAction.Raise, extra)
            self.assertIn(action, legal_actions(state))
            state = apply(state, action)
        self.assertEqual(state.round.betting.contributions, (15, 11))
        self.assertEqual(sum(state.round.betting.contributions), 26)
        state = apply(state, BettingAction(RoundAction.Call))
        self.assertEqual(len(state.history), 2)
        self.assertEqual(len(state.history[1].actions), 6)

    def test_chapter_444_overlap_at_judgement_causes_calamity_including_ante(self) -> None:
        state = check_check(betting_state(HAND_36_OVERLAP, HAND_15))
        self.assertEqual(state.history[0].winner, Team.ONE)
        self.assertTrue(state.history[0].calamity)
        # Winner gets the two-BIOS pot; loser destroys one more; both then ante two.
        self.assertEqual(tuple(player.sealed_bios for player in state.lower), (20, 23))
        self.assertEqual(sum(player.sealed_bios for player in state.lower), 43)

    def test_declared_fold_abstraction_skips_calamity_but_keeps_committed_cards(self) -> None:
        state = betting_state(HAND_36_OVERLAP, HAND_15)
        deck_after_commit = state.available_deck_mask
        state = apply(state, BettingAction(RoundAction.Raise, 1))
        state = apply(state, BettingAction(RoundAction.Fold))
        self.assertFalse(state.history[0].calamity)
        self.assertEqual(state.available_deck_mask, deck_after_commit)
        self.assertEqual(sum(player.sealed_bios for player in state.lower), 44)

    def test_nonpayment_of_calamity_penalty_is_immediate_defeat(self) -> None:
        state = betting_state(HAND_36_OVERLAP, HAND_15)
        lower_0 = replace(state.lower[0], sealed_bios=0)
        state = replace(state, lower=(lower_0, state.lower[1]))
        state = check_check(state)
        self.assertEqual(state.phase, Phase.TERMINAL)
        self.assertEqual(state.winner, Team.ONE)
        self.assertEqual(terminal_returns(state), (-1.0, 1.0))

    def test_inability_to_pay_the_next_ante_ends_the_game(self) -> None:
        state = betting_state()
        lower_0 = replace(state.lower[0], sealed_bios=1)
        state = replace(state, lower=(lower_0, state.lower[1]))
        state = check_check(state)
        self.assertEqual(state.phase, Phase.TERMINAL)
        self.assertEqual(state.winner, Team.ONE)

    def test_illegal_actions_are_rejected_without_mutating_state(self) -> None:
        state = betting_state()
        original = state
        illegal = (
            BettingAction(RoundAction.Call),
            BettingAction(RoundAction.Fold),
            BettingAction(RoundAction.Raise, 0),
            BettingAction(RoundAction.Raise, 2),
        )
        for action in illegal:
            with self.subTest(action=action), self.assertRaises(ValueError):
                apply(state, action)
        self.assertEqual(state, original)


class MatchTerminationTests(TestCase):
    def play_wins_for_team_zero(self, rounds: int) -> GameState:
        state = initial_state()
        for _ in range(rounds):
            target_0 = state.lower[0].remaining_metal_cards[0]
            target_1 = state.lower[1].remaining_metal_cards[0]
            state = apply(apply(state, target_0), target_1)
            valid_hand = generate_legal_hands(target_0, state.available_deck_mask)[0]
            state = check_check(apply(state, (valid_hand, None)))
        return state

    def test_chapter_430_match_has_at_most_five_rounds_and_rising_antes(self) -> None:
        state = initial_state()
        sealed_before_round = []
        for round_index in range(5):
            sealed_before_round.append(tuple(player.sealed_bios for player in state.lower))
            target_0 = state.lower[0].remaining_metal_cards[0]
            target_1 = state.lower[1].remaining_metal_cards[0]
            state = apply(apply(state, target_0), target_1)
            valid_hands = generate_legal_hands(target_0, state.available_deck_mask)
            self.assertTrue(valid_hands)
            state = apply(state, (valid_hands[0], None))
            state = check_check(state)
            if round_index < 4:
                self.assertEqual(state.phase, Phase.NUMBER_CHOICE_0)
        self.assertEqual(len(state.history), 5)
        self.assertEqual(state.phase, Phase.TERMINAL)
        self.assertTrue(all(not player.remaining_metal_cards for player in state.lower))
        self.assertEqual(terminal_returns(state), (1.0, -1.0))
        self.assertEqual(current_actor(state), None)
        self.assertEqual(legal_actions(state), ())
        with self.assertRaises(ValueError):
            apply(state, 25)
        # The losing side starts each later round lower by the next required ante.
        self.assertEqual([before[1] for before in sealed_before_round], [23, 21, 18, 14, 9])

    def test_terminal_returns_cover_both_winners_and_modeled_tie(self) -> None:
        state = initial_state()
        self.assertEqual(
            terminal_returns(replace(state, phase=Phase.TERMINAL, winner=Team.ZERO)),
            (1.0, -1.0),
        )
        self.assertEqual(
            terminal_returns(replace(state, phase=Phase.TERMINAL, winner=Team.ONE)),
            (-1.0, 1.0),
        )
        self.assertEqual(
            terminal_returns(replace(state, phase=Phase.TERMINAL, winner=None)),
            (0.0, 0.0),
        )
        with self.assertRaises(ValueError):
            terminal_returns(state)

    def test_declared_round_five_air_formula_uses_sealed_bios_and_residual_seconds(self) -> None:
        state = self.play_wins_for_team_zero(4)
        target_0 = state.lower[0].remaining_metal_cards[0]
        target_1 = state.lower[1].remaining_metal_cards[0]
        state = apply(apply(state, target_0), target_1)
        valid_hand = generate_legal_hands(target_0, state.available_deck_mask)[0]
        state = apply(state, (valid_hand, None))
        # Team 0 receives the ten-BIOS pot, producing ten sealed BIOS on each side;
        # Team 1's one residual second must therefore decide the modeled endurance tie.
        lower_0 = replace(state.lower[0], sealed_bios=0, remaining_air_seconds=0)
        lower_1 = replace(state.lower[1], sealed_bios=10, remaining_air_seconds=1)
        state = replace(state, lower=(lower_0, lower_1))
        state = check_check(state)
        self.assertEqual(state.phase, Phase.TERMINAL)
        self.assertEqual(state.winner, Team.ONE)
        self.assertEqual(terminal_returns(state), (-1.0, 1.0))
