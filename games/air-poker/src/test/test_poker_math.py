"""Exhaustive and boundary tests for the card-mask and poker evaluator core."""

from __future__ import annotations

from itertools import combinations
from unittest import TestCase

from src.engine.poker_math import (
    CARD_MASKS,
    CATEGORY_NAMES,
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_DECK_MASK,
    FULL_HOUSE,
    HAND_SIZE,
    HIGH_CARD,
    NUM_CARDS,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    card_mask,
    card_rank,
    card_suit,
    cards_to_mask,
    category_of_rank,
    evaluate_mask,
    format_card,
    generate_legal_hands,
    iter_mask,
    legal_cards_mask,
    make_card,
    make_hand_mask,
    mask_contains,
    mask_popcount,
    parse_card,
    rank_sum,
    remove_cards,
)


def hand(*cards: str) -> int:
    """Build a five-card mask from readable card names."""
    return make_hand_mask(parse_card(card) for card in cards)


class CardIndexTests(TestCase):
    def test_all_52_cards_round_trip_through_every_representation(self) -> None:
        for suit in range(4):
            for rank in range(13):
                with self.subTest(rank=rank, suit=suit):
                    card = make_card(rank, suit)
                    self.assertEqual(card, suit * 13 + rank)
                    self.assertEqual(card_rank(card), rank)
                    self.assertEqual(card_suit(card), suit)
                    self.assertEqual(parse_card(format_card(card)), card)
                    self.assertEqual(card_mask(card), 1 << card)

    def test_canonical_index_examples_match_the_rust_layout(self) -> None:
        self.assertEqual(parse_card("2c"), 0)
        self.assertEqual(parse_card("Ac"), 12)
        self.assertEqual(parse_card("2d"), 13)
        self.assertEqual(parse_card("Th"), 34)
        self.assertEqual(parse_card("As"), 51)

    def test_parser_normalizes_case_but_rejects_ambiguous_text(self) -> None:
        self.assertEqual(parse_card("aC"), parse_card("Ac"))
        for invalid in ("10h", "A", "Ace", "1c", "Tx", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_card(invalid)
        with self.assertRaises(TypeError):
            parse_card(12)  # type: ignore[arg-type]

    def test_card_boundaries_and_bool_inputs_are_rejected(self) -> None:
        invalid_calls = (
            lambda: make_card(-1, 0),
            lambda: make_card(13, 0),
            lambda: make_card(0, -1),
            lambda: make_card(0, 4),
            lambda: make_card(True, 0),
            lambda: card_mask(-1),
            lambda: card_mask(52),
            lambda: card_mask("Ac"),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()


class CardMaskTests(TestCase):
    def test_full_deck_occupies_exactly_the_low_52_bits(self) -> None:
        self.assertEqual(FULL_DECK_MASK, (1 << NUM_CARDS) - 1)
        self.assertEqual(mask_popcount(FULL_DECK_MASK), 52)
        self.assertEqual(tuple(iter_mask(FULL_DECK_MASK)), tuple(range(52)))

    def test_mask_construction_membership_iteration_and_removal(self) -> None:
        cards = (0, 18, 31, 51)
        mask = cards_to_mask(reversed(cards))
        self.assertEqual(mask, sum(1 << card for card in cards))
        self.assertEqual(tuple(iter_mask(mask)), cards)
        self.assertTrue(all(mask_contains(mask, card) for card in cards))
        self.assertFalse(mask_contains(mask, 1))
        self.assertEqual(mask_popcount(mask), 4)
        self.assertEqual(remove_cards(mask, (18, 31)), cards_to_mask((0, 51)))
        self.assertEqual(
            remove_cards(mask, card_mask(18) | card_mask(31)),
            cards_to_mask((0, 51)),
        )

    def test_legal_mask_is_the_bounded_complement_of_both_exclusions(self) -> None:
        used = cards_to_mask((0, 12, 24))
        forbidden = cards_to_mask((1, 13, 25))
        legal = legal_cards_mask(used, forbidden)
        self.assertEqual(legal, FULL_DECK_MASK & ~(used | forbidden))
        self.assertEqual(mask_popcount(legal), 46)

    def test_hand_masks_require_five_distinct_cards(self) -> None:
        self.assertEqual(mask_popcount(make_hand_mask(range(HAND_SIZE))), HAND_SIZE)
        with self.assertRaises(ValueError):
            make_hand_mask(range(4))
        with self.assertRaises(ValueError):
            make_hand_mask(range(6))
        with self.assertRaises(ValueError):
            cards_to_mask((0, 0))
        with self.assertRaises(TypeError):
            cards_to_mask("Ac Kd")

    def test_masks_reject_negative_and_out_of_deck_bits(self) -> None:
        for invalid in (-1, 1 << 52, FULL_DECK_MASK | (1 << 60), True):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                mask_popcount(invalid)


class AirPokerArithmeticTests(TestCase):
    def test_air_poker_values_are_ace_one_through_king_thirteen(self) -> None:
        expected_values = tuple(range(2, 14)) + (1,)
        for rank, expected in enumerate(expected_values):
            with self.subTest(rank=rank):
                four_suits = cards_to_mask(make_card(rank, suit) for suit in range(4))
                self.assertEqual(rank_sum(four_suits), expected * 4)

    def test_rank_sum_is_independent_of_suit(self) -> None:
        clubs = hand("Ac", "2c", "3c", "4c", "5c")
        mixed = hand("As", "2d", "3h", "4s", "5d")
        self.assertEqual(rank_sum(clubs), 15)
        self.assertEqual(rank_sum(mixed), 15)

    def test_minimum_and_maximum_five_card_sums_are_six_and_sixty_four(self) -> None:
        minimum = generate_legal_hands(6, FULL_DECK_MASK)
        maximum = generate_legal_hands(64, FULL_DECK_MASK)
        self.assertEqual(len(minimum), 4)  # four aces plus one of four deuces
        self.assertEqual(len(maximum), 4)  # four kings plus one of four queens
        self.assertTrue(all(rank_sum(candidate) == 6 for candidate in minimum))
        self.assertTrue(all(rank_sum(candidate) == 64 for candidate in maximum))
        self.assertEqual(generate_legal_hands(5, FULL_DECK_MASK), ())
        self.assertEqual(generate_legal_hands(65, FULL_DECK_MASK), ())

    def test_generator_matches_brute_force_for_every_sum_on_a_small_deck(self) -> None:
        available_cards = tuple(map(parse_card, (
            "Ac", "2c", "3c", "4c", "5c", "6d", "7d", "8h", "Qs", "Ks",
        )))
        available_mask = cards_to_mask(available_cards)
        expected: dict[int, set[int]] = {}
        for cards in combinations(available_cards, 5):
            candidate = cards_to_mask(cards)
            expected.setdefault(rank_sum(candidate), set()).add(candidate)
        for target in range(6, 65):
            with self.subTest(target=target):
                actual = generate_legal_hands(target, available_mask)
                self.assertEqual(set(actual), expected.get(target, set()))
                self.assertEqual(len(actual), len(set(actual)))

    def test_generator_respects_depletion_and_returns_deterministic_order(self) -> None:
        only_hand = hand("Ac", "2d", "3h", "4s", "5c")
        self.assertEqual(generate_legal_hands(15, only_hand), (only_hand,))
        depleted = remove_cards(only_hand, parse_card("Ac"))
        self.assertEqual(generate_legal_hands(15, depleted), ())
        first = generate_legal_hands(15, FULL_DECK_MASK)
        second = generate_legal_hands(15, FULL_DECK_MASK)
        self.assertEqual(first, second)
        self.assertTrue(all(mask & ~FULL_DECK_MASK == 0 for mask in first))


class FiveCardEvaluatorTests(TestCase):
    def test_representative_categories_are_ordered_weakest_to_strongest(self) -> None:
        hands = (
            hand("Ac", "Kd", "Qh", "9s", "3c"),
            hand("Ac", "As", "Kd", "Qh", "Jh"),
            hand("Ac", "As", "Kd", "Kc", "Jh"),
            hand("Ac", "As", "Ad", "Kd", "Qh"),
            hand("Ac", "Kc", "Qd", "Jh", "Td"),
            hand("Ac", "Kc", "Qc", "9c", "2c"),
            hand("Ac", "Ad", "Ah", "Kc", "Kd"),
            hand("Ac", "Ad", "Ah", "As", "Kd"),
            hand("Ac", "Kc", "Qc", "Jc", "Tc"),
        )
        scores = tuple(map(evaluate_mask, hands))
        self.assertEqual(scores, tuple(sorted(scores)))
        self.assertEqual(tuple(map(category_of_rank, scores)), tuple(range(9)))
        self.assertEqual(len(CATEGORY_NAMES), 9)

    def assert_stronger(self, stronger: int, weaker: int) -> None:
        self.assertGreater(evaluate_mask(stronger), evaluate_mask(weaker))

    def test_every_category_uses_standard_tie_breakers(self) -> None:
        cases = (
            (hand("Ac", "Kc", "Qc", "Jc", "Tc"), hand("Kc", "Qc", "Jc", "Tc", "9c")),
            (hand("Ac", "Ad", "Ah", "As", "Kd"), hand("Kc", "Kd", "Kh", "Ks", "Ad")),
            (hand("Ac", "Ad", "Ah", "Kc", "Kd"), hand("Kc", "Kd", "Kh", "As", "Ad")),
            (hand("Ac", "Ad", "Ah", "Kc", "Qd"), hand("As", "Ah", "Ac", "Kd", "Jc")),
            (hand("Ac", "Ad", "Kc", "Kd", "Qh"), hand("As", "Ah", "Qc", "Qd", "Kh")),
            (hand("Ac", "Ad", "Kc", "Qd", "Jh"), hand("As", "Ah", "Kd", "Qc", "Th")),
            (hand("Ac", "Qc", "9c", "6c", "3c"), hand("Ad", "Qd", "9d", "5d", "4d")),
            (hand("Ac", "Qd", "9h", "6s", "3c"), hand("As", "Qh", "9d", "5s", "4c")),
            (hand("6h", "5d", "4c", "3s", "2d"), hand("5h", "4d", "3c", "2s", "As")),
        )
        for stronger, weaker in cases:
            with self.subTest(stronger=stronger, weaker=weaker):
                self.assert_stronger(stronger, weaker)

    def test_suits_do_not_break_equal_poker_strength(self) -> None:
        clubs_high = hand("Ac", "Qh", "9d", "6s", "3c")
        spades_high = hand("As", "Qd", "9h", "6c", "3s")
        self.assertEqual(evaluate_mask(clubs_high), evaluate_mask(spades_high))

    def test_evaluator_rejects_every_non_five_card_shape(self) -> None:
        for size in (0, 1, 4, 6, 52):
            with self.subTest(size=size), self.assertRaises(ValueError):
                evaluate_mask(cards_to_mask(range(size)))
        with self.assertRaises(ValueError):
            evaluate_mask((1 << 52) | cards_to_mask(range(5)))

    def test_category_extraction_rejects_malformed_values(self) -> None:
        self.assertEqual(category_of_rank(evaluate_mask(hand("Ac", "Kc", "Qc", "Jc", "Tc"))), 8)
        for invalid in (-1, 9 << 24, (STRAIGHT_FLUSH << 24) | 1, True):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                category_of_rank(invalid)


class ExhaustiveFiveCardEvaluatorTests(TestCase):
    def test_all_2_598_960_hands_match_known_category_counts(self) -> None:
        """Exhaustively validate category classification and 7,462 rank classes."""
        expected = (
            1_302_540,  # high card
            1_098_240,  # one pair
            123_552,    # two pair
            54_912,     # three of a kind
            10_200,     # straight (excluding straight flushes)
            5_108,      # flush (excluding straight flushes)
            3_744,      # full house
            624,        # four of a kind
            40,         # straight flush
        )
        counts = [0] * len(expected)
        distinct_strengths: set[int] = set()
        for bits in combinations(CARD_MASKS, 5):
            mask = bits[0] | bits[1] | bits[2] | bits[3] | bits[4]
            score = evaluate_mask(mask)
            counts[score >> 24] += 1
            distinct_strengths.add(score)
        self.assertEqual(tuple(counts), expected)
        self.assertEqual(sum(counts), 2_598_960)
        self.assertEqual(len(distinct_strengths), 7_462)
        self.assertEqual(
            tuple(index for index, count in enumerate(counts) if count),
            (HIGH_CARD, ONE_PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT, FLUSH,
             FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH),
        )
