"""Compact pure-Python card masks and exact five-card poker evaluation.
"""

from __future__ import annotations

from timeit import timeit


NUM_CARDS = 52
HAND_SIZE = 5
FULL_DECK_MASK = (1 << NUM_CARDS) - 1

# These aliases document intent without wrapping integers or adding hot-path
# allocations. CardMask represents any subset of the deck, including a hand.
# Construct values through make_card/cards_to_mask/make_hand_mask when
# validation is required at an API boundary.
Card = int
CardMask = int
HandRank = int

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
CARD_RANK = tuple(i % 13 for i in range(NUM_CARDS))
CARD_SUIT = tuple(i // 13 for i in range(NUM_CARDS))
CARD_MASKS = tuple(1 << i for i in range(NUM_CARDS))
SUIT_MASKS = tuple(((1 << 13) - 1) << (13 * suit) for suit in range(4))

HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8
CATEGORY_NAMES = (
    "high card",
    "one pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)

__all__ = (
    "Card",
    "CardMask",
    "HandRank",
    "NUM_CARDS",
    "HAND_SIZE",
    "FULL_DECK_MASK",
    "CARD_RANK",
    "CARD_SUIT",
    "CARD_MASKS",
    "HIGH_CARD",
    "ONE_PAIR",
    "TWO_PAIR",
    "THREE_OF_A_KIND",
    "STRAIGHT",
    "FLUSH",
    "FULL_HOUSE",
    "FOUR_OF_A_KIND",
    "STRAIGHT_FLUSH",
    "CATEGORY_NAMES",
    "make_card",
    "parse_card",
    "format_card",
    "card_rank",
    "card_suit",
    "card_mask",
    "cards_to_mask",
    "make_hand_mask",
    "mask_contains",
    "mask_popcount",
    "iter_mask",
    "remove_cards",
    "legal_cards_mask",
    "rank_sum",
    "generate_legal_hands",
    "evaluate_mask",
    "evaluate_five",
    "category_of_rank",
)


def _build_straight_table() -> tuple[int, ...]:
    out = [0] * (1 << 13)
    for rank_mask in range(1 << 13):
        if rank_mask & 0x100F == 0x100F:  # A-2-3-4-5
            out[rank_mask] = 4
        else:
            for high in range(12, 3, -1):
                if rank_mask >> (high - 4) & 0x1F == 0x1F:
                    out[rank_mask] = high + 1
                    break
    return tuple(out)


STRAIGHT_HIGH_BY_MASK = _build_straight_table()


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def _validate_card(card_id: object) -> int:
    card = _require_int(card_id, "card_id")
    if not 0 <= card < NUM_CARDS:
        raise ValueError("card_id must be in range 0..51")
    return card


def _validate_mask(mask: object, name: str = "mask") -> int:
    value = _require_int(mask, name)
    if value < 0 or value & ~FULL_DECK_MASK:
        raise ValueError(f"{name} contains bits outside the 52-card deck")
    return value


def make_card(rank: int, suit: int) -> Card:
    """Return the compact card ID for integer rank 0..12 and suit 0..3."""
    rank = _require_int(rank, "rank")
    suit = _require_int(suit, "suit")
    if not 0 <= rank < 13 or not 0 <= suit < 4:
        raise ValueError("rank must be 0..12 and suit must be 0..3")
    return suit * 13 + rank


def parse_card(text: str) -> Card:
    """Parse exactly two characters such as ``Ac``; ambiguous forms are rejected."""
    if type(text) is not str:
        raise TypeError("text must be a str")
    if len(text) != 2:
        raise ValueError("card text must contain exactly two characters")
    rank_ch, suit_ch = text[0].upper(), text[1].lower()
    rank = RANK_CHARS.find(rank_ch)
    suit = SUIT_CHARS.find(suit_ch)
    if rank < 0 or suit < 0:
        raise ValueError(f"invalid card: {text!r}")
    return suit * 13 + rank


def format_card(card_id: Card) -> str:
    """Format a compact card ID using the Rust spelling, for example ``Th``."""
    card = _validate_card(card_id)
    return RANK_CHARS[CARD_RANK[card]] + SUIT_CHARS[CARD_SUIT[card]]


def card_rank(card_id: Card) -> int:
    """Return rank 0..12 (2..A) for a validated compact card ID."""
    return CARD_RANK[_validate_card(card_id)]


def card_suit(card_id: Card) -> int:
    """Return suit 0..3 (clubs, diamonds, hearts, spades)."""
    return CARD_SUIT[_validate_card(card_id)]


def card_mask(card_id: Card) -> CardMask:
    return CARD_MASKS[_validate_card(card_id)]


def cards_to_mask(cards) -> CardMask:
    """Build a mask from card IDs, rejecting strings and duplicate cards."""
    if isinstance(cards, (str, bytes, bytearray)):
        raise TypeError("cards must be an iterable of integer card IDs")
    mask = 0
    try:
        iterator = iter(cards)
    except TypeError as exc:
        raise TypeError("cards must be an iterable of integer card IDs") from exc
    for card_id in iterator:
        bit = CARD_MASKS[_validate_card(card_id)]
        if mask & bit:
            raise ValueError("duplicate card")
        mask |= bit
    return mask


def make_hand_mask(cards) -> CardMask:
    """Build the game's legal hand representation: exactly five distinct cards."""
    mask = cards_to_mask(cards)
    if mask.bit_count() != HAND_SIZE:
        raise ValueError("a hand requires exactly 5 distinct cards")
    return mask


def mask_contains(mask: CardMask, card_id: Card) -> bool:
    return bool(_validate_mask(mask) & CARD_MASKS[_validate_card(card_id)])


def mask_popcount(mask: CardMask) -> int:
    return _validate_mask(mask).bit_count()


def iter_mask(mask: CardMask):
    """Yield contained card IDs in ascending Rust card-index order."""
    bits = _validate_mask(mask)
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def remove_cards(mask: CardMask, cards_or_mask) -> CardMask:
    """Remove either a validated integer mask or an iterable of card IDs."""
    mask = _validate_mask(mask)
    if type(cards_or_mask) is int:
        removed = _validate_mask(cards_or_mask, "cards_or_mask")
    else:
        removed = cards_to_mask(cards_or_mask)
    return mask & ~removed & FULL_DECK_MASK


def legal_cards_mask(used_mask: CardMask, forbidden_mask: CardMask = 0) -> CardMask:
    used = _validate_mask(used_mask, "used_mask")
    forbidden = _validate_mask(forbidden_mask, "forbidden_mask")
    return FULL_DECK_MASK & ~(used | forbidden)


def rank_sum(mask: CardMask) -> int:
    """Sum Air Poker values in a card set: ace is 1 and king is 13."""
    total = 0
    for card in iter_mask(mask):
        rank = CARD_RANK[card]
        total += 1 if rank == 12 else rank + 2
    return total


def generate_legal_hands(target: int, available_mask: CardMask) -> tuple[CardMask, ...]:
    """Generate every legal five-card hand for an Air Poker number.

    Given integer ``target`` and the mask of cards still available in the shared
    deck, return every five-card ``CardMask`` satisfying
    ``sum(value(card) for card in hand) == target``, where ace is 1 and jack,
    queen, and king are 11, 12, and 13. Suits do not alter the numerical sum,
    but they distinguish exact cards and determine flushes and straight flushes.
    Every returned hand is a subset of ``available_mask``; an impossible target
    returns an empty tuple.
    """
    target = _require_int(target, "target")
    available_mask = _validate_mask(available_mask, "available_mask")
    if not 6 <= target <= 64 or available_mask.bit_count() < HAND_SIZE:
        return ()

    cards = tuple(iter_mask(available_mask))
    values = tuple(1 if CARD_RANK[card] == 12 else CARD_RANK[card] + 2 for card in cards)
    count = len(cards)
    impossible_low = 10**9
    impossible_high = -10**9
    minimum = [[impossible_low] * (HAND_SIZE + 1) for _ in range(count + 1)]
    maximum = [[impossible_high] * (HAND_SIZE + 1) for _ in range(count + 1)]
    for start in range(count + 1):
        minimum[start][0] = maximum[start][0] = 0

    for start in range(count - 1, -1, -1):
        for needed in range(1, HAND_SIZE + 1):
            minimum[start][needed] = minimum[start + 1][needed]
            maximum[start][needed] = maximum[start + 1][needed]
            if minimum[start + 1][needed - 1] != impossible_low:
                minimum[start][needed] = min(
                    minimum[start][needed],
                    values[start] + minimum[start + 1][needed - 1],
                )
                maximum[start][needed] = max(
                    maximum[start][needed],
                    values[start] + maximum[start + 1][needed - 1],
                )

    hands: list[CardMask] = []

    def search(start: int, needed: int, subtotal: int, mask: CardMask) -> None:
        if needed == 0:
            if subtotal == target:
                hands.append(mask)
            return
        if count - start < needed:
            return
        remainder = target - subtotal
        if remainder < minimum[start][needed] or remainder > maximum[start][needed]:
            return
        for index in range(start, count - needed + 1):
            search(
                index + 1,
                needed - 1,
                subtotal + values[index],
                mask | CARD_MASKS[cards[index]],
            )

    search(0, HAND_SIZE, 0, 0)
    return tuple(hands)


def _evaluate_five_unchecked(mask: CardMask) -> HandRank:
    """Evaluate a trusted five-bit in-deck mask using Rust HandRank packing."""
    bits = mask
    rank_counts = 0  # thirteen independent four-bit counters
    rank_mask = 0
    card_rank = CARD_RANK
    while bits:
        low = bits & -bits
        rank = card_rank[low.bit_length() - 1]
        rank_counts += 1 << (rank << 2)
        rank_mask |= 1 << rank
        bits ^= low

    is_flush = any(mask & suit_mask == mask for suit_mask in SUIT_MASKS)
    straight_code = STRAIGHT_HIGH_BY_MASK[rank_mask]
    straight_high = straight_code - 1

    single0 = single1 = single2 = single3 = single4 = 0
    pair0 = pair1 = trip = quad = 0
    has_trip = has_quad = False
    singles = pairs = 0
    for rank in range(12, -1, -1):
        count = rank_counts >> (rank << 2) & 0xF
        if count == 4:
            quad = rank
            has_quad = True
        elif count == 3:
            trip = rank
            has_trip = True
        elif count == 2:
            if pairs == 0:
                pair0 = rank
            else:
                pair1 = rank
            pairs += 1
        elif count == 1:
            if singles == 0:
                single0 = rank
            elif singles == 1:
                single1 = rank
            elif singles == 2:
                single2 = rank
            elif singles == 3:
                single3 = rank
            else:
                single4 = rank
            singles += 1

    if is_flush and straight_code:
        return (STRAIGHT_FLUSH << 24) | (straight_high << 20)
    if has_quad:
        return (FOUR_OF_A_KIND << 24) | (quad << 20) | (single0 << 16)
    if has_trip:
        if pairs:
            return (FULL_HOUSE << 24) | (trip << 20) | (pair0 << 16)
        return (
            (THREE_OF_A_KIND << 24)
            | (trip << 20)
            | (single0 << 16)
            | (single1 << 12)
        )
    if pairs == 2:
        return (TWO_PAIR << 24) | (pair0 << 20) | (pair1 << 16) | (single0 << 12)
    if pairs == 1:
        return (
            (ONE_PAIR << 24)
            | (pair0 << 20)
            | (single0 << 16)
            | (single1 << 12)
            | (single2 << 8)
        )
    kickers = (
        (single0 << 20)
        | (single1 << 16)
        | (single2 << 12)
        | (single3 << 8)
        | (single4 << 4)
    )
    if is_flush:
        return (FLUSH << 24) | kickers
    if straight_code:
        return (STRAIGHT << 24) | (straight_high << 20)
    return kickers


def evaluate_mask(mask: CardMask) -> HandRank:
    """Evaluate exactly five cards; larger comparable integers are stronger."""
    mask = _validate_mask(mask)
    if mask.bit_count() != HAND_SIZE:
        raise ValueError("evaluate_mask requires exactly 5 distinct cards")
    return _evaluate_five_unchecked(mask)


# Direct Rust-facing concept aliases without extra wrappers.
evaluate_five = evaluate_mask


def category_of_rank(hand_rank: HandRank) -> int:
    rank = _require_int(hand_rank, "hand_rank")
    category = rank >> 24
    if rank < 0 or category > STRAIGHT_FLUSH or rank & 0xF:
        raise ValueError("invalid packed hand rank")
    return category


def _hand(*names: str) -> CardMask:
    return make_hand_mask(parse_card(name) for name in names)


def _smoke_test() -> None:
    # Representative masks match game.rs: 2c=0, Ac=12, 2d=13, Th=34, As=51.
    assert tuple(map(parse_card, ("2c", "Ac", "2d", "Th", "As"))) == (0, 12, 13, 34, 51)
    assert tuple(card_mask(i) for i in (0, 12, 13, 34, 51)) == tuple(1 << i for i in (0, 12, 13, 34, 51))
    assert mask_popcount(0) == 0 and mask_popcount(FULL_DECK_MASK) == 52
    sample = cards_to_mask((0, 18, 51))
    assert tuple(iter_mask(sample)) == (0, 18, 51)
    assert mask_contains(sample, 18) and not mask_contains(sample, 19)
    assert remove_cards(sample, card_mask(18)) == cards_to_mask((0, 51))
    assert legal_cards_mask(sample, card_mask(19)) == FULL_DECK_MASK & ~(sample | card_mask(19))
    summed_hand = _hand("Ac", "2d", "3h", "4s", "5c")
    assert rank_sum(summed_hand) == 15
    assert generate_legal_hands(15, summed_hand) == (summed_hand,)

    hands = (
        _hand("Ac", "Kd", "Qh", "9s", "3c"),
        _hand("Ac", "As", "Kd", "Qh", "Jh"),
        _hand("Ac", "As", "Kd", "Kc", "Jh"),
        _hand("Ac", "As", "Ad", "Kd", "Qh"),
        _hand("Ac", "Kc", "Qd", "Jh", "Td"),
        _hand("Ac", "Kc", "Qc", "9c", "2c"),
        _hand("Ac", "Ad", "Ah", "Kc", "Kd"),
        _hand("Ac", "Ad", "Ah", "As", "Kd"),
        _hand("Ac", "Kc", "Qc", "Jc", "Tc"),
    )
    scores = tuple(map(evaluate_mask, hands))
    assert scores == tuple(sorted(scores))
    assert tuple(map(category_of_rank, scores)) == tuple(range(9))
    assert evaluate_mask(_hand("5h", "4d", "3c", "2s", "As")) < evaluate_mask(
        _hand("6h", "5d", "4c", "3s", "2d")
    )
    assert evaluate_mask(_hand("Ac", "Ad", "Kc", "Qh", "Js")) > evaluate_mask(
        _hand("Ac", "Ad", "Kc", "Qh", "Ts")
    )
    assert evaluate_mask(_hand("Ac", "Qh", "9d", "6s", "3c")) == evaluate_mask(
        _hand("As", "Qd", "9h", "6c", "3s")
    )

    invalid_calls = (
        lambda: card_mask(-1),
        lambda: card_mask(52),
        lambda: card_mask("Ac"),
        lambda: cards_to_mask((0, 0)),
        lambda: cards_to_mask("Ac Kd"),
        lambda: mask_popcount(1 << 52),
        lambda: legal_cards_mask(-1),
        lambda: evaluate_mask(_hand("Ac", "Kd", "Qh", "9s", "3c") | (1 << 52)),
        lambda: evaluate_mask(cards_to_mask((0, 1, 2, 3))),
    )
    for call in invalid_calls:
        try:
            call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input was accepted")


def _benchmark(repeats: int = 100_000) -> tuple[float, float]:
    """Return mask-iteration and five-card-evaluation operations per second."""
    hand = _hand("Ac", "Ad", "Ah", "Kc", "Kd")
    iteration_seconds = timeit(lambda: tuple(iter_mask(hand)), number=repeats)
    evaluation_seconds = timeit(lambda: evaluate_mask(hand), number=repeats)
    return repeats / iteration_seconds, repeats / evaluation_seconds


if __name__ == "__main__":
    _smoke_test()
    iteration_rate, evaluation_rate = _benchmark()
    print("smoke tests: ok")
    print(f"mask iteration: {iteration_rate:,.0f} hands/s")
    print(f"five-card evaluation: {evaluation_rate:,.0f} hands/s")
