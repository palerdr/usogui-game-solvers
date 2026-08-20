"""Compact card masks and exact five-card poker evaluation.

Card IDs use ``card_id = suit * 13 + poker_rank``. ``poker_rank`` is ordered
``2=0, 3=1, ..., K=11, A=12`` so larger indices are stronger for ordinary
poker comparison. Suits are ``clubs=0, diamonds=1, hearts=2, spades=3``.
Consequently IDs 0..12 are clubs, 13..25 diamonds, 26..38 hearts, and 39..51
spades. A ``CardMask`` stores card ``c`` in bit ``1 << c``; the full deck is the
low 52 bits set.

Air Poker's arithmetic is deliberately separate from poker ordering. Its card
values are ``A=1, 2=2, ..., K=13`` and are computed by :func:`rank_sum`.
Keeping the two mappings explicit prevents an ace from accidentally becoming 13
in the number-sum game merely because it is poker's highest rank.

Five-card evaluations are packed into integers. Higher categories occupy higher
bits, followed by tie-breaking ranks from most to least significant, so ordinary
integer comparison is exact poker comparison: ``evaluate_mask(a) >
evaluate_mask(b)`` precisely when hand ``a`` beats hand ``b``.
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

# Index order makes both card IDs and packed poker ranks lexically comparable.
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
    """Precompute the straight high card for every 13-bit rank-presence mask.

    Bit ``r`` means poker rank index ``r`` is present. Five consecutive one-bits
    are detected with ``(rank_mask >> (high - 4)) & 0b11111``. The wheel is the
    exceptional set ``A,2,3,4,5``: bits 12 and 0..3, hexadecimal ``0x100F``.

    Table entries store ``high_rank + 1`` so zero can mean "not a straight".
    Thus a wheel stores 4 and later decodes to poker rank index 3 (the five),
    while an ace-high straight stores 13 and decodes to index 12.
    """
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
    """Encode ``rank`` and ``suit`` as ``suit * 13 + rank``.

    Rank indices are ``2=0, ..., K=11, A=12`` and suits are clubs, diamonds,
    hearts, spades as 0..3. Grouping each suit into 13 consecutive IDs makes
    ``card_id % 13`` recover the rank and ``card_id // 13`` recover the suit.
    """
    rank = _require_int(rank, "rank")
    suit = _require_int(suit, "suit")
    if not 0 <= rank < 13 or not 0 <= suit < 4:
        raise ValueError("rank must be 0..12 and suit must be 0..3")
    return suit * 13 + rank


def parse_card(text: str) -> Card:
    """Parse a two-character card and encode it with :func:`make_card`'s layout.

    The first character is one of ``23456789TJQKA`` and the second is one of
    ``cdhs``. Case is normalized, but ambiguous spellings such as ``10h`` are
    rejected so every formatted card has one canonical two-character form.
    """
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
    """Invert the compact card index into canonical text such as ``Th``.

    ``CARD_RANK[card_id]`` is ``card_id % 13`` and ``CARD_SUIT[card_id]`` is
    ``card_id // 13``; those indices select from ``RANK_CHARS`` and
    ``SUIT_CHARS`` respectively.
    """
    card = _validate_card(card_id)
    return RANK_CHARS[CARD_RANK[card]] + SUIT_CHARS[CARD_SUIT[card]]


def card_rank(card_id: Card) -> int:
    """Return ``card_id % 13``: poker rank index 0..12 for 2 through ace.

    This is a poker-ordering index, not the Air Poker numerical value. Use
    :func:`rank_sum` for the latter because ace maps to 1 there, not 12 or 14.
    """
    return CARD_RANK[_validate_card(card_id)]


def card_suit(card_id: Card) -> int:
    """Return ``card_id // 13``: clubs, diamonds, hearts, or spades as 0..3."""
    return CARD_SUIT[_validate_card(card_id)]


def card_mask(card_id: Card) -> CardMask:
    """Return the singleton set for a card, represented by ``1 << card_id``."""
    return CARD_MASKS[_validate_card(card_id)]


def cards_to_mask(cards) -> CardMask:
    """Build the set union ``sum(1 << card for card in cards)``.

    Since each card owns one bit, bitwise OR represents set union, AND represents
    intersection, and ``mask.bit_count()`` is cardinality. Duplicate input is
    rejected rather than silently collapsed so malformed five-card hands cannot
    masquerade as smaller sets. Strings are rejected even though iterable.
    """
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
    """Build a mask and require the poker invariant ``mask.bit_count() == 5``."""
    mask = cards_to_mask(cards)
    if mask.bit_count() != HAND_SIZE:
        raise ValueError("a hand requires exactly 5 distinct cards")
    return mask


def mask_contains(mask: CardMask, card_id: Card) -> bool:
    """Test set membership with ``bool(mask & (1 << card_id))``."""
    return bool(_validate_mask(mask) & CARD_MASKS[_validate_card(card_id)])


def mask_popcount(mask: CardMask) -> int:
    """Return the card-set cardinality using Python's integer population count."""
    return _validate_mask(mask).bit_count()


def iter_mask(mask: CardMask):
    """Yield set bits as ascending card IDs without scanning all 52 positions.

    For a nonzero two's-complement integer, ``bits & -bits`` isolates its least
    significant set bit. Its index is ``bit_length() - 1``. Removing that bit and
    repeating costs O(k) for a mask containing k cards and naturally yields the
    IDs in ascending order.
    """
    bits = _validate_mask(mask)
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def remove_cards(mask: CardMask, cards_or_mask) -> CardMask:
    """Compute card-set difference as ``mask & ~removed & FULL_DECK_MASK``.

    ``cards_or_mask`` may already be a mask or may be an iterable converted by
    :func:`cards_to_mask`. The final full-deck AND documents and enforces that
    Python's infinite-width complement cannot introduce bits above card 51.
    """
    mask = _validate_mask(mask)
    if type(cards_or_mask) is int:
        removed = _validate_mask(cards_or_mask, "cards_or_mask")
    else:
        removed = cards_to_mask(cards_or_mask)
    return mask & ~removed & FULL_DECK_MASK


def legal_cards_mask(used_mask: CardMask, forbidden_mask: CardMask = 0) -> CardMask:
    """Return ``FULL_DECK_MASK & ~(used_mask | forbidden_mask)``.

    This is the complement, within the 52-card universe, of the union of cards
    already consumed and cards excluded for some caller-specific reason.
    """
    used = _validate_mask(used_mask, "used_mask")
    forbidden = _validate_mask(forbidden_mask, "forbidden_mask")
    return FULL_DECK_MASK & ~(used | forbidden)


def rank_sum(mask: CardMask) -> int:
    """Return the Air Poker sum ``Σ value(card)`` for every set bit.

    The stored poker rank is ``r=0..12`` for 2 through ace. Air Poker instead
    assigns ``value(A)=1`` and ``value(r)=r+2`` for every other rank, producing
    2 through 13 for deuce through king. Suits contribute nothing to the sum.
    """
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

    The mathematical result is

    ``{H ⊆ available_mask : |H| = 5 and Σ(card ∈ H) value(card) = target}``.

    The smallest possible five-card sum is 6 (four aces and a two); the largest
    is 64 (four kings and a queen), so targets outside 6..64 can return at once.

    Enumeration uses a depth-first combination search rather than testing all
    ``C(n, 5)`` hands blindly. Before that search, two dynamic-programming tables
    are built for every suffix ``cards[start:]`` and remaining choice count
    ``needed``:

    ``minimum[start][needed]`` = least achievable sum from that suffix
    ``maximum[start][needed]`` = greatest achievable sum from that suffix

    Each cell considers skipping the current card or taking it, giving the
    recurrence ``best(start, k) = best(best(start+1, k), value[start] +
    best(start+1, k-1))`` with min or max as appropriate. During DFS, a branch is
    discarded whenever ``target - subtotal`` falls outside those bounds. Card
    indices always advance, so every exact five-card combination is emitted once
    and no permutation duplicates are produced. Runtime is output-sensitive: the
    bounds prune impossible branches, but returning many valid hands necessarily
    costs at least O(number of returned hands).
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
    """Pack a trusted five-card mask into a totally ordered poker integer.

    Layout (most significant fields first)::

        bits 27..24  category: high card=0 through straight flush=8
        bits 23..20  primary rank or highest kicker
        bits 19..16  secondary rank or next kicker
        bits 15..12  tertiary rank or next kicker
        bits 11..8   fourth kicker when required
        bits 7..4    fifth kicker when required
        bits 3..0    always zero

    Rank fields use the poker indices ``2=0, ..., A=12``. Category is compared
    first because it occupies the highest nibble; within a category, decisive
    ranks are packed from most to least significant. Consequently normal integer
    ordering implements the complete five-card tie-break rules without tuples or
    allocations in the hot path.

    Rank multiplicities are accumulated in one integer containing thirteen
    independent four-bit counters. Adding ``1 << (rank * 4)`` increments a rank's
    nibble; a five-card hand can contain at most four of one rank, so no counter
    carries into its neighbor. ``rank_mask`` separately records presence for the
    precomputed straight lookup. A flush is true when all hand bits are contained
    in any one of the four 13-bit suit masks.

    The category-specific packed tie breakers are:

    - straight/straight flush: high card (a wheel is five-high);
    - four of a kind: quad rank, then kicker;
    - full house: trip rank, then pair rank;
    - three of a kind: trip rank, then two descending kickers;
    - two pair: high pair, low pair, then kicker;
    - one pair: pair rank, then three descending kickers;
    - flush/high card: all five ranks descending.

    Validation is intentionally performed by :func:`evaluate_mask`; this private
    function assumes exactly five in-deck bits.
    """
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
    """Evaluate exactly five cards into an integer whose ordering is poker order.

    ``a > b`` means hand ``a`` wins, ``a == b`` means an exact poker-strength tie,
    and ``a < b`` means ``b`` wins. Suits never break ties: two distinct masks can
    therefore receive the same ``HandRank``.

    The evaluator makes one pass over the five set bits to build both a 13-bit
    rank-presence mask and thirteen packed four-bit rank counters. Suit masks test
    whether all five cards share a suit; the rank-presence mask indexes a table
    containing the straight high card, including the ace-low wheel exception.
    Ranks are then scanned from ace down to deuce to collect pairs, trips, quads,
    and descending kickers.

    The result packs ``category`` into bits 24..27, followed by decisive ranks in
    descending significance at bits 20, 16, 12, 8, and 4. Those fields encode the
    standard comparisons: straight high card; quad then kicker; trip then pair;
    trip then kickers; high pair then low pair then kicker; pair then kickers; or
    all five descending ranks for a flush/high card. Because the fields appear in
    comparison order, one integer comparison is equivalent to lexicographically
    comparing ``(category, tie_breaker_1, tie_breaker_2, ...)``.

    Only exact five-card masks are accepted; this evaluator does not select the
    best five cards from a larger Hold'em-style set.
    """
    mask = _validate_mask(mask)
    if mask.bit_count() != HAND_SIZE:
        raise ValueError("evaluate_mask requires exactly 5 distinct cards")
    return _evaluate_five_unchecked(mask)


# Direct Rust-facing concept aliases without extra wrappers.
evaluate_five = evaluate_mask


def category_of_rank(hand_rank: HandRank) -> int:
    """Extract the category nibble from an evaluator-produced ``HandRank``.

    Categories are integers 0..8 in increasing strength from ``HIGH_CARD`` to
    ``STRAIGHT_FLUSH``. Since the category occupies bits 24..27, extraction is
    simply ``hand_rank >> 24``. The low nibble is reserved as zero by the packing
    layout and is checked here as a cheap malformed-value guard.
    """
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
