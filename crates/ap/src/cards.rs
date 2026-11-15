//! Card masks, Air Poker sums, legal hands and exact five-card poker evaluation.
//!
//! Card `suit * 13 + rank` is bit `1 << card` of a `u64` mask; ranks run `2 = 0` to `A = 12`
//! and suits are clubs, diamonds, hearts, spades. Poker order and Air Poker arithmetic are
//! separate: in the sum an ace is 1 and a king 13.

use crate::Error;

/// The 52-card deck as a mask.
pub const FULL_DECK: u64 = (1 << 52) - 1;

/// Air Poker value of a card: ace 1, two 2, ..., king 13.
#[must_use]
pub const fn value(card: u32) -> u32 {
    let rank = card % 13;
    if rank == 12 { 1 } else { rank + 2 }
}

/// The cards of a mask in ascending order.
pub fn cards(mut mask: u64) -> impl Iterator<Item = u32> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let card = mask.trailing_zeros();
            mask &= mask - 1;
            card
        })
    })
}

/// Parse two-character card text such as `Th` (rank `23456789TJQKA`, suit `cdhs`, any case).
///
/// # Errors
/// [`Error::Card`] for any other text.
pub fn parse_card(text: &str) -> Result<u32, Error> {
    let bad = || Error::Card(text.to_owned());
    let &[rank, suit] = text.as_bytes() else {
        return Err(bad());
    };
    let rank = b"23456789TJQKA"
        .iter()
        .position(|&c| c == rank.to_ascii_uppercase());
    let suit = b"cdhs".iter().position(|&c| c == suit.to_ascii_lowercase());
    match (rank, suit) {
        (Some(rank), Some(suit)) => Ok((suit * 13 + rank) as u32),
        _ => Err(bad()),
    }
}

/// The Air Poker sum of the cards in `mask`.
#[must_use]
pub fn rank_sum(mask: u64) -> u32 {
    cards(mask).map(value).sum()
}

/// Every five-card subset of `deck` whose values sum to `target`, in lexicographic card order.
///
/// A depth-first search over card indices prunes a branch when the remainder falls outside the
/// least and greatest sums the remaining suffix can still make.
#[must_use]
pub fn legal_hands(target: u32, deck: u64) -> Vec<u64> {
    let ids: Vec<u32> = cards(deck & FULL_DECK).collect();
    let n = ids.len();
    let mut hands = Vec::new();
    if !(6..=64).contains(&target) || n < 5 {
        return hands;
    }
    // bounds[start][k] = (least, greatest) sum of k cards from ids[start..].
    let mut bounds = vec![[None::<(u32, u32)>; 6]; n + 1];
    for row in &mut bounds {
        row[0] = Some((0, 0));
    }
    for start in (0..n).rev() {
        for k in 1..=5 {
            let skip = bounds[start + 1][k];
            let take = bounds[start + 1][k - 1]
                .map(|(lo, hi)| (lo + value(ids[start]), hi + value(ids[start])));
            bounds[start][k] = match (skip, take) {
                (Some(s), Some(t)) => Some((s.0.min(t.0), s.1.max(t.1))),
                (s, t) => s.or(t),
            };
        }
    }
    let mut stack = vec![(0, 5, target, 0u64)];
    while let Some((start, needed, remainder, mask)) = stack.pop() {
        if needed == 0 {
            if remainder == 0 {
                hands.push(mask);
            }
            continue;
        }
        let Some((lo, hi)) = bounds[start][needed] else {
            continue;
        };
        if remainder < lo || remainder > hi {
            continue;
        }
        for index in (start..=n - needed).rev() {
            let v = value(ids[index]);
            if v <= remainder {
                stack.push((index + 1, needed - 1, remainder - v, mask | 1 << ids[index]));
            }
        }
    }
    hands
}

/// Exact poker strength of a five-card hand; a larger value is a stronger hand.
///
/// The category (high card 0 to straight flush 8) sits in bits 24..27. Below it the deciding
/// ranks follow in four-bit fields from bit 20 down, most significant first: by multiplicity,
/// then by rank. A straight stores only its high card, and the wheel A-2-3-4-5 is five-high.
/// Suits never break ties.
///
/// # Errors
/// [`Error::Illegal`] unless `hand` holds exactly five cards of the deck.
pub fn evaluate(hand: u64) -> Result<u32, Error> {
    if hand.count_ones() != 5 || hand & !FULL_DECK != 0 {
        return Err(Error::Illegal(
            "a hand holds exactly five cards of the deck",
        ));
    }
    let mut counts = [0u32; 13];
    let mut present = 0u32;
    for card in cards(hand) {
        counts[(card % 13) as usize] += 1;
        present |= 1 << (card % 13);
    }
    let mut packed = 0;
    let mut shift = 20;
    for multiplicity in (1..=4).rev() {
        for rank in (0..13).rev() {
            if counts[rank as usize] == multiplicity {
                packed |= rank << shift;
                shift -= 4;
            }
        }
    }
    let flush = (0..4).any(|suit| hand & (0x1FFF << (13 * suit)) == hand);
    let straight = if present & 0x100F == 0x100F {
        Some(3)
    } else {
        (4..13)
            .rev()
            .find(|&high| present >> (high - 4) & 0x1F == 0x1F)
    };
    let largest = counts.iter().max().copied().unwrap_or(0);
    let category = match (present.count_ones(), largest, flush, straight) {
        (2, 4, ..) => 7,
        (2, ..) => 6,
        (3, 3, ..) => 3,
        (3, ..) => 2,
        (4, ..) => 1,
        (_, _, true, Some(high)) => return Ok(8 << 24 | high << 20),
        (_, _, false, Some(high)) => return Ok(4 << 24 | high << 20),
        (_, _, true, None) => 5,
        _ => 0,
    };
    Ok(category << 24 | packed)
}
