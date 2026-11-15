//! The trajectory sampler with a counter-based generator and dense prefix counts.
//!
//! A trajectory plays the five rounds of one number sequence from the full deck. Each round
//! builds the deck's [`HandTable`], draws each hand by one bounded random integer and an
//! unrank, classifies the pair as the engine does and removes the union. The class of round `r`
//! lands in the row of its revealed prefix (the classes of the earlier rounds as a base-5 code),
//! so one trajectory samples every prefix on its path at its chance reach.
//!
//! The stream of trajectory `i` of batch `b` under `seed` starts at `mix64` of `(seed, b, i)`
//! and steps by the splitmix64 increment, so every batch is reproducible and the counts are
//! identical to the Python reference for the same seed, batch and size.

use rayon::prelude::*;

use crate::cards::{FULL_DECK, evaluate};
use crate::dp::HandTable;
use crate::{NOJ, ROUNDS, TIE, W0, W0C, W1, W1C, digit};

/// Counts of the six result classes.
pub type Counts = [u64; 6];
/// First row of each round's revealed prefixes in a sequence's count array.
pub const ROW_OFFSET: [usize; ROUNDS] = [0, 1, 6, 31, 156];
/// Rows of a sequence's count array: `1 + 5 + 25 + 125 + 625`.
pub const ROWS: usize = 781;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The splitmix64 output function, a bijection on 64-bit words.
#[must_use]
pub const fn mix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The initial generator state of trajectory `index` of `batch` under `seed`.
#[must_use]
pub const fn stream(seed: u64, batch: u64, index: u64) -> u64 {
    let state = mix64(seed.wrapping_add(GOLDEN));
    let state = mix64(state.wrapping_add(batch.wrapping_mul(GOLDEN)));
    mix64(state.wrapping_add(index.wrapping_mul(GOLDEN)))
}

/// The next 64-bit word of a stream.
pub fn next64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN);
    mix64(*state)
}

/// An integer exactly uniform in `[0, n)`, `n >= 1`: the top bits of a word, with rejection.
pub fn bounded(state: &mut u64, n: u64) -> u64 {
    if n == 1 {
        return 0;
    }
    let shift = (n - 1).leading_zeros();
    loop {
        let u = next64(state) >> shift;
        if u < n {
            return u;
        }
    }
}

/// One round on `deck`: the class, the next deck and both hands (`None` for no-rank).
///
/// Both targets impossible is no judgement with the deck unchanged; one impossible target
/// gives the other side a plain win.
pub fn play_round(deck: u64, targets: (u8, u8), state: &mut u64) -> (usize, u64, [Option<u64>; 2]) {
    let table = HandTable::new(deck);
    let targets = [targets.0, targets.1].map(u32::from);
    let mut hands = [None; 2];
    for (hand, &target) in hands.iter_mut().zip(&targets) {
        let count = table.count(target);
        if count > 0 {
            *hand = table.unrank(target, bounded(state, count));
        }
    }
    let class = match hands {
        [None, None] => return (NOJ, deck, hands),
        [None, Some(_)] => W1,
        [Some(_), None] => W0,
        [Some(h0), Some(h1)] => {
            let [r0, r1] =
                [h0, h1].map(|hand| evaluate(hand).expect("unranked hands hold five cards"));
            let overlap = h0 & h1 != 0;
            match r0.cmp(&r1) {
                std::cmp::Ordering::Equal => TIE,
                std::cmp::Ordering::Greater if overlap => W0C,
                std::cmp::Ordering::Greater => W0,
                std::cmp::Ordering::Less if overlap => W1C,
                std::cmp::Ordering::Less => W1,
            }
        }
    };
    (
        class,
        deck & !(hands[0].unwrap_or(0) | hands[1].unwrap_or(0)),
        hands,
    )
}

/// Dense counts of `n` trajectories of one number sequence: one row per revealed prefix of each
/// round, and the number of trajectories cut short by an unreachable class.
#[must_use]
pub fn sample_sequence(
    targets: &[(u8, u8); ROUNDS],
    seed: u64,
    batch: u64,
    n: u64,
) -> (Vec<Counts>, u64) {
    // Blocks of trajectories run in parallel; a short sequence is one block, so many short
    // sequences can run side by side without splitting overhead.
    const BLOCK: u64 = 4096;
    let empty = || (vec![[0; 6]; ROWS], 0);
    (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|block| {
            let (mut counts, mut errors) = empty();
            for i in block * BLOCK..n.min((block + 1) * BLOCK) {
                let mut state = stream(seed, batch, i);
                let (mut deck, mut prefix) = (FULL_DECK, 0);
                for (r, &pair) in targets.iter().enumerate() {
                    let (class, next, _) = play_round(deck, pair, &mut state);
                    counts[ROW_OFFSET[r] + prefix][class] += 1;
                    let Some(d) = digit(class) else {
                        errors += 1;
                        break;
                    };
                    deck = next;
                    prefix = prefix * 5 + d;
                }
            }
            (counts, errors)
        })
        .reduce(empty, |(mut a, ea), (b, eb)| {
            a.iter_mut()
                .zip(&b)
                .for_each(|(x, y)| x.iter_mut().zip(y).for_each(|(x, y)| *x += y));
            (a, ea + eb)
        })
}

/// The class of each of `n` single rounds on `deck`, in stream order.
#[must_use]
pub fn round_classes(deck: u64, targets: (u8, u8), seed: u64, batch: u64, n: u64) -> Vec<usize> {
    (0..n)
        .into_par_iter()
        .map(|i| play_round(deck, targets, &mut stream(seed, batch, i)).0)
        .collect()
}

/// Class counts of `n` single rounds on `deck`.
#[must_use]
pub fn sample_rounds(deck: u64, targets: (u8, u8), seed: u64, batch: u64, n: u64) -> Counts {
    (0..n)
        .into_par_iter()
        .fold(
            || [0; 6],
            |mut counts, i| {
                counts[play_round(deck, targets, &mut stream(seed, batch, i)).0] += 1;
                counts
            },
        )
        .reduce(|| [0; 6], |a, b| std::array::from_fn(|c| a[c] + b[c]))
}

/// The row of the entry for round `r` (0-based) after the revealed classes `prefix`.
#[must_use]
pub fn row(r: usize, prefix: &[usize]) -> usize {
    ROW_OFFSET[r]
        + prefix.iter().fold(0, |code, &class| {
            code * 5 + digit(class).expect("a revealed class has a digit")
        })
}
