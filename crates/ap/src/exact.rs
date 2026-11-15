//! The exact result kernel of one round by complete enumeration of hand pairs.
//!
//! Under uniform upper policies every pair in `L(D, n0) x L(D, n1)` has the same probability,
//! so the kernel is a ratio of integer counts. On the full deck the 25 canonical pairs total
//! 52,341,125,760 hand pairs.

use rayon::prelude::*;

use crate::cards::{FULL_DECK, evaluate, legal_hands};
use crate::sampler::Counts;
use crate::{NOJ, TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS, TIE, W0, W0C, W1, W1C};

/// Class counts over every hand pair of `deck` for the targets `(n0, n1)`.
///
/// One impossible target gives the whole support to the other side's plain win; two
/// impossible targets give a single no-judgement count, as the engine's chance outcomes do.
#[must_use]
pub fn kernel_counts(deck: u64, targets: (u8, u8)) -> Counts {
    let ranked = |target: u8| -> (Vec<u64>, Vec<u32>) {
        let hands = legal_hands(u32::from(target), deck);
        let ranks = hands
            .iter()
            .map(|&hand| evaluate(hand).expect("legal hands hold five cards"))
            .collect();
        (hands, ranks)
    };
    let ((m0, r0), (m1, r1)) = (ranked(targets.0), ranked(targets.1));
    let mut counts = [0; 6];
    match (m0.len(), m1.len()) {
        (0, 0) => counts[NOJ] = 1,
        (0, n) => counts[W1] = n as u64,
        (n, 0) => counts[W0] = n as u64,
        _ => {
            // [wins, wins with overlap, ties, ties with overlap, overlaps]
            let tally = m0
                .par_iter()
                .zip(&r0)
                .map(|(&h0, &s0)| {
                    let mut t = [0u64; 5];
                    for (&h1, &s1) in m1.iter().zip(&r1) {
                        let (win, tie, overlap) = (
                            u64::from(s0 > s1),
                            u64::from(s0 == s1),
                            u64::from(h0 & h1 != 0),
                        );
                        t[0] += win;
                        t[1] += win & overlap;
                        t[2] += tie;
                        t[3] += tie & overlap;
                        t[4] += overlap;
                    }
                    t
                })
                .reduce(|| [0; 5], |a, b| std::array::from_fn(|i| a[i] + b[i]));
            let total = (m0.len() * m1.len()) as u64;
            counts[W0C] = tally[1];
            counts[W0] = tally[0] - tally[1];
            counts[TIE] = tally[2];
            counts[W1C] = tally[4] - tally[1] - tally[3];
            counts[W1] = total - tally[0] - tally[2] - counts[W1C];
        }
    }
    counts
}

/// The exact round-1 kernel counts of the 25 canonical number pairs on the full deck, in the
/// order `(25, 8), (25, 15), ..., (45, 63)`.
#[must_use]
pub fn round_one() -> Vec<((u8, u8), Counts)> {
    TEAM_0_METAL_CARDS
        .iter()
        .flat_map(|&a| TEAM_1_METAL_CARDS.iter().map(move |&b| (a, b)))
        .map(|pair| (pair, kernel_counts(FULL_DECK, pair)))
        .collect()
}
