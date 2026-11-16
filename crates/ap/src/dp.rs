//! Count and unrank the legal hands of a deck by a dynamic program over card values.
//!
//! `L(D, n)` is the set of five-card subsets of deck `D` whose values sum to `n`. For value
//! `v = 1..=13` the deck holds `a[v]` cards, and
//!
//! ```text
//! C[v][k][s] = sum over j of binom(a[v], j) * C[v + 1][k - j][s - j * v],   C[14][0][0] = 1
//! ```
//!
//! counts the `k`-card subsets of the cards with value at least `v` that sum to `s`, so
//! `|L(D, n)| = C[1][5][n]`. Unranking walks the values, picks each multiplicity by cumulative
//! counts and the suits by the combinatorial number system, so every legal hand has exactly one
//! rank and one bounded random integer is one exact uniform draw.

use crate::cards::{FULL_DECK, cards, value};

/// `BINOM[a][j]`: ways to choose `j` of the `a <= 4` cards of one value.
const BINOM: [[u32; 6]; 5] = [
    [1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 2, 1, 0, 0, 0],
    [1, 3, 3, 1, 0, 0],
    [1, 4, 6, 4, 1, 0],
];

/// The completion-count table of one deck; it serves every target.
pub struct HandTable {
    deck: u64,
    a: [usize; 15],
    c: [[[u32; 65]; 6]; 15],
}

impl HandTable {
    /// Build the table of `deck`.
    #[must_use]
    pub fn new(deck: u64) -> Self {
        let deck = deck & FULL_DECK;
        let mut a = [0; 15];
        for card in cards(deck) {
            a[value(card) as usize] += 1;
        }
        let mut c = [[[0; 65]; 6]; 15];
        c[14][0][0] = 1;
        for v in (1..=13).rev() {
            // Add binom(a[v], j) times C[v + 1] shifted by j cards and j * v in the sum.
            let (lower, upper) = c.split_at_mut(v + 1);
            for j in 0..=a[v] {
                for k in j..=5 {
                    let rows = lower[v][k][j * v..].iter_mut().zip(&upper[0][k - j]);
                    rows.for_each(|(x, &y)| *x += BINOM[a[v]][j] * y);
                }
            }
        }
        Self { deck, a, c }
    }

    /// The number of legal hands for target `n`, zero outside `6..=64`.
    #[must_use]
    pub fn count(&self, n: u32) -> u64 {
        if n > 64 {
            0
        } else {
            u64::from(self.c[1][5][n as usize])
        }
    }

    /// The `u`-th legal hand for target `n`, or `None` when `u >= count(n)`.
    #[must_use]
    pub fn unrank(&self, n: u32, mut u: u64) -> Option<u64> {
        if n > 64 {
            return None;
        }
        let (mut k, mut s, mut mask) = (5, n as usize, 0u64);
        for v in 1..=13 {
            let av = self.a[v];
            let mut j = 0;
            loop {
                if j > av || j > k || j * v > s {
                    return None;
                }
                let ways = u64::from(BINOM[av][j] * self.c[v + 1][k - j][s - j * v]);
                if u < ways {
                    break;
                }
                u -= ways;
                j += 1;
            }
            let combos = u64::from(BINOM[av][j]);
            let mut r = u % combos;
            u /= combos;
            let rank = if v == 1 { 12 } else { v - 2 };
            let (mut remaining, mut need) = (av, j);
            for suit in 0..4 {
                let card = suit * 13 + rank;
                if need == 0 || self.deck >> card & 1 == 0 {
                    continue;
                }
                let including = u64::from(BINOM[remaining - 1][need - 1]);
                if r < including {
                    mask |= 1 << card;
                    need -= 1;
                } else {
                    r -= including;
                }
                remaining -= 1;
            }
            k -= j;
            s -= j * v;
        }
        Some(mask)
    }
}
