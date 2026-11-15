//! The entry table: sampled counts keyed by number prefix and public class prefix.
//!
//! A kernel entry is `(h, a, b)`: the numbers and classes of the rounds so far and the next
//! pair. An entry gets samples from every number sequence that extends its number prefix, and a
//! hidden class (a fold) pools the five revealed classes at its position. Class prefixes are
//! base-6 codes, first round most significant, with digit 5 for hidden.

use std::collections::HashMap;

use crate::sampler::{Counts, ROW_OFFSET};
use crate::{ROUNDS, TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS, digit};

/// The class digit of a fold, which hides the class.
pub const HIDDEN: usize = 5;
/// Number pairs played so far, in order.
pub type Prefix = Vec<(u8, u8)>;

/// The 14,400 number sequences and the index of every number prefix.
pub struct Sequences {
    /// Sequence `s = i0 * 120 + i1` pairs the `i0`-th and `i1`-th orders of the two teams.
    pub targets: Vec<[(u8, u8); ROUNDS]>,
    /// Per length `r + 1`, the index of each number prefix in order of first appearance.
    pub prefix_index: Vec<HashMap<Prefix, usize>>,
    /// `prefix_of_sequence[r][s]`: the index of the length-`r + 1` prefix of sequence `s`.
    pub prefix_of_sequence: Vec<Vec<usize>>,
}

/// The 120 orders of five cards in lexicographic position order.
fn permutations(items: &[u8]) -> Vec<Vec<u8>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }
    (0..items.len())
        .flat_map(|i| {
            let rest: Vec<u8> = items
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &c)| c)
                .collect();
            permutations(&rest).into_iter().map(move |mut tail| {
                tail.insert(0, items[i]);
                tail
            })
        })
        .collect()
}

impl Sequences {
    /// Enumerate the sequences and index their prefixes.
    #[must_use]
    pub fn new() -> Self {
        let (orders_0, orders_1) = (
            permutations(&TEAM_0_METAL_CARDS),
            permutations(&TEAM_1_METAL_CARDS),
        );
        let targets: Vec<_> = orders_0
            .iter()
            .flat_map(|o0| {
                orders_1
                    .iter()
                    .map(move |o1| std::array::from_fn(|r| (o0[r], o1[r])))
            })
            .collect();
        let mut prefix_index = vec![HashMap::new(); ROUNDS];
        let prefix_of_sequence = (0..ROUNDS)
            .map(|r| {
                let index = &mut prefix_index[r];
                targets
                    .iter()
                    .map(|t: &[(u8, u8); ROUNDS]| {
                        let next = index.len();
                        *index.entry(t[..=r].to_vec()).or_insert(next)
                    })
                    .collect()
            })
            .collect();
        Self {
            targets,
            prefix_index,
            prefix_of_sequence,
        }
    }
}

impl Default for Sequences {
    fn default() -> Self {
        Self::new()
    }
}

/// Sampled counts per round over revealed class prefixes, with hidden pooling on demand.
pub struct KernelCounts {
    /// `revealed[r][prefix * 5^r + code]`: counts of round `r` after a revealed class prefix.
    pub revealed: Vec<Vec<Counts>>,
}

/// Entries the solver reads in rounds 2 to 5: number prefixes times base-6 class prefixes.
pub const ENTRIES: usize = 400 * 6 + 3600 * 36 + 14_400 * 216 + 14_400 * 1296;

impl KernelCounts {
    /// An empty table.
    #[must_use]
    pub fn new(sequences: &Sequences) -> Self {
        Self {
            revealed: (0..ROUNDS)
                .map(|r| vec![[0; 6]; sequences.prefix_index[r].len() * 5usize.pow(r as u32)])
                .collect(),
        }
    }

    /// Add one sequence's sampler output into the entries of its number prefixes.
    pub fn add(&mut self, sequences: &Sequences, sequence: usize, counts: &[Counts]) {
        for r in 0..ROUNDS {
            let width = 5usize.pow(r as u32);
            let start = sequences.prefix_of_sequence[r][sequence] * width;
            let rows = &counts[ROW_OFFSET[r]..ROW_OFFSET[r] + width];
            for (entry, row) in self.revealed[r][start..start + width].iter_mut().zip(rows) {
                entry.iter_mut().zip(row).for_each(|(x, y)| *x += y);
            }
        }
    }

    /// Round `r` counts of one number prefix over all `6^r` class prefixes, hidden classes pooled.
    #[must_use]
    pub fn pooled(&self, r: usize, prefix: usize) -> Vec<Counts> {
        let width = 5usize.pow(r as u32);
        let revealed = &self.revealed[r][prefix * width..(prefix + 1) * width];
        let mut pooled = vec![[0; 6]; 6usize.pow(r as u32)];
        for (code5, counts) in revealed.iter().enumerate() {
            let code6 = (0..r).rev().fold(0, |code, axis| {
                code * 6 + code5 / 5usize.pow(axis as u32) % 5
            });
            pooled[code6] = *counts;
        }
        // A code with hidden digits is filled at its last hidden axis from rows whose earlier
        // hidden axes are already filled.
        for axis in 0..r {
            let place = 6usize.pow((r - 1 - axis) as u32);
            for code in (0..pooled.len()).filter(|code| code / place % 6 == HIDDEN) {
                pooled[code] = (0..5).fold([0; 6], |sum, d| {
                    let source = &pooled[code - (HIDDEN - d) * place];
                    std::array::from_fn(|c| sum[c] + source[c])
                });
            }
        }
        pooled
    }
}

/// Base-6 code of a public class prefix; `None` is a hidden class.
#[must_use]
pub fn class_code(prefix: &[Option<usize>]) -> usize {
    prefix.iter().fold(0, |code, class| {
        code * 6 + class.map_or(HIDDEN, |c| digit(c).expect("a revealed class has a digit"))
    })
}
