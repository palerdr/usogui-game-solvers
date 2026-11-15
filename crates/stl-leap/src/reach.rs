//! Forward reachability: the states of each key that play can reach from the opening.
//!
//! The opening is `H1_12` with both players at profile 0. A state `(checker, dropper)` moves on
//! a successful check to `(dropper, succ[checker][k])` at the success child, and on a survived
//! failed check to `(dropper, fail[checker])` at the revival child of dose `st + 60`. Each key's
//! successor sets are unioned before the key is counted, so every state counts once.

use crate::Error;
use crate::rules::{Key, N, N_ALIVE, RESETS, profiles};
use std::collections::BTreeMap;

/// The reachable states of one key: bit `(checker, column)`, row-major, 64-bit words per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bits {
    cols: usize,
    stride: usize,
    words: Vec<u64>,
}

impl Bits {
    /// Return an empty bitmap with `N` checker rows and `cols` dropper columns.
    #[must_use]
    pub fn new(cols: usize) -> Self {
        let stride = cols.div_ceil(64);
        Self {
            cols,
            stride,
            words: vec![0; N * stride],
        }
    }

    /// Read the packed artifact layout: bit `row * cols + col` of `flat`, least significant first.
    pub(crate) fn from_flat(cols: usize, flat: &[u8]) -> Self {
        let mut bits = Self::new(cols);
        for row in 0..N {
            for col in (0..cols)
                .filter(|&col| flat[(row * cols + col) / 8] >> ((row * cols + col) % 8) & 1 == 1)
            {
                bits.set(row, col);
            }
        }
        bits
    }

    /// Return the packed artifact layout of this bitmap.
    pub(crate) fn to_flat(&self) -> Vec<u8> {
        let mut flat = vec![0_u8; (N * self.cols).div_ceil(8)];
        for row in 0..N {
            for col in self.columns(row) {
                flat[(row * self.cols + col) / 8] |= 1 << ((row * self.cols + col) % 8);
            }
        }
        flat
    }

    /// Return whether the state `(row, col)` is reachable.
    pub(crate) fn get(&self, row: usize, col: usize) -> bool {
        self.words[row * self.stride + col / 64] >> (col % 64) & 1 == 1
    }

    /// Mark the state `(row, col)` reachable.
    pub fn set(&mut self, row: usize, col: usize) {
        self.words[row * self.stride + col / 64] |= 1 << (col % 64);
    }

    /// Return the reachable columns of one checker row, in increasing order.
    pub(crate) fn columns(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        self.row(row).iter().enumerate().flat_map(|(w, &word)| {
            let mut word = word;
            std::iter::from_fn(move || {
                (word != 0).then(|| {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    64 * w + bit
                })
            })
        })
    }

    /// Return the number of reachable states.
    pub(crate) fn count(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    fn row(&self, row: usize) -> &[u64] {
        &self.words[row * self.stride..][..self.stride]
    }

    fn or_row(&mut self, row: usize, source: &[u64]) {
        or(&mut self.words[row * self.stride..][..self.stride], source);
    }
}

/// Set in `target` every bit set in `source`.
fn or(target: &mut [u64], source: &[u64]) {
    for (word, &x) in target.iter_mut().zip(source) {
        *word |= x;
    }
}

/// Transpose a 64-by-64 bit block in place: bit `j` of word `i` moves to bit `i` of word `j`.
fn transpose64(a: &mut [u64; 64]) {
    let mut width = 32;
    let mut mask: u64 = 0x0000_0000_FFFF_FFFF;
    while width != 0 {
        for k in (0..64).filter(|k| k & width == 0) {
            let t = ((a[k] >> width) ^ a[k + width]) & mask;
            a[k] ^= t << width;
            a[k + width] ^= t;
        }
        width >>= 1;
        mask ^= mask << width;
    }
}

/// Transpose `rows` rows of `words` (row length `stride`) into `cols` rows over `rows` bits.
fn transpose(words: &[u64], stride: usize, rows: usize, cols: usize) -> (Vec<u64>, usize) {
    let out_stride = rows.div_ceil(64);
    let mut out = vec![0; cols * out_stride];
    let mut block = [0; 64];
    for rb in 0..out_stride {
        for cb in 0..stride {
            for (i, x) in block.iter_mut().enumerate() {
                let row = 64 * rb + i;
                *x = if row < rows {
                    words[row * stride + cb]
                } else {
                    0
                };
            }
            transpose64(&mut block);
            for (i, &x) in block.iter().enumerate().take(cols - 64 * cb) {
                out[(64 * cb + i) * out_stride + rb] = x;
            }
        }
    }
    (out, out_stride)
}

/// Return the rows of the successors: row `q` unions the rows of every checker whose successful
/// check can leave it at profile `q`.
///
/// Within a TTD block profile ids follow the vial, so the 60 squandered times are 60 row shifts,
/// which six doubling steps (1, 2, 4, 8, 16, 28) compose.
fn successors(parent: &Bits) -> Bits {
    let stride = parent.stride;
    let mut target = Bits::new(parent.cols);
    for &(start, stop) in profiles().blocks.iter().chain(&[(N_ALIVE, N)]) {
        let length = stop - start;
        let size = (length + 60).min(300);
        let mut expanded = vec![0_u64; size * stride];
        for s in 1..=length.min(299) {
            expanded[s * stride..][..stride].copy_from_slice(parent.row(start + s - 1));
        }
        for shift in [1, 2, 4, 8, 16, 28] {
            for s in (shift..size).rev() {
                for w in 0..stride {
                    expanded[s * stride + w] |= expanded[(s - shift) * stride + w];
                }
            }
        }
        for s in 0..size {
            let row = if start != N_ALIVE && s < length {
                start + s
            } else {
                N_ALIVE + s
            };
            target.or_row(row, &expanded[s * stride..][..stride]);
        }
    }
    target
}

/// Mark the reachable states of every key whose clock is at most `limit`, by sweeping the keys
/// forward in clock order.
///
/// # Errors
///
/// [`Error::Invalid`] when `limit` exceeds 3600, where the leap game ends.
///
/// # Panics
///
/// Only on a broken profile invariant: a revival always resets the vial to a reset profile.
pub fn forward(limit: u32) -> Result<BTreeMap<Key, Bits>, Error> {
    if limit > 3600 {
        return Err(Error::Invalid("reachability ends at 3600".into()));
    }
    let p = profiles();
    let mut opening = Bits::new(N);
    opening.set(0, 0);
    let mut pending = BTreeMap::from([(Key::H1(12), opening)]);
    let mut reached = BTreeMap::new();
    for key in Key::all()
        .into_iter()
        .take_while(|key| key.clock() <= limit)
    {
        let Some(bits) = pending.remove(&key).filter(|bits| bits.count() > 0) else {
            continue;
        };
        let within = |child: Option<Key>| child.filter(|c| c.clock() <= limit);
        if let Some(child) = within(key.child(0)) {
            // Column c of the successor rows becomes the child row of dropper c.
            let (columns, stride) = transpose(&successors(&bits).words, bits.stride, N, bits.cols);
            let child_bits = pending
                .entry(child)
                .or_insert_with(|| Bits::new(child.columns()));
            for col in 0..bits.cols {
                child_bits.or_row(key.dropper(col), &columns[col * stride..][..stride]);
            }
        }
        // Group the revivals by child key; a group's rows are the checkers' reset profiles.
        let mut groups: BTreeMap<Key, Vec<u64>> = BTreeMap::new();
        for pc in (0..N_ALIVE).filter(|&pc| bits.row(pc).iter().any(|&w| w != 0)) {
            let Some(child) = within(key.child(u32::from(p.st[pc]) + 60)) else {
                continue;
            };
            let merged = groups
                .entry(child)
                .or_insert_with(|| vec![0; RESETS * bits.stride]);
            let reset = p
                .reset_column(p.fail[pc] as usize)
                .expect("a revival resets the vial");
            or(
                &mut merged[reset * bits.stride..][..bits.stride],
                bits.row(pc),
            );
        }
        for (child, merged) in groups {
            let (columns, stride) = transpose(&merged, bits.stride, RESETS, bits.cols);
            let child_bits = pending
                .entry(child)
                .or_insert_with(|| Bits::new(child.columns()));
            for col in 0..bits.cols {
                let resets = &columns[col * stride..][..stride];
                if matches!(child, Key::Rev(_)) {
                    child_bits.or_row(key.dropper(col), resets);
                } else {
                    for j in (0..RESETS).filter(|&j| resets[j / 64] >> (j % 64) & 1 == 1) {
                        child_bits.set(key.dropper(col), p.resets[j] as usize);
                    }
                }
            }
        }
        reached.insert(key, bits);
    }
    Ok(reached)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_moves_every_bit() {
        let mut words = vec![0_u64; 70 * 2];
        for (r, c) in [(0, 0), (1, 65), (69, 3), (64, 127)] {
            words[r * 2 + c / 64] |= 1 << (c % 64);
        }
        let (out, stride) = transpose(&words, 2, 70, 128);
        let set: Vec<(usize, usize)> = (0..128)
            .flat_map(|c| (0..70).map(move |r| (c, r)))
            .filter(|&(c, r)| out[c * stride + r / 64] >> (r % 64) & 1 == 1)
            .collect();
        assert_eq!(set, [(0, 0), (3, 69), (65, 1), (127, 64)]);
    }

    #[test]
    fn flat_layout_round_trips() {
        let mut bits = Bits::new(RESETS);
        bits.set(0, 0);
        bits.set(5, 182);
        bits.set(N - 1, 64);
        assert_eq!(Bits::from_flat(RESETS, &bits.to_flat()), bits);
        assert_eq!(bits.columns(5).collect::<Vec<_>>(), [182]);
    }

    #[test]
    fn forward_counts_match_the_reference() {
        let count = |limit| {
            forward(limit)
                .expect("a valid limit")
                .values()
                .map(Bits::count)
                .sum::<u64>()
        };
        assert_eq!(count(1500), 570_612);
        assert_eq!(count(1800), 12_971_150);
        assert!(forward(3601).is_err());
    }
}
