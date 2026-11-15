//! Dense working tables, their packed form, and rank-select reads of packed tables.
//!
//! A dense table `KEY.npy` holds `N` rows of `columns + 1` values: NaN for an unreachable state
//! and `-1` in the last column, the `WIN` child. A finished key packs into `KEY.bits.npy`, the
//! reachable states as bits `row * columns + col` (least significant first), and
//! `KEY.values.npy`, their values in the same order. The Python reference reads and writes the same
//! files.

use crate::Error;
use crate::npy::{Npy, NpyMut};
use crate::reach::Bits;
use crate::rules::{Key, N};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const BLOCK: usize = 4096;

/// The working directory of a build: dense tables of the keys still needed as children, and
/// packed tables of the rest.
pub struct Store {
    dir: PathBuf,
    source: PathBuf,
    dense: BTreeSet<Key>,
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error {
    let path = path.to_path_buf();
    move |source| Error::Io { path, source }
}

impl Store {
    /// Open a store that writes to `dir` and reads packed children from `source`, which is `dir`
    /// itself in a full build.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] when `dir` cannot be created.
    pub fn new(dir: &Path, source: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(dir).map_err(io(dir))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            source: source.to_path_buf(),
            dense: BTreeSet::new(),
        })
    }

    fn path(&self, key: Key) -> PathBuf {
        self.dir.join(format!("{key}.npy"))
    }

    /// Create the dense table of `key`, unsolved everywhere.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] when the file cannot be created.
    pub(crate) fn create(&mut self, key: Key) -> Result<NpyMut, Error> {
        self.dense.insert(key);
        blank(&self.path(key), key)
    }

    /// Map the dense table of `key`, unpacking it from `source` when no dense copy exists.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] or [`Error::Format`] when the table is missing or malformed.
    pub(crate) fn load(&self, key: Key) -> Result<Npy, Error> {
        let path = self.path(key);
        if !path.exists() {
            let packed = Packed::open(&self.source, key)?;
            let mut table = blank(&path, key)?;
            let (cols, dense) = (key.columns(), table.data::<f64>());
            for (flat, &value) in packed.positions().zip(packed.values()) {
                dense[flat / cols * (cols + 1) + flat % cols] = value;
            }
            table.flush()?;
        }
        let table = Npy::open::<f64>(&path)?;
        if table.shape != [N, key.columns() + 1] {
            return Err(Error::Format {
                path,
                reason: "dense table shape mismatch".into(),
            });
        }
        Ok(table)
    }

    /// Pack the dense table of `key` and delete it.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] or [`Error::Format`] when a file cannot be read or written.
    pub fn pack(&mut self, key: Key) -> Result<(), Error> {
        let dense = self.load(key)?;
        let (cols, width) = (key.columns(), key.columns() + 1);
        let table = dense.data::<f64>();
        let count = table
            .chunks(width)
            .flat_map(|row| &row[..cols])
            .filter(|x| x.is_finite())
            .count();
        let mut bits = NpyMut::create::<u8>(
            &self.dir.join(format!("{key}.bits.npy")),
            &[(N * cols).div_ceil(8)],
        )?;
        let mut values =
            NpyMut::create::<f64>(&self.dir.join(format!("{key}.values.npy")), &[count])?;
        let (flat, out) = (bits.data::<u8>(), values.data::<f64>());
        let mut next = 0;
        for (position, &value) in table.chunks(width).flat_map(|row| &row[..cols]).enumerate() {
            if value.is_finite() {
                flat[position / 8] |= 1 << (position % 8);
                out[next] = value;
                next += 1;
            }
        }
        bits.flush()?;
        values.flush()?;
        drop(dense);
        std::fs::remove_file(self.path(key)).map_err(io(&self.path(key)))?;
        self.dense.remove(&key);
        Ok(())
    }

    /// Pack every dense table whose key starts after `clock`, beyond the remaining sweep's
    /// dependency horizon.
    ///
    /// # Errors
    ///
    /// As [`Store::pack`].
    pub fn pack_after(&mut self, clock: u32) -> Result<(), Error> {
        let late: Vec<Key> = self
            .dense
            .iter()
            .copied()
            .filter(|key| key.clock() > clock)
            .collect();
        late.into_iter().try_for_each(|key| self.pack(key))
    }
}

/// Create the dense table of `key` at `path`: NaN for every state and -1 in the `WIN` column.
fn blank(path: &Path, key: Key) -> Result<NpyMut, Error> {
    let width = key.columns() + 1;
    let mut table = NpyMut::create::<f64>(path, &[N, width])?;
    for row in table.data::<f64>().chunks_mut(width) {
        row.fill(f64::NAN);
        row[width - 1] = -1.0;
    }
    Ok(table)
}

/// One packed table, with the bit count before each 4096-byte block for rank queries.
pub(crate) struct Packed {
    key: Key,
    bits: Npy,
    values: Npy,
    ranks: Vec<usize>,
}

impl Packed {
    /// Map the packed table of `key` in `dir` and check its schema.
    pub(crate) fn open(dir: &Path, key: Key) -> Result<Self, Error> {
        let path = dir.join(format!("{key}.bits.npy"));
        let bits = Npy::open::<u8>(&path)?;
        let values = Npy::open::<f64>(&dir.join(format!("{key}.values.npy")))?;
        let format = |reason: &str| Error::Format {
            path: path.clone(),
            reason: reason.into(),
        };
        if bits.shape != [(N * key.columns()).div_ceil(8)] || values.shape.len() != 1 {
            return Err(format("packed schema mismatch"));
        }
        let mut ranks = vec![0];
        for block in bits.data::<u8>().chunks(BLOCK) {
            ranks.push(
                ranks[ranks.len() - 1]
                    + block.iter().map(|b| b.count_ones() as usize).sum::<usize>(),
            );
        }
        if ranks[ranks.len() - 1] != values.shape[0] {
            return Err(format("packed bitmap and value count mismatch"));
        }
        Ok(Self {
            key,
            bits,
            values,
            ranks,
        })
    }

    pub(crate) fn flat(&self) -> &[u8] {
        self.bits.data()
    }

    pub(crate) fn values(&self) -> &[f64] {
        self.values.data()
    }

    /// Return the flat positions `row * columns + col` of the reachable states, in value order.
    pub(crate) fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.flat().iter().enumerate().flat_map(|(byte, &b)| {
            (0..8)
                .filter(move |bit| b >> bit & 1 == 1)
                .map(move |bit| 8 * byte + bit)
        })
    }

    /// Return the stored value of `(row, col)`; the `WIN` column reads -1.
    pub(crate) fn get(&self, row: usize, col: usize) -> Result<f64, Error> {
        let width = self.key.columns();
        if row >= N || col > width {
            return Err(Error::Invalid(format!(
                "{}: packed index ({row}, {col}) out of range",
                self.key
            )));
        }
        if col == width {
            return Ok(-1.0);
        }
        let (flat, position) = (self.flat(), row * width + col);
        let (byte, bit) = (position / 8, position % 8);
        if flat[byte] >> bit & 1 == 0 {
            return Err(Error::Invalid(format!(
                "{}: unreachable packed state ({row}, {col})",
                self.key
            )));
        }
        let block = byte / BLOCK;
        let before: usize = flat[block * BLOCK..byte]
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum();
        let value = self.values()
            [self.ranks[block] + before + (flat[byte] & ((1 << bit) - 1)).count_ones() as usize];
        if !value.is_finite() {
            return Err(Error::Invalid(format!(
                "{}: packed values must be finite",
                self.key
            )));
        }
        Ok(value)
    }

    /// Return the state `(row, col)` of the value with this rank.
    pub(crate) fn select(&self, rank: usize) -> (usize, usize) {
        let block = self.ranks.partition_point(|&r| r <= rank) - 1;
        let mut left = rank - self.ranks[block];
        let flat = self.flat();
        for (offset, &b) in flat[block * BLOCK..].iter().enumerate() {
            let ones = b.count_ones() as usize;
            if left < ones {
                let bit = (0..8)
                    .filter(|bit| b >> bit & 1 == 1)
                    .nth(left)
                    .expect("a set bit");
                let position = 8 * (block * BLOCK + offset) + bit;
                return (position / self.key.columns(), position % self.key.columns());
            }
            left -= ones;
        }
        unreachable!("rank {rank} lies below the value count")
    }

    /// Check the packed table against the reachable states of its key.
    pub(crate) fn matches(&self, reach: &Bits) -> bool {
        self.flat() == reach.to_flat().as_slice()
    }
}

/// Reads values of packed tables in a directory, keeping each opened table.
pub struct PackedReader {
    dir: PathBuf,
    open: Mutex<HashMap<Key, Arc<Packed>>>,
}

impl PackedReader {
    /// Read the packed tables in `dir`.
    #[must_use]
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            open: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn table(&self, key: Key) -> Result<Arc<Packed>, Error> {
        let mut open = self
            .open
            .lock()
            .expect("no reader panicked while holding the lock");
        if let Some(table) = open.get(&key) {
            return Ok(Arc::clone(table));
        }
        let table = Arc::new(Packed::open(&self.dir, key)?);
        open.insert(key, Arc::clone(&table));
        Ok(table)
    }

    /// Return the stored value of `(checker, col)` in the table of `key`; the column after the
    /// last reads -1, the value of a dead Checker.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for an index out of range or an unreachable state, and the errors of
    /// opening the table.
    pub fn get(&self, key: Key, checker: usize, col: usize) -> Result<f64, Error> {
        self.table(key)?.get(checker, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::RESETS;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stl-leap-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn pack_round_trips_and_keeps_the_win_column() {
        let dir = scratch("pack");
        let mut store = Store::new(&dir, &dir).expect("a scratch store");
        let key = Key::Rev(3420);
        let mut table = store.create(key).expect("a dense table");
        table.data::<f64>()[3 * (RESETS + 1) + 4] = 0.25;
        table.data::<f64>()[8 * (RESETS + 1) + 9] = -0.4;
        table.flush().expect("a flush");
        drop(table);
        store.pack(key).expect("a packed table");
        assert!(!dir.join("REV_3420.npy").exists());
        let reader = PackedReader::new(&dir);
        assert_eq!(reader.get(key, 3, 4).ok(), Some(0.25));
        assert_eq!(reader.get(key, 0, RESETS).ok(), Some(-1.0));
        assert!(reader.get(key, 0, 0).is_err());
        let restored = store.load(key).expect("an unpacked table");
        let values = restored.data::<f64>();
        assert_eq!(
            (values[3 * (RESETS + 1) + 4], values[8 * (RESETS + 1) + 9]),
            (0.25, -0.4)
        );
        assert!(values[0].is_nan() && values.chunks(RESETS + 1).all(|row| row[RESETS] == -1.0));
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn pack_after_keeps_the_dependency_horizon() {
        let dir = scratch("horizon");
        let mut store = Store::new(&dir, &dir).expect("a scratch store");
        for key in [Key::Rev(3599), Key::Rev(3600)] {
            let mut table = store.create(key).expect("a dense table");
            table.data::<f64>()[0] = 0.3;
        }
        store.pack_after(3599).expect("packing");
        assert!(dir.join("REV_3599.npy").exists() && !dir.join("REV_3600.npy").exists());
        store.pack_after(0).expect("packing");
        assert!(!dir.join("REV_3599.npy").exists());
        assert_eq!(
            PackedReader::new(&dir).get(Key::Rev(3599), 0, 0).ok(),
            Some(0.3)
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn packed_reader_ranks_and_selects() {
        let dir = scratch("rank");
        let mut store = Store::new(&dir, &dir).expect("a scratch store");
        let key = Key::Rev(3600);
        let mut table = store.create(key).expect("a dense table");
        let mut reach = Bits::new(RESETS);
        let mut expected = vec![];
        // About 17% of the states, spread over many 4096-byte blocks.
        let mut state = 384_u64;
        for row in 0..N {
            for col in 0..RESETS {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if state >> 61 == 0 || (state >> 58) & 7 == 1 {
                    let value = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                    table.data::<f64>()[row * (RESETS + 1) + col] = value;
                    reach.set(row, col);
                    expected.push((row, col, value));
                }
            }
        }
        drop(table);
        store.pack(key).expect("a packed table");
        let packed = Packed::open(&dir, key).expect("a packed table");
        assert!(packed.matches(&reach));
        for (rank, &(row, col, value)) in expected.iter().enumerate().step_by(97) {
            assert_eq!(packed.get(row, col).ok(), Some(value));
            assert_eq!(packed.select(rank), (row, col));
        }
        assert!(packed.get(N, 0).is_err() && packed.get(0, RESETS + 1).is_err());
        // One more reachable state than the packed table holds breaks the membership check.
        let (row, col) = (0..RESETS)
            .map(|col| (0, col))
            .find(|&(r, c)| !reach.get(r, c))
            .expect("a gap");
        assert!(packed.get(row, col).is_err());
        reach.set(row, col);
        assert!(!packed.matches(&reach));
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
