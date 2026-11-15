//! One player's behavioral strategy keyed by information state: the hand-off format between a
//! solver and the certificate.
//!
//! A key packs (own beads, ordered transcript of announced totals) into one integer. Column a of
//! a row announces own beads + a + 1.
//!
//! # File format
//!
//! Little-endian, no padding: the 8 bytes `TOKTABLE`, the row count n as `u64`, the 10 bead-pick
//! probabilities as `f64`, n keys as `u64`, then n rows of 10 probabilities as `f64`.
//! `tests/data/npz_to_table.py` writes this format from a Python `StrategyTable` `.npz`.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::rules::InformationState;
use crate::traversal::Profile;
use crate::tree::Layer;
use crate::{Error, HANDS, TOTALS};

const MAGIC: &[u8; 8] = b"TOKTABLE";

/// The key of own beads after a transcript of `len` announcements with base-19 code `code`.
fn pack(code: u64, len: usize, own_beads: u8) -> u64 {
    (code * 8 + len as u64) * HANDS as u64 + u64::from(own_beads) - 1
}

/// The key of the information state (own beads, transcript). The format holds at most 7
/// announcements, which covers K <= 4.
#[must_use]
pub fn info_key(own_beads: u8, totals: impl ExactSizeIterator<Item = u8>) -> u64 {
    let len = totals.len();
    let code = totals.fold(0, |code, g| code * TOTALS as u64 + u64::from(g - 2));
    pack(code, len, own_beads)
}

/// One player's strategy: a bead-pick distribution and one row per announcement information set.
#[derive(Debug, PartialEq)]
pub struct StrategyTable {
    /// Bead-pick probabilities, indexed by beads - 1.
    pub choose: Vec<f64>,
    /// The [`info_key`] of each row.
    pub keys: Vec<u64>,
    /// Rows of 10 action probabilities, row-major.
    pub probs: Vec<f64>,
    rows: HashMap<u64, usize>,
}

impl StrategyTable {
    /// A table from its parts.
    ///
    /// # Errors
    ///
    /// [`Error::Artifact`] when the lengths disagree or a key repeats.
    pub fn new(choose: Vec<f64>, keys: Vec<u64>, probs: Vec<f64>) -> Result<Self, Error> {
        if choose.len() != HANDS || probs.len() != HANDS * keys.len() {
            return Err(Error::Artifact("table lengths disagree".into()));
        }
        let rows: HashMap<_, _> = keys.iter().enumerate().map(|(r, &k)| (k, r)).collect();
        if rows.len() != keys.len() {
            return Err(Error::Artifact("table repeats a key".into()));
        }
        Ok(Self {
            choose,
            keys,
            probs,
            rows,
        })
    }

    /// The row of the key, if present.
    #[must_use]
    pub fn row_of(&self, key: u64) -> Option<&[f64]> {
        self.rows
            .get(&key)
            .map(|&r| &self.probs[HANDS * r..HANDS * (r + 1)])
    }

    /// The row the observer of `info` plays, if `info` is an announcement information state in
    /// the table.
    #[must_use]
    pub fn row(&self, info: &InformationState) -> Option<&[f64]> {
        let totals = info.history.iter().map(|miss| miss.total);
        self.row_of(info_key(info.own_beads?, totals))
    }

    /// Write the table in the file format of this module.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] when the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&(self.keys.len() as u64).to_le_bytes())?;
        for x in &self.choose {
            out.write_all(&x.to_le_bytes())?;
        }
        for k in &self.keys {
            out.write_all(&k.to_le_bytes())?;
        }
        for x in &self.probs {
            out.write_all(&x.to_le_bytes())?;
        }
        Ok(out.flush()?)
    }

    /// Read a table written in the file format of this module.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] when the file cannot be read, [`Error::Artifact`] when it is malformed.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path)?;
        let bad = || Error::Artifact(format!("{} is not a strategy table", path.display()));
        let (magic, rest) = bytes.split_first_chunk::<8>().ok_or_else(bad)?;
        let (n, rest) = rest.split_first_chunk::<8>().ok_or_else(bad)?;
        let n = u64::from_le_bytes(*n) as usize;
        let (words, tail) = rest.as_chunks::<8>();
        if magic != MAGIC || !tail.is_empty() || words.len() != HANDS + n + HANDS * n {
            return Err(bad());
        }
        let (choose, rest) = words.split_at(HANDS);
        let (keys, probs) = rest.split_at(n);
        let float = |w: &[u8; 8]| f64::from_le_bytes(*w);
        let keys = keys.iter().map(|w| u64::from_le_bytes(*w)).collect();
        Self::new(
            choose.iter().map(float).collect(),
            keys,
            probs.iter().map(float).collect(),
        )
    }
}

/// Base-19 code of each public node's transcript, layer by layer.
fn transcript_codes(layers: &[Layer]) -> Vec<Vec<u64>> {
    let mut codes = vec![vec![0]];
    for (d, layer) in layers.iter().enumerate().skip(1) {
        let up = &codes[d - 1];
        let code = |(&n, &g): (&u32, &u8)| up[n as usize] * TOTALS as u64 + u64::from(g - 2);
        let next = layer.parent.iter().zip(&layer.total).map(code).collect();
        codes.push(next);
    }
    codes
}

/// The keys of player p's slots on each of p's layers.
fn layer_keys(layers: &[Layer], p: usize) -> Result<Vec<(usize, Vec<u64>)>, Error> {
    if layers.len() > 8 {
        return Err(Error::Artifact("strategy tables hold at most K = 4".into()));
    }
    let codes = transcript_codes(layers);
    let mut out = Vec::new();
    for (d, layer) in layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| layer.actor == p)
    {
        let slots = layer.slot_node[p].iter().zip(&layer.slot_hand[p]);
        out.push((
            d,
            slots
                .map(|(&n, &h)| pack(codes[d][n as usize], d, h))
                .collect(),
        ));
    }
    Ok(out)
}

/// Collect player p's part of `profile` from the arrays.
///
/// # Errors
///
/// [`Error::Artifact`] when the horizon exceeds the key format.
pub fn export(layers: &[Layer], profile: &Profile, p: usize) -> Result<StrategyTable, Error> {
    let (mut keys, mut probs) = (Vec::new(), Vec::new());
    for (d, k) in layer_keys(layers, p)? {
        keys.extend(k);
        probs.extend_from_slice(&profile.behavior[d]);
    }
    StrategyTable::new(profile.choose[p].clone(), keys, probs)
}

/// The profile on the arrays that plays `tables[p]` for each player p.
///
/// # Errors
///
/// [`Error::Artifact`] when a table lacks an information set of the arrays.
pub fn import(layers: &[Layer], tables: [&StrategyTable; 2]) -> Result<Profile, Error> {
    let mut behavior = vec![Vec::new(); layers.len()];
    for (p, table) in tables.into_iter().enumerate() {
        for (d, keys) in layer_keys(layers, p)? {
            let missing = || Error::Artifact(format!("P{p}'s table lacks a slot of layer {d}"));
            for key in keys {
                behavior[d].extend_from_slice(table.row_of(key).ok_or_else(missing)?);
            }
        }
    }
    Ok(Profile {
        behavior,
        choose: tables.map(|table| table.choose.clone()),
    })
}
