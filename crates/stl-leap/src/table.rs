//! The read path into a finished leap artifact.

use crate::Error;
use crate::build::{TOTALS, stage_inputs};
use crate::kernel::GAP;
use crate::lp::{self, Certificate};
use crate::rules::{Key, WIN, profiles};
use crate::store::PackedReader;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use stl_dth::Table as Dth;

/// The fields of `report.json` that mark a complete, audited build.
#[derive(Deserialize)]
struct Completed {
    states: u64,
    rows: BTreeMap<String, Rows>,
    max_bellman_residual: f64,
}

#[derive(Deserialize)]
struct Rows {
    states: u64,
}

/// A stored stage value and its fresh certificate.
#[derive(Clone, Debug, Serialize)]
pub struct Stage {
    /// The stored value, from the Dropper's side.
    pub stored_value: f64,
    /// `|stored_value - certificate.value|`.
    pub bellman_residual: f64,
    /// The audit LP's certificate of the stage rebuilt from the stored children.
    #[serde(flatten)]
    pub certificate: Certificate,
}

/// The stored values and stage certificates of one complete leap artifact.
pub struct LeapTable {
    reader: PackedReader,
    dth: Dth,
}

/// Open a complete leap artifact (the directory holding `report.json` and `tables/`) and the DTH
/// table that supplies its tail.
///
/// # Errors
///
/// [`Error::Io`] when the report is missing, [`Error::Format`] when it does not record full
/// coverage and a Bellman residual of at most `1e-6`, and the errors of [`Dth::open`].
pub fn open_leap(artifact: &Path, dth: &Path) -> Result<LeapTable, Error> {
    let path = artifact.join("report.json");
    let text = std::fs::read_to_string(&path).map_err(crate::io(&path))?;
    let complete = serde_json::from_str::<Completed>(&text).is_ok_and(|report| {
        report.states == TOTALS.iter().map(|&(_, n)| n).sum::<u64>()
            && report.rows.len() == TOTALS.len()
            && TOTALS
                .iter()
                .all(|&(kind, n)| report.rows.get(kind).is_some_and(|row| row.states == n))
            && (0.0..=GAP).contains(&report.max_bellman_residual)
    });
    if !complete {
        return Err(Error::Format {
            path,
            reason: "no complete, numerically checked leap result".into(),
        });
    }
    Ok(LeapTable {
        reader: PackedReader::new(&artifact.join("tables")),
        dth: Dth::open(dth, 0)?,
    })
}

impl LeapTable {
    /// Return the stored value of the state `(checker, dropper)` of `key`, from the Dropper's
    /// side; the dropper profile `WIN` reads -1.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a profile out of range, a dropper that a `REV` key does not store,
    /// or an unreachable state.
    pub fn value(&self, key: Key, checker: usize, dropper: usize) -> Result<f64, Error> {
        if dropper > WIN {
            return Err(Error::Invalid(format!(
                "dropper profile {dropper} is out of range"
            )));
        }
        let column = match key {
            Key::Rev(_) => profiles().reset_column(dropper).ok_or_else(|| {
                Error::Invalid(format!(
                    "{key} stores no column for dropper profile {dropper}"
                ))
            })?,
            _ => dropper,
        };
        self.reader.get(key, checker, column)
    }

    /// Re-certify one stored value with an independent LP on its stage matrix, rebuilt from the
    /// stored children and the DTH tail.
    ///
    /// # Errors
    ///
    /// The errors of [`LeapTable::value`], and [`Error::Uncertified`] when the LP finds no
    /// certificate within `1e-6` or the stored value misses it by more than `1e-6`.
    pub fn stage(&self, key: Key, checker: usize, dropper: usize) -> Result<Stage, Error> {
        let stored_value = self.value(key, checker, dropper)?;
        let (s, f) = stage_inputs(&self.reader, key, checker, dropper, &self.dth)?;
        let place = format!("{key} checker {checker} dropper {dropper}");
        let certificate = lp::certify(&s, f, key.is_window())
            .map_err(|error| Error::Uncertified(format!("{place}: {error}")))?;
        let bellman_residual = (stored_value - certificate.value).abs();
        if bellman_residual > GAP {
            return Err(Error::Uncertified(format!(
                "{place}: Bellman residual {bellman_residual:.3e} exceeds 1e-6"
            )));
        }
        Ok(Stage {
            stored_value,
            bellman_residual,
            certificate,
        })
    }
}
