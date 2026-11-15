//! The finished table, memory-mapped, and its certified stage games (DTH.md 4.10, 6).

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::ladder::certify;
use crate::{
    ANCHORS, Certificate, Error, MAX_SADDLE_GAP, Matrix, Mix, N, Profiles, State, full_matrix,
    recheck, solve_class, strided,
};

/// One certified stage game: Dropper rows, Checker columns, values from the Dropper's side.
#[derive(Clone, Debug)]
pub struct Stage {
    /// The state `(s_c, t_c, s_d, t_d)`.
    pub state: State,
    /// The stored value `V[pc, pd]`.
    pub value: f64,
    /// `M[d][c]` for the seconds `d, c` in 1..=60.
    pub matrix: Matrix,
    /// A certified Dropper mix.
    pub drop: Mix,
    /// A certified Checker mix.
    pub check: Mix,
    /// `max(M · check) − min(drop · M)`, at most the gate.
    pub gap: f64,
}

impl Stage {
    /// Check `cert` against the full matrix of `(s, f)` and the stored `value` against the result.
    fn certify(
        state: State,
        value: f64,
        s: &Mix,
        f: f64,
        cert: &Certificate,
    ) -> Result<Self, Error> {
        let matrix = full_matrix(s, f);
        let c = certify(&matrix, &cert.drop, &cert.check, cert.rung);
        if !c.certified() {
            return Err(Error::StageGap(c.gap()));
        }
        let Certificate {
            lower,
            upper,
            drop,
            check,
            ..
        } = c;
        if !(lower - MAX_SADDLE_GAP <= value && value <= upper + MAX_SADDLE_GAP) {
            return Err(Error::OutsideCertificate(value, lower, upper));
        }
        Ok(Self {
            state,
            value,
            matrix,
            drop,
            check,
            gap: (upper - lower).max(0.0),
        })
    }
}

/// The finished `V.npy`, memory-mapped read-only.
pub struct Table {
    map: Mmap,
    profiles: Profiles,
}

impl Table {
    /// Byte offset of the first value.
    const DATA: usize = 128;

    /// Open and check a table: the exact `numpy.save` header and size, the six anchors, and
    /// `rechecks` strided classes re-derived from their children.
    ///
    /// # Errors
    ///
    /// [`Error::Io`], [`Error::Format`], [`Error::Anchors`], or a recheck failure.
    pub fn open(path: &Path, rechecks: usize) -> Result<Self, Error> {
        let file = File::open(path)?;
        // SAFETY: the map is read-only, and nothing writes the table file while it is open.
        #[allow(unsafe_code)]
        let map = unsafe { Mmap::map(&file)? };
        if map.len() != Self::DATA + 8 * N * (N + 1)
            || map[..Self::DATA] != crate::sweep::npy_header()
        {
            return Err(Error::Format(path.display().to_string()));
        }
        let table = Self {
            map,
            profiles: Profiles::build(),
        };
        let wrong: Vec<State> = ANCHORS
            .iter()
            .filter(|&&(state, expected)| {
                !table
                    .value(state)
                    .is_ok_and(|v| (v - expected).abs() <= MAX_SADDLE_GAP)
            })
            .map(|&(state, _)| state)
            .collect();
        if !wrong.is_empty() {
            return Err(Error::Anchors(wrong));
        }
        for c in strided(rechecks) {
            recheck(c, |i| table.get(i), &table.profiles)?;
        }
        Ok(table)
    }

    /// The entry at flat index `i`.
    fn get(&self, i: usize) -> f64 {
        let at = Self::DATA + 8 * i;
        f64::from_le_bytes(self.map[at..at + 8].try_into().expect("a slice of 8 bytes"))
    }

    /// The certified value of a state, from the current Dropper's side.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] or [`Error::OffDomain`].
    pub fn value(&self, state: State) -> Result<f64, Error> {
        let (pc, pd) = self.profiles.classes(state)?;
        Ok(self.get(pc * (N + 1) + pd))
    }

    /// Rebuild a state's stage game from its stored children, solve it, and check the certificate
    /// and the stored value against the full matrix.
    ///
    /// # Errors
    ///
    /// A domain error, [`Error::Unsolved`], [`Error::Uncertified`], [`Error::StageGap`] or
    /// [`Error::OutsideCertificate`].
    pub fn stage(&self, state: State) -> Result<Stage, Error> {
        let (pc, pd) = self.profiles.classes(state)?;
        let (s, f) = self.profiles.class_values(pc, pd, |i| self.get(i))?;
        let cert = solve_class(&s, f).ok_or(Error::Uncertified(pc, pd))?;
        Stage::certify(state, self.get(pc * (N + 1) + pd), &s, f, &cert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LAGS, Rung};

    #[test]
    fn stage_is_the_certified_full_matrix() {
        let s: Mix = std::array::from_fn(|k| -0.9 + 1.7 * (k as f64 / 59.0).powi(2));
        let cert = solve_class(&s, 0.3).expect("certified");
        let stage = Stage::certify([0; 4], cert.value(), &s, 0.3, &cert).expect("certified stage");
        assert_eq!(
            (stage.value, stage.matrix),
            (cert.value(), full_matrix(&s, 0.3))
        );
        assert!(stage.gap <= MAX_SADDLE_GAP);
        assert!((stage.drop.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((stage.check.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stage_fails_closed_when_the_saddle_gap_is_too_large() {
        let s: Mix = std::array::from_fn(|k| -1.0 + 2.0 * k as f64 / 59.0);
        let first: Mix = std::array::from_fn(|k| f64::from(u8::from(k == 0)));
        let cert = Certificate {
            lower: 0.0,
            upper: 0.0,
            rung: Rung::Pure,
            drop: first,
            check: first,
        };
        assert!(matches!(
            Stage::certify([0; 4], 0.0, &s, 0.0, &cert),
            Err(Error::StageGap(_))
        ));
    }

    #[test]
    fn stage_fails_closed_when_the_stored_value_leaves_its_certificate() {
        let cert = solve_class(&[0.25; LAGS], 0.25).expect("pure");
        assert!(matches!(
            Stage::certify([0; 4], 0.5, &[0.25; LAGS], 0.25, &cert),
            Err(Error::OutsideCertificate(..))
        ));
    }
}
