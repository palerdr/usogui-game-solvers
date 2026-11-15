//! The finished table, memory-mapped, and its certified stage games (DTH.md 4.10, 6).

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::ladder::{certify, solve_class};
use crate::{
    ANCHORS, Certificate, Error, MAX_SADDLE_GAP, Mix, N, Profiles, State, full_matrix, recheck,
    strided,
};

/// The certified solution of one stage game, from the Dropper's side.
#[derive(Clone, Debug)]
pub struct Stage {
    /// The stored value `V[pc, pd]`.
    pub value: f64,
    /// A certified Dropper mix.
    pub drop: Mix,
    /// A certified Checker mix.
    pub check: Mix,
    /// `max(M · check) − min(drop · M)`, at most the gate.
    pub gap: f64,
}

impl Stage {
    /// Check `cert` against the full matrix of `(s, f)` and the stored `value` against the result.
    fn certify(value: f64, s: &Mix, f: f64, cert: &Certificate) -> Result<Self, Error> {
        let c = certify(&full_matrix(s, f), &cert.drop, &cert.check, cert.rung);
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
            value,
            drop,
            check,
            gap: (upper - lower).max(0.0),
        })
    }
}

const _: () = assert!(
    cfg!(target_endian = "little"),
    "the table is read in place as little-endian f64"
);

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
        let values = table.values();
        for c in strided(rechecks) {
            recheck(c, |i| values[i], &table.profiles)?;
        }
        Ok(table)
    }

    /// The whole table in row-major order: `values()[pc * (N + 1) + pd]`, with column `N` the
    /// "mover wins" sentinel.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        // SAFETY: the map is page-aligned and the body starts at byte 128, so it is f64-aligned;
        // `open` checked its length; every bit pattern is an f64; the header declares `<f8` and
        // the target is little-endian.
        #[allow(unsafe_code)]
        let (head, body, tail) = unsafe { self.map[Self::DATA..].align_to::<f64>() };
        assert!(
            head.is_empty() && tail.is_empty(),
            "the table body is f64-aligned"
        );
        body
    }

    /// The certified value of a state, from the current Dropper's side.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] or [`Error::OffDomain`].
    pub fn value(&self, state: State) -> Result<f64, Error> {
        let (pc, pd) = self.profiles.classes(state)?;
        Ok(self.values()[pc * (N + 1) + pd])
    }

    /// Rebuild a state's stage game from its stored children, solve it, and check the certificate
    /// and the stored value against the full matrix.
    ///
    /// # Errors
    ///
    /// A domain error, [`Error::Unsolved`], [`Error::Uncertified`], [`Error::StageGap`] or
    /// [`Error::OutsideCertificate`].
    pub fn stage(&self, state: State) -> Result<Stage, Error> {
        let values = self.values();
        let (pc, pd) = self.profiles.classes(state)?;
        let (s, f) = self.profiles.class_values(pc, pd, |i| values[i])?;
        let cert = solve_class(&s, f).ok_or(Error::Uncertified(pc, pd))?;
        Stage::certify(values[pc * (N + 1) + pd], &s, f, &cert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LAGS, Rung};

    #[test]
    fn stage_certifies_the_solved_class() {
        let s: Mix = std::array::from_fn(|k| -0.9 + 1.7 * (k as f64 / 59.0).powi(2));
        let cert = solve_class(&s, 0.3).expect("certified");
        let stage = Stage::certify(cert.value(), &s, 0.3, &cert).expect("certified stage");
        assert_eq!(stage.value, cert.value());
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
            Stage::certify(0.0, &s, 0.0, &cert),
            Err(Error::StageGap(_))
        ));
    }

    #[test]
    fn stage_fails_closed_when_the_stored_value_leaves_its_certificate() {
        let cert = solve_class(&[0.25; LAGS], 0.25).expect("pure");
        assert!(matches!(
            Stage::certify(0.5, &[0.25; LAGS], 0.25, &cert),
            Err(Error::OutsideCertificate(..))
        ));
    }
}
