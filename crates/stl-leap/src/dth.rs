//! The leap-free DTH value table, the exact tail of every state after clock 3600.
//!
//! This is the minimal reader the leap solver needs; the `stl-dth` crate owns the table itself.
//! `V[checker, dropper]` is the Dropper's value of the DTH state, with the leap profile numbering
//! of [`crate::rules`], and column `WIN` holds `-1`.

use crate::Error;
use crate::npy::Npy;
use crate::rules::{N, profile_index};
use std::path::Path;

/// The six certified DTH values of DTH.md section 6, as `[checker st, ttd, dropper st, ttd]`.
/// TTD 301 names the profile that a failed check kills.
const ANCHORS: [([u16; 4], f64); 6] = [
    ([0, 0, 0, 0], 0.089_850_072_809_510_46),
    ([240, 301, 240, 301], 0.337_213_216_629_109_3),
    ([10, 60, 200, 0], -0.794_442_891_646_929_7),
    ([150, 90, 30, 120], 0.724_409_303_635_678_5),
    ([250, 301, 40, 0], 0.998_115_281_738_196_9),
    ([100, 140, 100, 140], 0.187_738_637_827_619_3),
];

/// A memory-mapped DTH table `V.npy` of shape `(17011, 17012)`.
pub struct Dth(Npy);

impl Dth {
    /// Open `V.npy` and check its shape, its type and the six certified values.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] when the file cannot be mapped and [`Error::Format`] when its type, shape or
    /// anchors differ.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let table = Npy::open::<f64>(path).map_err(|error| match error {
            Error::Format { path, reason } => Error::Format {
                path,
                reason: format!("{reason}; is not float64 (17011, 17012)"),
            },
            other => other,
        })?;
        if table.shape != [N, N + 1] {
            let reason = format!("shape {:?} is not float64 (17011, 17012)", table.shape);
            return Err(Error::Format {
                path: path.to_path_buf(),
                reason,
            });
        }
        let dth = Self(table);
        for ([cs, ct, ds, dt], expected) in ANCHORS {
            let checker = profile_index(cs, ct)?;
            let got = dth.values()[checker * (N + 1) + profile_index(ds, dt)?];
            if (got - expected).abs() > 1e-6 || !got.is_finite() {
                let reason = format!("anchor ({cs}, {ct}, {ds}, {dt}) is {got}, not {expected}");
                return Err(Error::Format {
                    path: path.to_path_buf(),
                    reason,
                });
            }
        }
        Ok(dth)
    }

    /// Return the table in row-major order: `values()[checker * 17012 + dropper]`.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        self.0.data()
    }
}
