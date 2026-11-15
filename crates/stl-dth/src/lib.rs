//! The complete solution of pure Drop the Handkerchief (DTH), the duel inside Surpassing the
//! Leader from the manga *Usogui*.
//!
//! # The game
//!
//! Two players take turns as Dropper and Checker. Each turn both secretly pick a second in
//! 1..=60. When the Checker's second is at least the Dropper's, the check succeeds and the
//! Checker's vial gains the inclusive elapsed time `c − d + 1` of squandered time (ST). A vial that
//! reaches 300 seconds kills the Checker and the Dropper wins. Otherwise the check fails: the
//! Checker takes a lethal dose of `ST + 60` seconds and revives with the frozen probability
//! `0.95 · (1 − ST/240) · 0.75^(TTD/60)`, where TTD is the toxin time accrued by earlier
//! revivals. A revived Checker restarts at ST 0 with TTD grown by the dose. After every turn the
//! roles swap. A state is `(s_c, t_c, s_d, t_d)`, Checker first, and its value is the current
//! Dropper's expected payoff in `[−1, 1]`.
//!
//! # The state classes
//!
//! TTD matters only through the revival probability. A player who can no longer survive a
//! failed check never will again, so every dead TTD collapses to one sentinel per ST. The
//! 16,711 alive `(ST, TTD)` pairs and 300 dead sentinels form 17,011 profiles, and a class is a
//! (Checker profile, Dropper profile) pair: 289,374,121 classes in place of 5.27 billion raw
//! states.
//!
//! # The solve
//!
//! The potential `phi = ST + TTD` (301 for TTD once dead) of the two players grows on every
//! move, so solving the 1,201 layers of equal potential from the highest down reads only
//! finished children. Each class is a 60 × 60 matrix game with only 61 distinct payoffs. A
//! three-rung ladder solves it: a pure saddle test, an equalizer recurrence, and a linear
//! program. Every answer must pass one gate, a saddle gap of at most `1e-6` against the full
//! matrix, and the stored value is the certificate midpoint. The finished table `V.npy` holds
//! one `f64` per class plus a column that reads `-1` for "the mover wins". The recipe is
//! `docs/DTH.md` of the Python reference.

mod ladder;
mod profiles;
mod sweep;
mod table;

pub use ladder::{Certificate, Matrix, Mix, Rung, full_matrix, lp, pure, support};
pub use profiles::Profiles;
pub use sweep::{Values, build, finalize};
pub use table::{Stage, Table};

/// Largest squandered time a live vial holds.
pub(crate) const MAX_ST: usize = 299;
/// Largest accrued toxin time of the domain.
pub(crate) const MAX_TTD: usize = 300;
/// A successful check that grows the Checker's ST to this wins for the Dropper.
pub(crate) const CAPACITY: usize = 300;
/// The fixed penalty a failed check adds to the Checker's dose.
pub(crate) const DOSE: usize = 60;
/// Actions are the seconds 1..=LAGS.
pub const LAGS: usize = 60;
/// Alive profiles, ids `0..N_ALIVE` in the normative order.
pub const N_ALIVE: usize = 16_711;
/// Dead sentinels, id `N_ALIVE + st`.
pub(crate) const N_DEAD: usize = 300;
/// All profiles.
pub const N: usize = N_ALIVE + N_DEAD;
/// State classes: one per (Checker profile, Dropper profile).
pub(crate) const CLASSES: usize = N * N;
/// The child column that means "the mover wins": `V[pd, WIN] == -1`.
pub(crate) const WIN: usize = N;
/// The potential offset, and stored TTD, of a dead profile.
pub(crate) const DEAD_PHI: usize = 301;
/// Largest profile potential.
pub(crate) const MAX_PHI: usize = MAX_ST + DEAD_PHI;
/// Largest class potential.
pub(crate) const MAX_LAYER: usize = 2 * MAX_PHI;
/// The certificate gate: the largest accepted saddle gap.
pub const MAX_SADDLE_GAP: f64 = 1e-6;
/// Classes that `finalize` re-derives from their stored children.
pub const RECHECK_SAMPLES: usize = 1200;

/// A role-relative state `(s_c, t_c, s_d, t_d)`: Checker first.
pub type State = [usize; 4];

/// Six certified reference values (DTH.md section 6); the first is the root of the game.
pub const ANCHORS: [(State, f64); 6] = [
    ([0, 0, 0, 0], 0.089_850_072_809_510_46),
    ([240, 0, 240, 0], 0.337_213_216_629_109_3),
    ([10, 60, 200, 0], -0.794_442_891_646_929_7),
    ([150, 90, 30, 120], 0.724_409_303_635_678_5),
    ([250, 300, 40, 0], 0.998_115_281_738_196_9),
    ([100, 140, 100, 140], 0.187_738_637_827_619_3),
];

/// Bad input or a bad artifact.
#[derive(Debug)]
pub enum Error {
    /// Reading or writing a table file failed.
    Io(std::io::Error),
    /// `(st, ttd)` lies outside ST 0..=299 or TTD 0..=300.
    OutOfRange(usize, usize),
    /// `(st, ttd)` is alive with TTD in 1..=59, which no play reaches.
    OffDomain(usize, usize),
    /// The file at this path is not a float64 `(N, N + 1)` table.
    Format(String),
    /// Anchor states whose stored values differ from the certified ones.
    Anchors(Vec<State>),
    /// Class `(pc, pd)` reads an unsolved child, or stores a value outside `[-1, 1]`.
    Unsolved(usize, usize),
    /// No rung certifies class `(pc, pd)`.
    Uncertified(usize, usize),
    /// The stored value of flat class `pc * N + pd` differs from its re-derivation by `diff`.
    Recheck(usize, f64),
    /// A stage certificate is wider than the gate.
    StageGap(f64),
    /// The stored value lies outside its stage certificate `[lower, upper]`.
    OutsideCertificate(f64, f64, f64),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::OutOfRange(st, ttd) => {
                write!(
                    f,
                    "({st}, {ttd}) lies outside ST 0..={MAX_ST} and TTD 0..={MAX_TTD}"
                )
            }
            Self::OffDomain(st, ttd) => {
                write!(f, "({st}, {ttd}): alive TTD in 1..59 is off-domain")
            }
            Self::Format(path) => write!(f, "{path}: not a float64 ({N}, {}) table", N + 1),
            Self::Anchors(states) => write!(
                f,
                "the anchors {states:?} differ from their certified values"
            ),
            Self::Unsolved(pc, pd) => {
                write!(f, "class ({pc}, {pd}) reads or stores an unsolved value")
            }
            Self::Uncertified(pc, pd) => write!(f, "no rung certifies class ({pc}, {pd})"),
            Self::Recheck(class, diff) => write!(f, "class {class} fails its recheck by {diff:e}"),
            Self::StageGap(gap) => write!(
                f,
                "the stage has saddle gap {gap:e} above {MAX_SADDLE_GAP:e}"
            ),
            Self::OutsideCertificate(value, lower, upper) => {
                write!(
                    f,
                    "the stored value {value} lies outside its certificate [{lower}, {upper}]"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// `n` class indices evenly strided over `0..CLASSES`, first and last included.
fn strided(n: usize) -> impl Iterator<Item = usize> {
    (0..n).map(move |i| i * (CLASSES - 1) / (n - 1).max(1))
}

/// Re-derive one class from its stored children and return `|re-derived − stored|` (DTH.md
/// 4.10). `value` reads the flat table `V[row * (N + 1) + col]`.
///
/// # Errors
///
/// [`Error::Unsolved`], [`Error::Uncertified`], or [`Error::Recheck`] when the difference exceeds
/// the gate.
pub fn recheck(
    class: usize,
    value: impl Fn(usize) -> f64,
    profiles: &Profiles,
) -> Result<f64, Error> {
    let (pc, pd) = (class / N, class % N);
    let diff = (profiles.solve(pc, pd, &value)?.value() - value(pc * (N + 1) + pd)).abs();
    if diff <= MAX_SADDLE_GAP {
        Ok(diff)
    } else {
        Err(Error::Recheck(class, diff))
    }
}
