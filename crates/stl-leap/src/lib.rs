//! The certified leap table of Surpassing the Leader, the clock game of *Usogui*.
//!
//! # The game
//!
//! Hal and Baku alternate as Dropper and Checker. In each half-round the Dropper picks a second
//! `1..=60` to drop and the Checker picks a second to check, both in secret. A check at or after
//! the drop succeeds, and the Checker's vial gains the squandered time `check - drop + 1`; a
//! full vial (300 s) kills the Checker. A check before the drop fails: the Checker takes a dose
//! and survives with a fixed revival probability, then carries the dose as time-to-death (TTD).
//! The half-rounds run on a clock that starts at 8:12. The minute 8:59 holds a leap second, and
//! a half-round that starts in `[3540, 3600]` lasts 61 seconds, so Baku as Dropper may also drop
//! at second 61, which no check can catch.
//!
//! # The formulation
//!
//! A public state is the pair of player profiles `(checker, dropper)` at a table key: the half
//! `H1` or `H2` of a round that starts at a whole minute, or `REV`, the half that resumes after a
//! revival at an arbitrary clock. A profile is a vial content and a TTD; [`rules`] numbers the
//! 17,011 profiles. Each state is a zero-sum 60-by-60 matrix game (61 rows in a leap window)
//! whose entries are the negated values of the child states one half-round later. Children past
//! clock 3600 can no longer meet the leap second, so their values come from the leap-free DTH
//! table, which [`dth`] reads. The table value is from the Dropper's side: +1 for a Dropper win.
//!
//! # The solve
//!
//! [`reach`] marks the 9,453,333,117 states reachable from the opening. [`build`] then solves the
//! keys in decreasing clock order, so every child is already stored. For each state the stage
//! kernel in [`kernel`] tries, in order: pure-action bounds; the equalizer that the Toeplitz
//! structure of the matrix gives in closed form; and for the residue, the support of the previous
//! residue stage moved with its payoff kink ([`support`]), then a packing LP that starts from the
//! same recurrence ([`packing`]). A candidate counts only when both saddle bounds, checked against
//! every row and column, lie within `1e-6`. Stages that all rungs miss go to `HiGHS` ([`lp`]). The
//! [`store`] packs each finished key into a bitmap of reachable states and their values, and the
//! audit re-derives sampled stages with an independent LP before the report is written.
//! [`table`] opens a finished artifact and re-certifies single stages.
//!
//! The kernel's certificates assume round-to-nearest binary64 arithmetic with gradual underflow.
//! Rust code runs in that default floating-point environment (changing it is undefined behavior),
//! and aarch64 keeps flush-to-zero off by default, so the reference's runtime probes of the
//! environment have no counterpart here. Rust never contracts `a * b + c` into a fused
//! multiply-add on its own, so the kernel writes each fused operation as `mul_add`, and `min` and
//! `max` are IEEE `minNum` and `maxNum`, as in the reference.

pub mod build;
pub mod kernel;
pub mod lp;
mod npy;
pub mod packing;
pub mod reach;
pub mod rules;
pub mod store;
pub mod support;
pub mod table;

use std::path::PathBuf;

pub use table::{LeapTable, open_leap};

/// A failure to build, audit or read the leap table.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read or written.
    Io {
        /// The file.
        path: PathBuf,
        /// The operating-system error.
        source: std::io::Error,
    },
    /// A file does not have the layout or contents this crate requires.
    Format {
        /// The file.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// A caller passed an argument outside the game.
    Invalid(String),
    /// A stage has no certificate within the saddle gate.
    Uncertified(String),
    /// The DTH table failed to open or certify.
    Dth(stl_dth::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Format { path, reason } => write!(f, "{}: {reason}", path.display()),
            Self::Invalid(reason) | Self::Uncertified(reason) => f.write_str(reason),
            Self::Dth(error) => write!(f, "DTH table: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Dth(error) => Some(error),
            _ => None,
        }
    }
}

impl From<stl_dth::Error> for Error {
    fn from(error: stl_dth::Error) -> Self {
        Self::Dth(error)
    }
}
