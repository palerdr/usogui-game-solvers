//! Tower of Karma, the game `D_K` from *Usogui*: rules, solvers, certificate and an abstraction
//! study.
//!
//! # The game
//!
//! Each player secretly picks 1 to 10 beads. P0 picks first and P1 does not see the pick. Then the
//! players take turns to announce the total, P0 first, for K turns each. An announcement must lie
//! between own beads + 1 and own beads + 10. The first player who names the true total wins 1 and
//! the other loses 1. When all 2K announcements miss, the game is a draw. The game is zero-sum
//! and has no chance moves, so it has one value: P0's equilibrium payoff.
//!
//! # The public tree
//!
//! The full game tree of `D_3` has 66,430,111 nodes. [`tree`] stores it as arrays instead. A public
//! node is the ordered transcript of misses. It keeps the bead pairs (x0, x1) that are still alive
//! there: every announcement so far was legal for its actor and missed. Layer d holds the
//! decisions of turn d / 2 + 1; P0 acts on even layers and P1 on odd layers. A slot is a pair
//! (public node, own beads) of one player, and the actor's slots are exactly its information
//! sets. One alive pair is one decision node of the full tree. `D_3` has 937,252 public nodes and
//! 3,175,048 information sets.
//!
//! # The solve
//!
//! - [`lp`] writes the sequence-form linear program on the arrays and solves it with `HiGHS`. P1's
//!   strategy comes back as the row duals.
//! - [`cfr`] runs CFR+ with full traversals of the arrays: [`traversal`] passes reach
//!   probabilities down and counterfactual values up. [`traversal::saddle_gap`] bounds the game
//!   value between the two best-response values of the average strategy.
//! - [`certificate`] proves those bounds again through the rules engine [`rules`] alone, reading
//!   each strategy from a [`table::StrategyTable`] by information state. It never reads the
//!   arrays, so a bug in [`tree`] cannot certify itself.
//! - [`abstraction`] asks how much a player loses when its last decisions are keyed by coarser
//!   state than the full information set.

pub mod abstraction;
pub mod certificate;
pub mod cfr;
pub mod lp;
pub mod rules;
pub mod table;
pub mod traversal;
pub mod tree;

/// Bead counts run 1..=10, and every decision has 10 actions.
pub const HANDS: usize = 10;
/// Announced totals run 2..=20.
pub const TOTALS: usize = 19;

/// Bad input or a bad artifact.
#[derive(Debug)]
pub enum Error {
    /// The horizon K must be positive.
    Horizon,
    /// The action is not legal in the state.
    IllegalAction(rules::Action),
    /// A file could not be read or written.
    Io(std::io::Error),
    /// An artifact file is malformed or does not fit the game.
    Artifact(String),
    /// `HiGHS` did not reach an optimum.
    Lp(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Horizon => write!(f, "the horizon must be a positive integer"),
            Self::IllegalAction(action) => write!(f, "illegal action {action:?}"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Artifact(message) => write!(f, "bad artifact: {message}"),
            Self::Lp(status) => write!(f, "LP not solved: {status}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
