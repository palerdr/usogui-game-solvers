//! Air Poker from Toshio Sako's *Usogui*, solved as a public-information robust game.
//!
//! # The game
//!
//! Two lower players (Team 0, Baku; Team 1, Lalo) each hold five metal cards with numbers:
//! Baku `25 26 36 39 45`, Lalo `8 15 44 47 63`. Each of five rounds starts with an ante of
//! 1 to 5 BIOS (air canisters that double as chips). Both players pick an unused metal card
//! in secret, then reveal. Each player's upper ally then builds a five-card poker hand from one
//! shared 52-card deck whose card values (ace 1 to king 13) sum to that number. The lower players
//! do not see the hands. One betting street follows (check, call, fold, raise up to half the
//! pot); at showdown the better poker hand wins the pot. If the two hands share an exact card
//! (calamity), the loser also destroys BIOS equal to its contribution. Used cards leave the deck.
//! After round 5 the player with more BIOS wins. [`engine`] holds the rules; [`cards`] holds
//! the card arithmetic and the exact poker evaluator.
//!
//! # The public-state reformulation
//!
//! The upper players are fixed chance: each draws uniformly from the legal hands of the current
//! deck. Once both numbers are revealed, the two lower players know the same things, so the game
//! is a public-information stochastic game. Each round is a matrix game over the remaining number
//! pairs (at most 5 x 5), a chance draw that the lower players see only as one of five result
//! classes (Team 0 wins, wins with calamity, Team 1 wins, wins with calamity, no judgement), and
//! a perfect-information betting street. A fold hides the class.
//!
//! # The kernel
//!
//! The hidden hands enter only through the result kernel: the probability of each class given
//! the public prefix (numbers and classes so far) and the next number pair. Round 1 is exact by
//! enumeration of every hand pair ([`exact`]). Later rounds are sampled ([`sampler`]): a dynamic
//! program over card values counts and unranks legal hands ([`dp`]), a counter-based generator
//! makes every trajectory reproducible, and [`table`] pools the counts by public prefix.
//! [`confidence`] turns counts into time-uniform confidence boxes that hold under any adaptive
//! allocation of samples.
//!
//! # The robust solve
//!
//! [`robust`] runs backward induction over public states. At every chance node nature picks a
//! kernel inside the entry's box: against Team 0 for the lower bound and for Team 0 for the upper
//! bound, so the true root value lies in the interval. Number choice is a matrix game solved
//! exactly ([`matrix`]); betting is a max-min over a DAG per chip configuration ([`betting`]).
//! [`selfplay`] plays the robust strategies through the engine, and [`allocate`] spends the next
//! sampling budget where the equilibria go.

// Library panics come only from `expect` on broken internal invariants (CLAUDE.md); bad input
// returns `Error`. A `# Panics` section on every such function would document impossibilities.
#![allow(clippy::missing_panics_doc)]

pub mod allocate;
pub mod betting;
pub mod cards;
pub mod confidence;
pub mod dp;
pub mod engine;
pub mod exact;
pub mod matrix;
pub mod robust;
pub mod sampler;
pub mod selfplay;
pub mod table;

use std::fmt;

/// Baku's metal cards (Team 0).
pub const TEAM_0_METAL_CARDS: [u8; 5] = [25, 26, 36, 39, 45];
/// Lalo's metal cards (Team 1).
pub const TEAM_1_METAL_CARDS: [u8; 5] = [8, 15, 44, 47, 63];
/// The ante of each round.
pub const ANTES: [u32; ROUNDS] = [1, 2, 3, 4, 5];
/// Rounds in a game.
pub const ROUNDS: usize = 5;

/// Result class: Team 0 wins the judgement.
pub const W0: usize = 0;
/// Result class: Team 0 wins with calamity.
pub const W0C: usize = 1;
/// Result class: Team 1 wins the judgement.
pub const W1: usize = 2;
/// Result class: Team 1 wins with calamity.
pub const W1C: usize = 3;
/// Result class: equal poker strength, unreachable with distinct metal cards.
pub const TIE: usize = 4;
/// Result class: both targets impossible, the round ends with no judgement.
pub const NOJ: usize = 5;
/// The six class names in index order.
pub const CLASS_NAMES: [&str; 6] = ["W0", "W0c", "W1", "W1c", "T", "N"];

/// Base-5 digit of a revealed class in a prefix code; the tie has none.
#[must_use]
pub const fn digit(class: usize) -> Option<usize> {
    match class {
        W0..=W1C => Some(class),
        NOJ => Some(4),
        _ => None,
    }
}

/// Bad input to the engine or the card parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Card text that is not one rank character and one suit character.
    Card(String),
    /// A starting configuration outside the rules.
    Setup(&'static str),
    /// A move or query that the current state does not allow.
    Illegal(&'static str),
    /// An upper policy that is not a probability distribution over legal hands.
    Policy(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Card(text) => write!(f, "invalid card {text:?}"),
            Self::Setup(why) => write!(f, "invalid setup: {why}"),
            Self::Illegal(why) => write!(f, "illegal: {why}"),
            Self::Policy(why) => write!(f, "invalid upper policy: {why}"),
        }
    }
}

impl std::error::Error for Error {}
