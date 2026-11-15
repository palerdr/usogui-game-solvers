//! Self-play through the engine with the robust strategies.
//!
//! The solve keeps values and number strategies for number prefixes of up to two rounds; deeper
//! decisions are recomputed on demand from the kernel boxes and cached per number prefix.
//! Chance comes from the kernel sampler and every move goes through the engine. Number choices
//! draw from a counter-based stream of their own, so a pure strategy plays as in the Python
//! reference while a mixed one draws differently from the `NumPy` generator.

use std::cell::RefCell;
use std::rc::Rc;

use rayon::prelude::*;

use crate::engine::{Bet, GameState, Move, Phase, Record, Team};
use crate::robust::{Chance, Kept, Solver, Stage, Values};
use crate::sampler::{next64, play_round, stream};
use crate::table::HIDDEN;
use crate::{ANTES, ROUNDS};

/// Cached stages per self-play worker; a round-5 table alone is 46 MB.
const CACHE: usize = 4;

type Cache<T> = RefCell<Vec<(Vec<(u8, u8)>, Rc<T>)>>;

/// The base-6 digit of a completed round as the lower players see it.
#[must_use]
pub fn record_digit(record: &Record) -> usize {
    if matches!(record.actions.last(), Some((_, Bet::Fold))) {
        return HIDDEN;
    }
    match (record.winner, record.calamity) {
        (None, _) => 4,
        (Some(team), calamity) => 2 * team.index() + usize::from(calamity),
    }
}

/// The engine bet of DAG action `slot`: check or fold, then call when facing a wager, then
/// raises by 1, 2, ...
#[must_use]
pub fn slot_action(slot: usize, to_call: u32) -> Bet {
    match (slot, to_call) {
        (0, 0) => Bet::Check,
        (0, _) => Bet::Fold,
        (1, 1..) => Bet::Call,
        (slot, 0) => Bet::Raise(slot as u32),
        (slot, _) => Bet::Raise(slot as u32 - 1),
    }
}

/// The value at `key` in a small least-recently-used cache, computed on a miss.
fn cached<T>(store: &Cache<T>, key: &[(u8, u8)], compute: impl FnOnce() -> T) -> Rc<T> {
    let mut store = store.borrow_mut();
    let item = if let Some(position) = store.iter().position(|(k, _)| k == key) {
        store.remove(position)
    } else {
        if store.len() >= CACHE {
            store.remove(0);
        }
        (key.to_vec(), Rc::new(compute()))
    };
    let value = Rc::clone(&item.1);
    store.push(item);
    value
}

/// Engine moves from one robust equilibrium: `use_hi` picks the upper-bound game.
pub struct Policy<'a> {
    solver: &'a Solver,
    kept: &'a Kept,
    use_hi: bool,
    subtrees: Cache<Stage>,
    round_five: Cache<Values>,
}

impl<'a> Policy<'a> {
    /// A policy over a solve's kept stages, which must cover number prefixes of two rounds.
    #[must_use]
    pub fn new(solver: &'a Solver, kept: &'a Kept, use_hi: bool) -> Self {
        Self {
            solver,
            kept,
            use_hi,
            subtrees: RefCell::default(),
            round_five: RefCell::default(),
        }
    }

    /// The round-4 stage after a number prefix of three rounds.
    fn subtree(&self, prefix: &[(u8, u8)]) -> Rc<Stage> {
        cached(&self.subtrees, prefix, || {
            self.solver.v(prefix, 3, &mut Kept::default())
        })
    }

    /// The round-5 values after a number prefix of four rounds.
    fn round_five(&self, prefix: &[(u8, u8)]) -> Rc<Values> {
        cached(&self.round_five, prefix, || {
            self.solver.v(prefix, 4, &mut Kept::default()).values
        })
    }

    /// The metal card to play, drawn from the equilibrium mix.
    fn number(&self, state: &GameState, rng: &mut u64) -> u8 {
        let (r, prefix, code) = public(state);
        let team = state.actor().expect("a choice phase has an actor").index();
        if r == ROUNDS - 1 {
            return state.lower[team].metal_cards[0];
        }
        let holder;
        let stage = if r <= self.kept.depth {
            &self.kept.stages[&prefix]
        } else {
            holder = self.subtree(&prefix);
            &*holder
        };
        let chips = (
            state.lower[0].sealed_bios as usize,
            state.lower[1].sealed_bios as usize,
        );
        let n = stage
            .chips
            .iter()
            .position(|&c| c == chips)
            .expect("the engine's chips are reachable");
        let mix = &stage.mixes[code * stage.chips.len() + n];
        let (p, options) = match (team, self.use_hi) {
            (0, false) => (&mix.x_lo, &stage.rows),
            (0, true) => (&mix.x_hi, &stage.rows),
            (_, false) => (&mix.y_lo, &stage.cols),
            (_, true) => (&mix.y_hi, &stage.cols),
        };
        let p = &p[..options.len()];
        let total: f64 = p.iter().sum();
        let u = (next64(rng) >> 11) as f64 / (1u64 << 53) as f64;
        let mut cumulative = 0.0;
        let pick = p.iter().position(|&q| {
            cumulative += q / total;
            u < cumulative
        });
        options[pick.unwrap_or(options.len() - 1)]
    }

    /// The robust best bet at the current betting node.
    fn bet(&self, state: &GameState) -> Bet {
        let (r, mut prefix, code) = public(state);
        let betting = state
            .round
            .betting
            .as_ref()
            .expect("the betting phase has a betting state");
        let [c0, c1] = betting.contributions;
        let ante = ANTES[r];
        let chips = (
            (state.lower[0].sealed_bios + c0 - ante) as usize,
            (state.lower[1].sealed_bios + c1 - ante) as usize,
        );
        let [a, b] = state
            .round
            .chosen
            .map(|card| card.expect("betting follows both choices"));
        prefix.push((a, b));
        let entry = self.solver.sequences.prefix_index[r][&prefix] * 6usize.pow(r as u32) + code;
        let (holder_stage, holder_values);
        let next = match r + 1 {
            ROUNDS => None,
            n if n <= self.kept.depth => Some(&self.kept.stages[&prefix].values),
            3 => {
                holder_stage = self.subtree(&prefix);
                Some(&holder_stage.values)
            }
            _ => {
                holder_values = self.round_five(&prefix);
                Some(&*holder_values)
            }
        };
        let chance = Chance {
            r,
            first: betting.first.index(),
            code,
            lo: &self.solver.box_lo[r][entry],
            hi: &self.solver.box_hi[r][entry],
            next,
        };
        let actor = betting.actor.index();
        let slot = self.solver.arena.betting_slot(
            &chance,
            chips,
            (c0, c1, actor, betting.checked),
            self.use_hi,
        );
        slot_action(
            slot,
            betting.contributions[1 - actor].saturating_sub(betting.contributions[actor]),
        )
    }
}

/// `(round, number prefix, class code)` of the completed rounds.
fn public(state: &GameState) -> (usize, Vec<(u8, u8)>, usize) {
    let prefix = state
        .history
        .iter()
        .map(|record| (record.chosen[0], record.chosen[1]))
        .collect();
    let code = state
        .history
        .iter()
        .fold(0, |code, record| code * 6 + record_digit(record));
    (state.history.len(), prefix, code)
}

/// One game through the engine: Team 0's utility and the number pairs played.
///
/// Chance of draw `d` uses stream `(seed, 0, 16 * index + d)`; number choices use stream
/// `(seed, 1, index)`.
#[must_use]
pub fn play_game(policies: [&Policy<'_>; 2], seed: u64, index: u64) -> (f64, Vec<(u8, u8)>) {
    let mut rng = stream(seed, 1, index);
    let mut state = GameState::canonical();
    let mut draw = 0;
    while state.phase != Phase::Terminal {
        let mv = match state.phase {
            Phase::NumberChoice0 => Move::Number(policies[0].number(&state, &mut rng)),
            Phase::NumberChoice1 => Move::Number(policies[1].number(&state, &mut rng)),
            Phase::UpperChance => {
                let [a, b] = state
                    .round
                    .chosen
                    .map(|card| card.expect("chance follows both choices"));
                let (_, _, hands) =
                    play_round(state.deck, (a, b), &mut stream(seed, 0, index * 16 + draw));
                draw += 1;
                Move::Hands(hands)
            }
            Phase::Betting | Phase::Terminal => {
                let actor = state.actor().expect("the betting phase has an actor");
                Move::Bet(policies[actor.index()].bet(&state))
            }
        };
        state = state
            .apply(mv)
            .expect("self-play moves are legal engine moves");
    }
    let (_, played, _) = public(&state);
    (
        state.returns().expect("the game ended")[Team::Zero.index()],
        played,
    )
}

/// Reach of each number sequence over `games` games under each robust equilibrium, and the mean
/// return of Team 0 under the lower and the upper equilibrium.
///
/// A game that ends early played a number prefix; its visit is shared equally among the
/// sequences that extend that prefix.
#[must_use]
pub fn sequence_reach(solver: &Solver, kept: &Kept, games: u64, seed: u64) -> (Vec<f64>, [f64; 2]) {
    let sequences = &solver.sequences;
    let mut reach = vec![0.0; sequences.targets.len()];
    let mut returns = [0.0; 2];
    for (offset, use_hi) in [false, true].into_iter().enumerate() {
        // Games run in parallel with a policy cache per worker; the visits are added in game
        // order, so the sums do not depend on the schedule.
        let results: Vec<_> = (0..games)
            .into_par_iter()
            .map_init(
                || Policy::new(solver, kept, use_hi),
                |policy, index| play_game([&*policy, &*policy], seed + offset as u64, index),
            )
            .collect();
        let mut total = 0.0;
        for (utility, played) in results {
            total += utility;
            let r = played.len() - 1;
            let key = sequences.prefix_index[r][&played];
            let members: Vec<usize> = (0..reach.len())
                .filter(|&s| sequences.prefix_of_sequence[r][s] == key)
                .collect();
            for &s in &members {
                reach[s] += 1.0 / members.len() as f64;
            }
        }
        returns[offset] = total / games as f64;
    }
    (reach, returns)
}
