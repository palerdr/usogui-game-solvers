//! Robust backward induction over public states.
//!
//! A public round-start state is the round `r`, the number prefix (pairs played), the base-6
//! class prefix code and the sealed BIOS `(s0, s1)` after the ante; the first bettor follows
//! from the round-1 pair and alternates.
//!
//! ```text
//! V_r(P, C, s) = matrix game over the remaining pairs (a, b) of W_r(P + (a, b), C, s)
//! W_r(P', C, s) = t * value after no judgement + (1 - t) * betting value,  t = P(no judgement)
//! ```
//!
//! The betting value is the perfect-information value of the configuration's betting DAG. A
//! showdown pays the expectation over the four win classes of the next round's value or the
//! terminal utility; a fold continues with the class hidden. Nature picks inside the entry's
//! confidence box at every chance node independently: against Team 0 for the lower bound and
//! for it for the upper bound. The true kernel lies in every box, so the true value lies in the
//! interval. Rounds are solved depth first over number prefixes, so live memory is one value
//! table per level; tables up to a chosen depth are kept for self-play.

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::betting::{
    BettingDag, CHIPS, End, ROOT_CHIPS, Settled, reachable_configurations, settle,
};
use crate::confidence::entry_box;
use crate::matrix::{MAX_SIZE, Matrix, solve};
use crate::sampler::Counts;
use crate::table::{ENTRIES, HIDDEN, KernelCounts, Sequences};
use crate::{ANTES, NOJ, ROUNDS, TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS, W0, W0C, W1, W1C, digit};

const WIN: [usize; 4] = [W0, W0C, W1, W1C];

/// The index of `(code, s0, s1)` in a dense value table.
#[must_use]
pub const fn at(code: usize, s0: usize, s1: usize) -> usize {
    (code * CHIPS + s0) * CHIPS + s1
}

/// Every reachable betting DAG.
pub struct Arena {
    /// `chips[r][first]`: the reachable sealed BIOS `(s0, s1)` at the start of round `r` with
    /// this first bettor.
    pub chips: [[Vec<(usize, usize)>; 2]; ROUNDS],
    config: Vec<usize>,
    dags: Vec<BettingDag>,
}

impl Arena {
    /// Build the DAG of every reachable configuration.
    #[must_use]
    pub fn new() -> Self {
        let reach = reachable_configurations();
        let chips: [[Vec<_>; 2]; ROUNDS] = std::array::from_fn(|r| {
            std::array::from_fn(|first| {
                reach[r]
                    .iter()
                    .filter(|c| c.2 == first)
                    .map(|&(s0, s1, _)| (s0, s1))
                    .collect()
            })
        });
        let mut config = vec![usize::MAX; ROUNDS * CHIPS * CHIPS * 2];
        let mut jobs = Vec::new();
        for (r, by_first) in chips.iter().enumerate() {
            for (first, list) in by_first.iter().enumerate() {
                for &(s0, s1) in list {
                    config[((r * CHIPS + s0) * CHIPS + s1) * 2 + first] = jobs.len();
                    jobs.push((r, s0, s1, first));
                }
            }
        }
        let dags = jobs
            .par_iter()
            .map(|&(r, s0, s1, first)| BettingDag::new(s0 as u32, s1 as u32, ANTES[r], first))
            .collect();
        Self {
            chips,
            config,
            dags,
        }
    }

    /// The DAG of round `r` with sealed BIOS `(s0, s1)` and this first bettor.
    fn dag(&self, r: usize, s0: usize, s1: usize, first: usize) -> &BettingDag {
        &self.dags[self.config[((r * CHIPS + s0) * CHIPS + s1) * 2 + first]]
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower and upper values over `(class prefix code, s0, s1)`; `NaN` where unreachable.
#[derive(Clone, Debug)]
pub struct Values {
    /// Lower bounds, indexed by [`at`].
    pub lo: Vec<f64>,
    /// Upper bounds, indexed by [`at`].
    pub hi: Vec<f64>,
}

impl Values {
    fn new(codes: usize) -> Self {
        Self {
            lo: vec![f64::NAN; codes * CHIPS * CHIPS],
            hi: vec![f64::NAN; codes * CHIPS * CHIPS],
        }
    }
}

/// Number strategies of both players in the lower-bound and the upper-bound games.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mixes {
    /// Team 0's mix over the remaining rows in the lower-bound game.
    pub x_lo: [f64; MAX_SIZE],
    /// Team 1's mix over the remaining columns in the lower-bound game.
    pub y_lo: [f64; MAX_SIZE],
    /// Team 0's mix in the upper-bound game.
    pub x_hi: [f64; MAX_SIZE],
    /// Team 1's mix in the upper-bound game.
    pub y_hi: [f64; MAX_SIZE],
}

/// The round-start values and number strategies of one number prefix.
#[derive(Clone, Debug)]
pub struct Stage {
    /// `V_r` over class prefix codes and chips.
    pub values: Values,
    /// Team 0's remaining metal cards (matrix rows).
    pub rows: Vec<u8>,
    /// Team 1's remaining metal cards (matrix columns).
    pub cols: Vec<u8>,
    /// The reachable chips of this round, in strategy order.
    pub chips: Vec<(usize, usize)>,
    /// `mixes[code * chips.len() + n]`: strategies at class code `code` and chips `chips[n]`.
    pub mixes: Vec<Mixes>,
    /// The largest saddle gap of any matrix game in this subtree.
    pub gap: f64,
}

/// Stages kept for self-play: every number prefix of length up to `depth`.
#[derive(Default)]
pub struct Kept {
    /// The deepest round whose stages are kept.
    pub depth: usize,
    /// Stages by number prefix.
    pub stages: HashMap<Vec<(u8, u8)>, Stage>,
}

/// The result of a full solve.
pub struct Solution {
    /// Lower bound of Team 0's root value.
    pub root_lo: f64,
    /// Upper bound of Team 0's root value.
    pub root_hi: f64,
    /// The largest saddle gap of any matrix game.
    pub max_gap: f64,
    /// Kept stages, the root under the empty prefix.
    pub kept: Kept,
    /// Wall time of the solve.
    pub seconds: f64,
}

/// The chance node of one round entry: its box and the next round's values.
pub struct Chance<'a> {
    /// The round, 0-based.
    pub r: usize,
    /// The first bettor.
    pub first: usize,
    /// The class prefix code before this round.
    pub code: usize,
    /// Lower ends of the entry's kernel box.
    pub lo: &'a [f64; 6],
    /// Upper ends of the entry's kernel box.
    pub hi: &'a [f64; 6],
    /// The next round's values; `None` in the last round.
    pub next: Option<&'a Values>,
}

/// The extreme of `sum q[c] u[c]` over the box `[q_lo, q_hi]` intersected with the simplex.
///
/// Mass starts at the lower ends; the rest goes to the classes in payoff order, worst first
/// when minimizing, up to their upper ends.
fn robust_expectation(u: &[f64; 4], q_lo: &[f64; 4], q_hi: &[f64; 4], minimize: bool) -> f64 {
    let mut remaining = 1.0;
    let mut total = 0.0;
    for c in 0..4 {
        remaining -= q_lo[c];
        total += q_lo[c] * u[c];
    }
    let mut open = *q_hi;
    for _ in 0..4 {
        if remaining <= 0.0 {
            break;
        }
        let pick = (0..4).filter(|&c| open[c] > q_lo[c]).reduce(|b, c| {
            if (minimize && u[c] < u[b]) || (!minimize && u[c] > u[b]) {
                c
            } else {
                b
            }
        });
        let Some(c) = pick else { break };
        let extra = (open[c] - q_lo[c]).min(remaining);
        total += extra * u[c];
        remaining -= extra;
        open[c] = q_lo[c];
    }
    total
}

/// The pairs `(rows, cols)` of metal cards neither team has played in `prefix`.
fn remaining(prefix: &[(u8, u8)]) -> (Vec<u8>, Vec<u8>) {
    let rows = TEAM_0_METAL_CARDS
        .into_iter()
        .filter(|&a| prefix.iter().all(|p| p.0 != a))
        .collect();
    let cols = TEAM_1_METAL_CARDS
        .into_iter()
        .filter(|&b| prefix.iter().all(|p| p.1 != b))
        .collect();
    (rows, cols)
}

/// The full-simplex box of an entry with no samples.
const TRIVIAL: ([f64; 6], [f64; 6]) = ([0.0; 6], [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]);

/// The robust solver: arena, sequences and kernel boxes per round.
pub struct Solver {
    /// The number sequences and prefix indices.
    pub sequences: Sequences,
    /// Every betting DAG.
    pub arena: Arena,
    /// `box_lo[r][prefix * 6^r + code]`: lower ends of the kernel box of an entry.
    pub box_lo: Vec<Vec<[f64; 6]>>,
    /// Upper ends, laid out as `box_lo`.
    pub box_hi: Vec<Vec<[f64; 6]>>,
}

impl Solver {
    /// A solver whose round-1 kernel is the exact one and whose later boxes are the whole simplex.
    #[must_use]
    pub fn new(round_one: &[((u8, u8), Counts)]) -> Self {
        let sequences = Sequences::new();
        let mut exact = vec![[0.0; 6]; 25];
        for &(pair, counts) in round_one {
            let total = counts.iter().sum::<u64>() as f64;
            exact[sequences.prefix_index[0][&vec![pair]]] = counts.map(|c| c as f64 / total);
        }
        let mut box_lo = vec![exact.clone()];
        let mut box_hi = vec![exact];
        for r in 1..ROUNDS {
            let entries = sequences.prefix_index[r].len() * 6usize.pow(r as u32);
            box_lo.push(vec![TRIVIAL.0; entries]);
            box_hi.push(vec![TRIVIAL.1; entries]);
        }
        Self {
            sequences,
            arena: Arena::new(),
            box_lo,
            box_hi,
        }
    }

    /// Kernel boxes for rounds 2 to 5 from sampled counts, all entries jointly at level `delta`
    /// by a union bound.
    pub fn set_boxes(&mut self, counts: &KernelCounts, delta: f64) {
        for r in 1..ROUNDS {
            let codes = 6usize.pow(r as u32);
            self.box_lo[r]
                .par_chunks_mut(codes)
                .zip(self.box_hi[r].par_chunks_mut(codes))
                .enumerate()
                .for_each(|(prefix, (lo, hi))| {
                    for (code, counts) in counts.pooled(r, prefix).iter().enumerate() {
                        (lo[code], hi[code]) = entry_box(counts, delta / ENTRIES as f64);
                    }
                });
        }
    }

    /// The first bettor of round `r` after a number prefix that starts with the round-1 pair.
    #[must_use]
    pub fn first(prefix: &[(u8, u8)], r: usize) -> usize {
        usize::from(prefix[0].0 > prefix[0].1) ^ (r & 1)
    }

    /// Solve the whole game, keeping stages of number prefixes up to length `keep_depth`.
    #[must_use]
    pub fn solve(&self, keep_depth: usize) -> Solution {
        let start = Instant::now();
        let mut kept = Kept {
            depth: keep_depth,
            stages: HashMap::new(),
        };
        let root = self.v(&[], 0, &mut kept);
        let i = at(0, ROOT_CHIPS.0, ROOT_CHIPS.1);
        let (root_lo, root_hi, max_gap) = (root.values.lo[i], root.values.hi[i], root.gap);
        kept.stages.insert(Vec::new(), root);
        Solution {
            root_lo,
            root_hi,
            max_gap,
            kept,
            seconds: start.elapsed().as_secs_f64(),
        }
    }

    /// `V_r` and the number strategies after the number prefix of length `r`.
    #[must_use]
    pub fn v(&self, prefix: &[(u8, u8)], r: usize, kept: &mut Kept) -> Stage {
        let (rows, cols) = remaining(prefix);
        let k = rows.len();
        let mut gap = 0.0_f64;
        let mut w = Vec::with_capacity(k * k);
        for &a in &rows {
            for &b in &cols {
                let (values, g) = self.w(&[prefix, &[(a, b)]].concat(), r, kept);
                gap = gap.max(g);
                w.push(values);
            }
        }
        let chips = if prefix.is_empty() {
            vec![ROOT_CHIPS]
        } else {
            self.arena.chips[r][Self::first(prefix, r)].clone()
        };
        let mut values = Values::new(6usize.pow(r as u32));
        let mut mixes = vec![Mixes::default(); values.lo.len() / (CHIPS * CHIPS) * chips.len()];
        let gaps: Vec<f64> = values
            .lo
            .par_chunks_mut(CHIPS * CHIPS)
            .zip(values.hi.par_chunks_mut(CHIPS * CHIPS))
            .zip(mixes.par_chunks_mut(chips.len()))
            .enumerate()
            .map(|(code, ((lo, hi), mixes))| {
                let mut worst = 0.0_f64;
                for (mix, &(s0, s1)) in mixes.iter_mut().zip(&chips) {
                    let (i, j) = (at(code, s0, s1), s0 * CHIPS + s1);
                    if k == 1 {
                        (lo[j], hi[j]) = (w[0].lo[i], w[0].hi[i]);
                        let pure = std::array::from_fn(|n| f64::from(u8::from(n == 0)));
                        *mix = Mixes {
                            x_lo: pure,
                            y_lo: pure,
                            x_hi: pure,
                            y_hi: pure,
                        };
                        continue;
                    }
                    let game = |upper: bool| {
                        let mut m: Matrix = [[0.0; MAX_SIZE]; MAX_SIZE];
                        for (a, row) in m.iter_mut().enumerate().take(k) {
                            for (b, cell) in row.iter_mut().enumerate().take(k) {
                                let table = &w[a * k + b];
                                *cell = if upper { table.hi[i] } else { table.lo[i] };
                            }
                        }
                        solve(&m, k)
                    };
                    let (low, high) = (game(false), game(true));
                    (lo[j], hi[j]) = (low.low, high.high);
                    worst = worst.max(low.high - low.low).max(high.high - high.low);
                    *mix = Mixes {
                        x_lo: low.x,
                        y_lo: low.y,
                        x_hi: high.x,
                        y_hi: high.y,
                    };
                }
                worst
            })
            .collect();
        Stage {
            values,
            rows,
            cols,
            chips,
            mixes,
            gap: gaps.into_iter().fold(gap, f64::max),
        }
    }

    /// `W_r` over class codes and chips for the number prefix of length `r + 1`, with the
    /// largest saddle gap below it.
    #[must_use]
    pub fn w(&self, prefix: &[(u8, u8)], r: usize, kept: &mut Kept) -> (Values, f64) {
        let first = Self::first(prefix, r);
        let next = (r + 1 < ROUNDS).then(|| self.v(prefix, r + 1, kept));
        let codes = 6usize.pow(r as u32);
        let base = self.sequences.prefix_index[r][prefix] * codes;
        // In the last round W depends on the entry's box alone, so codes with equal boxes share
        // the row of the first such code.
        let mut first_with_box = HashMap::new();
        let source: Vec<usize> = (0..codes)
            .map(|code| {
                let bits = |b: &[f64; 6]| b.map(f64::to_bits);
                let key = (
                    bits(&self.box_lo[r][base + code]),
                    bits(&self.box_hi[r][base + code]),
                );
                if next.is_some() {
                    code
                } else {
                    *first_with_box.entry(key).or_insert(code)
                }
            })
            .collect();
        let mut values = Values::new(codes);
        values
            .lo
            .par_chunks_mut(CHIPS * CHIPS)
            .zip(values.hi.par_chunks_mut(CHIPS * CHIPS))
            .enumerate()
            .for_each(|(code, (lo, hi))| {
                if source[code] != code {
                    return;
                }
                let chance = Chance {
                    r,
                    first,
                    code,
                    lo: &self.box_lo[r][base + code],
                    hi: &self.box_hi[r][base + code],
                    next: next.as_ref().map(|stage| &stage.values),
                };
                let mut scratch = Vec::new();
                for &(s0, s1) in &self.arena.chips[r][first] {
                    (lo[s0 * CHIPS + s1], hi[s0 * CHIPS + s1]) =
                        self.arena.round_value(&chance, s0, s1, &mut scratch);
                }
            });
        for (code, &from) in source
            .iter()
            .enumerate()
            .filter(|&(code, &from)| from != code)
        {
            let rows = from * CHIPS * CHIPS..(from + 1) * CHIPS * CHIPS;
            values.lo.copy_within(rows.clone(), code * CHIPS * CHIPS);
            values.hi.copy_within(rows, code * CHIPS * CHIPS);
        }
        let gap = next.as_ref().map_or(0.0, |stage| stage.gap);
        if let Some(stage) = next
            && r < kept.depth
        {
            kept.stages.insert(prefix.to_vec(), stage);
        }
        (values, gap)
    }
}

impl Arena {
    /// `(W_lo, W_hi)` of one round after its pair is revealed, at chips `(s0, s1)`; `scratch`
    /// is reused for the `(lo, hi)` value slots of the DAG.
    #[must_use]
    pub fn round_value(
        &self,
        chance: &Chance<'_>,
        s0: usize,
        s1: usize,
        scratch: &mut Vec<[f64; 2]>,
    ) -> (f64, f64) {
        let dag = self.dag(chance.r, s0, s1, chance.first);
        let [t_lo, t_hi, n_lo, n_hi] = Self::payoffs(chance, dag, s0, s1, scratch);
        let [b_lo, b_hi] = dag.evaluate(scratch);
        let w_lo = (t_lo * n_lo + (1.0 - t_lo) * b_lo).min(t_hi * n_lo + (1.0 - t_hi) * b_lo);
        let w_hi = (t_lo * n_hi + (1.0 - t_lo) * b_hi).max(t_hi * n_hi + (1.0 - t_hi) * b_hi);
        (w_lo, w_hi)
    }

    /// Fill the showdown and fold slots of `dag` with `(lo, hi)` payoffs; return
    /// `[t_lo, t_hi, n_lo, n_hi]`, the no-judgement probability bounds and continuation values.
    fn payoffs(
        chance: &Chance<'_>,
        dag: &BettingDag,
        s0: usize,
        s1: usize,
        values: &mut Vec<[f64; 2]>,
    ) -> [f64; 4] {
        let (t_lo, t_hi) = (chance.lo[NOJ], chance.hi[NOJ]);
        let q_lo = WIN.map(|c| {
            if t_lo >= 1.0 {
                0.0
            } else {
                chance.lo[c] / (1.0 - t_lo)
            }
        });
        let q_hi = WIN.map(|c| {
            if t_hi >= 1.0 {
                1.0
            } else {
                (chance.hi[c] / (1.0 - t_hi)).min(1.0)
            }
        });
        let continuation = |c0: u32, c1: u32, end: End, child: usize| match settle(
            chance.r,
            s0,
            s1,
            c0 as usize,
            c1 as usize,
            end,
        ) {
            Settled::Terminal(u) => [u, u],
            Settled::Next(t0, t1) => {
                let next = chance.next.expect("every round but the last continues");
                let i = at(chance.code * 6 + child, t0, t1);
                [next.lo[i], next.hi[i]]
            }
        };
        values.resize(dag.slots(), [0.0; 2]);
        let showdowns = dag.showdowns.iter().map(|&(c0, c1)| {
            let u = WIN.map(|class| continuation(c0, c1, End::Class(class), class));
            [
                robust_expectation(&u.map(|x| x[0]), &q_lo, &q_hi, true),
                robust_expectation(&u.map(|x| x[1]), &q_lo, &q_hi, false),
            ]
        });
        let folds = dag
            .folds
            .iter()
            .map(|&(c0, c1, who)| continuation(c0, c1, End::Fold(who), HIDDEN));
        for (slot, payoff) in values[dag.nodes.len()..]
            .iter_mut()
            .zip(showdowns.chain(folds))
        {
            *slot = payoff;
        }
        let ante = ANTES[chance.r];
        let [n_lo, n_hi] = continuation(
            ante,
            ante,
            End::Class(NOJ),
            digit(NOJ).expect("no judgement has a digit"),
        );
        [t_lo, t_hi, n_lo, n_hi]
    }

    /// The slot, in DAG action order, of the robust best action at betting node
    /// `(c0, c1, actor, checked)`; `use_hi` selects the upper-bound game (Team 1's side of the
    /// interval) instead of the lower.
    #[must_use]
    pub fn betting_slot(
        &self,
        chance: &Chance<'_>,
        chips: (usize, usize),
        node: (u32, u32, usize, bool),
        use_hi: bool,
    ) -> usize {
        let dag = self.dag(chance.r, chips.0, chips.1, chance.first);
        let mut values = Vec::new();
        Self::payoffs(chance, dag, chips.0, chips.1, &mut values);
        dag.evaluate(&mut values);
        let index = dag
            .nodes
            .iter()
            .position(|&n| n == node)
            .expect("the engine's betting node is in the DAG");
        dag.best(index, &values, usize::from(use_hi))
    }
}
