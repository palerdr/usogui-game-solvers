//! The betting street as a DAG per chip configuration, and round settlement.
//!
//! Once the round's ante and both players' sealed BIOS at round start are fixed, the legal bets
//! depend only on `(contribution 0, contribution 1, actor, previous check)`. Each reachable
//! configuration `(round, sealed 0, sealed 1, first bettor)` gets one DAG built from the same
//! rules as the engine's legal moves. [`BettingDag::evaluate`] computes the perfect-information
//! value from terminal payoffs, and [`settle`] applies the engine's settlement to one outcome.

use std::collections::{BTreeSet, HashMap};

use crate::engine::STARTING_BIOS;
use crate::{ANTES, NOJ, ROUNDS, W0, W0C, W1, W1C};

/// Chip counts `0..CHIPS` one side can hold after round 1: at most 46 sealed BIOS.
pub const CHIPS: usize = 2 * (STARTING_BIOS as usize - 1) - 2 * ANTES[0] as usize + 1;

/// The keys of an id map in id order.
fn by_id<K>(map: HashMap<K, u32>) -> Vec<K> {
    let mut items: Vec<_> = map.into_iter().collect();
    items.sort_by_key(|&(_, id)| id);
    items.into_iter().map(|(key, _)| key).collect()
}

/// Where a betting action leads while the DAG is built.
#[derive(Clone, Copy)]
enum Target {
    Node(u32),
    Showdown(u32),
    Fold(u32),
}

/// One configuration's betting DAG, nodes in topological order from the root.
///
/// A value array holds one slot per node, then one per showdown, then one per fold; an action
/// target is a slot.
pub struct BettingDag {
    /// `(c0, c1, actor, previous check)` of each node.
    pub nodes: Vec<(u32, u32, usize, bool)>,
    /// Actions of node `i` are `targets[starts[i]..starts[i + 1]]`: check or fold, then call,
    /// then raises by 1, 2, ... in order.
    pub starts: Vec<usize>,
    /// The slot each action leads to.
    pub targets: Vec<u32>,
    /// Contributions `(c0, c1)` at each showdown terminal.
    pub showdowns: Vec<(u32, u32)>,
    /// Contributions and folding team at each fold terminal.
    pub folds: Vec<(u32, u32, usize)>,
}

type Key = (u32, u32, usize, bool);

impl BettingDag {
    /// The DAG of round ante `ante` with sealed BIOS `(s0, s1)` after the ante.
    #[must_use]
    pub fn new(s0: u32, s1: u32, ante: u32, first: usize) -> Self {
        let root = (ante, ante, first, false);
        let mut index = HashMap::from([(root, 0)]);
        let mut order = vec![root];
        let mut actions = Vec::new();
        let (mut showdowns, mut folds) = (HashMap::new(), HashMap::new());
        let mut node = 0;
        while node < order.len() {
            let (c0, c1, i, checked) = order[node];
            let c = [c0, c1];
            let sealed = [s0 + ante - c0, s1 + ante - c1];
            let to_call = c[1 - i] - c[i].min(c[1 - i]);
            let mut visit = |key: Key, order: &mut Vec<Key>| {
                let next = order.len();
                Target::Node(*index.entry(key).or_insert_with(|| {
                    order.push(key);
                    next
                }) as u32)
            };
            let mut showdown = |c: (u32, u32)| {
                let next = showdowns.len() as u32;
                Target::Showdown(*showdowns.entry(c).or_insert(next))
            };
            let mut list = Vec::new();
            if to_call > 0 {
                let next = folds.len() as u32;
                list.push(Target::Fold(*folds.entry((c0, c1, i)).or_insert(next)));
            } else if checked {
                list.push(showdown((c0, c1)));
            } else {
                list.push(visit((c0, c1, 1 - i, true), &mut order));
            }
            if sealed[i] >= to_call {
                let raised = |pay: u32| {
                    if i == 0 {
                        (c0 + pay, c1)
                    } else {
                        (c0, c1 + pay)
                    }
                };
                if to_call > 0 {
                    list.push(showdown(raised(to_call)));
                }
                let max_extra = (sealed[i] - to_call)
                    .min(sealed[1 - i])
                    .min((c0 + c1 + to_call) / 2);
                for extra in 1..=max_extra {
                    let (n0, n1) = raised(to_call + extra);
                    list.push(visit((n0, n1, 1 - i, false), &mut order));
                }
            }
            actions.push(list);
            node += 1;
        }
        // Contributions only grow and a check only sets the flag, so this order is topological.
        let mut rank: Vec<usize> = (0..order.len()).collect();
        rank.sort_by_key(|&n| (order[n].0 + order[n].1, order[n].3));
        let mut position = vec![0; order.len()];
        for (new, &old) in rank.iter().enumerate() {
            position[old] = new as u32;
        }
        let (n, s) = (order.len() as u32, showdowns.len() as u32);
        let mut starts = vec![0];
        let mut targets = Vec::new();
        for &old in &rank {
            targets.extend(actions[old].iter().map(|&t| match t {
                Target::Node(k) => position[k as usize],
                Target::Showdown(k) => n + k,
                Target::Fold(k) => n + s + k,
            }));
            starts.push(targets.len());
        }
        Self {
            nodes: rank.iter().map(|&n| order[n]).collect(),
            starts,
            targets,
            showdowns: by_id(showdowns),
            folds: by_id(folds),
        }
    }

    /// Slots of a value array: nodes, showdowns, folds.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.nodes.len() + self.showdowns.len() + self.folds.len()
    }

    /// The root values for Team 0 of two payoff sets at once (Team 0 maximizes, Team 1
    /// minimizes): `values` carries the terminal payoffs in its showdown and fold slots and
    /// receives the node values.
    pub fn evaluate(&self, values: &mut [[f64; 2]]) -> [f64; 2] {
        for node in (0..self.nodes.len()).rev() {
            let mut slots = self.targets[self.starts[node]..self.starts[node + 1]]
                .iter()
                .map(|&t| values[t as usize]);
            let first = slots.next().expect("a betting node has an action");
            values[node] = if self.nodes[node].2 == 0 {
                slots.fold(first, |a, b| [a[0].max(b[0]), a[1].max(b[1])])
            } else {
                slots.fold(first, |a, b| [a[0].min(b[0]), a[1].min(b[1])])
            };
        }
        values[0]
    }

    /// The slot of the best action at `node` for payoff set `side` of evaluated `values`; ties
    /// go to the earliest slot.
    #[must_use]
    pub fn best(&self, node: usize, values: &[[f64; 2]], side: usize) -> usize {
        let maximize = self.nodes[node].2 == 0;
        let options = self.targets[self.starts[node]..self.starts[node + 1]]
            .iter()
            .map(|&t| values[t as usize][side]);
        let mut best = (0, f64::NAN);
        for (slot, v) in options.enumerate() {
            if slot == 0 || (maximize && v > best.1) || (!maximize && v < best.1) {
                best = (slot, v);
            }
        }
        best.0
    }
}

/// How a round ends: a judged or no-judgement class, or a fold by one team.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// A result class at showdown, or [`NOJ`].
    Class(usize),
    /// The team with this index folds.
    Fold(usize),
}

/// What follows a settled round.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Settled {
    /// The game ends with this utility for Team 0.
    Terminal(f64),
    /// The next round starts with these sealed BIOS after its ante.
    Next(usize, usize),
}

/// The engine's settlement of round `r` from sealed BIOS `(s0, s1)` after the ante, with
/// contributions `(c0, c1)`.
#[must_use]
pub fn settle(r: usize, s0: usize, s1: usize, c0: usize, c1: usize, end: End) -> Settled {
    let ante = ANTES[r] as usize;
    let (mut s0, mut s1) = (s0 + ante - c0, s1 + ante - c1);
    if end == End::Class(NOJ) {
        s0 += c0;
        s1 += c1;
    } else {
        if matches!(end, End::Class(W0 | W0C) | End::Fold(1)) {
            s0 += c0 + c1;
        } else {
            s1 += c0 + c1;
        }
        match end {
            End::Class(W0C) if s1 < c1 => return Settled::Terminal(1.0),
            End::Class(W0C) => s1 -= c1,
            End::Class(W1C) if s0 < c0 => return Settled::Terminal(-1.0),
            End::Class(W1C) => s0 -= c0,
            _ => {}
        }
    }
    let sign = |a: bool, b: bool| f64::from(i8::from(a) - i8::from(b));
    if r == ROUNDS - 1 {
        return Settled::Terminal(sign(s0 > s1, s0 < s1));
    }
    let next = ANTES[r + 1] as usize;
    let (pay0, pay1) = (s0 >= next, s1 >= next);
    if pay0 && pay1 {
        Settled::Next(s0 - next, s1 - next)
    } else {
        Settled::Terminal(sign(pay0, pay1))
    }
}

/// Per round, the reachable `(sealed 0, sealed 1, first bettor)` after the ante.
#[must_use]
pub fn reachable_configurations() -> Vec<BTreeSet<(usize, usize, usize)>> {
    let start = (STARTING_BIOS - 1 - ANTES[0]) as usize;
    let mut reach = vec![BTreeSet::from([(start, start, 0), (start, start, 1)])];
    for r in 0..ROUNDS - 1 {
        let mut next = BTreeSet::new();
        for &(s0, s1, first) in &reach[r] {
            let dag = BettingDag::new(s0 as u32, s1 as u32, ANTES[r], first);
            let showdowns = dag.showdowns.iter().flat_map(|&(c0, c1)| {
                [W0, W0C, W1, W1C, NOJ].map(|class| (c0, c1, End::Class(class)))
            });
            let folds = dag
                .folds
                .iter()
                .map(|&(c0, c1, folder)| (c0, c1, End::Fold(folder)));
            for (c0, c1, end) in showdowns.chain(folds) {
                if let Settled::Next(t0, t1) = settle(r, s0, s1, c0 as usize, c1 as usize, end) {
                    next.insert((t0, t1, 1 - first));
                }
            }
        }
        reach.push(next);
    }
    reach
}
