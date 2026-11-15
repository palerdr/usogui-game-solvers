//! Coarse keys for a player's last decision layer: how much value does player i lose when all
//! slots with one key must play one shared behavior?
//!
//! - [`merged_best_response`] fixes the opponent and i's earlier layers and picks the best action
//!   per key. Its value bounds the merge step from above.
//! - [`CfrPlus::merged`] solves the merge step: earlier layers fixed, last layer merged, opponent
//!   free. [`CfrPlus::fixed_last`] solves the earlier-layers step: last layer fixed, every other
//!   decision free.
//! - [`alternate`] runs the two steps in rounds. The certificate then proves the guarantee of the
//!   best lifted strategy on the full game.

use crate::cfr::{CfrPlus, expand, key_sums};
use crate::traversal::{Profile, best_response_value, reach, row_dots, values};
use crate::tree::{Layer, last_layer};
use crate::{HANDS, TOTALS};

/// What a last-layer key remembers besides own beads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Key {
    /// The information set itself.
    Full,
    /// The opponent hands still alive.
    Support,
    /// The set of own totals and the set of opponent totals.
    Hard,
    /// [`Key::Hard`] plus P0's first announced total.
    OpeningHard,
}

/// Announced totals of public node `node` on layer `d`, in order.
fn transcript(layers: &[Layer], d: usize, node: u32) -> Vec<u8> {
    let mut totals = vec![0; d];
    let mut node = node as usize;
    for (e, total) in totals.iter_mut().enumerate().rev() {
        *total = layers[e + 1].total[node];
        node = layers[e + 1].parent[node] as usize;
    }
    totals
}

/// The set of `totals` as a 19-bit mask.
fn bits<'a>(totals: impl Iterator<Item = &'a u8>) -> u64 {
    totals.fold(0, |mask, &g| mask | 1 << (g - 2))
}

/// The dense key of each slot on player i's last layer, and the number of keys.
#[must_use]
pub fn slot_keys(layers: &[Layer], i: usize, kind: Key) -> (Vec<u32>, usize) {
    let d = last_layer(layers, i);
    let layer = &layers[d];
    let hand = |s: usize| u64::from(layer.slot_hand[i][s]);
    let raw: Vec<u64> = match kind {
        Key::Full => return ((0..layer.slots(i) as u32).collect(), layer.slots(i)),
        Key::Support => {
            let mut mask = vec![0; layer.slots(i)];
            for (&s, &h) in layer.pair_slot[i].iter().zip(&layer.pair_hand[1 - i]) {
                mask[s as usize] |= 1 << (h - 1);
            }
            mask.iter()
                .enumerate()
                .map(|(s, m)| hand(s) + HANDS as u64 * m)
                .collect()
        }
        Key::Hard | Key::OpeningHard => (0..layer.slots(i))
            .map(|s| {
                let totals = transcript(layers, d, layer.slot_node[i][s]);
                let own = bits(totals.iter().skip(i).step_by(2));
                let opp = bits(totals.iter().skip(1 - i).step_by(2));
                let opening = if kind == Key::OpeningHard {
                    u64::from(totals[0]) << 45
                } else {
                    0
                };
                hand(s) + HANDS as u64 * (own + (opp << TOTALS)) + opening
            })
            .collect(),
    };
    let mut unique = raw.clone();
    unique.sort_unstable();
    unique.dedup();
    let key = raw.iter().map(|r| unique.partition_point(|u| u < r) as u32);
    (key.collect(), unique.len())
}

/// Player i's best expected payoff against the opponent's part of `profile` when i keeps its
/// earlier layers and plays one action per key on its last layer.
#[must_use]
pub fn merged_best_response(
    layers: &[Layer],
    i: usize,
    profile: &Profile,
    key: &[u32],
    keys: usize,
) -> f64 {
    let last = last_layer(layers, i);
    let (opp, own) = (reach(layers, profile, 1 - i), reach(layers, profile, i));
    let root = values(layers, i, &opp, &profile.behavior, |d, cfv| {
        if d != last {
            return row_dots(&profile.behavior[d], &cfv);
        }
        let totals = key_sums(key, keys, &own[d], &cfv);
        let mut best = vec![0.0; keys * HANDS];
        for (row, pick) in totals.chunks(HANDS).zip(best.chunks_mut(HANDS)) {
            pick[(0..HANDS).fold(0, |b, a| if row[a] > row[b] { a } else { b })] = 1.0;
        }
        row_dots(&expand(key, &best), &cfv)
    });
    profile.choose[i]
        .iter()
        .zip(&root)
        .map(|(p, v)| p * v)
        .sum()
}

/// Player i's worst-case payoff when i plays its part of `profile`.
#[must_use]
pub fn guarantee(layers: &[Layer], i: usize, profile: &Profile) -> f64 {
    -best_response_value(layers, 1 - i, profile)
}

/// One step of [`alternate`], in player i's payoffs.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Step {
    /// `merge<r>` or `early<r>` for round r.
    pub name: String,
    /// The guarantee of i's strategy after the step.
    pub guarantee: f64,
    /// For a merge step, the bound [`merged_best_response`] puts on its restricted value.
    pub cap: Option<f64>,
}

/// Alternate the merge step and the earlier-layers step for `rounds` rounds of `iterations` CFR+
/// iterations each, starting from `profile`. Returns the steps and the profile whose strategy for
/// i has the best guarantee; `log` sees each step as it ends.
pub fn alternate(
    layers: &[Layer],
    i: usize,
    profile: &Profile,
    key: &[u32],
    keys: usize,
    (rounds, iterations): (usize, usize),
    mut log: impl FnMut(&Step),
) -> (Vec<Step>, Profile) {
    let last = last_layer(layers, i);
    let (mut plan, mut steps) = (profile.clone(), Vec::new());
    let mut best = (f64::NEG_INFINITY, profile.clone());
    let mut record = |name: String, profile: &Profile, cap: Option<f64>| {
        let guarantee = guarantee(layers, i, profile);
        let step = Step {
            name,
            guarantee,
            cap,
        };
        log(&step);
        steps.push(step);
        if guarantee > best.0 {
            best = (guarantee, profile.clone());
        }
    };
    for r in 1..=rounds {
        let mut merge = CfrPlus::merged(layers, i, &plan, key, keys);
        (0..iterations).for_each(|_| merge.iterate());
        let merged = merge.average();
        let cap = merged_best_response(layers, i, &merged, key, keys);
        record(format!("merge{r}"), &merged, Some(cap));
        let mut early = CfrPlus::fixed_last(layers, i, merged.behavior[last].clone());
        (0..iterations).for_each(|_| early.iterate());
        plan = early.average();
        record(format!("early{r}"), &plan, None);
    }
    (steps, best.1)
}
