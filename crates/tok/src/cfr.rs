//! Exact full-traversal CFR+ on the public-tree arrays.
//!
//! Updates alternate: P0 updates against P1's current strategy, then P1 updates against P0's new
//! one. Regret matching+ clips cumulative regrets at zero. The average strategy weights
//! iteration t by t and by the player's own reach. A per-layer [`Rule`] lets the abstraction
//! study fix or merge one player's decisions inside the same routine.

use rayon::prelude::*;

use crate::HANDS;
use crate::traversal::{Profile, reach, row_dots, sum, values};
use crate::tree::{Layer, last_layer};

/// Each 10-action row of `weights`, clipped at zero and normalized; a row without positive
/// weight becomes uniform.
#[must_use]
pub(crate) fn normalize(weights: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; weights.len()];
    out.par_chunks_mut(HANDS)
        .zip(weights.par_chunks(HANDS))
        .for_each(|(row, w)| {
            row.iter_mut().zip(w).for_each(|(x, w)| *x = w.max(0.0));
            let total = sum(row);
            for x in row {
                *x = if total > 0.0 {
                    *x / total
                } else {
                    1.0 / HANDS as f64
                };
            }
        });
    out
}

/// How CFR+ treats the decisions of one layer, or one player's bead pick (a single row).
#[derive(Debug)]
enum Rule {
    /// Regret matching+ on every row.
    Free { regret: Vec<f64>, weight: Vec<f64> },
    /// A fixed behavior.
    Fixed(Vec<f64>),
    /// Regret matching+ on one shared row per key; `key` maps each slot to its row.
    Merged {
        key: Vec<u32>,
        regret: Vec<f64>,
        weight: Vec<f64>,
    },
}

/// The row of each slot's key, for slots keyed by `key`.
pub(crate) fn expand(key: &[u32], rows: &[f64]) -> Vec<f64> {
    let row = |k: u32| &rows[HANDS * k as usize..HANDS * (k as usize + 1)];
    key.iter().flat_map(|&k| row(k)).copied().collect()
}

/// Own reach times action values, summed per key: `keys` rows of 10.
pub(crate) fn key_sums(key: &[u32], keys: usize, own: &[f64], cfv: &[f64]) -> Vec<f64> {
    let mut totals = vec![0.0; keys * HANDS];
    for (s, &k) in key.iter().enumerate() {
        for a in 0..HANDS {
            totals[HANDS * k as usize + a] += own[s] * cfv[HANDS * s + a];
        }
    }
    totals
}

impl Rule {
    fn free(rows: usize) -> Self {
        Self::Free {
            regret: vec![0.0; rows * HANDS],
            weight: vec![0.0; rows * HANDS],
        }
    }

    fn current(&self) -> Vec<f64> {
        match self {
            Self::Free { regret, .. } => normalize(regret),
            Self::Fixed(behavior) => behavior.clone(),
            Self::Merged { key, regret, .. } => expand(key, &normalize(regret)),
        }
    }

    fn average(&self) -> Vec<f64> {
        match self {
            Self::Free { weight, .. } => normalize(weight),
            Self::Fixed(behavior) => behavior.clone(),
            Self::Merged { key, weight, .. } => expand(key, &normalize(weight)),
        }
    }

    /// Update on action values `cfv` with current behavior `sigma` and own reach `own`; return
    /// the slot values under `sigma`.
    fn update(&mut self, cfv: &[f64], sigma: &[f64], own: &[f64], t: f64) -> Vec<f64> {
        let v = row_dots(sigma, cfv);
        match self {
            Self::Free { regret, weight } => {
                let rows = regret
                    .par_chunks_mut(HANDS)
                    .zip(weight.par_chunks_mut(HANDS));
                rows.enumerate().for_each(|(k, (r, w))| {
                    let at = HANDS * k;
                    for a in 0..HANDS {
                        r[a] = (r[a] + cfv[at + a] - v[k]).max(0.0);
                        w[a] += t * own[k] * sigma[at + a];
                    }
                });
            }
            Self::Fixed(_) => {}
            Self::Merged {
                key,
                regret,
                weight,
            } => {
                let totals = key_sums(key, regret.len() / HANDS, own, cfv);
                let merged = normalize(regret);
                let vk = row_dots(&merged, &totals);
                for (j, (r, w)) in regret.iter_mut().zip(weight.iter_mut()).enumerate() {
                    *r = (*r + (totals[j] - vk[j / HANDS])).max(0.0);
                    *w += t * merged[j];
                }
            }
        }
        v
    }
}

/// CFR+ state: one [`Rule`] per layer and per bead pick.
#[derive(Debug)]
pub struct CfrPlus<'a> {
    layers: &'a [Layer],
    rules: Vec<Rule>,
    roots: [Rule; 2],
    /// Completed iterations.
    pub iterations: usize,
}

impl<'a> CfrPlus<'a> {
    /// CFR+ with every decision free.
    #[must_use]
    pub fn new(layers: &'a [Layer]) -> Self {
        Self::with(layers, |_| None, [Rule::free(1), Rule::free(1)])
    }

    /// CFR+ on the merge step: player i plays `plan` on its bead pick and earlier layers and one
    /// shared row per key on its last layer; the opponent is free. Player i then has one
    /// decision per play, so the restricted game has perfect recall and CFR+ solves it.
    #[must_use]
    pub fn merged(layers: &'a [Layer], i: usize, plan: &Profile, key: &[u32], keys: usize) -> Self {
        let last = last_layer(layers, i);
        let rule = |d: usize| match (layers[d].actor == i, d == last) {
            (true, true) => Some(Rule::Merged {
                key: key.to_vec(),
                regret: vec![0.0; keys * HANDS],
                weight: vec![0.0; keys * HANDS],
            }),
            (true, false) => Some(Rule::Fixed(plan.behavior[d].clone())),
            (false, _) => None,
        };
        let mut roots = [Rule::free(1), Rule::free(1)];
        roots[i] = Rule::Fixed(plan.choose[i].clone());
        Self::with(layers, rule, roots)
    }

    /// CFR+ on the earlier-layers step: player i's last layer plays `last_behavior`; every other
    /// decision is free.
    #[must_use]
    pub fn fixed_last(layers: &'a [Layer], i: usize, last_behavior: Vec<f64>) -> Self {
        let last = last_layer(layers, i);
        let mut fixed = Some(last_behavior);
        let rule = |d: usize| fixed.take_if(|_| d == last).map(Rule::Fixed);
        Self::with(layers, rule, [Rule::free(1), Rule::free(1)])
    }

    /// Layers where `rule` returns `None` are free.
    fn with(
        layers: &'a [Layer],
        mut rule: impl FnMut(usize) -> Option<Rule>,
        roots: [Rule; 2],
    ) -> Self {
        let rules = (layers.iter().enumerate())
            .map(|(d, layer)| rule(d).unwrap_or_else(|| Rule::free(layer.slots(layer.actor))))
            .collect();
        Self {
            layers,
            rules,
            roots,
            iterations: 0,
        }
    }

    /// The current strategy: regret matching+ on the cumulative regrets.
    #[must_use]
    pub fn current(&self) -> Profile {
        Profile {
            behavior: self.rules.iter().map(Rule::current).collect(),
            choose: self.roots.each_ref().map(Rule::current),
        }
    }

    /// The average strategy, which converges to an equilibrium.
    #[must_use]
    pub fn average(&self) -> Profile {
        Profile {
            behavior: self.rules.iter().map(Rule::average).collect(),
            choose: self.roots.each_ref().map(Rule::average),
        }
    }

    fn update(&mut self, i: usize) {
        let t = (self.iterations + 1) as f64;
        let sigma = self.current();
        let opp = reach(self.layers, &sigma, 1 - i);
        let own = reach(self.layers, &sigma, i);
        let rules = &mut self.rules;
        let root = values(self.layers, i, &opp, &sigma.behavior, |d, cfv| {
            rules[d].update(&cfv, &sigma.behavior[d], &own[d], t)
        });
        self.roots[i].update(&root, &sigma.choose[i], &[1.0], t);
    }

    /// One iteration: P0 updates, then P1.
    pub fn iterate(&mut self) {
        self.update(0);
        self.update(1);
        self.iterations += 1;
    }
}
