//! Forward reach and backward counterfactual values on the public-tree arrays, and the
//! best responses built from them.
//!
//! No chance node exists, so the reach of an alive pair is r0(x0) * r1(x1). The best responses
//! here are the CFR stopping test. They are not the certificate: [`crate::certificate`] computes
//! best responses through the rules engine without these arrays.

use rayon::prelude::*;

use crate::HANDS;
use crate::tree::{Layer, NONE};

/// A behavioral strategy profile on the arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// Per layer: the actor's slots x 10 action probabilities, row-major.
    pub behavior: Vec<Vec<f64>>,
    /// The bead-pick probabilities of each player, indexed by beads - 1.
    pub choose: [Vec<f64>; 2],
}

// CFR+ meets exact ties (an action whose regret is zero in exact arithmetic), and rounding
// decides whether such a regret stays at +1 ulp or clips to zero, which flips the next strategy.
// `dot` and `sum` therefore round as the Python reference does on this machine: numpy's einsum
// with two fused multiply-add lanes, and numpy's pairwise sum.

/// The dot product of two 10-action rows, in numpy's einsum order on aarch64.
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    let lane = |l: usize| {
        let body = [3, 2, 1, 0]
            .iter()
            .fold(0.0, |acc, &i| x[2 * i + l].mul_add(y[2 * i + l], acc));
        x[8 + l].mul_add(y[8 + l], body)
    };
    lane(0) + lane(1)
}

/// The sum of a 10-action row, in numpy's pairwise order.
pub(crate) fn sum(x: &[f64]) -> f64 {
    (((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]))) + x[8] + x[9]
}

/// The dot product of each 10-action row of `a` with the same row of `b`.
pub(crate) fn row_dots(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.par_chunks(HANDS)
        .zip(b.par_chunks(HANDS))
        .map(|(x, y)| dot(x, y))
        .collect()
}

/// Probability of player q's own actions to each of q's slots, layer by layer.
#[must_use]
pub fn reach(layers: &[Layer], profile: &Profile, q: usize) -> Vec<Vec<f64>> {
    let root = layers[0].slot_hand[q]
        .iter()
        .map(|&h| profile.choose[q][usize::from(h) - 1]);
    let mut out = vec![root.collect::<Vec<_>>()];
    for (d, layer) in layers.iter().enumerate().skip(1) {
        let (up, acted) = (&out[d - 1], layers[d - 1].actor == q);
        let next = (layer.slot_parent[q].iter().enumerate())
            .map(|(s, &k)| {
                let r = up[k as usize];
                if acted {
                    r * profile.behavior[d - 1][HANDS * k as usize + layer.arrival(q, s)]
                } else {
                    r
                }
            })
            .collect();
        out.push(next);
    }
    out
}

/// Player i's counterfactual value at each of its root slots, indexed by beads - 1.
///
/// The value of slot (h, x) sums the opponent's reach times i's payoff over the opponent hands
/// alive at h. On i's layers, `own_layer(d, cfv)` receives the action values (slots x 10) and
/// returns the slot values; it may update regrets. The opponent plays `behavior` on its layers.
pub fn values(
    layers: &[Layer],
    i: usize,
    opp_reach: &[Vec<f64>],
    behavior: &[Vec<f64>],
    mut own_layer: impl FnMut(usize, Vec<f64>) -> Vec<f64>,
) -> Vec<f64> {
    let mut below: Vec<f64> = Vec::new();
    for (d, layer) in layers.iter().enumerate().rev() {
        let (p, n) = (layer.actor, layer.slots(i));
        below = if p == i {
            let mut cfv = vec![0.0; n * HANDS];
            for (k, &opp) in layer.pair_slot[1 - i].iter().enumerate() {
                cfv[layer.hit(k)] = opp_reach[d][opp as usize];
            }
            cfv.par_iter_mut()
                .zip(&layer.child_slot)
                .for_each(|(c, &child)| {
                    if child != NONE {
                        *c += below[child as usize];
                    }
                });
            own_layer(d, cfv)
        } else {
            let mut v = vec![0.0; n];
            if let Some(next) = layers.get(d + 1) {
                for (&up, b) in next.slot_parent[i].iter().zip(&below) {
                    v[up as usize] += b;
                }
            }
            let mut lost = vec![0.0; n];
            for (k, (&own, &opp)) in layer.pair_slot[i]
                .iter()
                .zip(&layer.pair_slot[p])
                .enumerate()
            {
                lost[own as usize] += opp_reach[d][opp as usize] * behavior[d][layer.hit(k)];
            }
            v.iter().zip(&lost).map(|(v, l)| v - l).collect()
        };
    }
    below
}

/// Player i's best-response value against the opponent's part of `profile`.
#[must_use]
pub fn best_response_value(layers: &[Layer], i: usize, profile: &Profile) -> f64 {
    let opp = reach(layers, profile, 1 - i);
    let root = values(layers, i, &opp, &profile.behavior, |_, cfv| {
        cfv.chunks(HANDS)
            .map(|row| row.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect()
    });
    root.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// (P0's guarantee, P0's cap) of `profile`: the game value lies between them.
#[must_use]
pub fn saddle_gap(layers: &[Layer], profile: &Profile) -> (f64, f64) {
    (
        -best_response_value(layers, 1, profile),
        best_response_value(layers, 0, profile),
    )
}

/// P0's expected payoff when both players follow `profile`.
#[must_use]
pub fn profile_value(layers: &[Layer], profile: &Profile) -> f64 {
    let opp = reach(layers, profile, 1);
    let root = values(layers, 0, &opp, &profile.behavior, |d, cfv| {
        row_dots(&profile.behavior[d], &cfv)
    });
    profile.choose[0]
        .iter()
        .zip(&root)
        .map(|(p, v)| p * v)
        .sum()
}
