//! The sequence-form linear program on the public-tree arrays, solved with `HiGHS`.
//!
//! Sequence 0 is empty and sequences 1..=10 pick beads. The announcement sequences of actor slot k
//! on layer d are `offset[d] + 10 k + a`. The LP is P0's primal: maximize P1's root value `u_0` over
//! P0's realization plans x, subject to `flow_0 x = e_0` and, for each P1 sequence j,
//! `(flow_1^T u)_j <= (payoff^T x)_j`. A last-layer P1 sequence that hits no alive pair only states
//! `u_I <= 0`, which every alive sequence of the same information set implies, so it is dropped.
//! `HiGHS` returns P1's plan as the negated duals of the remaining rows.

use highs::{ColProblem, HighsModelStatus, Sense};

use crate::cfr::normalize;
use crate::traversal::Profile;
use crate::tree::Layer;
use crate::{Error, HANDS};

/// The sequence form of `D_K`.
#[derive(Debug)]
pub struct SequenceForm {
    /// `offset[p][d]` is player p's first announcement sequence on layer d, where p acts.
    pub offset: [Vec<usize>; 2],
    /// The number of sequences of each player.
    pub count: [usize; 2],
    /// (P0 sequence, P1 sequence, P0's payoff) of each hit leaf.
    pub payoff: Vec<(usize, usize, f64)>,
    /// (row, sequence, coefficient) of each player's flow constraints: row 0 fixes the empty
    /// sequence at 1, and each information set's sequences sum to its parent sequence.
    pub flow: [Vec<(usize, usize, f64)>; 2],
    /// The number of flow rows of each player.
    pub flow_rows: [usize; 2],
}

/// Player q's sequence at each of its slots on each layer, before that layer acts.
fn sequences(layers: &[Layer], offset: &[Vec<usize>; 2], q: usize) -> Vec<Vec<usize>> {
    let mut out = vec![
        layers[0].slot_hand[q]
            .iter()
            .map(|&h| usize::from(h))
            .collect::<Vec<_>>(),
    ];
    for (d, layer) in layers.iter().enumerate().skip(1) {
        let acted = layers[d - 1].actor == q;
        let next = (layer.slot_parent[q].iter().enumerate())
            .map(|(s, &up)| {
                let up = up as usize;
                if acted {
                    offset[q][d - 1] + HANDS * up + layer.arrival(q, s)
                } else {
                    out[d - 1][up]
                }
            })
            .collect();
        out.push(next);
    }
    out
}

/// Build the sequence form of the game that `layers` represent.
#[must_use]
pub fn build(layers: &[Layer]) -> SequenceForm {
    let mut offset = [vec![0; layers.len()], vec![0; layers.len()]];
    let mut count = [1 + HANDS; 2];
    for (d, layer) in layers.iter().enumerate() {
        let p = layer.actor;
        offset[p][d] = count[p];
        count[p] += HANDS * layer.slots(p);
    }
    let seqs = [0, 1].map(|q| sequences(layers, &offset, q));
    let mut payoff = Vec::new();
    for (d, layer) in layers.iter().enumerate() {
        let (p, q) = (layer.actor, 1 - layer.actor);
        for (k, &slot) in layer.pair_slot[q].iter().enumerate() {
            let (own, opp) = (offset[p][d] + layer.hit(k), seqs[q][d][slot as usize]);
            payoff.push(if p == 0 {
                (own, opp, 1.0)
            } else {
                (opp, own, -1.0)
            });
        }
    }
    let mut flow_rows = [2; 2];
    let flow = [0, 1].map(|p| {
        let mut f = vec![(0, 0, 1.0), (1, 0, -1.0)];
        f.extend((1..=HANDS).map(|s| (1, s, 1.0)));
        for (d, layer) in layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.actor == p)
        {
            for (k, &parent) in seqs[p][d].iter().enumerate() {
                let row = flow_rows[p] + k;
                f.push((row, parent, -1.0));
                f.extend((0..HANDS).map(|a| (row, offset[p][d] + HANDS * k + a, 1.0)));
            }
            flow_rows[p] += layer.slots(p);
        }
        f
    });
    SequenceForm {
        offset,
        count,
        payoff,
        flow,
        flow_rows,
    }
}

/// An optimal solution of the LP.
#[derive(Debug)]
pub struct LpSolution {
    /// The game value: P0's equilibrium payoff.
    pub value: f64,
    /// The realization plan of each player, indexed by sequence.
    pub plans: [Vec<f64>; 2],
    /// Constraint rows.
    pub rows: usize,
    /// Variables.
    pub columns: usize,
    /// Nonzero matrix entries.
    pub nonzeros: usize,
}

/// Solve the LP with the `HiGHS` `solver` ("choose", "simplex" or "ipm") within `time_limit`
/// seconds.
///
/// # Errors
///
/// [`Error::Lp`] when `HiGHS` stops without an optimum.
pub fn solve(form: &SequenceForm, solver: &str, time_limit: f64) -> Result<LpSolution, Error> {
    let ([n0, n1], [m0, m1]) = (form.count, form.flow_rows);
    let last = form.offset[1].iter().copied().max().unwrap_or(n1); // P1 acts last
    let mut keep: Vec<bool> = (0..n1).map(|j| j < last).collect();
    for &(_, j, _) in &form.payoff {
        keep[j] = true;
    }
    let mut problem = ColProblem::new();
    let mut row = vec![None; n1];
    for j in (0..n1).filter(|&j| keep[j]) {
        row[j] = Some(problem.add_row(..=0.0));
    }
    let fixed: Vec<_> = (0..m0)
        .map(|r| f64::from(u8::from(r == 0)))
        .map(|b| problem.add_row(b..=b))
        .collect();
    let mut columns = vec![Vec::new(); n0 + m1];
    for &(s, j, v) in &form.payoff {
        columns[s].extend(row[j].map(|r| (r, -v)));
    }
    for &(r, s, v) in &form.flow[0] {
        columns[s].push((fixed[r], v));
    }
    for &(r, j, v) in &form.flow[1] {
        columns[n0 + r].extend(row[j].map(|row| (row, v)));
    }
    let nonzeros = columns.iter().map(Vec::len).sum();
    for (c, factors) in columns.into_iter().enumerate() {
        let lower = if c < n0 { 0.0 } else { f64::NEG_INFINITY };
        problem.add_column(if c == n0 { -1.0 } else { 0.0 }, lower.., factors);
    }
    let (rows, columns) = (problem.num_rows(), problem.num_cols());
    let mut model = problem.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("solver", solver);
    model.set_option("time_limit", time_limit);
    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(Error::Lp(format!("{:?}", solved.status())));
    }
    let solution = solved.get_solution();
    let mut p1 = vec![0.0; n1];
    for (r, j) in (0..n1).filter(|&j| keep[j]).enumerate() {
        p1[j] = -solution.dual_rows()[r];
    }
    let plans = [solution.columns()[..n0].to_vec(), p1];
    Ok(LpSolution {
        value: -solved.objective_value(),
        plans,
        rows,
        columns,
        nonzeros,
    })
}

/// The behavioral profile of the realization plans: each information set normalizes its
/// sequences. Negative noise is clipped; an unreached information set plays uniformly.
#[must_use]
pub fn plan_profile(form: &SequenceForm, layers: &[Layer], plans: [&[f64]; 2]) -> Profile {
    let behavior = (layers.iter().enumerate())
        .map(|(d, layer)| {
            let (p, start) = (layer.actor, form.offset[layer.actor][d]);
            normalize(&plans[p][start..start + HANDS * layer.slots(p)])
        })
        .collect();
    Profile {
        behavior,
        choose: plans.map(|plan| normalize(&plan[1..=HANDS])),
    }
}
