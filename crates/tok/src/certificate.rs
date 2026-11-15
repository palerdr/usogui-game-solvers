//! Best responses that walk the rules engine.
//!
//! Every transition comes from [`GameState::apply`], every action set from
//! [`GameState::legal_actions`], and every payoff from [`GameState::returns`]. The opponent's
//! behavior is read from a [`StrategyTable`] by information state. Nothing here reads the
//! public-tree arrays, so the certificate is independent of the solver.
//!
//! Histories are grouped by public transcript. A group's value for the responding player is a
//! vector over its own beads: the sum, over alive opponent hands, of opponent reach times payoff.
//! The responder's information set is (own beads, transcript), so a max per bead count is a best
//! response. Arithmetic is `f64`. Rayon walks sibling groups in parallel; the sums run in a fixed
//! order, so the result does not depend on the thread count.

use rayon::prelude::*;

use crate::rules::{Action, GameState, Player};
use crate::table::{MAX_HORIZON, StrategyTable};
use crate::{Error, HANDS, TOTALS};

type Group = Vec<(GameState, f64)>;

/// The responder's value of `group`, indexed by own beads - 1.
fn value(group: Group, player: Player, table: &StrategyTable) -> Result<[f64; HANDS], Error> {
    let actor = group[0].0.actor().expect("groups hold live states");
    let mut immediate = [[0.0; TOTALS]; HANDS];
    let mut children: [Group; TOTALS] = std::array::from_fn(|_| Vec::new());
    let mut legal: [Vec<usize>; HANDS] = std::array::from_fn(|_| Vec::new());
    for (state, r) in group {
        if r == 0.0 {
            continue;
        }
        let beads = state.beads();
        let own = usize::from(beads[player as usize].expect("announcers have beads")) - 1;
        let probs = if actor == player {
            if legal[own].is_empty() {
                legal[own] = state.legal_actions().map(|a| index(a, 2)).collect();
            }
            None
        } else {
            let info = state.information_state(actor);
            let missing = || Error::Artifact(format!("table lacks {info:?}"));
            Some(table.row(&info).ok_or_else(missing)?)
        };
        let first = beads[actor as usize].expect("announcers have beads") + 1;
        for action in state.legal_actions() {
            let w = probs.map_or(r, |p| r * p[index(action, first)]);
            if w == 0.0 {
                continue;
            }
            let child = state.apply(action)?;
            let g = index(action, 2);
            match child.returns() {
                Some(payoff) => immediate[own][g] += w * f64::from(payoff[player as usize]),
                None => children[g].push((child, w)),
            }
        }
    }
    let below: Vec<[f64; HANDS]> = (children.into_par_iter())
        .map(|group| {
            if group.is_empty() {
                Ok([0.0; HANDS])
            } else {
                value(group, player, table)
            }
        })
        .collect::<Result<_, _>>()?;
    Ok(std::array::from_fn(|x| {
        if actor == player {
            let line = |&g: &usize| immediate[x][g] + below[g][x];
            legal[x].iter().map(line).reduce(f64::max).unwrap_or(0.0)
        } else {
            immediate[x].iter().sum::<f64>() + below.iter().map(|v| v[x]).sum::<f64>()
        }
    }))
}

/// The announced total of a guess, minus `base`.
fn index(action: Action, base: u8) -> usize {
    let Action::Guess(total) = action else {
        unreachable!("groups hold announcement states")
    };
    usize::from(total - base)
}

/// The value to `player` of its best response to `table`, the opponent's strategy, in `D_horizon`.
///
/// # Errors
///
/// [`Error::Horizon`] for horizon 0, [`Error::Artifact`] when the horizon exceeds the table key
/// format (longer transcripts would alias shorter keys) or the table lacks an information state
/// the walk reaches.
pub fn best_response_value(
    horizon: usize,
    player: Player,
    table: &StrategyTable,
) -> Result<f64, Error> {
    if horizon > MAX_HORIZON {
        return Err(Error::Artifact(format!(
            "strategy tables hold at most K = {MAX_HORIZON}"
        )));
    }
    let root = GameState::new(horizon)?;
    let mut group = Group::new();
    for c0 in root.legal_actions() {
        let after = root.apply(c0)?;
        for c1 in after.legal_actions() {
            let (Action::Choose(x0), Action::Choose(x1)) = (c0, c1) else {
                unreachable!("the game opens with two bead picks")
            };
            let opp = if player == Player::P0 { x1 } else { x0 };
            group.push((after.apply(c1)?, table.choose[usize::from(opp) - 1]));
        }
    }
    Ok(value(group, player, table)?
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max))
}

/// (P0's guarantee, P0's cap) of the profile that plays `p0` and `p1`.
///
/// # Errors
///
/// As [`best_response_value`].
pub fn certify(
    horizon: usize,
    p0: &StrategyTable,
    p1: &StrategyTable,
) -> Result<(f64, f64), Error> {
    let lower = -best_response_value(horizon, Player::P1, p0)?;
    Ok((lower, best_response_value(horizon, Player::P0, p1)?))
}
