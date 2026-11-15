//! The adaptive allocation loop.
//!
//! Each iteration solves the robust game on the current boxes, estimates the reach of every
//! number sequence by self-play under both robust equilibria, gives the next sampling budget as
//! a uniform floor plus a share proportional to reach, samples, and records the root interval.
//! The confidence sequences stay valid under this allocation, so every iteration's interval is
//! a certificate on its own.

use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use crate::robust::Solver;
use crate::sampler::{Counts, sample_sequence};
use crate::selfplay::sequence_reach;
use crate::table::KernelCounts;

/// The settings of an allocation run.
#[derive(Clone, Debug, Serialize, clap::Args)]
pub struct Config {
    /// Solve-reach-sample iterations.
    #[arg(long, default_value_t = 3)]
    pub iterations: usize,
    /// Trajectories sampled per iteration.
    #[arg(long = "budget", default_value_t = 1e8)]
    pub budget_per_iteration: f64,
    /// Fraction of each budget spread uniformly over all sequences.
    #[arg(long = "floor", default_value_t = 0.2)]
    pub floor_fraction: f64,
    /// Self-play games per equilibrium for the reach estimate.
    #[arg(long = "games", default_value_t = 2000)]
    pub games_per_equilibrium: u64,
    /// Seed of the sampler and of self-play.
    #[arg(long, default_value_t = 20_261_029)]
    pub seed: u64,
    /// Joint error level of all kernel boxes.
    #[arg(long, default_value_t = 1e-3)]
    pub delta: f64,
}

/// One number sequence and its reach.
#[derive(Clone, Debug, Serialize)]
pub struct Top {
    /// The five number pairs.
    pub sequence: Vec<(u8, u8)>,
    /// Self-play visits.
    pub reach: f64,
}

/// The record of one iteration.
#[derive(Clone, Debug, Serialize)]
pub struct Iteration {
    /// Iteration index.
    pub iteration: usize,
    /// Trajectories in the table at the solve.
    pub samples_before: u64,
    /// Lower bound of Team 0's root value.
    pub root_lo: f64,
    /// Upper bound of Team 0's root value.
    pub root_hi: f64,
    /// `root_hi - root_lo`.
    pub width: f64,
    /// The largest saddle gap of any matrix game.
    pub max_saddle_gap: f64,
    /// Mean self-play return of Team 0 under the lower and the upper equilibrium.
    pub self_play_mean_return: [f64; 2],
    /// Sequences with positive reach.
    pub sequences_reached: usize,
    /// The ten sequences of largest reach.
    pub top_sequences: Vec<Top>,
    /// Wall time of the solve.
    pub solve_seconds: f64,
    /// Wall time of self-play.
    pub reach_seconds: f64,
    /// Wall time of sampling.
    pub sample_seconds: f64,
    /// Trajectories in the table after sampling.
    pub samples_after: u64,
}

/// The settings and every iteration so far.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    /// The settings.
    #[serde(flatten)]
    pub config: Config,
    /// The iterations.
    pub iterations: Vec<Iteration>,
}

/// Samples per sequence: a uniform floor plus the rest of the budget in proportion to `reach`.
#[must_use]
pub fn allocation(reach: &[f64], budget: u64, floor_fraction: f64) -> Vec<u64> {
    let n = reach.len() as u64;
    let floor = (budget as f64 * floor_fraction) as u64 / n;
    let rest = (budget - floor * n) as f64;
    let total: f64 = reach.iter().sum();
    let share = |r: f64| {
        if total > 0.0 {
            r / total
        } else {
            1.0 / n as f64
        }
    };
    reach
        .iter()
        .map(|&r| floor + (rest * share(r)).floor() as u64)
        .collect()
}

/// Run the loop from an empty table; `progress` sees the report after every iteration.
pub fn run(
    config: &Config,
    round_one: &[((u8, u8), Counts)],
    mut progress: impl FnMut(&Report),
) -> Report {
    let mut solver = Solver::new(round_one);
    let mut table = KernelCounts::new(&solver.sequences);
    let mut report = Report {
        config: config.clone(),
        iterations: Vec::new(),
    };
    let mut samples = 0;
    let budget = config.budget_per_iteration as u64;
    for iteration in 0..config.iterations {
        let clock = Instant::now();
        solver.set_boxes(&table, config.delta);
        let solution = solver.solve(2);
        let solve_seconds = clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let (reach, returns) = sequence_reach(
            &solver,
            &solution.kept,
            config.games_per_equilibrium,
            config.seed * 1000 + iteration as u64,
        );
        let reach_seconds = clock.elapsed().as_secs_f64();
        drop(solution.kept);
        let plan = allocation(&reach, budget, config.floor_fraction);
        let clock = Instant::now();
        let targets = &solver.sequences.targets;
        let jobs: Vec<(usize, u64)> = plan
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, n)| n > 0)
            .collect();
        // Sequences sample side by side in chunks; the counts are added in sequence order.
        for chunk in jobs.chunks(1024) {
            let sampled: Vec<_> = chunk
                .par_iter()
                .map(|&(s, n)| {
                    sample_sequence(
                        &targets[s],
                        config.seed,
                        (iteration * targets.len() + s) as u64,
                        n,
                    )
                })
                .collect();
            for (&(s, _), (counts, errors)) in chunk.iter().zip(sampled) {
                assert_eq!(errors, 0, "ties are impossible with distinct metal cards");
                table.add(&solver.sequences, s, &counts);
            }
        }
        let sampled: u64 = plan.iter().sum();
        let mut order: Vec<usize> = (0..reach.len()).collect();
        order.sort_by(|&a, &b| reach[b].total_cmp(&reach[a]));
        report.iterations.push(Iteration {
            iteration,
            samples_before: samples,
            root_lo: solution.root_lo,
            root_hi: solution.root_hi,
            width: solution.root_hi - solution.root_lo,
            max_saddle_gap: solution.max_gap,
            self_play_mean_return: returns,
            sequences_reached: reach.iter().filter(|&&r| r > 0.0).count(),
            top_sequences: order[..10]
                .iter()
                .map(|&s| Top {
                    sequence: targets[s].to_vec(),
                    reach: reach[s],
                })
                .collect(),
            solve_seconds,
            reach_seconds,
            sample_seconds: clock.elapsed().as_secs_f64(),
            samples_after: samples + sampled,
        });
        samples += sampled;
        progress(&report);
    }
    report
}
