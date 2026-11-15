//! Command line: the exact round-1 kernel, the full-scale sampler gate, the robust solve and the
//! allocation loop.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ap::allocate::{self, Config};
use ap::cards::FULL_DECK;
use ap::dp::HandTable;
use ap::exact::round_one;
use ap::robust::Solver;
use ap::sampler::sample_rounds;
use ap::{CLASS_NAMES, W0, W0C};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(about = "Air Poker: exact round-1 kernel, kernel sampler and robust public-state solve")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Count the exact round-1 kernel of the 25 canonical pairs on the full deck.
    Exact {
        #[arg(long, default_value = "kernel-round1.json")]
        out: PathBuf,
    },
    /// Sample single full-deck rounds of every pair against the exact kernel (gate 3).
    Sample {
        #[arg(long, default_value_t = 1e8)]
        per_pair: f64,
        #[arg(long, default_value_t = 20_261_029)]
        seed: u64,
        #[arg(long, default_value = "kernel-round1-parity.json")]
        out: PathBuf,
    },
    /// Solve the robust game with the exact round-1 kernel and no samples after it.
    Solve,
    /// Run the solve-reach-sample loop from an empty table.
    Allocate {
        #[command(flatten)]
        config: Config,
        #[arg(long, default_value = "allocation.json")]
        out: PathBuf,
    },
}

fn write(path: &Path, value: &impl serde::Serialize) -> Outcome {
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

fn by_class<T: Into<Value> + Copy>(values: [T; 6]) -> Value {
    CLASS_NAMES
        .iter()
        .zip(values)
        .map(|(name, v)| ((*name).to_owned(), v.into()))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

type Outcome = Result<(), Box<dyn std::error::Error>>;

fn main() -> Outcome {
    match Cli::parse().command {
        Command::Exact { out } => exact(&out),
        Command::Sample {
            per_pair,
            seed,
            out,
        } => sample(per_pair as u64, seed, &out),
        Command::Solve => {
            let solution = Solver::new(&round_one()).solve(0);
            println!(
                "root value of Team 0 in [{:.6}, {:.6}], max saddle gap {:.2e}, solve {:.1} s on {} threads",
                solution.root_lo,
                solution.root_hi,
                solution.max_gap,
                solution.seconds,
                rayon::current_num_threads()
            );
            Ok(())
        }
        Command::Allocate { config, out } => {
            let mut written = Ok(());
            allocate::run(&config, &round_one(), |report| {
                let it = report
                    .iterations
                    .last()
                    .expect("progress follows an iteration");
                println!(
                    "iteration {}: root in [{:.4}, {:.4}] width {:.4} from {:.2e} samples; solve {:.0} s, reach {:.0} s ({} sequences), sampling {:.0} s",
                    it.iteration,
                    it.root_lo,
                    it.root_hi,
                    it.width,
                    it.samples_before as f64,
                    it.solve_seconds,
                    it.reach_seconds,
                    it.sequences_reached,
                    it.sample_seconds
                );
                if written.is_ok() {
                    written = write(&out, report);
                }
            });
            written
        }
    }
}

fn exact(out: &Path) -> Outcome {
    let clock = Instant::now();
    let kernel = round_one();
    let seconds = clock.elapsed().as_secs_f64();
    let table = HandTable::new(FULL_DECK);
    let entries: Vec<Value> = kernel
        .iter()
        .map(|&((a, b), counts)| {
            let total: u64 = counts.iter().sum();
            let support = (table.count(a.into()), table.count(b.into()));
            assert_eq!(total, support.0 * support.1, "every hand pair is counted once");
            json!({"numbers": [a, b], "support": support, "pairs": total, "counts": by_class(counts),
                   "probabilities_float": by_class(counts.map(|c| c as f64 / total as f64))})
        })
        .collect();
    let pairs: u64 = kernel.iter().flat_map(|(_, c)| c).sum();
    println!(
        "{pairs} hand pairs over {} number pairs in {seconds:.1} s",
        kernel.len()
    );
    println!("P(Team 0 wins), rows Baku, columns Lalo");
    for row in kernel.chunks(5) {
        print!("{:>4} ", row[0].0.0);
        for (_, c) in row {
            print!(
                "{:8.4}",
                (c[W0] + c[W0C]) as f64 / c.iter().sum::<u64>() as f64
            );
        }
        println!();
    }
    let report = json!({"model": "canonical-uniform-v1", "round": 1, "deck": "full", "threads": rayon::current_num_threads(),
                        "wall_seconds": seconds, "total_pairs": pairs, "entries": entries});
    write(out, &report)
}

fn sample(n: u64, seed: u64, out: &Path) -> Outcome {
    let clock = Instant::now();
    let mut worst = 0.0_f64;
    let mut entries = Vec::new();
    for (batch, ((a, b), exact)) in round_one().into_iter().enumerate() {
        let counts = sample_rounds(FULL_DECK, (a, b), seed, batch as u64, n);
        let total = exact.iter().sum::<u64>() as f64;
        let z: [f64; 6] = std::array::from_fn(|c| {
            let p = exact[c] as f64 / total;
            if p == 0.0 || p == 1.0 {
                if counts[c] as f64 == (p * n as f64).round() {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                (counts[c] as f64 - n as f64 * p) / (n as f64 * p * (1.0 - p)).sqrt()
            }
        });
        let largest = z.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        worst = worst.max(largest);
        println!("({a:>2},{b:>2}) max |z| {largest:5.2}");
        entries
            .push(json!({"numbers": [a, b], "n": n, "counts": by_class(counts), "z": by_class(z)}));
    }
    let seconds = clock.elapsed().as_secs_f64();
    println!("max |z| {worst:.3} over 150 cells, {seconds:.0} s wall");
    let report = json!({"gate": 3, "seed": seed, "per_pair": n, "threads": rayon::current_num_threads(), "wall_seconds": seconds,
                        "max_abs_z": worst, "passed": worst < 4.0, "entries": entries});
    write(out, &report)
}
