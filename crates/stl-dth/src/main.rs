//! Build, verify and read the certified DTH table.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use stl_dth::{ANCHORS, Error, Profiles, State, Table, build, finalize};

#[derive(Parser)]
#[command(about = "The certified table of pure Drop the Handkerchief")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// The table file.
    #[arg(long, global = true, default_value = "artifacts/dth/V.npy")]
    table: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    /// Solve every layer, audit the result, write the table, and check its anchors.
    Build,
    /// Open the table with strided rechecks and print the anchors.
    Verify {
        /// Classes to re-derive from their children.
        #[arg(long, default_value_t = stl_dth::RECHECK_SAMPLES)]
        rechecks: usize,
    },
    /// Print the certified stage game of a state.
    Stage {
        /// `s_c t_c s_d t_d`, Checker first.
        #[arg(num_args = 4, required = true)]
        state: Vec<usize>,
    },
}

fn run(cli: Cli) -> Result<(), Error> {
    let start = Instant::now();
    match cli.command {
        Command::Build => {
            let profiles = Profiles::build();
            let (values, [pure, support, lp]) = build(&profiles, 0, |layer, classes| {
                if layer % 100 == 0 {
                    println!(
                        "layer {layer:4}: {classes:8} classes | {:6.1} s",
                        start.elapsed().as_secs_f64()
                    );
                }
            })?;
            println!(
                "solved in {:.1} s: pure={pure} support={support} lp={lp}",
                start.elapsed().as_secs_f64()
            );
            let worst = finalize(&values, &profiles)?;
            println!(
                "finalize: {} rechecks ok, worst |diff| = {worst:.3e}",
                stl_dth::RECHECK_SAMPLES
            );
            values.write(&cli.table)?;
            print_anchors(&Table::open(&cli.table, 0)?)?;
        }
        Command::Verify { rechecks } => print_anchors(&Table::open(&cli.table, rechecks)?)?,
        Command::Stage { state } => {
            let state: State = state.try_into().expect("clap passes four numbers");
            let stage = Table::open(&cli.table, 0)?.stage(state)?;
            println!(
                "state {state:?}: value {} gap {:.3e}",
                stage.value, stage.gap
            );
            for (who, mix) in [("drop", stage.drop), ("check", stage.check)] {
                let support: Vec<String> = (1..)
                    .zip(mix)
                    .filter(|&(_, w)| w > 0.0)
                    .map(|(s, w)| format!("{s}:{w:.6}"))
                    .collect();
                println!("{who:5} {}", support.join(" "));
            }
        }
    }
    println!("done in {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}

fn print_anchors(table: &Table) -> Result<(), Error> {
    for (state, expected) in ANCHORS {
        println!(
            "anchor {state:?}: expected {expected:.16} got {:.16}",
            table.value(state)?
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
