//! Build, audit and read the certified leap table of Surpassing the Leader.

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use stl_dth::Table as Dth;
use stl_leap::{Error, build, open_leap, reach};

#[derive(Parser)]
#[command(about = "Build, audit and read the certified leap table of Surpassing the Leader")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// A finished artifact and the DTH table of its tail.
#[derive(Args)]
struct Inputs {
    /// The artifact directory that holds `report.json` and `tables/`.
    #[arg(long)]
    artifact: PathBuf,
    /// The DTH table `V.npy`.
    #[arg(long)]
    dth: PathBuf,
}

#[derive(Subcommand)]
enum Command {
    /// Sweep the whole game into an empty directory, audit it and write report.json (~82 GB).
    Build {
        /// The DTH table `V.npy`.
        #[arg(long)]
        dth: PathBuf,
        /// The empty output directory.
        #[arg(long)]
        output: PathBuf,
    },
    /// Audit a finished artifact against fresh reachability, DTH and sampled LP certificates.
    Audit(Inputs),
    /// Solve stored keys again from the artifact's own children and compare the values.
    Resweep {
        #[command(flatten)]
        inputs: Inputs,
        /// A scratch directory for the new tables and the unpacked children.
        #[arg(long)]
        output: PathBuf,
        /// Keys such as `H2_57` or `REV_3000`.
        #[arg(required = true)]
        keys: Vec<String>,
    },
    /// Print one stored value and its fresh stage certificate.
    Read {
        #[command(flatten)]
        inputs: Inputs,
        /// The key, such as `H1_12`.
        key: String,
        /// The Checker's profile index.
        checker: usize,
        /// The Dropper's profile index.
        dropper: usize,
    },
}

fn json(value: &impl serde::Serialize) -> String {
    serde_json::to_string_pretty(value).expect("reports serialize")
}

fn run(command: Command) -> Result<String, Error> {
    Ok(match command {
        Command::Build { dth, output } => json(&build::run_full(&dth, &output)?),
        Command::Audit(Inputs { artifact, dth }) => {
            let reach = reach::forward(3600)?;
            json(&build::audit(
                &artifact.join("tables"),
                &reach,
                &Dth::open(&dth, 0)?,
            )?)
        }
        Command::Resweep {
            inputs,
            output,
            keys,
        } => {
            let dth = Dth::open(&inputs.dth, 0)?;
            let tables = inputs.artifact.join("tables");
            let mut lines = Vec::new();
            for key in keys {
                let line = json(&build::resweep(&tables, &dth, &output, key.parse()?)?);
                eprintln!("{line}");
                lines.push(line);
            }
            lines.join("\n")
        }
        Command::Read {
            inputs,
            key,
            checker,
            dropper,
        } => json(&open_leap(&inputs.artifact, &inputs.dth)?.stage(
            key.parse()?,
            checker,
            dropper,
        )?),
    })
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("stl-leap: {error}");
            ExitCode::FAILURE
        }
    }
}
