//! Solve, certify and study `D_K` from the command line.
//!
//! Each step reads and writes artifacts in `--dir` (default `artifacts/tok`):
//!
//! - `solve-lp K` and `solve-cfr K` write the strategy tables `lp_K_p0.tbl`, `lp_K_p1.tbl` or
//!   `cfr_K_p0.tbl`, `cfr_K_p1.tbl` (format in `tok::table`).
//! - `certify K SOURCE` reads `SOURCE_K_p*.tbl` and writes `SOURCE_K_certificate.json`.
//! - `abstraction K SOURCE PLAYER KEY` reads the tables and the certificate of SOURCE and writes
//!   `abstraction_K_pPLAYER_KEY.tbl` and `.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use serde_json::json;
use tok::abstraction::{self, Key};
use tok::cfr::CfrPlus;
use tok::rules::Player;
use tok::table::{self, StrategyTable};
use tok::traversal::saddle_gap;
use tok::{Error, certificate, lp, tree};

#[derive(Parser)]
#[command(about = "Solve and certify Tower of Karma (D_K)")]
struct Cli {
    /// Directory of the artifacts.
    #[arg(long, default_value = "artifacts/tok")]
    dir: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solve the sequence-form LP with `HiGHS`.
    SolveLp {
        /// Turns per player.
        k: usize,
        /// `HiGHS` solver: choose, simplex or ipm.
        #[arg(long, default_value = "choose")]
        solver: String,
        /// Time limit in seconds.
        #[arg(long, default_value_t = 3600.0)]
        seconds: f64,
    },
    /// Run CFR+ until the saddle gap reaches the target or time runs out.
    SolveCfr {
        /// Turns per player.
        k: usize,
        /// Target saddle gap.
        #[arg(long, default_value_t = 1e-6)]
        target: f64,
        /// Time limit in seconds.
        #[arg(long, default_value_t = 3600.0)]
        seconds: f64,
        /// Iterations between gap checks; the tables are saved at each check.
        #[arg(long, default_value_t = 250)]
        every: usize,
    },
    /// Certify a solved profile with best responses through the rules engine.
    Certify {
        /// Turns per player.
        k: usize,
        /// lp or cfr.
        source: String,
    },
    /// Key one player's last layer coarsely, re-solve, and certify the guarantee.
    Abstraction {
        /// Turns per player.
        k: usize,
        /// The certified profile to start from: lp or cfr.
        source: String,
        /// The player whose last layer is keyed: 0 or 1.
        player: usize,
        /// The key.
        key: Key,
        /// CFR+ iterations per step.
        #[arg(long, default_value_t = 2000)]
        iterations: usize,
        /// Rounds of merge and earlier-layers steps.
        #[arg(long, default_value_t = 3)]
        rounds: usize,
    },
}

fn tables(dir: &Path, name: &str) -> Result<[StrategyTable; 2], Error> {
    Ok([
        StrategyTable::load(&dir.join(format!("{name}_p0.tbl")))?,
        StrategyTable::load(&dir.join(format!("{name}_p1.tbl")))?,
    ])
}

fn save(
    dir: &Path,
    name: &str,
    layers: &[tree::Layer],
    profile: &tok::traversal::Profile,
) -> Result<(), Error> {
    for p in 0..2 {
        table::export(layers, profile, p)?.save(&dir.join(format!("{name}_p{p}.tbl")))?;
    }
    Ok(())
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Error> {
    let text = serde_json::to_string_pretty(value).expect("JSON values serialize");
    Ok(std::fs::write(path, text + "\n")?)
}

/// Key player i's last layer of `D_k` by `kind`, alternate the two CFR+ steps from the certified
/// profile `source`, and certify the best lifted strategy.
fn abstraction(
    dir: &Path,
    k: usize,
    source: &str,
    i: usize,
    kind: Key,
    rounds: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let layers = tree::build(k)?;
    let [p0, p1] = tables(dir, &format!("{source}_{k}"))?;
    let profile = table::import(&layers, [&p0, &p1])?;
    drop((p0, p1));
    let path = dir.join(format!("{source}_{k}_certificate.json"));
    let cert: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let bound = |name: &str| {
        cert[name]
            .as_f64()
            .ok_or_else(|| Error::Artifact(format!("{} lacks {name}", path.display())))
    };
    let (lower, upper) = (bound("lower")?, bound("upper")?);
    let (v_lo, v_hi) = if i == 0 {
        (lower, upper)
    } else {
        (-upper, -lower)
    };
    let (key, keys) = abstraction::slot_keys(&layers, i, kind);
    let (full, slots) = abstraction::slot_keys(&layers, i, Key::Full);
    let reference = abstraction::merged_best_response(&layers, i, &profile, &full, slots);
    let cap = abstraction::merged_best_response(&layers, i, &profile, &key, keys);
    let tag = format!("K={k} P{i} key={kind:?}");
    println!(
        "{tag} infosets={slots} keys={keys} value=[{v_lo:+.9},{v_hi:+.9}] one_sided_full={reference:+.9} one_sided_key={cap:+.9} round1_loss>={:+.3e}",
        v_lo - cap
    );
    let t = Instant::now();
    let log = |step: &abstraction::Step| {
        println!(
            "{tag} {} guarantee={:+.9} loss={:+.3e} cap={:+.9} t={:.0}s",
            step.name,
            step.guarantee,
            v_hi - step.guarantee,
            step.cap.unwrap_or(f64::NAN),
            start.elapsed().as_secs_f64()
        );
    };
    let (steps, best) =
        abstraction::alternate(&layers, i, &profile, &key, keys, (rounds, iterations), log);
    let cfr_seconds = t.elapsed().as_secs_f64();
    let name = format!("abstraction_{k}_p{i}_{kind:?}").to_lowercase();
    let lifted = table::export(&layers, &best, i)?;
    lifted.save(&dir.join(format!("{name}.tbl")))?;
    drop((layers, profile, best));
    let t = Instant::now();
    let responder = if i == 0 { Player::P1 } else { Player::P0 };
    let certified = -certificate::best_response_value(k, responder, &lifted)?;
    let certificate_seconds = t.elapsed().as_secs_f64();
    println!(
        "{tag} certified_guarantee={certified:+.9} loss in [{:+.3e}, {:+.3e}] cfr_seconds={cfr_seconds:.0} certificate_seconds={certificate_seconds:.0}",
        v_lo - certified,
        v_hi - certified
    );
    let record = json!({
        "K": k, "player": i, "key": format!("{kind:?}"), "infosets": slots, "keys": keys,
        "value": [v_lo, v_hi], "one_sided_key": cap, "one_sided_full": reference,
        "round1_loss_lower_bound": v_lo - cap, "steps": steps, "certified_guarantee": certified,
        "loss": [v_lo - certified, v_hi - certified], "iterations": iterations, "rounds": rounds,
        "cfr_seconds": cfr_seconds, "certificate_seconds": certificate_seconds,
    });
    write_json(&dir.join(format!("{name}.json")), &record)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let dir = cli.dir.as_path();
    std::fs::create_dir_all(dir)?;
    let start = Instant::now();
    match cli.command {
        Command::SolveLp { k, solver, seconds } => {
            let layers = tree::build(k)?;
            let form = lp::build(&layers);
            println!("K={k} built in {:.1}s", start.elapsed().as_secs_f64());
            let t = Instant::now();
            let sol = lp::solve(&form, &solver, seconds)?;
            println!(
                "value={:+.12} rows={} columns={} nonzeros={} lp_seconds={:.1}",
                sol.value,
                sol.rows,
                sol.columns,
                sol.nonzeros,
                t.elapsed().as_secs_f64()
            );
            let profile = lp::plan_profile(&form, &layers, &sol.plans);
            save(dir, &format!("lp_{k}"), &layers, &profile)?;
        }
        Command::SolveCfr {
            k,
            target,
            seconds,
            every,
        } => {
            let layers = tree::build(k)?;
            let mut cfr = CfrPlus::new(&layers);
            let mut solve_seconds = 0.0;
            loop {
                let t = Instant::now();
                (0..every).for_each(|_| cfr.iterate());
                solve_seconds += t.elapsed().as_secs_f64();
                let average = cfr.average();
                let (lower, upper) = saddle_gap(&layers, &average);
                println!(
                    "iterations={} lower={lower:+.12} upper={upper:+.12} gap={:.3e} cfr_seconds={solve_seconds:.0} wall={:.0}",
                    cfr.iterations,
                    upper - lower,
                    start.elapsed().as_secs_f64()
                );
                save(dir, &format!("cfr_{k}"), &layers, &average)?;
                if upper - lower <= target || start.elapsed().as_secs_f64() > seconds {
                    break;
                }
            }
        }
        Command::Certify { k, source } => {
            let [p0, p1] = tables(dir, &format!("{source}_{k}"))?;
            let t = Instant::now();
            let (lower, upper) = certificate::certify(k, &p0, &p1)?;
            let seconds = t.elapsed().as_secs_f64();
            println!(
                "K={k} source={source} lower={lower:+.12} upper={upper:+.12} gap={:.3e} certificate_seconds={seconds:.1}",
                upper - lower
            );
            let record = json!({"lower": lower, "upper": upper, "seconds": seconds});
            write_json(&dir.join(format!("{source}_{k}_certificate.json")), &record)?;
        }
        Command::Abstraction {
            k,
            source,
            player,
            key,
            iterations,
            rounds,
        } => {
            abstraction(dir, k, &source, player, key, rounds, iterations)?;
        }
    }
    Ok(())
}
