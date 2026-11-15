//! The full build: reachability, the backward sweep of every key, the audit and the report.

use crate::Error;
use crate::kernel::{self, Children, Residue, Table, kind};
use crate::lp;
use crate::reach::{self, Bits};
use crate::rules::{ACTIONS as A, Key, N, profiles};
use crate::store::{Packed, PackedReader, Store};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;
use stl_dth::Table as Dth;

/// The reachable states of each key type in the complete game.
pub(crate) const TOTALS: [(&str, u64); 3] = [
    ("H1", 4_180_634_990),
    ("H2", 4_188_677_815),
    ("REV", 1_084_020_312),
];
/// The reachable states of the leap-window keys.
const WINDOW_STATES: u64 = 470_336_555;
/// The kernel solves at most about this many classes per call, as the reference batches them.
const BATCH: usize = 1 << 20;

/// What the sweep of one key did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Record {
    /// The key, for example `H2_57`.
    pub key: String,
    /// Its start clock.
    pub clock: u32,
    /// Reachable states solved.
    pub states: u64,
    /// States that both closed-form rungs missed.
    pub failures: u64,
    /// Residue states the packing LP certified.
    pub packing_solves: u64,
    /// Residue states the kink seed certified.
    pub kink_solves: u64,
    /// Residue states `HiGHS` certified.
    pub lp_solves: u64,
    /// Seconds in the stage kernel.
    pub kernel_seconds: f64,
    /// Seconds in `HiGHS`.
    pub lp_seconds: f64,
    /// Seconds for the whole key, writes included.
    pub sweep_seconds: f64,
}

/// Group the reachable checkers of `key` by the child of a failed check, in order of first
/// appearance; a checker whom a failed check kills has no child.
fn failure_groups(key: Key, reach: &Bits) -> Vec<(Option<Key>, Vec<u32>)> {
    let p = profiles();
    let mut groups: Vec<(Option<Key>, Vec<u32>)> = Vec::new();
    for checker in (0..N).filter(|&pc| reach.columns(pc).next().is_some()) {
        let child = (p.rev[checker] > 0.0)
            .then(|| key.child(u32::from(p.st[checker]) + 60))
            .flatten();
        match groups.iter_mut().find(|(c, _)| *c == child) {
            Some((_, checkers)) => checkers.push(checker as u32),
            None => groups.push((child, vec![checker as u32])),
        }
    }
    groups
}

/// Solve every reachable state of `key` from its stored children and write its dense table.
///
/// # Errors
///
/// The store's errors, kernel errors for malformed children, and [`Error::Uncertified`] when
/// `HiGHS` cannot certify a residue stage.
pub fn sweep_key(key: Key, reach: &Bits, store: &mut Store, dth: &Dth) -> Result<Record, Error> {
    let started = Instant::now();
    let p = profiles();
    let mut table = store.create(key)?;
    let success_map = key.child(0).map(|child| store.load(child)).transpose()?;
    let success = Table {
        values: success_map.as_ref().map_or(dth.values(), |t| t.data()),
        width: N + 1,
    };
    let reset_columns: Vec<u32> = p
        .fail
        .iter()
        .map(|&c| p.reset_column(c as usize).map_or(u32::MAX, |c| c as u32))
        .collect();
    let mut record = Record {
        key: key.to_string(),
        clock: key.clock(),
        ..Record::default()
    };
    let (cols, width) = (key.columns(), key.columns() + 1);
    // REV keys keep the reference's plain packing path; H keys also try kink seeds.
    let residue = if matches!(key, Key::Rev(_)) {
        Residue::Packing
    } else {
        Residue::Kink
    };
    for (child, checkers) in &failure_groups(key, reach) {
        let failure_map = child.map(|child| store.load(child)).transpose()?;
        let failure_columns = if matches!(child, Some(Key::Rev(_))) {
            &reset_columns
        } else {
            &p.fail
        };
        let children = Children {
            success,
            failure: Table {
                values: failure_map.as_ref().map_or(dth.values(), |t| t.data()),
                width: child.map_or(N, Key::columns) + 1,
            },
            success_columns: &p.succ,
            failure_columns,
            revival: &p.rev,
        };
        let columns_per_batch = (BATCH / checkers.len()).min(cols).max(64);
        for start in (0..cols).step_by(columns_per_batch) {
            let classes: Vec<(u32, u32)> = (start..cols.min(start + columns_per_batch))
                .flat_map(|col| {
                    checkers
                        .iter()
                        .filter(move |&&pc| reach.get(pc as usize, col))
                        .map(move |&pc| (pc, col as u32))
                })
                .collect();
            if classes.is_empty() {
                continue;
            }
            let tick = Instant::now();
            let droppers: Vec<(u32, u32)> = classes
                .iter()
                .map(|&(pc, col)| (pc, key.dropper(col as usize) as u32))
                .collect();
            let mut solved = kernel::sweep(&droppers, &children, key.is_window(), residue)?;
            record.kernel_seconds += tick.elapsed().as_secs_f64();
            let tick = Instant::now();
            let misses: Vec<usize> = (0..solved.len())
                .filter(|&i| solved[i].kind == kind::UNSOLVED)
                .collect();
            let values = misses
                .par_iter()
                .map(|&i| {
                    let (s, f) = children.stage(droppers[i]);
                    lp::certify(&s, f, key.is_window()).map(|c| c.value)
                })
                .collect::<Result<Vec<f64>, Error>>()?;
            for (&i, value) in misses.iter().zip(values) {
                solved[i].value = value;
            }
            record.lp_seconds += tick.elapsed().as_secs_f64();
            let dense = table.data::<f64>();
            for (&(pc, col), result) in classes.iter().zip(&solved) {
                record.failures += u64::from(result.kind >= kind::PACKING);
                record.packing_solves += u64::from(result.kind == kind::PACKING);
                record.kink_solves += u64::from(result.kind == kind::KINK);
                record.lp_solves += u64::from(result.kind == kind::UNSOLVED);
                dense[pc as usize * width + col as usize] = result.value;
            }
            record.states += classes.len() as u64;
        }
    }
    table.flush()?;
    if record.states != reach.count() {
        return Err(Error::Invalid(format!(
            "{key}: the sweep missed reachable states"
        )));
    }
    record.sweep_seconds = started.elapsed().as_secs_f64();
    Ok(record)
}

/// A simple deterministic generator for audit samples (`SplitMix64`).
struct Samples(u64);

impl Samples {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }

    fn distinct(&mut self, count: usize, n: usize) -> Vec<usize> {
        let mut picked = Vec::with_capacity(count.min(n));
        while picked.len() < count.min(n) {
            let x = self.below(n);
            if !picked.contains(&x) {
                picked.push(x);
            }
        }
        picked
    }
}

/// Return the success payoffs and failure payoff of a stored stage, read from the packed children
/// and the DTH tail.
///
/// # Errors
///
/// The reader's errors when a child state is missing.
pub(crate) fn stage_inputs(
    reader: &PackedReader,
    key: Key,
    pc: usize,
    pd: usize,
    dth: &Dth,
) -> Result<([f64; A], f64), Error> {
    let p = profiles();
    let child = |dest: Option<Key>, column: usize| match dest {
        None => Ok(dth.values()[pd * (N + 1) + column]),
        Some(Key::Rev(clock)) => reader.get(
            Key::Rev(clock),
            pd,
            p.reset_column(column)
                .expect("revival children have reset columns"),
        ),
        Some(dest) => reader.get(dest, pd, column),
    };
    let mut s = [0.0; A];
    for (x, &column) in s.iter_mut().zip(&p.succ[pc]) {
        *x = -child(key.child(0), column as usize)?;
    }
    let rev = p.rev[pc];
    let f = if rev > 0.0 {
        rev * -child(key.child(u32::from(p.st[pc]) + 60), p.fail[pc] as usize)? + (1.0 - rev)
    } else {
        1.0
    };
    Ok((s, f))
}

/// Check every packed table against reachability and DTH, re-derive sampled stages with the
/// audit LP, and certify the opening.
///
/// The result holds `states_checked`, `lp_samples`, `max_bellman_residual`, `max_lp_gap`,
/// `changed_states` (states where the leap gains Baku over `1e-8` against DTH), the mean, least
/// and largest gain, and `root`: the opening's certificate, stored value, win probabilities, DTH
/// value and Baku's gain in percentage points.
///
/// # Errors
///
/// [`Error::Invalid`] for a packed table that disagrees with reachability or holds a bad value,
/// [`Error::Uncertified`] for a sample whose stored value misses its certificate by over `1e-6`.
pub fn audit(tables: &Path, reach: &BTreeMap<Key, Bits>, dth: &Dth) -> Result<Value, Error> {
    let reader = PackedReader::new(tables);
    let p = profiles();
    let keys: Vec<(&Key, &Bits)> = reach.iter().collect();
    let gains = keys
        .par_iter()
        .map(|&(&key, bits)| audit_key(&reader, key, bits, dth))
        .collect::<Result<Vec<_>, Error>>()?;
    let mut samples = Samples(93_482);
    let (mut lp_samples, mut max_bellman_residual, mut max_lp_gap) = (0, 0.0_f64, 0.0_f64);
    for &(&key, bits) in &keys {
        let table = reader.table(key)?;
        let count = if key == Key::H2(57) { 200 } else { 2 };
        for rank in samples.distinct(count, bits.count() as usize) {
            let (pc, col) = table.select(rank);
            let pd = key.dropper(col);
            let (s, f) = stage_inputs(&reader, key, pc, pd, dth)?;
            let certificate = lp::certify(&s, f, key.is_window())?;
            let residual = (table.get(pc, col)? - certificate.value).abs();
            if residual > kernel::GAP {
                return Err(Error::Uncertified(format!(
                    "Bellman audit failed: {key} {pc} {pd}: {residual}"
                )));
            }
            max_bellman_residual = max_bellman_residual.max(residual);
            max_lp_gap = max_lp_gap.max(certificate.gap);
            lp_samples += 1;
        }
    }
    let states_checked: u64 = gains.iter().map(|g| g.states).sum();
    let (s, f) = stage_inputs(&reader, Key::H1(12), 0, 0, dth)?;
    let opening = lp::certify(&s, f, false)?;
    let stored = reader.get(Key::H1(12), 0, 0)?;
    let baseline = dth.values()[0];
    let dth_s: [f64; A] = std::array::from_fn(|k| -dth.values()[p.succ[0][k] as usize]);
    let dth_f = p.rev[0] * -dth.values()[p.fail[0] as usize] + (1.0 - p.rev[0]);
    Ok(json!({
        "states_checked": states_checked,
        "lp_samples": lp_samples,
        "max_bellman_residual": max_bellman_residual,
        "max_lp_gap": max_lp_gap,
        "changed_states": gains.iter().map(|g| g.changed).sum::<u64>(),
        "mean_baku_gain_over_state_classes":
            gains.iter().map(|g| g.sum).sum::<f64>() / states_checked as f64,
        "min_baku_gain": gains.iter().map(|g| g.min).fold(f64::INFINITY, f64::min),
        "max_baku_gain": gains.iter().map(|g| g.max).fold(f64::NEG_INFINITY, f64::max),
        "root": {
            "value": opening.value,
            "lower": opening.lower,
            "upper": opening.upper,
            "gap": opening.gap,
            "drop": opening.drop,
            "check": opening.check,
            "stored_value": stored,
            "hal_win_probability": f64::midpoint(1.0, stored),
            "baku_win_probability": f64::midpoint(1.0, -stored),
            "dth_value": baseline,
            "baku_gain_probability_points": 50.0 * (baseline - stored),
            "success_payoffs": s.to_vec(),
            "failure_payoff": f,
            "dth_policies": lp::certify(&dth_s, dth_f, false)?,
        },
    }))
}

/// Baku's gain over DTH across one key's states.
struct Gain {
    states: u64,
    changed: u64,
    sum: f64,
    min: f64,
    max: f64,
}

fn audit_key(reader: &PackedReader, key: Key, reach: &Bits, dth: &Dth) -> Result<Gain, Error> {
    let table = reader.table(key)?;
    let bad = |reason: &str| Err(Error::Invalid(format!("{key}: {reason}")));
    if !table.matches(reach) || table.values().len() as u64 != reach.count() {
        return bad("packed reachability mismatch");
    }
    let mut gain = Gain {
        states: 0,
        changed: 0,
        sum: 0.0,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };
    for (position, &value) in table.positions().zip(table.values()) {
        if !value.is_finite() {
            return bad("packed values must be finite");
        }
        if value.abs() > 1.0 + 1e-9 {
            return bad("packed value outside utility range");
        }
        let (row, col) = (position / key.columns(), position % key.columns());
        let baseline = dth.values()[row * (N + 1) + key.dropper(col)];
        // Values are from the Dropper's side; Baku drops in H2 and after a Hal revival.
        let g = if matches!(key, Key::H1(_)) {
            baseline - value
        } else {
            value - baseline
        };
        gain.states += 1;
        gain.changed += u64::from(g > 1e-8);
        gain.sum += g;
        gain.min = gain.min.min(g);
        gain.max = gain.max.max(g);
    }
    // Accumulated child error can make a gain slightly negative; each stage still meets 1e-6.
    if gain.min < -1.3e-5 {
        return bad(&format!(
            "Baku action-set monotonicity failed: {}",
            gain.min
        ));
    }
    Ok(gain)
}

/// Build the complete leap table from the DTH table `dth` into the empty directory `output`,
/// audit it, and write `output/report.json`: the [`audit`] fields, the reachable states per key
/// type under `rows`, and the sweep totals of [`Record`].
///
/// # Errors
///
/// [`Error::Invalid`] for a nonempty output directory or a coverage mismatch, and the errors of
/// the sweep and the audit.
///
/// # Panics
///
/// Only if a record or the report fails to serialize; they hold numbers and strings alone.
pub fn run_full(dth: &Path, output: &Path) -> Result<Value, Error> {
    let started = Instant::now();
    if output
        .read_dir()
        .is_ok_and(|mut entries| entries.next().is_some())
    {
        return Err(Error::Invalid(format!(
            "{}: leap builds require an empty output directory",
            output.display()
        )));
    }
    let dth = Dth::open(dth, 0)?;
    if dth.values().chunks(N + 1).any(|row| row[N] != -1.0) {
        return Err(Error::Invalid("DTH table sentinel mismatch".into()));
    }
    let tables = output.join("tables");
    let mut store = Store::new(&tables, &tables)?;
    let reach = reach::forward(3600)?;
    let states = |kind: &str| {
        reach
            .iter()
            .filter(|(k, _)| k.kind() == kind)
            .map(|(_, b)| b.count())
            .sum::<u64>()
    };
    let window: u64 = reach
        .iter()
        .filter(|(k, _)| k.is_window())
        .map(|(_, b)| b.count())
        .sum();
    if TOTALS.iter().any(|&(kind, total)| states(kind) != total) || window != WINDOW_STATES {
        return Err(Error::Invalid("full reachability count mismatch".into()));
    }
    let mut records = Vec::with_capacity(reach.len());
    let mut order: Vec<Key> = reach.keys().copied().collect();
    order.sort_by_key(|key| std::cmp::Reverse(key.clock()));
    for key in order {
        let record = sweep_key(key, &reach[&key], &mut store, &dth)?;
        store.pack_after(key.clock() + 540)?;
        eprintln!(
            "{}",
            serde_json::to_string(&record).expect("a record serializes")
        );
        records.push(record);
    }
    store.pack_after(0)?;
    let total: u64 = TOTALS.iter().map(|&(_, n)| n).sum();
    if records.iter().map(|r| r.states).sum::<u64>() != total {
        return Err(Error::Invalid("full sweep coverage mismatch".into()));
    }
    let audit_started = Instant::now();
    let mut report = audit(&tables, &reach, &dth)?;
    let sum = |field: fn(&Record) -> f64| records.iter().map(field).sum::<f64>();
    let count = |field: fn(&Record) -> u64| records.iter().map(field).sum::<u64>();
    let rows: BTreeMap<&str, Value> = TOTALS
        .iter()
        .map(|&(kind, n)| (kind, json!({"states": n})))
        .collect();
    let totals = json!({
        "rows": rows,
        "states": total,
        "keys": records.len(),
        "failures": count(|r| r.failures),
        "packing_solves": count(|r| r.packing_solves),
        "kink_solves": count(|r| r.kink_solves),
        "lp_solves": count(|r| r.lp_solves),
        "kernel_seconds": sum(|r| r.kernel_seconds),
        "lp_seconds": sum(|r| r.lp_seconds),
        "audit_seconds": audit_started.elapsed().as_secs_f64(),
        "wall_seconds": started.elapsed().as_secs_f64(),
    });
    if let (Value::Object(report), Value::Object(totals)) = (&mut report, totals) {
        report.extend(totals);
    }
    let path = output.join("report.json");
    let text = serde_json::to_string_pretty(&report).expect("a report serializes") + "\n";
    std::fs::write(&path, text).map_err(crate::io(&path))?;
    Ok(report)
}

/// A stored key solved again from the same stored children.
#[derive(Clone, Debug, Serialize)]
pub struct Resweep {
    /// The sweep of the key.
    #[serde(flatten)]
    pub record: Record,
    /// States whose new value equals the stored value bit for bit.
    pub identical: u64,
    /// The largest distance between a new and a stored value.
    pub max_abs_difference: f64,
}

/// Solve `key` again from the children stored in the packed directory `tables`, pack it into
/// `output`, and compare it with the stored table. Children are unpacked into `output`.
///
/// # Errors
///
/// [`Error::Invalid`] when `output` is `tables`, and the errors of the store and the sweep.
pub fn resweep(tables: &Path, dth: &Dth, output: &Path, key: Key) -> Result<Resweep, Error> {
    let stored = Packed::open(tables, key)?;
    let mut store = Store::new(output, tables)?;
    if tables.canonicalize().ok() == output.canonicalize().ok() {
        return Err(Error::Invalid(
            "a re-sweep must not write into the artifact it reads".into(),
        ));
    }
    let record = sweep_key(
        key,
        &Bits::from_flat(key.columns(), stored.flat()),
        &mut store,
        dth,
    )?;
    store.pack(key)?;
    let fresh = Packed::open(output, key)?;
    let pairs = fresh.values().iter().zip(stored.values());
    let identical = pairs
        .clone()
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count() as u64;
    let max_abs_difference = pairs.map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    Ok(Resweep {
        record,
        identical,
        max_abs_difference,
    })
}
