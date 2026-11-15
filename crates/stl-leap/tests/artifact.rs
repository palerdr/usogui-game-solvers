//! Parity with the Python-built artifact and the DTH table. Every test here needs
//! `~/Desktop/projects/stl/artifacts/leap` (74 GB, built by stl at 55e342e's predecessor) and
//! `~/Desktop/projects/stl/artifacts/dth/V.npy`; run them with `cargo test -- --ignored`.

use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use stl_dth::Table as Dth;
use stl_leap::build::{audit, resweep};
use stl_leap::kernel::{Children, Residue, Table, kind, sweep};
use stl_leap::lp::{certify, stage_matrix};
use stl_leap::open_leap;
use stl_leap::reach::forward;
use stl_leap::rules::{ACTIONS as A, Key, N, WIN, profile_index, profiles};

fn artifacts() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts"))
}

fn dth() -> Dth {
    Dth::open(&artifacts().join("dth/V.npy"), 0).expect("the DTH table")
}

fn report() -> Value {
    let text = std::fs::read_to_string(artifacts().join("leap/report.json")).expect("the report");
    serde_json::from_str(&text).expect("a JSON report")
}

#[test]
#[ignore = "reads the leap build at artifacts/leap (74 GB) and artifacts/dth/V.npy"]
fn audit_reproduces_the_report() {
    let reach = forward(3600).expect("reachability");
    let audit = audit(&artifacts().join("leap/tables"), &reach, &dth()).expect("a passing audit");
    let report = report();
    // Coverage, membership and the gains over DTH are exact; the mean differs only in the order of
    // its 9.45e9-term sum. The LP samples are drawn independently, so only their residual bound
    // carries over.
    assert_eq!(audit["states_checked"], 9_453_333_117_u64);
    assert_eq!(audit["lp_samples"], report["lp_audit_samples"]);
    for name in ["changed_states", "min_baku_gain", "max_baku_gain"] {
        assert_eq!(audit[name], report[name], "{name}");
    }
    let number = |value: &Value| value.as_f64().expect("a number");
    let mean = "mean_baku_gain_over_state_classes";
    assert!((number(&audit[mean]) - number(&report[mean])).abs() < 1e-12);
    assert!(number(&audit["max_bellman_residual"]) <= 1e-6);
    let (root, stored) = (&audit["root"], &report["root"]);
    for name in [
        "stored_value",
        "dth_value",
        "failure_payoff",
        "baku_gain_probability_points",
        "success_payoffs",
    ] {
        assert_eq!(root[name], stored[name], "{name}");
    }
    assert!((number(&root["value"]) - number(&stored["value"])).abs() < 1e-12);
}

#[test]
#[ignore = "reads the leap build at artifacts/leap (74 GB) and artifacts/dth/V.npy"]
fn reader_recertifies_the_figure_3_records() {
    let artifacts = artifacts();
    let table = open_leap(&artifacts.join("leap"), &artifacts.join("dth/V.npy"))
        .expect("a complete artifact");
    let root = table.stage(Key::H1(12), 0, 0).expect("a certified opening");
    assert_eq!(
        root.stored_value,
        report()["root"]["stored_value"].as_f64().expect("a value")
    );
    let checker = profile_index(60, 120).expect("a profile");
    let dropper = profile_index(0, 180).expect("a profile");
    for minute in 44..60 {
        let record = table
            .stage(Key::H1(minute), checker, dropper)
            .expect("a certified record");
        assert!(record.bellman_residual <= 1e-6 && record.certificate.gap <= 1e-6);
        let mixtures = [&record.certificate.drop, &record.certificate.check];
        assert!(
            mixtures
                .iter()
                .all(|m| m.len() == A && (m.iter().sum::<f64>() - 1.0).abs() < 1e-9)
        );
    }
}

#[test]
#[ignore = "reads the leap build at artifacts/leap (74 GB) and artifacts/dth/V.npy"]
fn reader_maps_revival_columns_and_certifies_window_stages() {
    let artifacts = artifacts();
    let table = open_leap(&artifacts.join("leap"), &artifacts.join("dth/V.npy"))
        .expect("a complete artifact");
    let p = profiles();
    for key in [Key::Rev(3600), Key::H2(58), Key::H1(12)] {
        let droppers: Vec<usize> = if matches!(key, Key::Rev(_)) {
            p.resets.iter().map(|&r| r as usize).collect()
        } else {
            (0..N).collect()
        };
        let (checker, dropper) = (0..N)
            .flat_map(|c| droppers.iter().map(move |&d| (c, d)))
            .find(|&(c, d)| table.value(key, c, d).is_ok())
            .expect("a reachable state");
        let record = table
            .stage(key, checker, dropper)
            .expect("a certified stage");
        assert_eq!(
            record.certificate.drop.len(),
            A + usize::from(key.is_window())
        );
        assert!(record.bellman_residual <= 1e-6);
        assert_eq!(table.value(key, checker, WIN).ok(), Some(-1.0));
    }
}

#[test]
#[ignore = "reads ~/Desktop/projects/stl/artifacts/leap (74 GB) and writes 2.3 GB to the temp directory"]
fn resweeps_reproduce_stored_window_keys_bit_for_bit() {
    let output = std::env::temp_dir().join(format!("stl-leap-resweep-{}", std::process::id()));
    let dth = dth();
    for key in [Key::H2(57), Key::Rev(3600)] {
        let result =
            resweep(&artifacts().join("leap/tables"), &dth, &output, key).expect("a resweep");
        assert_eq!(result.identical, result.record.states, "{key}");
    }
    std::fs::remove_dir_all(&output).expect("cleanup");
}

#[test]
#[ignore = "reads ~/Desktop/projects/stl/artifacts/dth/V.npy"]
fn kernel_reproduces_dth() {
    let dth = dth();
    let p = profiles();
    let children = Children {
        success: Table {
            values: dth.values(),
            width: N + 1,
        },
        failure: Table {
            values: dth.values(),
            width: N + 1,
        },
        success_columns: &p.succ,
        failure_columns: &p.fail,
        revival: &p.rev,
    };
    let mut worst = 0.0_f64;
    for block in (0..N as u32).step_by(64) {
        let classes: Vec<(u32, u32)> = (block..(block + 64).min(N as u32))
            .flat_map(|pd| (0..N as u32).map(move |pc| (pc, pd)))
            .collect();
        let solved = sweep(&classes, &children, false, Residue::Defer).expect("valid DTH children");
        for (&(pc, pd), result) in classes.iter().zip(&solved) {
            assert_ne!(result.kind, kind::UNSOLVED, "({pc}, {pd})");
            worst =
                worst.max((result.value - dth.values()[pc as usize * (N + 1) + pd as usize]).abs());
        }
    }
    assert!(worst <= 1e-9, "worst {worst}");
}

/// The value of `(pc, pd)` at `key`, by recursion to the DTH tail with the kernel and `HiGHS`.
fn evaluate(
    key: Option<Key>,
    pc: usize,
    pd: usize,
    dth: &Dth,
    memo: &mut HashMap<(Key, usize, usize), f64>,
) -> f64 {
    let Some(key) = key else {
        return dth.values()[pc * (N + 1) + pd];
    };
    if let Some(&value) = memo.get(&(key, pc, pd)) {
        return value;
    }
    let p = profiles();
    let s: [f64; A] = std::array::from_fn(|k| match p.succ[pc][k] as usize {
        WIN => 1.0,
        child => -evaluate(key.child(0), pd, child, dth, memo),
    });
    let rev = p.rev[pc];
    let f = if rev > 0.0 {
        rev * -evaluate(
            key.child(u32::from(p.st[pc]) + 60),
            pd,
            p.fail[pc] as usize,
            dth,
            memo,
        ) + (1.0 - rev)
    } else {
        1.0
    };
    let value = solve(&s, f, key.is_window());
    memo.insert((key, pc, pd), value);
    value
}

fn solve(s: &[f64; A], f: f64, window: bool) -> f64 {
    let (success, failure, columns) = (s.map(|x| -x), [-f], [std::array::from_fn(|k| k as u32)]);
    let children = Children {
        success: Table {
            values: &success,
            width: A,
        },
        failure: Table {
            values: &failure,
            width: 1,
        },
        success_columns: &columns,
        failure_columns: &[0],
        revival: &[1.0],
    };
    let solved = sweep(&[(0, 0)], &children, window, Residue::Packing).expect("a valid stage")[0];
    if solved.kind == kind::UNSOLVED {
        lp(s, f, window)
    } else {
        solved.value
    }
}

fn lp(s: &[f64; A], f: f64, window: bool) -> f64 {
    certify(&stage_matrix(s, f, window), A + usize::from(window))
        .expect("a certificate")
        .value
}

fn pairs(seed: u64) -> Vec<(usize, usize)> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize % N
    };
    (0..200).map(|_| (next(), next())).collect()
}

#[test]
#[ignore = "reads ~/Desktop/projects/stl/artifacts/dth/V.npy"]
fn keys_beyond_the_leap_equal_dth_and_window_stages_match_lp() {
    let dth = dth();
    let mut memo = HashMap::new();
    // These keys cannot reach a leap window, so their values are DTH's.
    for key in [Key::H1(59), Key::H2(56), Key::H1(56)] {
        for (pc, pd) in pairs(826) {
            let value = evaluate(Some(key), pc, pd, &dth, &mut memo);
            assert!(
                (value - dth.values()[pc * (N + 1) + pd]).abs() <= 1e-9,
                "{key} ({pc}, {pd})"
            );
        }
    }
    let p = profiles();
    for (pc, pd) in pairs(296) {
        let value = evaluate(Some(Key::H2(57)), pc, pd, &dth, &mut memo);
        let row = &dth.values()[pd * (N + 1)..][..=N];
        let s: [f64; A] = std::array::from_fn(|k| -row[p.succ[pc][k] as usize]);
        let f = p.rev[pc] * -row[p.fail[pc] as usize] + (1.0 - p.rev[pc]);
        assert!((value - lp(&s, f, true)).abs() <= 1e-8, "({pc}, {pd})");
    }
}
