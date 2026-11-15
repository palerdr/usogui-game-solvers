//! The sweep driver and the reader on a sparse fake DTH table.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use stl_leap::build::{run_full, sweep_key};
use stl_leap::dth::Dth;
use stl_leap::kernel::{Children, Residue, Table, kind, sweep};
use stl_leap::lp::{certify, stage_matrix};
use stl_leap::reach::Bits;
use stl_leap::rules::{ACTIONS as A, Key, N, RESETS, WIN, profile_index, profiles};
use stl_leap::store::{PackedReader, Store};
use stl_leap::{Error, open_leap};

/// The six certified DTH values that `Dth::open` checks, in profile coordinates.
const ANCHORS: [([u16; 4], f64); 6] = [
    ([0, 0, 0, 0], 0.089_850_072_809_510_46),
    ([240, 301, 240, 301], 0.337_213_216_629_109_3),
    ([10, 60, 200, 0], -0.794_442_891_646_929_7),
    ([150, 90, 30, 120], 0.724_409_303_635_678_5),
    ([250, 301, 40, 0], 0.998_115_281_738_196_9),
    ([100, 140, 100, 140], 0.187_738_637_827_619_3),
];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stl-leap-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn npy_header(shape: &str) -> Vec<u8> {
    let mut text =
        format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}").into_bytes();
    text.resize(117, b' ');
    text.push(b'\n');
    let mut head = b"\x93NUMPY\x01\x00\x76\x00".to_vec();
    head.extend(text);
    head
}

/// Write a sparse `V.npy` of the full shape holding the anchors and `cells`, zero elsewhere.
fn fake_dth(path: &Path, cells: &[(usize, usize, f64)]) {
    let mut file = File::create(path).expect("a fake DTH file");
    file.write_all(&npy_header("(17011, 17012)"))
        .expect("a header");
    file.set_len(128 + (N * (N + 1) * 8) as u64)
        .expect("a sparse body");
    let anchors = ANCHORS.map(|([cs, ct, ds, dt], v)| {
        (
            profile_index(cs, ct).expect("a profile"),
            profile_index(ds, dt).expect("a profile"),
            v,
        )
    });
    for &(row, col, value) in anchors.iter().chain(cells) {
        file.seek(SeekFrom::Start(128 + 8 * (row * (N + 1) + col) as u64))
            .expect("a seek");
        file.write_all(&value.to_le_bytes()).expect("a cell");
    }
}

fn fake_artifact(dir: &Path, report: &str) -> PathBuf {
    let artifact = dir.join("leap");
    std::fs::create_dir_all(&artifact).expect("an artifact directory");
    std::fs::write(artifact.join("report.json"), report).expect("a report");
    artifact
}

const COMPLETE: &str = r#"{"states": 9453333117, "max_bellman_residual": 0.0,
    "rows": {"H1": {"states": 4180634990}, "H2": {"states": 4188677815}, "REV": {"states": 1084020312}}}"#;

#[test]
fn full_build_rejects_existing_output() {
    let dir = scratch("existing");
    std::fs::write(dir.join("saved.npy"), b"existing result").expect("a file");
    let result = run_full(&dir.join("absent-dth.npy"), &dir);
    assert!(matches!(result, Err(Error::Invalid(reason)) if reason.contains("empty output")));
    assert_eq!(
        std::fs::read(dir.join("saved.npy")).expect("the file"),
        b"existing result"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn open_leap_requires_a_complete_checked_result_and_a_valid_dth() {
    let dir = scratch("open");
    let dth = dir.join("V.npy");
    fake_dth(&dth, &[]);
    let artifact = fake_artifact(&dir, COMPLETE);
    let table = open_leap(&artifact, &dth).expect("a complete artifact");
    let index = profile_index(60, 120).expect("an eligible profile");
    let message =
        |result: Result<f64, Error>| result.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(message(table.value(Key::Rev(3600), 0, index)).contains("no column"));
    assert!(message(table.value(Key::H1(44), 0, WIN + 1)).contains("out of range"));
    assert!(matches!(
        open_leap(&dir.join("missing"), &dth),
        Err(Error::Io { .. })
    ));
    let incomplete = [
        COMPLETE.replace("9453333117", "1"),
        COMPLETE.replace(r#""H1": {"states": 4180634990}, "#, ""),
        COMPLETE.replace("0.0", "1e-5"),
        COMPLETE.replace("0.0", "NaN"),
    ];
    for report in incomplete {
        let artifact = fake_artifact(&dir, &report);
        let result = open_leap(&artifact, &dth);
        assert!(
            matches!(result, Err(Error::Format { reason, .. }) if reason.contains("complete, numerically checked"))
        );
    }
    let small = dir.join("small.npy");
    let mut file = File::create(&small).expect("a small table");
    file.write_all(&npy_header("(2, 3)")).expect("a header");
    file.write_all(&[0; 48]).expect("a body");
    let artifact = fake_artifact(&dir, COMPLETE);
    assert!(
        matches!(open_leap(&artifact, &small), Err(Error::Format { reason, .. }) if reason.contains("is not float64"))
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// A window stage with `f < s[0]` that the closed-form rungs miss, so `HiGHS` must solve it.
fn highs_stage(rev: f64) -> ([f64; A], f64) {
    let mut state = 7_u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    };
    loop {
        let s: [f64; A] = std::array::from_fn(|_| uniform());
        // The child value that the table stores, and the failure payoff it gives back.
        let child = (1.0 - rev - uniform()) / rev;
        let f = rev * -child + (1.0 - rev);
        let (success, failure) = (s.map(|x| -x), [-child]);
        let columns = [std::array::from_fn(|k| k as u32)];
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
            revival: &[rev],
        };
        let deferred = sweep(&[(0, 0)], &children, true, Residue::Packing).expect("a valid stage");
        if child.abs() <= 1.0 && f < s[0] && deferred[0].kind == kind::UNSOLVED {
            return (s, child);
        }
    }
}

#[test]
fn sweep_stores_kernel_and_highs_values() {
    let dir = scratch("sweep");
    let p = profiles();
    // REV_3600 reads only DTH. Checker 0 meets droppers resets[0] and resets[1].
    let (easy, hard) = (p.resets[0] as usize, p.resets[1] as usize);
    let (s, child) = highs_stage(p.rev[0]);
    let mut cells = vec![];
    for (k, &column) in p.succ[0].iter().enumerate() {
        cells.push((easy, column as usize, -0.5 + 0.01 * k as f64));
        cells.push((hard, column as usize, -s[k]));
    }
    cells.extend([
        (easy, p.fail[0] as usize, 0.2),
        (hard, p.fail[0] as usize, child),
    ]);
    fake_dth(&dir.join("V.npy"), &cells);
    let dth = Dth::open(&dir.join("V.npy")).expect("a fake DTH");
    let mut reach = Bits::new(RESETS);
    reach.set(0, 0);
    reach.set(0, 1);
    let tables = dir.join("tables");
    let mut store = Store::new(&tables, &tables).expect("a store");
    let record = sweep_key(Key::Rev(3600), &reach, &mut store, &dth).expect("a sweep");
    assert_eq!((record.states, record.lp_solves), (2, 1));
    store.pack(Key::Rev(3600)).expect("a packed table");
    let reader = PackedReader::new(&tables);
    for (col, dropper) in [(0, easy), (1, hard)] {
        let row = &dth.values()[dropper * (N + 1)..][..=N];
        let s: [f64; A] = std::array::from_fn(|k| -row[p.succ[0][k] as usize]);
        let f = p.rev[0] * -row[p.fail[0] as usize] + (1.0 - p.rev[0]);
        let expected = certify(&stage_matrix(&s, f, true), A + 1)
            .expect("a certificate")
            .value;
        let stored = reader.get(Key::Rev(3600), 0, col).expect("a stored value");
        assert!(
            (stored - expected).abs() <= 1e-8,
            "{col}: {stored} != {expected}"
        );
    }
    std::fs::remove_dir_all(&dir).expect("cleanup");
}
