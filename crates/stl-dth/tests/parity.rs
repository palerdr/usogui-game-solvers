//! Parity with the Python reference through the golden fixtures in `tests/data` (see its
//! README), and the reader against the real table.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use stl_dth::{
    ANCHORS, Certificate, Error, LAGS, MAX_SADDLE_GAP, Mix, N, N_ALIVE, Profiles, Rung, Table,
    build, full_matrix, lp, pure, support,
};

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn hex(digest: &[u8]) -> String {
    digest
        .iter()
        .fold(String::new(), |s, b| s + &format!("{b:02x}"))
}

fn f64s(name: &str) -> Vec<f64> {
    let bytes = std::fs::read(data(name)).expect("the fixture exists");
    bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect()
}

/// `ladder.bin` records: `s[0..60]`, `f`, the numba kernel's value and rung, and scipy's LP
/// value. Rows 0..300 are uniform, 300..600 sorted, 600..603 special, and 603.. real classes.
struct Case {
    s: Mix,
    f: f64,
    value: f64,
    rung: f64,
    lp: f64,
}

fn cases() -> Vec<Case> {
    f64s("ladder.bin")
        .chunks_exact(64)
        .map(|r| Case {
            s: r[..60].try_into().expect("60 values"),
            f: r[60],
            value: r[61],
            rung: r[62],
            lp: r[63],
        })
        .collect()
}

#[test]
fn profiles_match_the_reference_hashes() {
    let t = Profiles::build();
    let hex = |bytes: Vec<u8>| hex(&Sha256::digest(bytes));
    let ints = |v: &[usize]| hex(v.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect());
    let got = [
        ints(&t.st),
        ints(&t.ttd),
        hex(t.rev.iter().flat_map(|x| x.to_le_bytes()).collect()),
        hex(t.fail.iter().flat_map(|x| x.to_le_bytes()).collect()),
        hex(t
            .succ
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect()),
    ];
    let expected = std::fs::read_to_string(data("profiles.sha256")).expect("the fixture exists");
    for (line, got) in expected.lines().zip(got) {
        assert_eq!(line.split_whitespace().next(), Some(got.as_str()), "{line}");
    }
}

#[test]
fn ladder_matches_the_numba_kernel_bit_for_bit() {
    for (i, c) in cases().iter().enumerate() {
        let cert = Some(pure(&c.s, c.f))
            .filter(Certificate::certified)
            .or_else(|| support(&c.s, c.f).filter(Certificate::certified));
        match cert {
            Some(cert) => {
                assert_eq!(f64::from(cert.rung as u8), c.rung, "case {i}");
                assert_eq!(cert.value().to_bits(), c.value.to_bits(), "case {i}");
            }
            None => assert_eq!(c.rung, 2.0, "case {i} goes to the LP"),
        }
    }
}

#[test]
fn lp_agrees_with_scipy_and_every_rung_brackets_it() {
    // Both LP values are midpoints of certificates at most 1e-6 wide that contain the value.
    for (i, c) in cases().iter().enumerate() {
        let v = lp(&c.s, c.f)
            .filter(Certificate::certified)
            .expect("the LP certifies")
            .value();
        assert!(
            (v - c.lp).abs() <= MAX_SADDLE_GAP,
            "case {i}: {v} vs {}",
            c.lp
        );
        let r1 = pure(&c.s, c.f);
        assert!(r1.lower <= v + 2e-6 && r1.upper >= v - 2e-6, "case {i}");
        if let Some(r2) = support(&c.s, c.f) {
            assert!(r2.lower <= v + 2e-6 && r2.upper >= v - 2e-6, "case {i}");
        }
    }
}

#[test]
fn rungs_return_certifying_strategies() {
    for (i, c) in cases()[300..603].iter().enumerate() {
        let m = full_matrix(&c.s, c.f);
        for cert in [Some(pure(&c.s, c.f)), support(&c.s, c.f), lp(&c.s, c.f)]
            .into_iter()
            .flatten()
        {
            for mix in [cert.drop, cert.check] {
                assert!(
                    mix.iter().all(|&w| w >= 0.0) && (mix.iter().sum::<f64>() - 1.0).abs() < 1e-12,
                    "case {i}"
                );
            }
            let lower = (0..LAGS)
                .map(|k| (0..LAGS).map(|d| cert.drop[d] * m[d][k]).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            let upper = m
                .iter()
                .map(|row| row.iter().zip(&cert.check).map(|(a, q)| a * q).sum::<f64>())
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (lower - cert.lower).abs() < 1e-12 && (upper - cert.upper).abs() < 1e-12,
                "case {i} {:?}",
                cert.rung
            );
            if cert.rung == Rung::Support {
                assert!(
                    cert.drop.iter().eq(cert.check.iter().rev()),
                    "the mirror pair"
                );
            }
        }
    }
}

#[test]
fn dead_band_build_matches_the_reference() {
    const STOP: usize = 1082; // the dead-band anchor and its subtree: dead x dead classes
    let t = Profiles::build();
    let (values, routed) = build(&t, STOP, |_, _| ()).expect("the dead band certifies");
    let expected = f64s("dead_band.bin");
    let mut solved = Vec::new();
    for pc in 0..N {
        assert_eq!(values.get(pc * (N + 1) + N), -1.0, "the WIN column");
        for pd in 0..N {
            let v = values.get(pc * (N + 1) + pd);
            assert_eq!(v.is_nan(), t.phi[pc] + t.phi[pd] < STOP, "({pc}, {pd})");
            if !v.is_nan() {
                assert!(v.abs() <= 1.0 + 1e-9);
                solved.push((pc * N + pd, v));
            }
        }
    }
    assert_eq!(
        (solved.len(), routed.iter().sum::<u64>()),
        (expected.len(), 7140)
    );
    for (&(c, v), e) in solved.iter().zip(&expected) {
        assert_eq!(v.to_bits(), e.to_bits(), "class {c}");
        assert_eq!(
            stl_dth::recheck(c, |i| values.get(i), &t).expect("rechecks"),
            0.0
        );
    }
    let anchor = 16_951 * (N + 1) + 16_951;
    assert_eq!(
        t.classes(ANCHORS[1].0).expect("dead band"),
        (N_ALIVE + 240, N_ALIVE + 240)
    );
    assert!((values.get(anchor) - ANCHORS[1].1).abs() <= MAX_SADDLE_GAP);
}

#[test]
fn open_fails_closed_on_the_wrong_shape() {
    assert!(matches!(
        Table::open(&data("wrong_shape.npy"), 0),
        Err(Error::Format(_))
    ));
}

#[test]
fn open_fails_closed_on_the_anchors_and_the_rechecks() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("sparse-V.npy");
    let mut header =
        b"\x93NUMPY\x01\x00v\x00{'descr': '<f8', 'fortran_order': False, 'shape': (17011, 17012), }"
            .to_vec();
    header.resize(127, b' ');
    header.push(b'\n');
    let mut file = std::fs::File::create(&path).expect("a scratch file");
    file.write_all(&header).expect("header");
    file.set_len(128 + 8 * (N * (N + 1)) as u64)
        .expect("a sparse table of zeros");
    assert!(matches!(Table::open(&path, 0), Err(Error::Anchors(states)) if states.len() == 6));
    let t = Profiles::build();
    for (state, value) in ANCHORS {
        let (pc, pd) = t.classes(state).expect("anchors are on the domain");
        file.seek(SeekFrom::Start(128 + 8 * (pc * (N + 1) + pd) as u64))
            .expect("seek");
        file.write_all(&value.to_le_bytes()).expect("anchor");
    }
    let table = Table::open(&path, 0).expect("the anchors pass");
    assert_eq!(table.value([0; 4]).expect("root"), ANCHORS[0].1);
    assert!(
        matches!(Table::open(&path, 1), Err(Error::Recheck(0, _))),
        "class 0's children read 0"
    );
    std::fs::remove_file(path).expect("cleanup");
}

const REAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/dth/V.npy");

#[test]
#[ignore = "needs the built table at artifacts/dth/V.npy (cargo run --release -p stl-dth -- build)"]
fn the_real_table_serves_a_certified_root() {
    let table = Table::open(Path::new(REAL), 12).expect("the real table opens");
    let stage = table.stage([0; 4]).expect("the root stage certifies");
    assert_eq!(stage.value, table.value([0; 4]).expect("root"));
    assert!((stage.value - ANCHORS[0].1).abs() <= MAX_SADDLE_GAP && stage.gap <= MAX_SADDLE_GAP);
    assert!(stage.drop.iter().filter(|&&w| w > 0.0).count() > 1);
    assert!(stage.check.iter().filter(|&&w| w > 0.0).count() > 1);
}

#[test]
#[ignore = "needs the built table at artifacts/dth/V.npy (cargo run --release -p stl-dth -- build)"]
fn the_real_table_is_the_reference_byte_for_byte() {
    let mut file = std::fs::File::open(REAL).expect("the real table exists");
    let mut hasher = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        match file.read(&mut buf).expect("read") {
            0 => break,
            n => hasher.update(&buf[..n]),
        }
    }
    assert_eq!(
        hex(&hasher.finalize()),
        "e43977b7a2aa9be6cb7708465ed52989380bb8a787fcedd6f7929ee14ffd29b2"
    );
}
