//! The kernel reproduces the retired Rust kernel's outputs exactly (`tests/data/README.md`).
//!
//! Equal means equal as numbers, with NaN in the same places: a signed zero matches either sign,
//! because `minnum` and `maxnum` may return either zero of a tie.

use serde_json::Value;
use stl_leap::kernel::{Children, Residue, Solved, Table, kind, sweep};
use stl_leap::packing::{Packer, Start};
use stl_leap::rules::ACTIONS as A;

struct Golden {
    header: Value,
    blob: Vec<u8>,
}

impl Golden {
    fn load() -> Self {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let text = std::fs::read_to_string(dir.join("leap_kernel_golden.json"))
            .expect("the fixture header");
        let header = serde_json::from_str(&text).expect("a JSON header");
        let blob = std::fs::read(dir.join("leap_kernel_golden.bin")).expect("the fixture body");
        Self { header, blob }
    }

    fn bytes(&self, name: &str, dtype: &str) -> (&[u8], usize) {
        let entry = &self.header["arrays"][name];
        assert_eq!(entry["dtype"], dtype, "{name}");
        let count: usize = entry["shape"]
            .as_array()
            .expect("a shape")
            .iter()
            .map(|x| x.as_u64().expect("a dim") as usize)
            .product();
        let offset = entry["offset"].as_u64().expect("an offset") as usize;
        let width = dtype[2..].parse::<usize>().expect("an element size");
        (&self.blob[offset..offset + count * width], width)
    }

    fn f64s(&self, name: &str) -> Vec<f64> {
        let (bytes, width) = self.bytes(name, "<f8");
        bytes
            .chunks(width)
            .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect()
    }

    fn u32s(&self, name: &str) -> Vec<u32> {
        let (bytes, width) = self.bytes(name, "<i4");
        bytes
            .chunks(width)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect()
    }

    fn u64s(&self, name: &str) -> Vec<u64> {
        let (bytes, width) = self.bytes(name, "<u8");
        bytes
            .chunks(width)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect()
    }

    fn kinds(&self, name: &str) -> Vec<u8> {
        self.bytes(name, "|u1").0.to_vec()
    }

    fn count(&self, name: &str) -> usize {
        let (bytes, _) = self.bytes(name, "<i8");
        i64::from_le_bytes(bytes.try_into().expect("one count")) as usize
    }

    fn stages(&self) -> Vec<([f64; A], f64)> {
        let (s, f) = (self.f64s("stage_s"), self.f64s("stage_f"));
        s.chunks(A)
            .map(|s| s.try_into().expect("60 payoffs"))
            .zip(f)
            .collect()
    }
}

fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

fn assert_values(name: &str, got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len(), "{name}");
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(same(a, b), "{name}[{i}]: {a:e} != {b:e}");
    }
}

/// The sweep inputs of the golden fixture: owned tables, then the classes.
struct Inputs {
    success: Vec<f64>,
    success_width: usize,
    failure: Vec<f64>,
    failure_width: usize,
    success_columns: Vec<[u32; A]>,
    failure_columns: Vec<u32>,
    revival: Vec<f64>,
    classes: Vec<(u32, u32)>,
}

impl Inputs {
    fn new(golden: &Golden, label: &str) -> Self {
        if label == "tables" {
            let columns = golden.u32s("table_succ_col");
            let pcs = golden.u32s("table_pcs");
            Self {
                success: golden.f64s("table_succ_table"),
                success_width: 121,
                failure: golden.f64s("table_fail_table"),
                failure_width: 61,
                success_columns: columns
                    .chunks(A)
                    .map(|c| c.try_into().expect("60 columns"))
                    .collect(),
                failure_columns: golden.u32s("table_fail_col"),
                revival: golden.f64s("table_rev"),
                classes: pcs.into_iter().zip(golden.u32s("table_pds")).collect(),
            }
        } else {
            Self {
                success: golden.f64s("stage_s").iter().map(|x| -x).collect(),
                success_width: A,
                failure: golden.f64s("stage_f").iter().map(|x| -x).collect(),
                failure_width: 1,
                success_columns: vec![std::array::from_fn(|k| k as u32)],
                failure_columns: vec![0],
                revival: vec![1.0],
                classes: golden
                    .u32s("stage_order")
                    .into_iter()
                    .map(|pd| (0, pd))
                    .collect(),
            }
        }
    }

    fn sweep(&self, window: bool, residue: Residue) -> Vec<Solved> {
        let children = Children {
            success: Table {
                values: &self.success,
                width: self.success_width,
            },
            failure: Table {
                values: &self.failure,
                width: self.failure_width,
            },
            success_columns: &self.success_columns,
            failure_columns: &self.failure_columns,
            revival: &self.revival,
        };
        sweep(&self.classes, &children, window, residue).expect("valid golden inputs")
    }
}

#[test]
fn fixture_covers_every_kind_under_the_production_options() {
    let golden = Golden::load();
    let meta = &golden.header["meta"];
    assert_eq!(meta["engine"], "stl_solver_rs");
    // The build's residue options: crash starts and one kink attempt, which `Residue::Kink` is.
    assert_eq!(
        meta["production"],
        serde_json::json!({"crash": true, "kink_attempts": 1, "native": true})
    );
    let mut kinds = std::collections::BTreeSet::new();
    for (name, counts) in meta["coverage"].as_object().expect("a coverage map") {
        if name.starts_with("sweep_") {
            kinds.extend(counts.as_object().expect("counts").keys().cloned());
        }
    }
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        ["0", "1", "2", "255", "3", "4"]
    );
}

#[test]
fn sweep_matches_golden() {
    let golden = Golden::load();
    for label in ["stages", "tables"] {
        let inputs = Inputs::new(&golden, label);
        for window in [false, true] {
            for (mode, residue) in [("rungs", Residue::Defer), ("production", Residue::Kink)] {
                let name = format!(
                    "sweep_{label}_{}_{mode}",
                    if window { "window" } else { "square" }
                );
                let solved = inputs.sweep(window, residue);
                let kinds: Vec<u8> = solved.iter().map(|x| x.kind).collect();
                let failed = solved.iter().filter(|x| x.kind == kind::UNSOLVED).count();
                assert_eq!(failed, golden.count(&format!("{name}_failed")), "{name}");
                assert_eq!(kinds, golden.kinds(&format!("{name}_kind")), "{name}");
                let values: Vec<f64> = solved.iter().map(|x| x.value).collect();
                assert_values(&name, &values, &golden.f64s(&format!("{name}_value")));
            }
        }
    }
}

#[test]
fn packing_matches_golden() {
    let golden = Golden::load();
    for (start, label) in [(Start::Slack, "slack"), (Start::Crash, "crash")] {
        for window in [false, true] {
            let name = format!(
                "packing_{label}_{}",
                if window { "window" } else { "square" }
            );
            let mut packer = Packer::default();
            let (mut values, mut kinds, mut supports) = (vec![], vec![], vec![]);
            for (s, f) in golden.stages() {
                let (value, p, q) = packer.solve(&s, f, start).unwrap_or((f64::NAN, 0, 0));
                let certified = value.is_finite();
                // A window stage stores max(v60, f), the constant leap row's lift.
                values.push(if certified && window {
                    value.max(f)
                } else {
                    value
                });
                kinds.push(if certified {
                    kind::PACKING
                } else {
                    kind::UNSOLVED
                });
                supports.extend([p, q]);
            }
            let failed = values.iter().filter(|v| v.is_nan()).count();
            assert_eq!(failed, golden.count(&format!("{name}_failed")), "{name}");
            assert_eq!(kinds, golden.kinds(&format!("{name}_kind")), "{name}");
            assert_eq!(supports, golden.u64s(&format!("{name}_support")), "{name}");
            assert_values(&name, &values, &golden.f64s(&format!("{name}_value")));
        }
    }
}
