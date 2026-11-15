//! The engine certificate against the array best responses, hand-made strategies, the Python
//! reference and the stored `D_3` certificate.

use std::path::Path;

use tok::certificate::{best_response_value, certify};
use tok::cfr::CfrPlus;
use tok::rules::Player;
use tok::table::{self, StrategyTable};
use tok::traversal::{Profile, saddle_gap};
use tok::{HANDS, tree};

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("data/reference.json")).expect("reference.json parses")
}

fn tables(layers: &[tree::Layer], profile: &Profile) -> [StrategyTable; 2] {
    [0, 1].map(|p| table::export(layers, profile, p).expect("K <= 4"))
}

fn cfr_average(layers: &[tree::Layer], iterations: usize) -> Profile {
    let mut cfr = CfrPlus::new(layers);
    (0..iterations).for_each(|_| cfr.iterate());
    cfr.average()
}

#[test]
fn matches_the_array_gap_and_python() {
    for want in reference()["certificate"]
        .as_array()
        .expect("certificate records")
    {
        let k = want["horizon"].as_u64().expect("a horizon") as usize;
        let layers = tree::build(k).expect("positive horizon");
        let profile = cfr_average(
            &layers,
            want["iterations"].as_u64().expect("a count") as usize,
        );
        let [p0, p1] = tables(&layers, &profile);
        let (lower, upper) = certify(k, &p0, &p1).expect("complete tables");
        let (array_lower, array_upper) = saddle_gap(&layers, &profile);
        assert!(upper - lower > 1e-4);
        assert!((lower - array_lower).abs() < 1e-12 && (upper - array_upper).abs() < 1e-12);
        // The profiles differ from Python's by the CFR+ tolerance of tests/solvers.rs.
        let python = [&want["lower"], &want["upper"]].map(|v| v.as_f64().expect("a bound"));
        assert!((lower - python[0]).abs() < 1e-8 && (upper - python[1]).abs() < 1e-8);
    }
}

#[test]
fn parallel_matches_serial() {
    let layers = tree::build(2).expect("positive horizon");
    let [p0, p1] = tables(&layers, &cfr_average(&layers, 20));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a thread pool");
    let serial = pool
        .install(|| certify(2, &p0, &p1))
        .expect("complete tables");
    assert_eq!(serial, certify(2, &p0, &p1).expect("complete tables"));
}

/// `D_1` strategies from the tests of `solve.py`: uniform beads, then guess uniformly over the bead counts
/// the opponent can still hold, or announce 11 whenever 11 can still hit.
fn hand_table(layers: &[tree::Layer], p: usize, announce_eleven: bool) -> StrategyTable {
    let behavior = (layers.iter().enumerate())
        .map(|(d, layer)| {
            if layer.actor != p {
                return vec![0.0; HANDS * layer.slots(layer.actor)];
            }
            let rows =
                layer.slot_node[p]
                    .iter()
                    .zip(&layer.slot_hand[p])
                    .flat_map(|(&node, &x)| {
                        let (x, mut alive) = (i32::from(x), (1..=10).collect::<Vec<i32>>());
                        if d == 1 {
                            let g = i32::from(layers[1].total[node as usize]);
                            alive.retain(|&c| c != g - x && g - 10 <= c && c < g);
                        }
                        if announce_eleven && alive.contains(&(11 - x)) {
                            alive = vec![11 - x];
                        }
                        (1..=10).map(move |c| {
                            if alive.contains(&c) {
                                1.0 / alive.len() as f64
                            } else {
                                0.0
                            }
                        })
                    });
            rows.collect()
        })
        .collect();
    let profile = Profile {
        behavior,
        choose: [vec![0.1; HANDS], vec![0.1; HANDS]],
    };
    table::export(layers, &profile, p).expect("K <= 4")
}

#[test]
fn announcing_eleven_guarantees_zero() {
    let layers = tree::build(1).expect("positive horizon");
    let p0 = hand_table(&layers, 0, true);
    let p1 = hand_table(&layers, 1, false);
    let (lower, upper) = certify(1, &p0, &p1).expect("complete tables");
    assert!(lower.abs() < 1e-15 && upper.abs() < 1e-15);
}

#[test]
fn uniform_guessing_leaks_for_p0() {
    let layers = tree::build(1).expect("positive horizon");
    let p0 = hand_table(&layers, 0, false);
    let guarantee = -best_response_value(1, Player::P1, &p0).expect("complete table");
    assert!((guarantee + 0.09).abs() < 1e-15);
}

#[test]
fn tables_survive_a_round_trip() {
    let layers = tree::build(2).expect("positive horizon");
    let profile = cfr_average(&layers, 3);
    let dir = std::env::temp_dir().join(format!("tok-table-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    let saved = tables(&layers, &profile);
    let paths = [0, 1].map(|p| dir.join(format!("p{p}.tbl")));
    let loaded = [0, 1].map(|p| {
        saved[p].save(&paths[p]).expect("writable");
        StrategyTable::load(&paths[p]).expect("readable")
    });
    std::fs::remove_dir_all(&dir).expect("removable");
    assert_eq!(loaded, saved);
    assert_eq!(
        table::import(&layers, [&loaded[0], &loaded[1]]).expect("complete tables"),
        profile
    );
    assert!(StrategyTable::load(Path::new("Cargo.toml")).is_err());
}

#[test]
#[ignore = "needs artifacts/tok/cfr_3_p0.tbl and cfr_3_p1.tbl (tests/data/README.md)"]
fn stored_d3_profile_matches_its_certificate() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/tok");
    let load = |name: &str| StrategyTable::load(&dir.join(name)).expect("see tests/data/README.md");
    let (p0, p1) = (load("cfr_3_p0.tbl"), load("cfr_3_p1.tbl"));
    let json = &reference()["cfr_3_certificate"];
    let want = [&json["lower"], &json["upper"]].map(|v| v.as_f64().expect("a bound"));
    let close = |(lower, upper): (f64, f64)| {
        (lower - want[0]).abs() < 1e-12 && (upper - want[1]).abs() < 1e-12
    };
    assert!(close(certify(3, &p0, &p1).expect("complete tables")));
    let layers = tree::build(3).expect("positive horizon");
    assert!(close(saddle_gap(
        &layers,
        &table::import(&layers, [&p0, &p1]).expect("complete tables")
    )));
}
