//! The LP and CFR+ against the Python reference and against their own best responses, and the
//! merged solvers of the abstraction study at the `D_2` equilibrium.

use std::sync::OnceLock;

use tok::abstraction::{Key, guarantee, merged_best_response, slot_keys};
use tok::cfr::CfrPlus;
use tok::traversal::{Profile, profile_value, saddle_gap};
use tok::{HANDS, lp, tree};

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("data/reference.json")).expect("reference.json parses")
}

fn number(value: &serde_json::Value) -> f64 {
    value.as_f64().expect("a number")
}

fn lp_profile(layers: &[tree::Layer], solver: &str) -> (f64, Profile) {
    let form = lp::build(layers);
    let sol = lp::solve(&form, solver, 600.0).expect("HiGHS solves D_K");
    let profile = lp::plan_profile(&form, layers, &sol.plans);
    (sol.value, profile)
}

#[test]
fn d1_lp_is_zero_and_certified() {
    let layers = tree::build(1).expect("positive horizon");
    let form = lp::build(&layers);
    assert_eq!(form.count, tree::sizes(&layers).sequences);
    assert_eq!(form.payoff.len(), 1_000);
    assert_eq!(form.payoff.iter().filter(|e| e.2 == 1.0).count(), 100);
    let sol = lp::solve(&form, "choose", 60.0).expect("HiGHS solves D_1");
    assert!(sol.value.abs() < 1e-9);
    for p in 0..2 {
        let mut residual = vec![0.0; form.flow_rows[p]];
        for &(r, s, v) in &form.flow[p] {
            residual[r] += v * sol.plans[p][s].max(0.0);
        }
        assert!((residual[0] - 1.0).abs() < 1e-12);
        assert!(residual[1..].iter().all(|r| r.abs() < 1e-9));
    }
    let profile = lp::plan_profile(&form, &layers, &sol.plans);
    let (lower, upper) = saddle_gap(&layers, &profile);
    assert!(upper - lower < 1e-9);
    // P0 announces 11 at every bead count, as in the object-tree test of solve.py.
    for (x, row) in profile.behavior[0].chunks(HANDS).enumerate() {
        assert!(
            (row[HANDS - 1 - x] - 1.0).abs() < 1e-9,
            "x0 = {}: {row:?}",
            x + 1
        );
    }
}

/// The `D_2` LP solution, solved once: the solve takes most of the time of the two tests that
/// read it.
fn d2_lp() -> &'static (Vec<tree::Layer>, f64, Profile) {
    static D2: OnceLock<(Vec<tree::Layer>, f64, Profile)> = OnceLock::new();
    D2.get_or_init(|| {
        let layers = tree::build(2).expect("positive horizon");
        let (value, profile) = lp_profile(&layers, "ipm");
        (layers, value, profile)
    })
}

#[test]
fn lp_values_match_python() {
    let reference = reference();
    let d1 = tree::build(1).expect("positive horizon");
    let (d1_value, d1_profile) = lp_profile(&d1, "choose");
    let (d2, d2_value, d2_profile) = d2_lp();
    let solved = [
        (1, &d1, d1_value, &d1_profile),
        (2, d2, *d2_value, d2_profile),
    ];
    for (k, layers, value, profile) in solved {
        assert!((value - number(&reference["lp_value"][k.to_string()])).abs() < 1e-9);
        let (lower, upper) = saddle_gap(layers, profile);
        assert!(lower <= value + 1e-9 && value - 1e-9 <= upper && upper - lower < 1e-7);
    }
}

#[test]
fn merged_responses_and_cfr_keep_the_value() {
    let (layers, value, profile) = d2_lp();
    let value = *value;
    for p in 0..2 {
        let (key, keys) = slot_keys(layers, p, Key::Full);
        let got = merged_best_response(layers, p, profile, &key, keys);
        assert!((got - if p == 0 { value } else { -value }).abs() < 1e-6);
    }
    let (key, keys) = slot_keys(layers, 0, Key::Support);
    let mut cfr = CfrPlus::merged(layers, 0, profile, &key, keys);
    (0..1000).for_each(|_| cfr.iterate());
    let average = cfr.average();
    let held = guarantee(layers, 0, &average);
    assert!(held <= merged_best_response(layers, 0, &average, &key, keys) + 1e-12);
    assert!(held > value - 1e-4);
}

fn run(k: usize, checkpoints: &[usize]) -> (Vec<(f64, f64)>, Vec<tree::Layer>) {
    let layers = tree::build(k).expect("positive horizon");
    let mut cfr = CfrPlus::new(&layers);
    let mut bounds = Vec::new();
    for &target in checkpoints {
        while cfr.iterations < target {
            cfr.iterate();
        }
        bounds.push(saddle_gap(&layers, &cfr.average()));
    }
    (bounds, layers)
}

#[test]
fn d1_gap_declines_toward_zero() {
    let (bounds, _) = run(1, &[10, 100, 2000]);
    let gaps: Vec<f64> = bounds.iter().map(|(lower, upper)| upper - lower).collect();
    assert!(gaps[0] > gaps[1] && gaps[1] > gaps[2] && gaps[2] < 1e-4);
    assert!(bounds[2].0 <= 1e-12 && bounds[2].1 >= -1e-12);
}

#[test]
fn d2_brackets_the_lp_value() {
    let d2 = number(&reference()["lp_value"]["2"]);
    let layers = tree::build(2).expect("positive horizon");
    let mut cfr = CfrPlus::new(&layers);
    (0..30).for_each(|_| cfr.iterate());
    let (lo1, hi1) = saddle_gap(&layers, &cfr.average());
    (30..300).for_each(|_| cfr.iterate());
    let (lo2, hi2) = saddle_gap(&layers, &cfr.average());
    assert!(hi2 - lo2 < hi1 - lo1 && hi2 - lo2 < 1e-2);
    assert!(lo2 <= d2 && d2 <= hi2);
    let value = profile_value(&layers, &cfr.average());
    assert!(lo2 <= value + 1e-12 && value - 1e-12 <= hi2);
}

#[test]
fn strategies_are_distributions() {
    let layers = tree::build(2).expect("positive horizon");
    let mut cfr = CfrPlus::new(&layers);
    (0..5).for_each(|_| cfr.iterate());
    for profile in [cfr.current(), cfr.average()] {
        let rows = profile
            .behavior
            .iter()
            .chain(&profile.choose)
            .flat_map(|b| b.chunks(HANDS));
        for row in rows {
            assert!(row.iter().all(|&x| x >= 0.0) && (row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
}

/// Rust rounds every row sum and row dot product as numpy does, so ties break the same way; only
/// P0's 10-term root dot product (Accelerate's `ddot` in numpy) rounds differently. Those ulps
/// grow to 1.4e-9 in the worst entry after 100 iterations at K = 2, measured over every entry.
const CFR_TOLERANCE: f64 = 1e-8;

#[test]
fn cfr_average_matches_python() {
    let want = &reference()["cfr"];
    let k = want["horizon"].as_u64().expect("a horizon") as usize;
    let iterations = want["iterations"].as_u64().expect("a count") as usize;
    let layers = tree::build(k).expect("positive horizon");
    let mut cfr = CfrPlus::new(&layers);
    (0..iterations).for_each(|_| cfr.iterate());
    let average = cfr.average();
    let close = |got: f64, want: &serde_json::Value| (got - number(want)).abs() < CFR_TOLERANCE;
    let (lower, upper) = saddle_gap(&layers, &average);
    assert!(close(lower, &want["lower"]) && close(upper, &want["upper"]));
    assert!(close(profile_value(&layers, &average), &want["value"]));
    for p in 0..2 {
        assert!((0..HANDS).all(|a| close(average.choose[p][a], &want["choose"][p][a])));
    }
    let rows = want["rows"].as_array().expect("sampled rows");
    for row in rows {
        let (d, s) = (
            row[0].as_u64().expect("a layer") as usize,
            row[1].as_u64().expect("a slot") as usize,
        );
        assert!(
            (0..HANDS).all(|a| close(average.behavior[d][HANDS * s + a], &row[2][a])),
            "layer {d} slot {s}"
        );
    }
}
