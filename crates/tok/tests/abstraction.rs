//! Last-layer keys against keys computed from the rules, and the merged solvers against their
//! bounds.

use std::collections::HashSet;

use tok::abstraction::{Key, alternate, guarantee, merged_best_response, slot_keys};
use tok::cfr::CfrPlus;
use tok::traversal::Profile;
use tok::tree::{self, Layer, last_layer};
use tok::{HANDS, lp};

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("data/reference.json")).expect("reference.json parses")
}

const KEYS: [(Key, &str); 4] = [
    (Key::Full, "full"),
    (Key::Support, "support"),
    (Key::Hard, "hard"),
    (Key::OpeningHard, "opening+hard"),
];

/// The key of `order_blind.py` for player p with `own` beads after `totals`, from the rules.
fn reference_key(kind: Key, p: usize, own: u8, totals: &[u8]) -> (u8, Vec<u8>, Vec<u8>, u8) {
    let alive = (1..=10).filter(|&opp| {
        let hands = if p == 0 { [own, opp] } else { [opp, own] };
        let legal = totals
            .iter()
            .enumerate()
            .all(|(j, &g)| (hands[j % 2] + 1..=hands[j % 2] + 10).contains(&g));
        legal && totals.iter().all(|&g| g != hands[0] + hands[1])
    });
    let set = |start: usize| {
        let mut v: Vec<u8> = totals.iter().skip(start).step_by(2).copied().collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    match kind {
        Key::Full => unreachable!("the full key is the slot itself"),
        Key::Support => (own, alive.collect(), Vec::new(), 0),
        Key::Hard => (own, set(p), set(1 - p), 0),
        Key::OpeningHard => (own, set(p), set(1 - p), totals[0]),
    }
}

fn transcript(layers: &[Layer], d: usize, node: u32) -> Vec<u8> {
    let (mut totals, mut node) = (vec![0; d], node as usize);
    for e in (1..=d).rev() {
        totals[e - 1] = layers[e].total[node];
        node = layers[e].parent[node] as usize;
    }
    totals
}

#[test]
fn keys_partition_like_the_reference() {
    let layers = tree::build(2).expect("positive horizon");
    for p in 0..2 {
        let d = last_layer(&layers, p);
        let layer = &layers[d];
        for (kind, _) in &KEYS[1..] {
            let (key, keys) = slot_keys(&layers, p, *kind);
            let slots = layer.slot_node[p].iter().zip(&layer.slot_hand[p]);
            let want: Vec<_> = slots
                .map(|(&n, &h)| reference_key(*kind, p, h, &transcript(&layers, d, n)))
                .collect();
            assert_eq!(keys, want.iter().collect::<HashSet<_>>().len());
            assert_eq!(keys, key.iter().zip(&want).collect::<HashSet<_>>().len());
        }
    }
}

#[test]
fn key_counts_match_python() {
    let reference = reference();
    for (k, name) in [(2, "slot_keys_2"), (3, "slot_keys_3")] {
        let layers = tree::build(k).expect("positive horizon");
        for p in 0..2 {
            for (kind, label) in KEYS {
                if let Some(want) = reference[name][p][label].as_u64() {
                    assert_eq!(
                        slot_keys(&layers, p, kind).1,
                        want as usize,
                        "K={k} P{p} {label}"
                    );
                }
            }
        }
    }
}

fn lp_equilibrium(layers: &[Layer]) -> (f64, Profile) {
    let form = lp::build(layers);
    let sol = lp::solve(&form, "ipm", 600.0).expect("HiGHS solves D_2");
    (
        sol.value,
        lp::plan_profile(&form, layers, [&sol.plans[0], &sol.plans[1]]),
    )
}

#[test]
fn merged_responses_and_cfr_keep_the_value() {
    let layers = tree::build(2).expect("positive horizon");
    let (value, profile) = lp_equilibrium(&layers);
    for p in 0..2 {
        let (key, keys) = slot_keys(&layers, p, Key::Full);
        let got = merged_best_response(&layers, p, &profile, &key, keys);
        assert!((got - if p == 0 { value } else { -value }).abs() < 1e-6);
    }
    let (key, keys) = slot_keys(&layers, 0, Key::Support);
    let mut cfr = CfrPlus::merged(&layers, 0, &profile, &key, keys);
    (0..1000).for_each(|_| cfr.iterate());
    let average = cfr.average();
    let held = guarantee(&layers, 0, &average);
    assert!(held <= merged_best_response(&layers, 0, &average, &key, keys) + 1e-12);
    assert!(held > value - 1e-4);
}

#[test]
fn alternation_returns_a_key_measurable_strategy() {
    let layers = tree::build(2).expect("positive horizon");
    let mut cfr = CfrPlus::new(&layers);
    (0..200).for_each(|_| cfr.iterate());
    let (key, keys) = slot_keys(&layers, 1, Key::Support);
    let (steps, best) = alternate(&layers, 1, &cfr.average(), &key, keys, (1, 100), |_| {});
    assert_eq!(
        steps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
        ["merge1", "early1"]
    );
    let last = &best.behavior[last_layer(&layers, 1)];
    let mut rows = vec![None; keys];
    for (&k, row) in key.iter().zip(last.chunks(HANDS)) {
        let first = rows[k as usize].get_or_insert(row);
        assert!(first.iter().zip(row).all(|(a, b)| (a - b).abs() < 1e-12));
    }
    let top = steps
        .iter()
        .map(|s| s.guarantee)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((top - guarantee(&layers, 1, &best)).abs() < 1e-12);
    assert!(steps[0].guarantee <= steps[0].cap.expect("merge steps have a cap") + 1e-12);
}
