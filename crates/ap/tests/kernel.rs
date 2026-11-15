//! The kernel: DP count and unrank, the counter-based sampler, the exact enumeration, the
//! confidence sets and the entry table, with golden parity against the Python reference
//! (ports of `test_kernel_dp.py`, `test_kernel_sampler.py`, `test_exact_kernel.py`,
//! `test_kernel_confidence.py` and the reachable parts of `test_sensitivity_sampler.py`).

use std::collections::HashSet;

use ap::cards::{FULL_DECK, evaluate, legal_hands, rank_sum};
use ap::confidence::{entry_box, interval, log_mixture};
use ap::dp::HandTable;
use ap::engine::{Bet, GameState, Move, Phase, Team, uniform_upper_policy};
use ap::exact::{kernel_counts, round_one};
use ap::sampler::{
    Counts, ROW_OFFSET, ROWS, bounded, mix64, play_round, round_classes, row, sample_rounds,
    sample_sequence, stream,
};
use ap::table::{ENTRIES, HIDDEN, KernelCounts, Sequences, class_code};
use ap::{NOJ, TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS, TIE, W0, W0C, W1, W1C};
use rayon::prelude::*;
use serde_json::Value;

const REACHABLE: [usize; 5] = [W0, W0C, W1, W1C, NOJ];
/// All clubs and the diamonds 2..7: 19 cards; 8 and 63 are impossible.
const REDUCED: u64 = 0x1FFF | 0x3F << 13;
/// `REDUCED` without the clubs 8..A: 36 and 63 are both impossible.
const TINY: u64 = REDUCED & !(0x7F << 6);

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture exists"))
        .expect("fixture is JSON")
}

fn int(v: &Value) -> u64 {
    v.as_u64().expect("an unsigned integer")
}

fn pairs() -> Vec<(u8, u8)> {
    TEAM_0_METAL_CARDS
        .iter()
        .flat_map(|&a| TEAM_1_METAL_CARDS.iter().map(move |&b| (a, b)))
        .collect()
}

fn canonical_targets() -> Vec<u32> {
    let mut t: Vec<u32> = TEAM_0_METAL_CARDS
        .iter()
        .chain(&TEAM_1_METAL_CARDS)
        .map(|&c| c.into())
        .collect();
    t.sort_unstable();
    t
}

/// `n` decks with sizes uniform in `[low, high]`, cards drawn without replacement.
fn decks(seed: u64, n: u64, low: u64, high: u64) -> Vec<u64> {
    (0..n)
        .map(|i| {
            let mut state = stream(seed, 99, i);
            let size = low + bounded(&mut state, high - low + 1);
            let mut deck = 0u64;
            while u64::from(deck.count_ones()) < size {
                deck |= 1 << bounded(&mut state, 52);
            }
            deck
        })
        .collect()
}

/// Class and next deck of `hands` applied through the engine on `deck`.
fn engine_replay(deck: u64, targets: (u8, u8), hands: [Option<u64>; 2]) -> (usize, u64) {
    let mut state = engine_round(deck, targets);
    for mv in [
        Move::Hands(hands),
        Move::Hands(hands),
        Move::Bet(Bet::Check),
        Move::Bet(Bet::Check),
    ] {
        let fits = matches!(
            (state.phase, mv),
            (Phase::UpperChance, Move::Hands(_)) | (Phase::Betting, Move::Bet(_))
        );
        if fits {
            state = state.apply(mv).expect("legal replay");
        }
    }
    (record_class(&state), state.deck)
}

fn engine_round(deck: u64, (t0, t1): (u8, u8)) -> GameState {
    let base = GameState::new([[t0, 9, 10, 11, 12], [t1, 13, 14, 16, 17]], [300, 300])
        .expect("valid setup");
    let mut state = base
        .apply(Move::Number(t0))
        .and_then(|s| s.apply(Move::Number(t1)))
        .expect("legal");
    state.deck = deck;
    state
}

fn record_class(state: &GameState) -> usize {
    let record = state.history.last().expect("a settled round");
    match (record.winner, record.calamity) {
        (None, _) => NOJ,
        (Some(Team::Zero), c) => {
            if c {
                W0C
            } else {
                W0
            }
        }
        (Some(Team::One), c) => {
            if c {
                W1C
            } else {
                W1
            }
        }
    }
}

/// The class probabilities from the engine's own chance enumeration on `deck`.
fn engine_kernel(deck: u64, targets: (u8, u8)) -> [f64; 6] {
    let state = engine_round(deck, targets);
    let uniform = [&uniform_upper_policy as &ap::engine::UpperPolicy<'_>; 2];
    let mut mass = [0.0; 6];
    for (hands, p) in state.chance_outcomes(uniform).expect("valid policy") {
        let next = state.apply(Move::Hands(hands)).expect("legal");
        if next.phase == Phase::UpperChance {
            for (retry, q) in next.chance_outcomes(uniform).expect("valid policy") {
                let (class, _) = engine_replay(deck, targets, retry);
                mass[class] += p * q;
            }
        } else {
            mass[engine_replay(deck, targets, hands).0] += p;
        }
    }
    mass
}

/// `counts` agree with the kernel `exact / sum(exact)` within `sigma` standard errors.
fn assert_within(counts: &Counts, exact: &Counts, sigma: f64, context: &str) {
    let n = counts.iter().sum::<u64>() as f64;
    let total = exact.iter().sum::<u64>() as f64;
    for c in 0..6 {
        let p = exact[c] as f64 / total;
        if p == 0.0 || p == 1.0 {
            assert_eq!(counts[c] as f64, p * n, "{context} class {c}");
        } else {
            let z = (counts[c] as f64 - n * p) / (n * p * (1.0 - p)).sqrt();
            assert!(z.abs() < sigma, "{context} class {c}: z = {z}");
        }
    }
}

#[test]
fn dp_counts_and_unranks_equal_the_python_reference() {
    for case in fixture("dp.json").as_array().expect("cases") {
        let deck = int(&case["deck"]);
        let table = HandTable::new(deck);
        let counts: Vec<u64> = (0..70).map(|n| table.count(n)).collect();
        assert_eq!(
            counts,
            case["counts"]
                .as_array()
                .expect("counts")
                .iter()
                .map(int)
                .collect::<Vec<_>>()
        );
        for r in case["unrank"].as_array().expect("ranks") {
            assert_eq!(
                table.unrank(int(&r[0]) as u32, int(&r[1])),
                Some(int(&r[2]))
            );
        }
    }
}

#[test]
fn dp_counts_equal_enumeration_on_full_and_depleted_decks_and_vanish_exactly_when_empty() {
    let full = HandTable::new(FULL_DECK);
    for &t in &canonical_targets() {
        assert_eq!(full.count(t), legal_hands(t, FULL_DECK).len() as u64);
    }
    for deck in decks(20_261_029, 200, 5, 42) {
        let table = HandTable::new(deck);
        for &t in &canonical_targets() {
            assert_eq!(
                table.count(t),
                legal_hands(t, deck).len() as u64,
                "{deck:#x} {t}"
            );
        }
    }
    for deck in std::iter::once(REDUCED).chain(decks(7, 10, 4, 12)) {
        let table = HandTable::new(deck);
        for t in 0..70 {
            assert_eq!(
                table.count(t),
                legal_hands(t, deck).len() as u64,
                "{deck:#x} {t}"
            );
        }
    }
}

#[test]
fn unranking_every_rank_yields_every_legal_hand_once() {
    for deck in std::iter::once(REDUCED).chain(decks(11, 20, 8, 14)) {
        let table = HandTable::new(deck);
        for t in 6..=64 {
            let hands: HashSet<u64> = (0..table.count(t))
                .map(|u| table.unrank(t, u).expect("in range"))
                .collect();
            assert_eq!(hands.len() as u64, table.count(t));
            assert_eq!(
                hands,
                legal_hands(t, deck).into_iter().collect(),
                "{deck:#x} {t}"
            );
        }
    }
    let table = HandTable::new(FULL_DECK);
    for &t in &canonical_targets() {
        let total = table.count(t);
        let hands: HashSet<u64> = (0..total.min(2000))
            .map(|i| {
                table
                    .unrank(t, i * total / total.min(2000))
                    .expect("in range")
            })
            .collect();
        assert_eq!(hands.len() as u64, total.min(2000));
        assert!(
            hands
                .iter()
                .all(|&h| h.count_ones() == 5 && h & !FULL_DECK == 0 && rank_sum(h) == t)
        );
    }
    let reduced = HandTable::new(REDUCED);
    assert_eq!(reduced.unrank(25, reduced.count(25)), None);
    assert_eq!(reduced.unrank(8, 0), None);
}

#[test]
fn generator_words_and_bounded_draws_equal_the_python_reference() {
    let data = fixture("sampler.json");
    for s in data["streams"].as_array().expect("streams") {
        assert_eq!(stream(int(&s[0]), int(&s[1]), int(&s[2])), int(&s[3]));
    }
    for d in data["bounded"].as_array().expect("draws") {
        let mut state = stream(int(&d[0]), int(&d[1]), int(&d[2]));
        let draws: Vec<u64> = (0..5).map(|_| bounded(&mut state, int(&d[3]))).collect();
        assert_eq!(
            draws,
            d[4].as_array()
                .expect("draws")
                .iter()
                .map(int)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn mix64_is_injective_on_a_sample_and_bounded_is_uniform() {
    assert_eq!(
        (0..50_000u64).map(mix64).collect::<HashSet<_>>().len(),
        50_000
    );
    for n in [1, 2, 3, 7, 100, 61_892] {
        let trials = if n < 1000 { 20_000 } else { 200_000 };
        let mut draws = vec![0u64; n as usize];
        for i in 0..trials {
            draws[bounded(&mut stream(5, 0, i), n) as usize] += 1;
        }
        if n < 1000 {
            let expected = trials as f64 / n as f64;
            assert!(
                draws
                    .iter()
                    .all(|&d| (d as f64 - expected).abs() < 5.0 * expected.sqrt() + 1.0),
                "{n}"
            );
        } else {
            assert!(*draws.iter().max().expect("draws") < 20);
            assert!(draws.iter().filter(|&&d| d > 0).count() as f64 > 0.9 * n as f64);
        }
    }
    let a = stream(1, 0, 0);
    assert!(a != stream(1, 0, 1) && a != stream(1, 1, 0) && a != stream(2, 0, 0));
}

#[test]
fn sampled_counts_and_hands_are_bit_identical_to_the_python_reference() {
    let data = fixture("sampler.json");
    for case in data["sequences"].as_array().expect("sequences") {
        let t = case["targets"].as_array().expect("targets");
        let targets = std::array::from_fn(|r| (int(&t[r][0]) as u8, int(&t[r][1]) as u8));
        let (counts, errors) = sample_sequence(
            &targets,
            int(&case["seed"]),
            int(&case["batch"]),
            int(&case["n"]),
        );
        assert_eq!(errors, int(&case["errors"]));
        let mut expected = vec![[0u64; 6]; ROWS];
        for cell in case["counts"].as_array().expect("cells") {
            expected[int(&cell[0]) as usize][int(&cell[1]) as usize] = int(&cell[2]);
        }
        assert_eq!(counts, expected);
    }
    for case in data["rounds"].as_array().expect("rounds") {
        let targets = (int(&case["t0"]) as u8, int(&case["t1"]) as u8);
        let counts = sample_rounds(
            int(&case["deck"]),
            targets,
            int(&case["seed"]),
            int(&case["batch"]),
            int(&case["n"]),
        );
        assert_eq!(
            counts.to_vec(),
            case["counts"]
                .as_array()
                .expect("counts")
                .iter()
                .map(int)
                .collect::<Vec<_>>()
        );
    }
    for case in data["hands"].as_array().expect("hands") {
        let targets = (int(&case["t0"]) as u8, int(&case["t1"]) as u8);
        let (class, next, hands) = play_round(
            int(&case["deck"]),
            targets,
            &mut stream(int(&case["seed"]), 0, int(&case["index"])),
        );
        assert_eq!(
            (class as u64, next),
            (int(&case["class"]), int(&case["next"]))
        );
        assert_eq!(
            hands.to_vec(),
            case["hands"]
                .as_array()
                .expect("hands")
                .iter()
                .map(Value::as_u64)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn gate_3_single_rounds_and_trajectory_rows_match_the_exact_round_one_kernel() {
    let exact: Vec<Counts> = fixture("kernel_round1.json")["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|e| std::array::from_fn(|c| int(&e[2][c])))
        .collect();
    for (batch, (&pair, exact)) in pairs().iter().zip(&exact).enumerate() {
        assert_within(
            &sample_rounds(FULL_DECK, pair, 20_261_029, batch as u64, 400_000),
            exact,
            4.0,
            &format!("{pair:?}"),
        );
    }
    let (counts, errors) = sample_sequence(
        &[(36, 15), (25, 8), (45, 63), (26, 44), (39, 47)],
        3,
        0,
        200_000,
    );
    assert_eq!(errors, 0);
    assert_within(&counts[0], &exact[2 * 5 + 1], 4.0, "(36, 15) row");
}

#[test]
fn gate_4_single_rounds_on_50_depleted_decks_match_the_exact_counter() {
    let pairs = pairs();
    for (i, deck) in decks(20_261_030, 50, 14, 32).into_iter().enumerate() {
        let pair = pairs[bounded(&mut stream(4, 0, i as u64), 25) as usize];
        assert_within(
            &sample_rounds(deck, pair, 11, 0, 100_000),
            &kernel_counts(deck, pair),
            4.0,
            &format!("{deck:#x} {pair:?}"),
        );
    }
}

#[test]
fn gate_5_sampled_rounds_replay_through_the_engine_to_the_same_class_and_deck() {
    let pairs = pairs();
    let mut cases = vec![
        (FULL_DECK, (25, 8)),
        (REDUCED, (26, 8)),
        (REDUCED, (25, 63)),
        (TINY, (36, 63)),
    ];
    let random = decks(6, 40, 10, 40);
    cases.extend((0..200).map(|i| {
        (
            random[i % 40],
            pairs[bounded(&mut stream(5, 1, i as u64), 25) as usize],
        )
    }));
    let mut seen = HashSet::new();
    for (index, &(deck, pair)) in cases.iter().enumerate() {
        let (class, next, hands) = play_round(deck, pair, &mut stream(77, 0, index as u64));
        assert_eq!(
            (class, next),
            engine_replay(deck, pair, hands),
            "{deck:#x} {pair:?}"
        );
        seen.insert(class);
    }
    assert_eq!(seen, REACHABLE.into_iter().collect());
    // One impossible target always loses; two give no judgement with the deck unchanged.
    let clubs = 0x1FFF;
    for i in 0..20 {
        let (class, next, _) = play_round(clubs, (63, 15), &mut stream(3, 0, i));
        assert_eq!((class, (clubs & !next).count_ones()), (W1, 5));
    }
    assert_eq!(
        play_round(clubs, (63, 63), &mut stream(4, 0, 0)),
        (NOJ, clubs, [None, None])
    );
}

#[test]
fn dense_rows_add_up_across_rounds_and_reproduce_under_the_same_seed() {
    let targets = [(25, 15), (36, 8), (45, 44), (26, 47), (39, 63)];
    let (counts, errors) = sample_sequence(&targets, 42, 7, 50_000);
    assert_eq!((errors, counts.len()), (0, ROWS));
    assert_eq!(counts.iter().map(|c| c[TIE]).sum::<u64>(), 0);
    assert_eq!(counts[0].iter().sum::<u64>(), 50_000);
    let mut prefixes: Vec<Vec<usize>> = vec![vec![]];
    for r in 0..4 {
        let mut longer = Vec::new();
        for prefix in &prefixes {
            let parent = counts[row(r, prefix)];
            for class in REACHABLE {
                let child = [prefix.as_slice(), &[class]].concat();
                assert_eq!(
                    counts[row(r + 1, &child)].iter().sum::<u64>(),
                    parent[class],
                    "{r} {child:?}"
                );
                longer.push(child);
            }
        }
        prefixes = longer;
    }
    assert_eq!(sample_sequence(&targets, 42, 7, 50_000).0, counts);
    assert_ne!(sample_sequence(&targets, 42, 8, 50_000).0, counts);
}

#[test]
fn the_exact_round_one_kernel_equals_the_python_counts_for_all_25_pairs() {
    let data = fixture("kernel_round1.json");
    let expected: Vec<((u8, u8), Counts)> = data["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|e| {
            (
                (int(&e[0]) as u8, int(&e[1]) as u8),
                std::array::from_fn(|c| int(&e[2][c])),
            )
        })
        .collect();
    assert_eq!(round_one(), expected);
}

#[test]
fn the_exact_counter_agrees_with_the_engines_chance_enumeration() {
    for pair in [(25, 15), (36, 44), (39, 47), (45, 15), (26, 8), (25, 63)] {
        let counts = kernel_counts(REDUCED, pair);
        let support = (
            legal_hands(pair.0.into(), REDUCED).len(),
            legal_hands(pair.1.into(), REDUCED).len(),
        );
        assert_eq!(
            counts.iter().sum::<u64>(),
            (support.0.max(1) * support.1.max(1)) as u64
        );
        let total = counts.iter().sum::<u64>() as f64;
        let engine = engine_kernel(REDUCED, pair);
        for c in 0..6 {
            assert!(
                (counts[c] as f64 / total - engine[c]).abs() < 1e-9,
                "{pair:?} class {c}"
            );
        }
    }
    assert_eq!(
        kernel_counts(REDUCED, (26, 8))[W0],
        legal_hands(26, REDUCED).len() as u64
    );
    assert_eq!(kernel_counts(REDUCED, (25, 63))[W1], 0);
    assert_ne!(kernel_counts(REDUCED, (36, 63))[NOJ], 1);
    assert_eq!(kernel_counts(TINY, (36, 63)), [0, 0, 0, 0, 0, 1]);
    assert_eq!(
        engine_kernel(TINY, (36, 63)),
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
    let full = kernel_counts(FULL_DECK, (25, 63));
    assert_eq!(
        (full.iter().sum::<u64>(), full[TIE], full[NOJ]),
        (61_892 * 28, 0, 0)
    );
}

#[test]
fn distinct_metal_cards_never_tie() {
    // Equal strength needs an equal rank multiset, hence an equal sum.
    let strengths: Vec<HashSet<u32>> = canonical_targets()
        .iter()
        .map(|&t| {
            legal_hands(t, FULL_DECK)
                .into_iter()
                .map(|h| evaluate(h).expect("five cards"))
                .collect()
        })
        .collect();
    for (i, a) in strengths.iter().enumerate() {
        for b in &strengths[i + 1..] {
            assert!(a.is_disjoint(b));
        }
    }
}

#[test]
fn intervals_equal_the_python_reference() {
    // The reference calls Apple's libm lgamma through numba; libm's fdlibm lgamma differs by up
    // to 3 ulp of lgamma(n + 1), which moves an end by up to 2e-13 at n = 1e6, 2.3e-12 at
    // n = 1e8 and 4.8e-12 at n = 3e8 (measured); 59% of the rows are bitwise equal.
    let data = fixture("confidence.json");
    for row in data["intervals"].as_array().expect("rows") {
        let (k, n, alpha) = (int(&row[0]), int(&row[1]), row[2].as_f64().expect("alpha"));
        let (lo, hi) = interval(k, n, alpha);
        let (want_lo, want_hi) = (row[3].as_f64().expect("lo"), row[4].as_f64().expect("hi"));
        let tolerance = 1e-12 * (1.0 + n as f64 / 5e7);
        assert!(
            (lo - want_lo).abs() <= tolerance && (hi - want_hi).abs() <= tolerance,
            "{k} {n} {alpha}: ({lo}, {hi}) vs ({want_lo}, {want_hi})"
        );
    }
    for b in data["boxes"].as_array().expect("boxes") {
        let counts: Counts = std::array::from_fn(|c| int(&b["counts"][c]));
        let (lo, hi) = entry_box(&counts, 1e-3 / ENTRIES as f64);
        for c in 0..6 {
            assert!(
                (lo[c] - b["lo"][c].as_f64().expect("lo")).abs() <= 1e-12,
                "{counts:?} lo {c}"
            );
            assert!(
                (hi[c] - b["hi"][c].as_f64().expect("hi")).abs() <= 1e-12,
                "{counts:?} hi {c}"
            );
        }
    }
}

#[test]
fn intervals_are_whole_when_empty_closed_form_at_zero_and_symmetric() {
    assert_eq!(interval(0, 0, 0.01), (0.0, 1.0));
    for n in [1, 10, 1000] {
        let (lo, hi) = interval(0, n, 0.01);
        assert_eq!(lo, 0.0);
        assert!((hi - (1.0 - ((0.01f64.ln() + log_mixture(0, n)) / n as f64).exp())).abs() < 1e-12);
        assert!(hi < (100f64.ln() + 0.5 * (n as f64).ln() + 1.0) / n as f64);
    }
    let mut widths = Vec::new();
    for n in [10, 100, 1000, 10_000] {
        let k = n / 3;
        let (lo, hi) = interval(k, n, 1e-11);
        assert!(lo < k as f64 / n as f64 && hi > k as f64 / n as f64);
        widths.push(hi - lo);
        let (lo_r, hi_r) = interval(n - k, n, 1e-11);
        assert!((lo_r - (1.0 - hi)).abs() < 1e-12 && (hi_r - (1.0 - lo)).abs() < 1e-12);
    }
    assert!(widths.is_sorted_by(|a, b| a > b));
    assert!(widths[3] < 0.08);
}

#[test]
fn intervals_cover_the_exact_kernel_at_every_sample_size_on_1000_small_decks() {
    let (alpha, draws) = (0.05, 2000);
    let pairs = pairs();
    let results: Vec<(bool, f64)> = decks(19, 1000, 19, 19)
        .into_par_iter()
        .enumerate()
        .map(|(seed, deck)| {
            let pair = pairs[bounded(&mut stream(20_261_029, 0, seed as u64), 25) as usize];
            let exact = kernel_counts(deck, pair);
            let total = exact.iter().sum::<u64>() as f64;
            let classes = round_classes(deck, pair, 1000 + seed as u64, 0, draws);
            let mut counts = [0u64; 6];
            let mut covered = true;
            for (n, &class) in classes.iter().enumerate() {
                counts[class] += 1;
                for c in REACHABLE {
                    let (lo, hi) = interval(counts[c], n as u64 + 1, alpha / 5.0);
                    let p = exact[c] as f64 / total;
                    covered &= lo <= p && p <= hi;
                }
            }
            let width: f64 = REACHABLE
                .iter()
                .map(|&c| interval(counts[c], draws, alpha / 5.0))
                .map(|(lo, hi)| hi - lo)
                .sum();
            (covered, width / 5.0)
        })
        .collect();
    let violations = results.iter().filter(|r| !r.0).count();
    assert!(
        violations as f64 <= alpha * 1000.0,
        "{violations} seeds with a violation"
    );
    assert!(results.iter().map(|r| r.1).sum::<f64>() / 1000.0 < 0.1);
}

#[test]
fn boxes_pin_the_tie_and_contain_the_frequencies() {
    for counts in [[900, 50, 40, 10, 0, 0], [0, 0, 0, 0, 0, 1000]] {
        let (lo, hi) = entry_box(&counts, 1e-3);
        assert_eq!((lo[TIE], hi[TIE]), (0.0, 0.0));
        let n = counts.iter().sum::<u64>() as f64;
        for c in 0..6 {
            let p = counts[c] as f64 / n;
            assert!(lo[c] <= p + 1e-15 && hi[c] >= p - 1e-15);
        }
    }
    assert!(
        entry_box(&[0, 0, 0, 0, 0, 1000], 1e-3).1[..5]
            .iter()
            .all(|&h| h < 0.03)
    );
}

#[test]
fn the_table_counts_sequences_prefixes_and_entries() {
    let sequences = Sequences::new();
    assert_eq!(sequences.targets.len(), 14_400);
    assert_eq!(
        sequences
            .prefix_index
            .iter()
            .map(std::collections::HashMap::len)
            .collect::<Vec<_>>(),
        [25, 400, 3600, 14_400, 14_400]
    );
    assert_eq!(ENTRIES, 21_904_800);
    assert_eq!(
        sequences.targets[0],
        [(25, 8), (26, 15), (36, 44), (39, 47), (45, 63)]
    );
    assert_eq!(
        (
            HIDDEN,
            class_code(&[]),
            class_code(&[Some(W0C), None, Some(NOJ)])
        ),
        (5, 0, 36 + 5 * 6 + 4)
    );
}

#[test]
fn sequences_sharing_a_prefix_add_into_one_entry_and_hidden_classes_pool() {
    let sequences = Sequences::new();
    let mut table = KernelCounts::new(&sequences);
    let shared: Vec<usize> = (0..14_400)
        .filter(|&s| sequences.targets[s][..2] == [(25, 8), (26, 15)])
        .collect();
    assert_eq!(shared.len(), 36);
    let mut outputs = Vec::new();
    for &s in &shared[..3] {
        let (counts, errors) = sample_sequence(&sequences.targets[s], 9, s as u64, 20_000);
        assert_eq!(errors, 0);
        table.add(&sequences, s, &counts);
        outputs.push(counts);
    }
    let p1 = sequences.prefix_of_sequence[1][shared[0]];
    for (d, entry) in table.revealed[1][p1 * 5..p1 * 5 + 5].iter().enumerate() {
        let expected: Counts =
            std::array::from_fn(|c| outputs.iter().map(|o| o[ROW_OFFSET[1] + d][c]).sum());
        assert_eq!(*entry, expected);
    }
    assert_eq!(
        table.revealed[1][p1 * 5..p1 * 5 + 5]
            .iter()
            .flatten()
            .sum::<u64>(),
        60_000
    );
    let p2 = sequences.prefix_of_sequence[2][shared[0]];
    let pooled = table.pooled(2, p2);
    let revealed = &table.revealed[2][p2 * 25..p2 * 25 + 25];
    let sum = |rows: &mut dyn Iterator<Item = &Counts>| {
        rows.fold([0; 6], |s: Counts, r| std::array::from_fn(|c| s[c] + r[c]))
    };
    for first in REACHABLE {
        let d = ap::digit(first).expect("revealed");
        assert_eq!(
            pooled[class_code(&[Some(first), None])],
            sum(&mut revealed[d * 5..d * 5 + 5].iter())
        );
    }
    let both_hidden = pooled[class_code(&[None, None])];
    assert_eq!(both_hidden, sum(&mut revealed.iter()));
    let same_three = shared[..3]
        .iter()
        .filter(|&&s| sequences.prefix_of_sequence[2][s] == p2)
        .count();
    assert_eq!(same_three, 2);
    assert_eq!(both_hidden.iter().sum::<u64>(), 20_000 * same_three as u64);
}
