//! The solver: matrix games, the betting DAG and settlement against the engine, the robust round
//! value against an independent reference, robust subtrees, golden parity with the Python solve,
//! self-play through the engine and the allocation rule (ports of `test_matrix_game.py`,
//! `test_betting_dag.py`, `test_robust.py` and `test_selfplay.py`).

use std::collections::HashMap;

use ap::allocate::allocation;
use ap::betting::{BettingDag, CHIPS, End, Settled, reachable_configurations, settle};
use ap::cards::{FULL_DECK, evaluate, legal_hands};
use ap::engine::{Bet, Betting, GameState, HandPair, Lower, Move, Phase, Record, Round, Team};
use ap::matrix::{MAX_SIZE, Matrix, solve};
use ap::robust::{Arena, Chance, Kept, Mixes, Scratch, Solver, Stage, Values, at};
use ap::sampler::{Counts, next64, sample_sequence, stream};
use ap::selfplay::{Policy, play_game, record_digit, slot_action};
use ap::table::{HIDDEN, KernelCounts};
use ap::{ANTES, NOJ, W0, W0C, W1, W1C, digit};
use serde_json::Value;

const WIN: [usize; 4] = [W0, W0C, W1, W1C];

/// Uniform draws from a counter-based stream.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(stream(seed, 0, 0))
    }
    fn uniform(&mut self) -> f64 {
        (next64(&mut self.0) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize
    }
    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture exists"))
        .expect("fixture is JSON")
}

fn round_one() -> Vec<((u8, u8), Counts)> {
    let data = fixture("kernel_round1.json");
    let int = |v: &Value| v.as_u64().expect("count");
    data["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|e| {
            (
                (int(&e[0]) as u8, int(&e[1]) as u8),
                std::array::from_fn(|c| int(&e[2][c])),
            )
        })
        .collect()
}

fn check_game(m: &Matrix, n: usize, cols: usize) {
    let s = solve(m, n, cols);
    let scale = m[..n]
        .iter()
        .flat_map(|r| &r[..cols])
        .fold(1.0_f64, |a, v| a.max(v.abs()));
    // By weak duality low <= value <= high, so a small gap brackets the value.
    assert!(s.high - s.low <= 1e-9 * scale, "{m:?}");
    assert!(
        (s.x[..n].iter().sum::<f64>() - 1.0).abs() < 1e-12
            && (s.y[..cols].iter().sum::<f64>() - 1.0).abs() < 1e-12
    );
    assert!(s.x.iter().chain(&s.y).all(|&p| p >= 0.0));
    let low = (0..cols)
        .map(|j| (0..n).map(|i| s.x[i] * m[i][j]).sum::<f64>())
        .fold(f64::INFINITY, f64::min);
    let high = (0..n)
        .map(|i| (0..cols).map(|j| m[i][j] * s.y[j]).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((low - s.low).abs() <= 1e-12 * scale && (high - s.high).abs() <= 1e-12 * scale);
}

#[test]
fn matrix_games_of_every_shape_close_their_saddle_gap() {
    let mut rng = Rng::new(1);
    for _ in 0..300 {
        let (n, cols) = (1 + rng.below(5), 1 + rng.below(5));
        let scale = [1.0, 0.01, 100.0][rng.below(3)];
        let mut m = [[0.0; MAX_SIZE]; MAX_SIZE];
        m.iter_mut().take(n).for_each(|row| {
            row.iter_mut()
                .take(cols)
                .for_each(|v| *v = rng.between(-2.0, 2.0) * scale);
        });
        check_game(&m, n, cols);
        m.iter_mut().take(n).for_each(|row| {
            row.iter_mut()
                .take(cols)
                .for_each(|v| *v = rng.below(3) as f64 - 1.0);
        });
        check_game(&m, n, cols);
    }
    let pad = |rows: &[&[f64]]| -> Matrix {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| rows.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0))
        })
    };
    check_game(&pad(&[]), 5, 5);
    check_game(&pad(&[&[0.25; 4], &[0.25; 4], &[0.25; 4]]), 3, 4);
    check_game(
        &pad(&[&[0.0, 1.0, -1.0], &[-1.0, 0.0, 1.0], &[1.0, -1.0, 0.0]]),
        3,
        3,
    );
    check_game(&pad(&[&[0.3, 0.3], &[0.3, 0.3], &[0.1, 0.9]]), 3, 2);
    let pennies = solve(&pad(&[&[1.0, -1.0], &[-1.0, 1.0]]), 2, 2);
    assert_eq!(
        (&pennies.x[..2], &pennies.y[..2], pennies.low, pennies.high),
        (&[0.5; 2][..], &[0.5; 2][..], 0.0, 0.0)
    );
    let saddle = solve(&pad(&[&[3.0, 5.0], &[1.0, 2.0]]), 2, 2);
    assert_eq!(
        (&saddle.x[..2], &saddle.y[..2], saddle.low, saddle.high),
        (&[1.0, 0.0][..], &[1.0, 0.0][..], 3.0, 3.0)
    );
}

/// A full-deck hand pair for targets (25, 44) whose showdown gives `class`.
fn hands_of_class(class: usize) -> HandPair {
    let (h0, h1) = (legal_hands(25, FULL_DECK), legal_hands(44, FULL_DECK));
    for &a in h0.iter().step_by(97) {
        for &b in h1.iter().step_by(89) {
            let win = evaluate(a).expect("five") > evaluate(b).expect("five");
            if [W0, W0C, W1, W1C][2 * usize::from(!win) + usize::from(a & b != 0)] == class {
                return [Some(a), Some(b)];
            }
        }
    }
    unreachable!("every class occurs")
}

/// An engine state at the betting phase of round `r` with the given public facts.
fn betting_state(
    r: usize,
    sealed: [u32; 2],
    first: usize,
    contributions: [u32; 2],
    actor: usize,
    checked: bool,
    hands: HandPair,
) -> GameState {
    let lower = |s, cards: &[u8]| Lower {
        sealed_bios: s,
        air_seconds: 300,
        metal_cards: cards.to_vec(),
    };
    let record = Record {
        chosen: [25, 44],
        first: Team::from_index(1 - first),
        actions: vec![],
        winner: Some(Team::Zero),
        calamity: false,
    };
    let betting = Betting {
        first: Team::from_index(first),
        actor: Team::from_index(actor),
        contributions,
        checked,
        actions: vec![],
    };
    GameState {
        phase: Phase::Betting,
        lower: [
            lower(sealed[0], &[26, 36, 39, 45]),
            lower(sealed[1], &[8, 15, 47, 63]),
        ],
        deck: FULL_DECK & !(hands[0].unwrap_or(0) | hands[1].unwrap_or(0)),
        round: Round {
            chosen: [Some(25), Some(44)],
            hands: Some(hands),
            retry: false,
            betting: Some(betting),
        },
        history: vec![record; r],
        winner: None,
    }
}

/// Perfect-information value over the engine's own legal moves with table payoffs.
fn engine_value(
    state: &GameState,
    dag: &BettingDag,
    show: &[f64],
    fold: &[f64],
    memo: &mut HashMap<([u32; 2], usize, bool), f64>,
) -> f64 {
    let b = state.round.betting.as_ref().expect("betting");
    let (actor, c) = (b.actor.index(), b.contributions);
    if let Some(&v) = memo.get(&(c, actor, b.checked)) {
        return v;
    }
    let mut best: Option<f64> = None;
    for mv in state.legal_moves() {
        let next = state.apply(mv).expect("legal");
        let v = if next.history.len() > state.history.len() {
            if mv == Move::Bet(Bet::Fold) {
                fold[dag
                    .folds
                    .iter()
                    .position(|&f| f == (c[0], c[1], actor))
                    .expect("fold terminal")]
            } else {
                let mut called = c;
                called[actor] = c[0].max(c[1]);
                show[dag
                    .showdowns
                    .iter()
                    .position(|&s| s == (called[0], called[1]))
                    .expect("showdown terminal")]
            }
        } else {
            engine_value(&next, dag, show, fold, memo)
        };
        best = Some(best.map_or(v, |b| if actor == 0 { b.max(v) } else { b.min(v) }));
    }
    let v = best.expect("a betting node has moves");
    memo.insert((c, actor, b.checked), v);
    v
}

#[test]
fn reachable_configurations_match_the_measured_reductions() {
    let reach = reachable_configurations();
    assert_eq!(
        reach
            .iter()
            .map(std::collections::BTreeSet::len)
            .collect::<Vec<_>>(),
        [2, 134, 1066, 1394, 1458]
    );
    assert_eq!(BettingDag::new(23, 23, 1, 0).nodes.len(), 290);
}

#[test]
fn dag_values_match_engine_expectimax_on_sampled_configurations() {
    let reach = reachable_configurations();
    let mut rng = Rng::new(3);
    let hands = hands_of_class(W0);
    let mut checked = 0;
    for (r, configs) in reach.iter().enumerate() {
        let configs: Vec<_> = configs.iter().collect();
        for _ in 0..configs.len().min(6) {
            let &&(s0, s1, first) = &configs[rng.below(configs.len())];
            let dag = BettingDag::new(s0 as u32, s1 as u32, ANTES[r], first);
            let show: Vec<f64> = (0..dag.showdowns.len())
                .map(|_| rng.between(-1.0, 1.0))
                .collect();
            let fold: Vec<f64> = (0..dag.folds.len())
                .map(|_| rng.between(-1.0, 1.0))
                .collect();
            let state = betting_state(
                r,
                [s0 as u32, s1 as u32],
                first,
                [ANTES[r]; 2],
                first,
                false,
                hands,
            );
            let expected = engine_value(&state, &dag, &show, &fold, &mut HashMap::new());
            let mut values: Vec<[f64; 2]> = vec![[0.0; 2]; dag.nodes.len()];
            values.extend(show.iter().chain(&fold).map(|&v| [v, v]));
            assert_eq!(
                dag.evaluate(&mut values),
                [expected; 2],
                "{r} {s0} {s1} {first}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 20);
}

fn compare(r: usize, s0: usize, s1: usize, c0: u32, c1: u32, end: End, next: &GameState) {
    match settle(r, s0, s1, c0 as usize, c1 as usize, end) {
        Settled::Terminal(u) => assert_eq!(
            (next.phase, Ok(u)),
            (Phase::Terminal, next.returns().map(|r| r[0])),
            "{r} {s0} {s1} {c0} {c1} {end:?}"
        ),
        Settled::Next(t0, t1) => {
            assert_eq!(
                next.phase,
                Phase::NumberChoice0,
                "{r} {s0} {s1} {c0} {c1} {end:?}"
            );
            assert_eq!(
                [t0, t1],
                next.lower.each_ref().map(|p| p.sealed_bios as usize)
            );
        }
    }
}

#[test]
fn settlement_matches_the_engine_for_every_class_fold_and_no_judgement() {
    let reach = reachable_configurations();
    let mut rng = Rng::new(4);
    let mut cases = 0;
    for (r, configs) in reach.iter().enumerate() {
        let configs: Vec<_> = configs.iter().copied().collect();
        let ante = ANTES[r];
        for _ in 0..configs.len().min(8) {
            let (s0, s1, first) = configs[rng.below(configs.len())];
            let dag = BettingDag::new(s0 as u32, s1 as u32, ante, first);
            let sealed = |c0: u32, c1: u32| [s0 as u32 + ante - c0, s1 as u32 + ante - c1];
            for _ in 0..dag.showdowns.len().min(5) {
                let (c0, c1) = dag.showdowns[rng.below(dag.showdowns.len())];
                for class in WIN {
                    let state = betting_state(
                        r,
                        sealed(c0, c1),
                        first,
                        [c0, c1],
                        1 - first,
                        true,
                        hands_of_class(class),
                    );
                    compare(
                        r,
                        s0,
                        s1,
                        c0,
                        c1,
                        End::Class(class),
                        &state.apply(Move::Bet(Bet::Check)).expect("legal"),
                    );
                    cases += 1;
                }
            }
            for _ in 0..dag.folds.len().min(3) {
                let (c0, c1, folder) = dag.folds[rng.below(dag.folds.len())];
                let state = betting_state(
                    r,
                    sealed(c0, c1),
                    first,
                    [c0, c1],
                    folder,
                    false,
                    hands_of_class(W0),
                );
                compare(
                    r,
                    s0,
                    s1,
                    c0,
                    c1,
                    End::Fold(folder),
                    &state.apply(Move::Bet(Bet::Fold)).expect("legal"),
                );
                cases += 1;
            }
        }
    }
    assert!(cases > 100);
    for r in [3, 4] {
        let (s0, s1, first) = *reach[r].iter().nth(7).expect("configurations");
        let lower = |s, card| Lower {
            sealed_bios: s as u32,
            air_seconds: 300,
            metal_cards: vec![card],
        };
        let record = Record {
            chosen: [25, 44],
            first: Team::from_index(1 - first),
            actions: vec![],
            winner: Some(Team::Zero),
            calamity: false,
        };
        let state = GameState {
            phase: Phase::UpperChance,
            lower: [lower(s0, 26), lower(s1, 8)],
            deck: 0,
            round: Round {
                chosen: [Some(36), Some(63)],
                hands: None,
                retry: true,
                betting: None,
            },
            history: vec![record; r],
            winner: None,
        };
        compare(
            r,
            s0,
            s1,
            ANTES[r],
            ANTES[r],
            End::Class(NOJ),
            &state.apply(Move::Hands([None, None])).expect("legal"),
        );
    }
}

/// A box around a random kernel on the five reachable classes; `point` collapses it to the
/// kernel while drawing the same numbers.
fn random_box(rng: &mut Rng, point: bool) -> ([f64; 6], [f64; 6]) {
    let power = [0.2, 1.0, 5.0][rng.below(3)];
    let mut k = [0.0; 6];
    for c in [W0, W0C, W1, W1C, NOJ] {
        k[c] = (-rng.uniform().max(1e-300).ln()).powf(1.0 / power);
    }
    if rng.uniform() < 0.5 {
        k[NOJ] = 0.0;
    }
    let total: f64 = k.iter().sum();
    for p in &mut k {
        *p /= total;
    }
    let width = rng.between(0.0, 0.3) * if point { 0.0 } else { 1.0 };
    let mut lo = k.map(|p| (p - width * rng.uniform()).clamp(0.0, 1.0));
    let mut hi = k.map(|p| (p + width * rng.uniform()).clamp(0.0, 1.0));
    (lo[4], hi[4]) = (0.0, 0.0);
    if point {
        (lo, hi) = (k, k);
    }
    (lo, hi)
}

/// The extreme of `q . u` over the box intersected with the simplex, by vertex enumeration.
fn vertex_extreme(u: [f64; 4], lo: [f64; 4], hi: [f64; 4], minimize: bool) -> f64 {
    let mut best: Option<f64> = None;
    for free in 0..4 {
        for mask in 0..8 {
            let mut q = [0.0; 4];
            let others: Vec<usize> = (0..4).filter(|&c| c != free).collect();
            for (bit, &c) in others.iter().enumerate() {
                q[c] = if mask >> bit & 1 == 1 { hi[c] } else { lo[c] };
            }
            q[free] = 1.0 - others.iter().map(|&c| q[c]).sum::<f64>();
            if q[free] < lo[free] - 1e-12 || q[free] > hi[free] + 1e-12 {
                continue;
            }
            let v: f64 = (0..4).map(|c| q[c] * u[c]).sum();
            best = Some(best.map_or(v, |b| if minimize { b.min(v) } else { b.max(v) }));
        }
    }
    best.expect("the box meets the simplex")
}

/// Plain `W` with vertex extremes and the same DAG, settlement and continuation rules.
fn w_reference(chance: &Chance<'_>, s0: usize, s1: usize) -> (f64, f64) {
    let dag = BettingDag::new(s0 as u32, s1 as u32, ANTES[chance.r], chance.first);
    let (t_lo, t_hi) = (chance.lo[NOJ], chance.hi[NOJ]);
    let q_lo = WIN.map(|c| {
        if t_lo >= 1.0 {
            0.0
        } else {
            chance.lo[c] / (1.0 - t_lo)
        }
    });
    let q_hi = WIN.map(|c| {
        if t_hi >= 1.0 {
            1.0
        } else {
            (chance.hi[c] / (1.0 - t_hi)).min(1.0)
        }
    });
    let cont = |c0: u32, c1: u32, end: End, child: usize| match settle(
        chance.r,
        s0,
        s1,
        c0 as usize,
        c1 as usize,
        end,
    ) {
        Settled::Terminal(u) => (u, u),
        Settled::Next(t0, t1) => {
            let next = chance.next.expect("a next round");
            (
                next.lo[at(chance.code * 6 + child, t0, t1)],
                next.hi[at(chance.code * 6 + child, t0, t1)],
            )
        }
    };
    let (mut show_lo, mut show_hi) = (Vec::new(), Vec::new());
    for &(c0, c1) in &dag.showdowns {
        let u = WIN.map(|c| cont(c0, c1, End::Class(c), c));
        show_lo.push(vertex_extreme(u.map(|x| x.0), q_lo, q_hi, true));
        show_hi.push(vertex_extreme(u.map(|x| x.1), q_lo, q_hi, false));
    }
    let fold: Vec<(f64, f64)> = dag
        .folds
        .iter()
        .map(|&(c0, c1, who)| cont(c0, c1, End::Fold(who), HIDDEN))
        .collect();
    let mut values = vec![[0.0; 2]; dag.nodes.len()];
    values.extend(
        show_lo
            .iter()
            .zip(&show_hi)
            .map(|(&l, &h)| [l, h])
            .chain(fold.iter().map(|&(l, h)| [l, h])),
    );
    let [b_lo, b_hi] = dag.evaluate(&mut values);
    let (n_lo, n_hi) = cont(
        ANTES[chance.r],
        ANTES[chance.r],
        End::Class(NOJ),
        digit(NOJ).expect("digit"),
    );
    let w_lo = [t_lo, t_hi].map(|t| t * n_lo + (1.0 - t) * b_lo);
    let w_hi = [t_lo, t_hi].map(|t| t * n_hi + (1.0 - t) * b_hi);
    (w_lo[0].min(w_lo[1]), w_hi[0].max(w_hi[1]))
}

#[test]
fn the_round_value_matches_a_vertex_enumeration_reference() {
    let arena = Arena::new();
    let mut rng = Rng::new(8);
    for r in 0..5 {
        for _ in 0..6 {
            let first = rng.below(2);
            let chips = &arena.chips[r][first];
            let (s0, s1) = chips[rng.below(chips.len())];
            let code = rng.below(6usize.pow(r as u32));
            let codes = 6usize.pow(r as u32 + 1);
            let lo: Vec<f64> = (0..codes * CHIPS * CHIPS)
                .map(|_| rng.between(-1.0, 1.0))
                .collect();
            let hi = lo
                .iter()
                .map(|&v| (v + rng.between(0.0, 0.5)).min(1.0))
                .collect();
            let next = Values { lo, hi };
            let (box_lo, box_hi) = random_box(&mut rng, false);
            let chance = Chance {
                r,
                first,
                code,
                lo: &box_lo,
                hi: &box_hi,
                next: (r < 4).then_some(&next),
            };
            let (w_lo, w_hi) = arena.round_value(&chance, s0, s1, &mut Scratch::default());
            let (e_lo, e_hi) = w_reference(&chance, s0, s1);
            assert!(
                (w_lo - e_lo).abs() < 1e-9 && (w_hi - e_hi).abs() < 1e-9,
                "{r} {first} {s0} {s1}"
            );
            assert!(w_lo <= w_hi + 1e-12);
        }
    }
}

/// A round-4 subtree under random boxes for the number prefixes below `prefix`.
fn subtree(solver: &mut Solver, rng: &mut Rng, point: bool) -> Stage {
    let prefix = solver.sequences.targets[7][..3].to_vec();
    for r in [3, 4] {
        let codes = 6usize.pow(r as u32);
        let mut indices: Vec<usize> = solver.sequences.prefix_index[r]
            .iter()
            .filter(|(k, _)| k[..3] == prefix[..])
            .map(|(_, &i)| i)
            .collect();
        indices.sort_unstable();
        for index in indices {
            for code in 0..codes {
                (
                    solver.box_lo[r][index * codes + code],
                    solver.box_hi[r][index * codes + code],
                ) = random_box(rng, point);
            }
        }
    }
    solver.v(
        &prefix,
        3,
        &mut Kept {
            depth: 0,
            stages: HashMap::new(),
        },
    )
}

#[test]
fn point_kernels_collapse_the_interval_and_wider_boxes_widen_it() {
    let mut solver = Solver::new(&round_one());
    let point = subtree(&mut solver, &mut Rng::new(10), true);
    let wide = subtree(&mut solver, &mut Rng::new(10), false);
    let first = Solver::first(&solver.sequences.targets[7], 3);
    let reachable: Vec<usize> = (0..216)
        .flat_map(|code| {
            solver.arena.chips[3][first]
                .iter()
                .map(move |&(s0, s1)| at(code, s0, s1))
        })
        .collect();
    let finite = point.values.lo.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, reachable.len());
    let mut width = 0.0;
    for &i in &reachable {
        let (lo, hi) = (point.values.lo[i], point.values.hi[i]);
        assert!((lo - hi).abs() < 1e-9 && lo.abs() <= 1.0 + 1e-12);
        assert!(wide.values.lo[i] <= lo + 1e-9 && wide.values.hi[i] >= hi - 1e-9);
        width += wide.values.hi[i] - wide.values.lo[i];
    }
    assert!(width / reachable.len() as f64 > 0.01);
    assert!(point.gap <= 1e-9);
    assert_eq!((point.rows.len(), point.cols.len()), (2, 2));
    assert!(
        point
            .mixes
            .iter()
            .all(|m| (m.x_lo.iter().sum::<f64>() - 1.0).abs() < 1e-12)
    );
}

#[test]
fn a_sampled_subtree_matches_the_python_solve() {
    let data = fixture("robust.json");
    let int = |v: &Value| v.as_u64().expect("integer");
    let pairs = |v: &Value| {
        v.as_array()
            .expect("pairs")
            .iter()
            .map(|p| (int(&p[0]) as u8, int(&p[1]) as u8))
            .collect::<Vec<_>>()
    };
    let mut solver = Solver::new(&round_one());
    let mut table = KernelCounts::new(&solver.sequences);
    for s in data["sampled"]
        .as_array()
        .expect("sampled")
        .iter()
        .map(|s| int(s) as usize)
    {
        let (counts, errors) = sample_sequence(
            &solver.sequences.targets[s],
            int(&data["seed"]),
            s as u64,
            int(&data["n"]),
        );
        assert_eq!(errors, 0);
        table.add(&solver.sequences, s, &counts);
    }
    solver.set_boxes(&table, data["delta"].as_f64().expect("delta"));
    let stage = solver.v(
        &pairs(&data["prefix"]),
        3,
        &mut Kept {
            depth: 0,
            stages: HashMap::new(),
        },
    );
    // Boxes differ from the reference by lgamma ulps and summation order (below 1e-13).
    let close = |a: f64, b: &Value| (a - b.as_f64().expect("float")).abs() < 1e-9;
    for v in data["v3"].as_array().expect("values") {
        let i = at(
            int(&v[0]) as usize,
            int(&v[1]) as usize,
            int(&v[2]) as usize,
        );
        assert!(
            close(stage.values.lo[i], &v[3]) && close(stage.values.hi[i], &v[4]),
            "{v}"
        );
    }
    assert!(stage.gap <= 1e-9 && data["max_gap"].as_f64().expect("gap") <= 1e-9);
    for m in data["strategies"].as_array().expect("strategies") {
        let n = stage
            .chips
            .iter()
            .position(|&c| c == (int(&m[1]) as usize, int(&m[2]) as usize))
            .expect("chips");
        let mix = stage.mixes[int(&m[0]) as usize * stage.chips.len() + n];
        for (got, want) in [mix.x_lo, mix.y_lo, mix.x_hi, mix.y_hi]
            .iter()
            .zip(&m.as_array().expect("row")[3..])
        {
            assert!(
                got[..2]
                    .iter()
                    .zip(want.as_array().expect("mix"))
                    .all(|(g, w)| close(*g, w)),
                "{m}"
            );
        }
    }
    let (w, _) = solver.w(
        &pairs(&data["w4_prefix"]),
        4,
        &mut Kept {
            depth: 0,
            stages: HashMap::new(),
        },
    );
    for v in data["w4"].as_array().expect("values") {
        let i = at(
            int(&v[0]) as usize,
            int(&v[1]) as usize,
            int(&v[2]) as usize,
        );
        assert!(close(w.lo[i], &v[3]) && close(w.hi[i], &v[4]), "{v}");
    }
}

#[test]
fn record_digits_and_slots_map_to_public_classes_and_legal_bets() {
    let record = |actions: Vec<(Team, Bet)>, winner, calamity| Record {
        chosen: [25, 8],
        first: Team::Zero,
        actions,
        winner,
        calamity,
    };
    assert_eq!(
        record_digit(&record(
            vec![(Team::One, Bet::Fold)],
            Some(Team::Zero),
            false
        )),
        5
    );
    assert_eq!(record_digit(&record(vec![], None, false)), 4);
    let digits = [
        (Team::Zero, false),
        (Team::Zero, true),
        (Team::One, false),
        (Team::One, true),
    ]
    .map(|(w, c)| record_digit(&record(vec![], Some(w), c)));
    assert_eq!(digits, [0, 1, 2, 3]);
    let mut rng = Rng::new(12);
    let hands = hands_of_class(W0);
    for _ in 0..40 {
        let r = rng.below(5);
        let ante = ANTES[r];
        let sealed = [
            ante + rng.below(30 - ante as usize) as u32,
            ante + rng.below(30 - ante as usize) as u32,
        ];
        let c = [ante + rng.below(6) as u32, ante + rng.below(6) as u32];
        let actor = rng.below(2);
        let state = betting_state(r, sealed, 0, c, actor, rng.below(2) == 1, hands);
        let legal = state.legal_moves();
        let to_call = c[1 - actor].saturating_sub(c[actor]);
        let translated: Vec<Move> = (0..legal.len())
            .map(|slot| Move::Bet(slot_action(slot, to_call)))
            .collect();
        let (mut a, mut b) = (legal.clone(), translated.clone());
        a.sort_by_key(|m| format!("{m:?}"));
        b.sort_by_key(|m| format!("{m:?}"));
        assert_eq!(a, b, "{r} {sealed:?} {c:?} {actor}");
        b.dedup();
        assert_eq!(b.len(), translated.len());
    }
}

#[test]
fn self_play_runs_whole_games_through_the_engine_reproducibly() {
    let mut rng = Rng::new(13);
    let mut solver = Solver::new(&round_one());
    for r in 1..5 {
        for (lo, hi) in solver.box_lo[r].iter_mut().zip(solver.box_hi[r].iter_mut()) {
            *lo = std::array::from_fn(|_| rng.between(0.0, 0.2));
            *hi = std::array::from_fn(|c| (lo[c] + rng.between(0.2, 0.6)).min(1.0));
            (lo[4], hi[4]) = (0.0, 0.0);
        }
    }
    let mut kept = Kept {
        depth: 2,
        stages: HashMap::new(),
    };
    let all = |w: f64| Mixes {
        x_lo: [w; 5],
        y_lo: [w; 5],
        x_hi: [w; 5],
        y_hi: [w; 5],
    };
    let root = Stage {
        values: Values {
            lo: vec![],
            hi: vec![],
        },
        rows: ap::TEAM_0_METAL_CARDS.to_vec(),
        cols: ap::TEAM_1_METAL_CARDS.to_vec(),
        chips: vec![ap::robust::ROOT_CHIPS],
        mixes: vec![all(0.2)],
        gap: 0.0,
    };
    kept.stages.insert(Vec::new(), root);
    for r in 1..3 {
        let mut prefixes: Vec<Vec<(u8, u8)>> = solver.sequences.prefix_index[r - 1]
            .keys()
            .cloned()
            .collect();
        prefixes.sort();
        for prefix in prefixes {
            let cells = 6usize.pow(r as u32) * CHIPS * CHIPS;
            let lo: Vec<f64> = (0..cells).map(|_| rng.between(-1.0, 1.0)).collect();
            let hi = lo.iter().map(|v| (v + 0.1).min(1.0)).collect();
            let rows: Vec<u8> = ap::TEAM_0_METAL_CARDS
                .into_iter()
                .filter(|a| prefix.iter().all(|p| p.0 != *a))
                .collect();
            let cols = ap::TEAM_1_METAL_CARDS
                .into_iter()
                .filter(|b| prefix.iter().all(|p| p.1 != *b))
                .collect();
            let chips = solver.arena.chips[r][Solver::first(&prefix, r)].clone();
            let mixes = vec![all(1.0 / rows.len() as f64); 6usize.pow(r as u32) * chips.len()];
            kept.stages.insert(
                prefix,
                Stage {
                    values: Values { lo, hi },
                    rows,
                    cols,
                    chips,
                    mixes,
                    gap: 0.0,
                },
            );
        }
    }
    let (lo, hi) = (
        Policy::new(&solver, &kept, false),
        Policy::new(&solver, &kept, true),
    );
    for index in 0..3 {
        let (utility, played) = play_game([&lo, &hi], 5, index);
        assert!([-1.0, 0.0, 1.0].contains(&utility));
        assert!((1..=5).contains(&played.len()));
        assert!(solver.sequences.prefix_index[played.len() - 1].contains_key(&played));
    }
    assert_eq!(play_game([&lo, &hi], 5, 1), play_game([&lo, &hi], 5, 1));
}

#[test]
fn allocation_gives_a_floor_plus_a_proportional_share_within_budget() {
    let mut reach = vec![0.0; 14_400];
    (reach[3], reach[7], reach[100]) = (1.0, 12.0, 1.0);
    let plan = allocation(&reach, 1_440_000, 0.5);
    assert_eq!(*plan.iter().min().expect("plan"), 50);
    assert!(plan.iter().sum::<u64>() <= 1_440_000);
    assert!(plan[7] > plan[3] && plan[3] == plan[100]);
    assert_eq!(allocation(&[0.0; 4], 100, 0.0), [25; 4]);
}

#[test]
#[ignore = "two full solves, 4,000 self-play games per iteration and 2e8 trajectories: run with --ignored"]
fn the_allocation_loop_reproduces_the_python_root_interval() {
    // docs/results/allocation.json of the Python reference, iterations 0 and 1.
    let config = ap::allocate::Config {
        iterations: 2,
        budget_per_iteration: 1e8,
        floor_fraction: 0.2,
        games_per_equilibrium: 2000,
        seed: 20_261_029,
        delta: 1e-3,
    };
    let report = ap::allocate::run(&config, &round_one(), |_| {});
    let [zero, one] = &report.iterations[..] else {
        unreachable!("two iterations")
    };
    // Iteration 0 plays pure strategies, so self-play draws the same games as the reference.
    assert_eq!(
        (zero.root_lo, zero.root_hi, zero.sequences_reached),
        (-1.0, 1.0, 1)
    );
    assert_eq!(zero.self_play_mean_return, [-0.4085, -0.3925]);
    assert_eq!(
        (one.samples_before, zero.samples_after),
        (100_000_000, 100_000_000)
    );
    assert!(
        (one.root_lo + 0.984_254_722_594_647).abs() < 1e-9,
        "{}",
        one.root_lo
    );
    assert!(
        (one.root_hi - 0.304_601_437_522_803_16).abs() < 1e-9,
        "{}",
        one.root_hi
    );
}
