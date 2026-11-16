//! The stage kernel and the packing LP against explicit-matrix `HiGHS` certificates.

use stl_leap::kernel::{Children, Residue, Solved, Table, kind, sweep};
use stl_leap::lp::certify;
use stl_leap::packing::{Packer, Start};
use stl_leap::rules::ACTIONS as A;

/// A seeded `SplitMix64` stream of uniform numbers.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        lo + (hi - lo) * ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// Stages `(s, f)` solved by one Checker profile against one Dropper row per stage.
struct Stages {
    success: Vec<f64>,
    failure: Vec<f64>,
}

impl Stages {
    fn new(stages: &[([f64; A], f64)]) -> Self {
        Self {
            success: stages.iter().flat_map(|(s, _)| s.map(|x| -x)).collect(),
            failure: stages.iter().map(|&(_, f)| -f).collect(),
        }
    }

    fn sweep(&self, window: bool, residue: Residue) -> Vec<Solved> {
        let columns = [std::array::from_fn(|k| k as u32)];
        let children = Children {
            success: Table {
                values: &self.success,
                width: A,
            },
            failure: Table {
                values: &self.failure,
                width: 1,
            },
            success_columns: &columns,
            failure_columns: &[0],
            revival: &[1.0],
        };
        let classes: Vec<(u32, u32)> = (0..self.failure.len() as u32).map(|pd| (0, pd)).collect();
        sweep(&classes, &children, window, residue).expect("valid stages")
    }
}

fn lp(s: &[f64; A], f: f64, window: bool) -> f64 {
    certify(s, f, window).expect("a certificate").value
}

/// The asymmetric-support stage of the reference tests; its value is VALUE.
#[rustfmt::skip]
const S: [f64; A] = [
    -0.1284722791674977, -0.12311238213031911, -0.11783938104329245, -0.11229663346467983,
    -0.10668796350604809, -0.10083315349960814, -0.09537470669625181, -0.08977130778795045,
    -0.08339761432886156, -0.07769516443214197, -0.07192164010554408, -0.06616436419483152,
    -0.06006093717963651, -0.053227655288462405, -0.047598391274268514, -0.041854762653576905,
    -0.03587453314612943, -0.03003252289938163, -0.023603057751273637, -0.017586484540291714,
    -0.011219842703011795, -0.005275671293213546, 0.0008117282032785167, 0.007153154358869429,
    0.013428233497556619, 0.01974489537252866, 0.026399180249675525, 0.03279969710950498,
    -0.00047749039889410927, 0.005131084056217761, 0.010850843327395387, 0.01664725521396604,
    0.022484377545262724, 0.02847221115882731, 0.03448311001780113, 0.04064462357322546,
    0.04664102425924954, 0.05273846839689167, 0.05896383364466534, 0.06518391929773504,
    0.07147635977875716, 0.07779174310726783, 0.08423778804274584, 0.09077531440470277,
    0.09702645256663872, 0.1033539948162649, 0.1097535240115865, 0.11616685105592794,
    0.12271761517131707, 0.1292247029577216, 0.13584770092279674, 0.14236099302853944,
    0.14894853172567968, 0.15561553076972012, 0.1622855862936653, 0.17099723315775567,
    0.1901691314113193, 0.19683910407608857, 0.20375096839599455, 0.2099309116592808,
];
const F: f64 = 0.813_835_962_781_682;
const VALUE: f64 = 0.150_404_773_005_883_6;

/// Slide a 60-second window over one kink, as consecutive Checker vials do.
fn shifted_stages(n: usize) -> Vec<([f64; A], f64)> {
    let mut sigma = [0.0; 100];
    let mut total = -0.35;
    for (k, x) in sigma.iter_mut().enumerate() {
        total += 0.004 + 0.004 * k as f64 / 99.0;
        *x = total - if k >= 70 { 0.04 } else { 0.0 };
    }
    (11..11 + n)
        .map(|j| (std::array::from_fn(|k| sigma[j + k]), 0.55))
        .collect()
}

/// Increasing payoffs with one drop, and a failure payoff above `s[0]`: packing-LP residue.
fn kinked_stages(n: usize, rng: &mut Rng) -> Vec<([f64; A], f64)> {
    (0..n)
        .map(|_| {
            let mut s = [0.0; A];
            let mut total = -rng.uniform(0.2, 0.5);
            let (cut, drop) = (rng.uniform(1.0, 60.0) as usize, rng.uniform(0.005, 0.08));
            for (k, x) in s.iter_mut().enumerate() {
                total += rng.uniform(0.002, 0.01);
                *x = total - if k >= cut { drop } else { 0.0 };
            }
            let f = (s[0] + rng.uniform(0.2, 1.0)).min(1.0);
            (s, f)
        })
        .collect()
}

/// Random stages until `count` have `f < s[0]` and need more than the closed-form rungs.
fn unsupported_residue(count: usize, rng: &mut Rng) -> Vec<([f64; A], f64)> {
    let mut found = vec![];
    while found.len() < count {
        let stage: ([f64; A], f64) = (
            std::array::from_fn(|_| rng.uniform(-1.0, 1.0)),
            rng.uniform(-1.0, 1.0),
        );
        if stage.1 < stage.0[0]
            && Stages::new(&[stage]).sweep(false, Residue::Defer)[0].kind == kind::UNSOLVED
        {
            found.push(stage);
        }
    }
    found
}

#[test]
fn kernel_matches_lp_with_spare_lanes() {
    let mut rng = Rng(1981);
    // Monotone payoffs exercise the closed-form equalizer; random ones the clipped check.
    let mut stages: Vec<([f64; A], f64)> = (0..23)
        .map(|_| {
            let mut s: [f64; A] = std::array::from_fn(|_| rng.uniform(-0.7, 0.7));
            s.sort_by(f64::total_cmp);
            (s, 0.8 * rng.uniform(0.91, 0.99) - 0.2)
        })
        .collect();
    stages.extend((0..30).map(|_| {
        (
            std::array::from_fn(|_| rng.uniform(-1.0, 1.0)),
            rng.uniform(-1.0, 1.0),
        )
    }));
    for n in [1, 15, 16, 17, stages.len()] {
        for window in [false, true] {
            let solved = Stages::new(&stages[..n]).sweep(window, Residue::Packing);
            for ((s, f), result) in stages[..n].iter().zip(&solved) {
                let value = if result.kind == kind::UNSOLVED {
                    lp(s, *f, window)
                } else {
                    result.value
                };
                assert!((value - lp(s, *f, window)).abs() <= 1e-8, "{result:?}");
            }
        }
    }
}

#[test]
fn kernel_rejects_bad_children() {
    let stages = Stages::new(&[(S, F)]);
    let columns = [std::array::from_fn(|k| k as u32)];
    let good = Children {
        success: Table {
            values: &stages.success,
            width: A,
        },
        failure: Table {
            values: &stages.failure,
            width: 1,
        },
        success_columns: &columns,
        failure_columns: &[0],
        revival: &[1.0],
    };
    assert!(sweep(&[(0, 0)], &good, false, Residue::Defer).is_ok());
    let nan = [f64::NAN; A];
    let bad_column = [std::array::from_fn(|k| k as u32 + 1)];
    let cases = [
        (
            Children {
                success: Table {
                    values: &nan,
                    width: A,
                },
                ..good
            },
            (0, 0),
        ),
        (
            Children {
                success_columns: &bad_column,
                ..good
            },
            (0, 0),
        ),
        (
            Children {
                revival: &[1.01],
                ..good
            },
            (0, 0),
        ),
        (good, (0, 1)),
        (good, (1, 0)),
    ];
    for (children, class) in cases {
        assert!(sweep(&[class], &children, false, Residue::Defer).is_err());
    }
}

#[test]
fn recurrence_overflow_is_left_to_highs() {
    let mut s = [-0.9; A];
    s[0] = 2e-6;
    let solved = Stages::new(&[(s, 0.0)]).sweep(false, Residue::Defer);
    assert_eq!(solved[0].kind, kind::UNSOLVED);
    assert!(solved[0].value.is_nan());
    let certificate = certify(&s, 0.0, false).expect("a certificate");
    assert!(certificate.gap <= 1e-6);
}

#[test]
fn residue_modes_agree_with_lp() {
    let mut rng = Rng(23);
    let mut stages = vec![(S, F)];
    stages.extend(shifted_stages(30));
    stages.extend(kinked_stages(200, &mut rng));
    stages.extend(unsupported_residue(3, &mut rng));
    let table = Stages::new(&stages);
    for window in [false, true] {
        let plain = table.sweep(window, Residue::Defer);
        for residue in [Residue::Packing, Residue::Kink] {
            let solved = table.sweep(window, residue);
            let mut kinks = 0;
            for (((s, f), before), after) in stages.iter().zip(&plain).zip(&solved) {
                if before.kind != kind::UNSOLVED {
                    assert_eq!(before.kind, after.kind);
                    assert_eq!(before.value.to_bits(), after.value.to_bits());
                } else if *f < s[0] {
                    // The packing LP needs f > s[0]; HiGHS takes these stages.
                    assert_eq!(after.kind, kind::UNSOLVED);
                } else {
                    // With f > s[0] the lifted pure bounds settle every window stage.
                    assert!(!window);
                    assert!([kind::PACKING, kind::KINK].contains(&after.kind));
                    assert!((after.value - lp(s, *f, window)).abs() <= 1e-8);
                    kinks += usize::from(after.kind == kind::KINK);
                }
            }
            assert_eq!(
                kinks > 10,
                residue == Residue::Kink && !window,
                "{residue:?} {window}: {kinks} kink solves"
            );
        }
    }
}

#[test]
fn kink_seeds_follow_the_shifted_kink_and_restart_per_chunk() {
    let stages = shifted_stages(30);
    let solved = Stages::new(&stages).sweep(false, Residue::Kink);
    assert_eq!(solved[0].kind, kind::PACKING);
    assert!(solved.iter().filter(|x| x.kind == kind::KINK).count() >= 20);
    for ((s, f), result) in stages.iter().zip(&solved) {
        assert!((result.value - lp(s, *f, false)).abs() <= 1e-8);
    }
    let tiled: Vec<_> = stages.iter().copied().cycle().take(30 * 80).collect();
    let solved = Stages::new(&tiled).sweep(false, Residue::Kink);
    // Each 1024-class chunk starts without a seed, so its first residue stage uses the LP.
    assert!(
        [0, 1024, 2048]
            .iter()
            .all(|&i| solved[i].kind == kind::PACKING)
    );
    assert!(
        solved
            .iter()
            .enumerate()
            .all(|(i, x)| (x.value - solved[i % 30].value).abs() <= 1e-6)
    );
}

#[test]
fn crash_packing_matches_slack_and_lp() {
    let mut rng = Rng(17);
    let mut stages = vec![(S, F)];
    stages.extend(shifted_stages(30));
    stages.extend(kinked_stages(300, &mut rng));
    let mut packer = Packer::default();
    for (i, (s, f)) in stages.iter().enumerate() {
        let (crash, p, q) = packer
            .solve(s, *f, Start::Crash)
            .expect("a certified basis");
        assert!(p != 0 && p.count_ones() == q.count_ones());
        let (slack, _, _) = packer
            .solve(s, *f, Start::Slack)
            .expect("a certified basis");
        assert!((crash - slack).abs() <= 1e-6);
        if i % 7 == 0 {
            assert!((crash - lp(s, *f, false)).abs() <= 1e-8);
        }
    }
    let (value, _, _) = packer
        .solve(&S, F, Start::Crash)
        .expect("a certified basis");
    assert!((value - VALUE).abs() < 1e-9);
}

#[test]
fn packing_scales_with_the_stage_and_rejects_stages_without_packing_form() {
    let mut packer = Packer::default();
    for i in 0..17 {
        let scale = 0.7 + 0.3 * f64::from(i) / 16.0;
        let (value, _, _) = packer
            .solve(&S.map(|x| scale * x), scale * F, Start::Slack)
            .expect("a certified basis");
        assert!((value - scale * VALUE).abs() < 1e-9);
    }
    // f <= s[0] + 1e-12 has no packing form; neither does a nonfinite payoff.
    for (s, f) in [
        ([1.0; A], 0.0),
        ([1.0; A], 1.0 + 1e-13),
        ([f64::NAN; A], 1.0),
    ] {
        for start in [Start::Slack, Start::Crash] {
            assert!(packer.solve(&s, f, start).is_none());
        }
    }
}
