//! Explicit-matrix certificates from `HiGHS`, for stages that the kernel leaves unsolved and for
//! the audit.
//!
//! The Dropper's LP is `max v` subject to `v <= (p^T M)_j` for every column and `sum p = 1`; the
//! row duals of the column constraints are the Checker's mixture. Both mixtures are clipped at
//! zero, normalized and checked on the original matrix, so the result does not depend on `HiGHS`
//! tolerances: it passes only when `max_i (Mq)_i - min_j (p^T M)_j <= 1e-6`. A failed attempt is
//! retried with the interior-point solver, then on the matrix shifted by `M[last, 0]` and by
//! `M[0, 0]`, which recentres badly scaled stages.

use crate::Error;
use crate::kernel::GAP;
use crate::rules::ACTIONS as A;
use highs::{ColProblem, HighsModelStatus, Sense};
use serde::Serialize;

/// A certified solution of a zero-sum matrix game, from the row player's side.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Certificate {
    /// The midpoint of the saddle bounds.
    pub value: f64,
    /// The row player's guaranteed payoff, `min_j (p^T M)_j`.
    pub lower: f64,
    /// The column player's guaranteed bound, `max_i (Mq)_i`.
    pub upper: f64,
    /// The saddle gap `upper - lower`.
    pub gap: f64,
    /// The row player's (Dropper's) mixture.
    pub drop: Vec<f64>,
    /// The column player's (Checker's) mixture.
    pub check: Vec<f64>,
}

/// Certify the stage `(s, f)`: `60 + window` Dropper rows of 60 Checker columns, `s[j - i]` where
/// the check `j` is at or after the drop `i` and `f` elsewhere. Row 61 is Baku's leap drop.
///
/// # Errors
///
/// [`Error::Invalid`] for a nonfinite payoff and [`Error::Uncertified`] when no attempt meets the
/// `1e-6` gate.
pub fn certify(s: &[f64; A], f: f64, window: bool) -> Result<Certificate, Error> {
    certify_matrix(&stage_matrix(s, f, window), A + usize::from(window))
}

fn stage_matrix(s: &[f64; A], f: f64, window: bool) -> Vec<f64> {
    let rows = A + usize::from(window);
    (0..rows * A)
        .map(|cell| {
            if cell / A <= cell % A {
                s[cell % A - cell / A]
            } else {
                f
            }
        })
        .collect()
}

/// Certify the game with `rows` rows whose payoffs `matrix` lists row by row.
fn certify_matrix(matrix: &[f64], rows: usize) -> Result<Certificate, Error> {
    if matrix.iter().any(|x| !x.is_finite()) {
        return Err(Error::Invalid("matrix must contain finite payoffs".into()));
    }
    let cols = matrix.len() / rows;
    let (last, first) = (matrix[(rows - 1) * cols], matrix[0]);
    [
        ("simplex", 0.0),
        ("ipm", 0.0),
        ("ipm", last),
        ("ipm", first),
    ]
    .into_iter()
    .find_map(|(solver, offset)| attempt(matrix, rows, cols, solver, offset))
    .ok_or_else(|| Error::Uncertified("HiGHS failed the 1e-6 saddle gate".into()))
}

fn attempt(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    solver: &str,
    offset: f64,
) -> Option<Certificate> {
    let mut problem = ColProblem::default();
    let constraints: Vec<_> = (0..cols).map(|_| problem.add_row(..=0.0)).collect();
    let total = problem.add_row(1.0..=1.0);
    for i in 0..rows {
        let factors = constraints
            .iter()
            .enumerate()
            .map(|(j, &row)| (row, offset - matrix[i * cols + j]));
        problem.add_column(0.0, 0.0.., factors.chain([(total, 1.0)]));
    }
    problem.add_column(
        -1.0,
        ..=f64::INFINITY,
        constraints.iter().map(|&row| (row, 1.0)),
    );
    let mut model = problem.try_optimise(Sense::Minimise).ok()?;
    model.make_quiet();
    for (option, value) in [
        ("primal_feasibility_tolerance", 1e-9),
        ("dual_feasibility_tolerance", 1e-9),
    ] {
        model.try_set_option(option, value).ok()?;
    }
    model.try_set_option("solver", solver).ok()?;
    model.try_set_option("threads", 1).ok()?;
    let solved = model.try_solve().ok()?;
    if solved.status() != HighsModelStatus::Optimal {
        return None;
    }
    let solution = solved.get_solution();
    let drop = normalized(solution.columns()[..rows].iter().map(|&x| x.max(0.0)))?;
    let check = normalized(solution.dual_rows()[..cols].iter().map(|&y| (-y).max(0.0)))?;
    let lower = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|i| drop[i] * matrix[i * cols + j])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    let upper = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| matrix[i * cols + j] * check[j])
                .sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let gap = upper - lower;
    (gap.is_finite() && gap <= GAP).then(|| Certificate {
        value: f64::midpoint(lower, upper),
        lower,
        upper,
        gap,
        drop,
        check,
    })
}

fn normalized(weights: impl Iterator<Item = f64>) -> Option<Vec<f64>> {
    let weights: Vec<f64> = weights.collect();
    let total: f64 = weights.iter().sum();
    (total.is_finite() && total > 0.0).then(|| weights.iter().map(|x| x / total).collect())
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)] // reference stages, copied digit for digit
mod tests {
    use super::*;

    #[test]
    fn matching_pennies_has_value_zero() {
        let result = certify_matrix(&[1.0, -1.0, -1.0, 1.0], 2).expect("a certificate");
        assert!(result.value.abs() < 1e-12 && result.gap <= 1e-6);
        assert!(
            result
                .drop
                .iter()
                .chain(&result.check)
                .all(|x| (x - 0.5).abs() < 1e-9)
        );
        assert!(certify_matrix(&[f64::NAN, 0.0], 1).is_err());
    }

    #[test]
    fn stage_matrix_places_success_on_and_above_the_diagonal() {
        let s: [f64; A] = std::array::from_fn(|k| k as f64);
        let m = stage_matrix(&s, -1.0, true);
        assert_eq!(m.len(), 61 * 60);
        assert_eq!(
            (m[0], m[59], m[60], m[61], m[60 * 60 + 5]),
            (0.0, 59.0, -1.0, 0.0, -1.0)
        );
    }

    /// The REV[2933] stage on which `HiGHS` once returned an unknown status.
    #[rustfmt::skip]
    const REV_2933: [f64; A] = [
        0.5607424418891126, 0.5655518491845215, 0.5703557003984117, 0.5751538917730719, 0.5799463026905874,
        0.584732834250341, 0.6483151058350379, 0.7086023579242888, 0.47224604612927623, 0.4767158090516911,
        0.48118325337218537, 0.4856483277407043, 0.4901109783322807, 0.49457114897107246, 0.4990287813388775,
        0.5034838152362253, 0.5079361889304526, 0.512385839581746, 0.5168327037531836, 0.5212767180054091,
        0.5257178195730805, 0.5301559879950309, 0.5345910960381096, 0.5390231125559077, 0.5434519856808143,
        0.5478776693322658, 0.5523001279516182, 0.5567193234040602, 0.5611352345647862, 0.565547849433565,
        0.5699571654486212, 0.5743631934830495, 0.5787659574049986, 0.5831654953098447, 0.5875618603403369,
        0.5919551211579275, 0.596417074977373, 0.6008109522654486, 0.6052003990242172, 0.6095857644792779,
        0.6139671369133468, 0.6183446727965163, 0.6227185920631692, 0.6270891738258304, 0.631456746960816,
        0.6358216779805593, 0.6401957524718873, 0.6445576496959142, 0.648917109413709, 0.6532747511938127,
        0.6576311560582244, 0.661988792656643, 0.6663442467358592, 0.6706995720939722, 0.6750555527746332,
        0.679411789005464, 0.6837688775376115, 0.6881268553096092, 0.692485841342741, 0.6968458766397327,
    ];

    /// The H2[46] stage that defeated both unshifted `HiGHS` solvers.
    #[rustfmt::skip]
    const H2_46: [f64; A] = [
        -0.6599336293341775, -0.6598371595049548, -0.6596519227840301, -0.6595688060954965, -0.6594876102392648,
        -0.6593841293334477, -0.6593010610142643, -0.6592300217172435, -0.6591642708311296, -0.6591013652732551,
        -0.6590401720591859, -0.6589796617506387, -0.6576852759956346, -0.6576660175796247, -0.657647521262862,
        -0.6576198116525441, -0.6576045536907242, -0.657587172316132, -0.6575733314790461, -0.6575611009335036,
        -0.6575499418399386, -0.6575391689487855, -0.6574303903079805, -0.65742772235835, -0.6574255733427329,
        -0.6574237748632162, -0.6574825326479494, -0.6574284951528107, -0.6574252466275179, -0.6574224647918139,
        -0.6574135443965261, -0.657412881064743, -0.6574113827295205, -0.657411064120967, -0.6574110056117766,
        -0.6574110112173694, -0.6574110407250207, -0.6574109953379883, -0.6574109935884253, -0.657410993581152,
        -0.657410993581152, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518,
        -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518,
        -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518,
        -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518,
    ];

    #[test]
    fn hard_archived_stages_certify() {
        let cases = [
            (REV_2933, 0.569_397_809_993_176_5, 0.569_397_809_603_054_1),
            (H2_46, 0.661_963_134_224_360_4, -0.657_410_993_831_167),
        ];
        for (s, f, expected) in cases {
            let result = certify(&s, f, false).expect("a certificate");
            assert!(result.gap <= 1e-8, "gap {}", result.gap);
            assert!(
                (result.value - expected).abs() < 1e-8,
                "{} != {expected}",
                result.value
            );
        }
        // Success payoffs that dip at action 13 and then stay flat, with f just below one.
        let mut s = [0.311_271_128_357_446_86; A];
        s[..14].copy_from_slice(&[
            0.305_016_041_477_605_14,
            0.305_435_155_836_465_7,
            0.305_850_712_240_700_93,
            0.306_637_393_087_305_2,
            0.307_002_876_682_356_2,
            0.307_632_803_669_328_5,
            0.308_161_209_117_677_2,
            0.308_617_115_152_986_23,
            0.309_018_455_340_374_1,
            0.309_463_237_890_532_4,
            0.309_880_926_562_751_2,
            0.310_271_051_955_546_8,
            0.310_633_129_809_331_2,
            0.297_847_114_549_782_63,
        ]);
        let result = certify(&s, 0.999_999_999_999_999_9, false).expect("a certificate");
        assert!(result.gap <= 1e-8 && (result.value - 0.311_271_128_333_051_04).abs() < 1e-9);
    }
}
