//! The certified solver ladder for one class (DTH.md 4.4-4.7).
//!
//! The arithmetic follows the reference's numba kernel operation by operation, and `min` and
//! `max` keep its tie rule, so the build reproduces the reference table bit for bit.

use highs::{ColProblem, HighsModelStatus, Sense};

use crate::{LAGS, MAX_SADDLE_GAP};

/// A mixed strategy over the seconds 1..=60, or the 60 successful-check values of a class.
pub type Mix = [f64; LAGS];
/// A stage payoff matrix `M[d][c]`: Dropper rows, Checker columns, from the Dropper's side.
pub type Matrix = [Mix; LAGS];

/// The rung that certified a class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rung {
    /// A pure saddle point.
    Pure,
    /// The full-support equalizer.
    Support,
    /// The linear program.
    Lp,
}

/// Bounds on a stage game's value and the strategies that prove them: `lower = min(drop · M)`
/// and `upper = max(M · check)`.
#[derive(Clone, Debug)]
pub struct Certificate {
    /// The Checker's best reply to `drop`: a lower bound on the value.
    pub lower: f64,
    /// The Dropper's best reply to `check`: an upper bound on the value.
    pub upper: f64,
    /// The rung that produced the certificate.
    pub rung: Rung,
    /// The Dropper's mix.
    pub drop: Mix,
    /// The Checker's mix.
    pub check: Mix,
}

impl Certificate {
    /// The saddle gap `upper − lower`.
    #[must_use]
    pub fn gap(&self) -> f64 {
        self.upper - self.lower
    }

    /// Whether the gap passes the gate; NaN bounds never do.
    #[must_use]
    pub fn certified(&self) -> bool {
        self.gap() <= MAX_SADDLE_GAP
    }

    /// The certificate midpoint, the stored value of a certified class.
    #[must_use]
    pub fn value(&self) -> f64 {
        0.5 * (self.lower + self.upper)
    }
}

/// The reference's `min`: the first argument wins ties, which fixes the sign of a zero.
fn min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// The reference's `max`: the first argument wins ties.
fn max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// The pure strategy that plays second `i + 1`.
fn unit(i: usize) -> Mix {
    std::array::from_fn(|k| f64::from(u8::from(k == i)))
}

/// The full matrix: `M[d][c] = s[c − d]` when the check succeeds (`c ≥ d`), else `f`.
#[must_use]
pub fn full_matrix(s: &Mix, f: f64) -> Matrix {
    std::array::from_fn(|d| std::array::from_fn(|c| if c >= d { s[c - d] } else { f }))
}

/// Rung 1, the pure saddle test. Row `d` of `M` is `d − 1` copies of `f` and then `s`, so the best
/// pure row is row 1 or row 60 and the best pure column is column 1 or column 60.
#[must_use]
pub fn pure(s: &Mix, f: f64) -> Certificate {
    let lo = s.iter().fold(s[0], |a, &b| min(a, b));
    let hi = s.iter().fold(s[0], |a, &b| max(a, b));
    Certificate {
        lower: max(lo, min(f, s[0])),
        upper: min(hi, max(f, s[0])),
        rung: Rung::Pure,
        drop: unit(if lo >= min(f, s[0]) { 0 } else { LAGS - 1 }),
        check: unit(if hi <= max(f, s[0]) { LAGS - 1 } else { 0 }),
    }
}

/// Rung 2, the full-support equalizer. A Checker mix that leaves the Dropper indifferent between
/// consecutive seconds solves a 59-step recurrence; negative weights are clipped. `M` is
/// persymmetric, so the reversed mix is the Dropper's and `max`/`min` of `M · check` bound both
/// sides. `None` when the recurrence breaks down.
#[must_use]
pub fn support(s: &Mix, f: f64) -> Option<Certificate> {
    let d0 = s[0] - f;
    if d0.abs() < 1e-12 {
        return None;
    }
    let mut r = [0.0; LAGS];
    r[0] = 1.0;
    for k in 1..LAGS {
        let acc = s
            .windows(2)
            .zip(r[..k].iter().rev())
            .fold(0.0, |acc, (w, &rr)| acc + (w[1] - w[0]) * rr);
        r[k] = -acc / d0;
    }
    if !r.iter().all(|x| x.is_finite()) {
        return None;
    }
    let mut check: Mix = std::array::from_fn(|k| max(0.0, r[LAGS - 1 - k]));
    let sum = check.iter().fold(0.0, |a, &q| a + q);
    if sum <= 0.0 {
        return None;
    }
    for q in &mut check {
        *q /= sum;
    }
    let (mut lower, mut upper, mut cum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for i in 0..LAGS {
        let y = s
            .iter()
            .zip(&check[i..])
            .fold(f * cum, |acc, (&sk, &qk)| acc + sk * qk);
        lower = min(lower, y);
        upper = max(upper, y);
        cum += check[i];
    }
    let mut drop = check;
    drop.reverse();
    Some(Certificate {
        lower,
        upper,
        rung: Rung::Support,
        drop,
        check,
    })
}

/// Clip a candidate mix to nonnegative weights and normalize it; `None` when nothing is left.
fn normalized(x: &[f64]) -> Option<Mix> {
    let mix: Mix = std::array::from_fn(|k| x[k].max(0.0));
    let sum: f64 = mix.iter().sum();
    (sum > 0.0).then(|| mix.map(|w| w / sum))
}

/// Rung 3, the linear program: maximize `v` subject to `Mᵀ p ≥ v`, `Σ p = 1`, `p ≥ 0` for the
/// Dropper's mix `p`, and read the Checker's mix from the row duals. Simplex first, then
/// interior point; the gate never changes. `None` when neither solve certifies.
#[must_use]
pub fn lp(s: &Mix, f: f64) -> Option<Certificate> {
    let m = full_matrix(s, f);
    for solver in ["simplex", "ipm"] {
        let mut problem = ColProblem::default();
        let rows: Vec<_> = (0..LAGS).map(|_| problem.add_row(0.0..)).collect();
        let total = problem.add_row(1.0..=1.0);
        for row in &m {
            problem.add_column(
                0.0,
                0.0..,
                rows.iter()
                    .copied()
                    .zip(row.iter().copied())
                    .chain([(total, 1.0)]),
            );
        }
        problem.add_column(1.0, f64::NEG_INFINITY.., rows.iter().map(|&r| (r, -1.0)));
        let mut model = problem.optimise(Sense::Maximise);
        model.make_quiet();
        model.set_option("solver", solver);
        let Ok(solved) = model.try_solve() else {
            continue;
        };
        if solved.status() != HighsModelStatus::Optimal {
            continue;
        }
        let solution = solved.get_solution();
        let duals: Vec<f64> = solution.dual_rows()[..LAGS].iter().map(|y| -y).collect();
        let (Some(drop), Some(check)) =
            (normalized(&solution.columns()[..LAGS]), normalized(&duals))
        else {
            continue;
        };
        let cert = certify(&m, &drop, &check, Rung::Lp);
        if cert.certified() {
            return Some(cert);
        }
    }
    None
}

/// The certificate of a strategy pair against the full matrix.
pub(crate) fn certify(m: &Matrix, drop: &Mix, check: &Mix, rung: Rung) -> Certificate {
    let lower = (0..LAGS)
        .map(|c| (0..LAGS).map(|d| drop[d] * m[d][c]).sum::<f64>())
        .fold(f64::INFINITY, f64::min);
    let upper = m
        .iter()
        .map(|row| row.iter().zip(check).map(|(a, q)| a * q).sum::<f64>())
        .fold(f64::NEG_INFINITY, f64::max);
    Certificate {
        lower,
        upper,
        rung,
        drop: *drop,
        check: *check,
    }
}

/// The ladder for one class: the first rung that certifies, or `None` when none does.
pub(crate) fn solve_class(s: &Mix, f: f64) -> Option<Certificate> {
    Some(pure(s, f))
        .filter(Certificate::certified)
        .or_else(|| support(s, f).filter(Certificate::certified))
        .or_else(|| lp(s, f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cert(lower: f64, upper: f64) -> Certificate {
        Certificate {
            lower,
            upper,
            rung: Rung::Support,
            drop: unit(0),
            check: unit(0),
        }
    }

    #[test]
    fn the_gate_and_the_midpoint() {
        let c = cert(0.25, 0.25 + 5e-7);
        assert!(
            c.certified()
                && (c.gap() - 5e-7).abs() < 1e-15
                && (c.value() - (0.25 + 2.5e-7)).abs() < 1e-15
        );
        assert!(!cert(0.0, 1e-3).certified());
        assert!(!cert(f64::NAN, f64::NAN).certified());
        assert!(
            cert(0.3, 0.3 - 1e-15).certified(),
            "rounding may cross the bounds"
        );
    }

    #[test]
    fn constant_and_monotone_stages_are_pure() {
        let linspace: Mix = std::array::from_fn(|k| -1.0 + 2.0 * k as f64 / 59.0);
        for (s, f, value) in [
            ([0.5; LAGS], 0.5, 0.5),
            ([1.0; LAGS], 1.0, 1.0),
            (linspace, -1.0, -1.0),
        ] {
            let c = solve_class(&s, f).expect("certified");
            assert_eq!((c.rung, c.value()), (Rung::Pure, value));
        }
    }
}
