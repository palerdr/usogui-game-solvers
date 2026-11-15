//! Time-uniform confidence sets for kernel entries from class counts alone.
//!
//! Each class indicator of an entry is a Bernoulli variable with mean `p`, so the mixture
//! likelihood ratio
//!
//! ```text
//! M_n(p) = m(k, n) / (p^k (1 - p)^(n - k)),   m(k, n) = B(k + 1/2, n - k + 1/2) / B(1/2, 1/2)
//! ```
//!
//! is a nonnegative martingale with mean one (Robbins' Beta(1/2, 1/2) mixture). By Ville's
//! inequality `C_n = { p : p^k (1 - p)^(n - k) >= alpha m(k, n) }` holds `p` at every `n` at
//! once with probability at least `1 - alpha`, whatever rule decides how many samples an entry
//! gets. A box takes each of the five reachable classes at level `alpha / 5`, pins the
//! impossible tie to zero and intersects with the simplex.

use crate::TIE;
use crate::sampler::Counts;

const ITERATIONS: usize = 64;

/// `log m(k, n)` of the Beta(1/2, 1/2) mixture of Bernoulli likelihoods.
#[must_use]
pub fn log_mixture(k: u64, n: u64) -> f64 {
    libm::lgamma(k as f64 + 0.5) + libm::lgamma((n - k) as f64 + 0.5)
        - libm::lgamma(n as f64 + 1.0)
        - std::f64::consts::PI.ln()
}

fn log_likelihood(p: f64, k: u64, n: u64) -> f64 {
    k as f64 * p.ln() + (n - k) as f64 * (-p).ln_1p()
}

/// `(lo, hi)` of `C_n` for `k` successes in `n` draws at level `alpha`.
///
/// Each end is the outer end of its final bisection bracket, so the interval contains `C_n`.
#[must_use]
pub fn interval(k: u64, n: u64, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let threshold = alpha.ln() + log_mixture(k, n);
    let mode = k as f64 / n as f64;
    // Bisect between a point inside the set and one outside; return the outside end.
    let bisect = |mut inside: f64, mut outside: f64| {
        for _ in 0..ITERATIONS {
            let mid = 0.5 * (inside + outside);
            if log_likelihood(mid, k, n) >= threshold {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        outside
    };
    let lo = if k == 0 { 0.0 } else { bisect(mode, 0.0) };
    let hi = if k == n { 1.0 } else { bisect(mode, 1.0) };
    (lo, hi)
}

/// The confidence box `(lo, hi)` of one entry at level `alpha`, intersected with the simplex:
/// `p_i >= 1 - sum of the other upper ends` and `p_i <= 1 - sum of the other lower ends`.
#[must_use]
pub fn entry_box(counts: &Counts, alpha: f64) -> ([f64; 6], [f64; 6]) {
    let n = counts.iter().sum();
    let mut lo = [0.0; 6];
    let mut hi = [0.0; 6];
    for class in (0..6).filter(|&c| c != TIE) {
        (lo[class], hi[class]) = interval(counts[class], n, alpha / 5.0);
    }
    let (sum_lo, sum_hi) = (lo.iter().sum::<f64>(), hi.iter().sum::<f64>());
    (
        std::array::from_fn(|c| lo[c].max(1.0 - (sum_hi - hi[c]))),
        std::array::from_fn(|c| hi[c].min(1.0 - (sum_lo - lo[c]))),
    )
}
