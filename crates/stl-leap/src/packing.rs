//! The packing LP of a residue stage, solved by a dense simplex and certified on the full matrix.
//!
//! For `d = f - s[0] > 0` let `A[i, j] = (f - s[j - i]) / d` for `j >= i` and 0 otherwise, so
//! the stage matrix is `M = f 11^T - d A`. The pair
//! `max {1^T y : A^T y <= 1, y >= 0} = min {1^T x : A x >= 1, x >= 0} = z`
//! gives the mixtures `p = y / z`, `q = x / z` and the value `f - d / z`. The tableau starts from
//! the slack basis, from the last certified basis (warm), or from the crash basis in which every
//! packing variable is basic. That basis matrix `A^T` is lower triangular Toeplitz, so its inverse
//! is the Toeplitz matrix of the series `c` with `c[0] = 1`, `c[k] = -sum_{m=1..k} a[m] c[k-m]`,
//! which the stage kernel's recurrence also computes. A crash start has wrong-signed reduced
//! costs and basic values on a residue stage; the solver zeroes the positive reduced costs, runs
//! the dual simplex to primal feasibility, restores the true costs and finishes with the primal
//! simplex. A solve that fails restarts from the slack basis. Every candidate passes only after
//! both saddle bounds, checked on all 60 rows and columns, lie within `1e-6`.

use crate::rules::ACTIONS as N;
use rayon::prelude::*;

const W: usize = N + 1;
const EPS: f64 = 1e-10;
const PIVOTS: u32 = 512;

/// Where a packing solve starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// The slack basis.
    Slack,
    /// The last certified basis of this solver, refactored for the new payoffs.
    Warm,
    /// The recurrence basis in which every packing variable is basic.
    Crash,
}

/// The outcome of one packing solve.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    /// The certified square-stage value, or NaN.
    pub value: f64,
    /// The certified saddle gap, or NaN.
    pub gap: f64,
    /// Simplex pivots: 512 when uncertified, 0 when the stage has no packing form.
    pub pivots: u32,
    /// Whether the solve began from a warm or crash basis.
    pub seeded: bool,
    /// Whether the seeded start failed and the solve restarted from the slack basis.
    pub restarted: bool,
}

/// A packing tableau and the support of its last certified basis.
pub struct Packer {
    t: Vec<[f64; W]>,
    basic: [usize; N],
    nonbasic: [usize; N],
    certified: Option<(u64, u64)>,
}

impl Default for Packer {
    fn default() -> Self {
        Self {
            t: vec![[0.0; W]; W],
            basic: [0; N],
            nonbasic: [0; N],
            certified: None,
        }
    }
}

fn coefficient(a: &[f64; N], row: usize, variable: usize) -> f64 {
    if variable >= N {
        f64::from(u8::from(variable - N == row))
    } else if row >= variable {
        a[row - variable]
    } else {
        0.0
    }
}

impl Packer {
    /// Return the basic Dropper actions and the nonbasic Checker slacks of the last solve, as two
    /// masks with equal bit counts, when that solve was certified.
    #[must_use]
    pub fn support(&self) -> Option<(u64, u64)> {
        self.certified
    }

    /// Solve the packing LP of `stage = [s[0..60], f]` and certify the square stage.
    pub fn solve(&mut self, stage: &[f64; W], start: Start) -> Run {
        let f = stage[N];
        let d = f - stage[0];
        if d <= 1e-12 || stage.iter().any(|x| !x.is_finite()) {
            self.certified = None;
            return Run {
                value: f64::NAN,
                gap: f64::NAN,
                pivots: 0,
                seeded: false,
                restarted: false,
            };
        }
        let a: [f64; N] = std::array::from_fn(|i| (f - stage[i]) / d);
        let seeded = match start {
            Start::Slack => false,
            Start::Warm => self.certified.is_some() && self.refactor(&a),
            Start::Crash => true,
        };
        // The pivot count and the certified (value, gap).
        let mut result = None;
        if start == Start::Crash {
            if self.crash(&a) {
                for x in self.t[N][..N].iter_mut().filter(|x| **x > 0.0) {
                    *x = 0.0;
                }
                if let Some(first) = self.optimize() {
                    self.objective();
                    result = self
                        .optimize()
                        .and_then(|n| self.certificate(stage).map(|c| (first + n, c)));
                }
            }
        } else {
            if !seeded {
                self.cold(&a);
            }
            result = self.optimize().zip(self.certificate(stage));
        }
        let restarted = seeded && result.is_none();
        if restarted {
            self.cold(&a);
            result = self.optimize().zip(self.certificate(stage));
        }
        self.certified = result.map(|_| self.basis_support());
        let (pivots, (value, gap)) = result.unwrap_or((PIVOTS, (f64::NAN, f64::NAN)));
        Run {
            value,
            gap,
            pivots,
            seeded,
            restarted,
        }
    }

    fn clear(&mut self) {
        for row in &mut self.t {
            *row = [0.0; W];
        }
    }

    fn cold(&mut self, a: &[f64; N]) {
        self.clear();
        for i in 0..N {
            self.basic[i] = N + i;
            self.nonbasic[i] = i;
            self.t[i][N] = 1.0;
            self.t[N][i] = 1.0;
            for j in 0..=i {
                self.t[i][j] = a[i - j];
            }
        }
    }

    /// Keep the basic and nonbasic variables, and rebuild the tableau for the new payoffs.
    fn refactor(&mut self, a: &[f64; N]) -> bool {
        let mut work = vec![[0.0; 2 * N + 1]; N];
        for (i, row) in work.iter_mut().enumerate() {
            for j in 0..N {
                row[j] = coefficient(a, i, self.basic[j]);
                row[N + j] = coefficient(a, i, self.nonbasic[j]);
            }
            row[2 * N] = 1.0;
        }
        for col in 0..N {
            // The last row of largest magnitude.
            let pivot = (col..N)
                .max_by(|&i, &j| work[i][col].abs().total_cmp(&work[j][col].abs()))
                .expect("a nonempty range");
            if !work[pivot][col].is_finite() || work[pivot][col].abs() < 1e-12 {
                return false;
            }
            work.swap(col, pivot);
            let scale = work[col][col];
            for x in &mut work[col][col..] {
                *x /= scale;
            }
            let source = work[col];
            for (i, row) in work.iter_mut().enumerate() {
                if i != col {
                    let factor = row[col];
                    for j in col..=2 * N {
                        row[j] = (-factor).mul_add(source[j], row[j]);
                    }
                }
            }
        }
        self.clear();
        for (row, source) in self.t.iter_mut().zip(&work) {
            row.copy_from_slice(&source[N..]);
        }
        self.objective();
        self.t.iter().flatten().all(|x| x.is_finite())
    }

    /// Make every packing variable basic; basic value `i` is `c[0] + ... + c[i]` and the reduced
    /// cost of slack `j` is `-(c[0] + ... + c[59 - j])`.
    fn crash(&mut self, a: &[f64; N]) -> bool {
        let mut c = [0.0; N];
        c[0] = 1.0;
        for k in 1..N {
            c[k] = (1..=k).fold(0.0, |x, m| (-a[m]).mul_add(c[k - m], x));
        }
        self.clear();
        let mut prefix = 0.0;
        for i in 0..N {
            prefix += c[i];
            self.basic[i] = i;
            self.nonbasic[i] = N + i;
            for j in 0..=i {
                self.t[i][j] = c[i - j];
            }
            self.t[i][N] = prefix;
        }
        self.objective();
        self.t.iter().flatten().all(|x| x.is_finite())
    }

    /// Rebuild the objective row for the true costs: one per packing variable, zero per slack.
    fn objective(&mut self) {
        let mut objective = [0.0; W];
        for (cost, &variable) in objective.iter_mut().zip(&self.nonbasic) {
            *cost = f64::from(u8::from(variable < N));
        }
        for (row, &variable) in self.t.iter().zip(&self.basic) {
            if variable < N {
                for (cost, x) in objective.iter_mut().zip(row) {
                    *cost -= x;
                }
            }
        }
        self.t[N] = objective;
    }

    fn pivot(&mut self, row: usize, col: usize) {
        let scale = self.t[row][col];
        for x in &mut self.t[row] {
            *x /= scale;
        }
        self.t[row][col] = 1.0 / scale;
        let source = self.t[row];
        for (i, line) in self.t.iter_mut().enumerate() {
            if i != row {
                let factor = line[col];
                for j in 0..W {
                    line[j] = (-factor).mul_add(source[j], line[j]);
                }
                line[col] = -factor / scale;
            }
        }
        std::mem::swap(&mut self.basic[row], &mut self.nonbasic[col]);
    }

    /// Return the pivot count at an optimum, or `None` when the simplex stops without one.
    fn optimize(&mut self) -> Option<u32> {
        let t = |s: &Self, i: usize, j: usize| s.t[i][j];
        for iteration in 0..PIVOTS {
            let primal = (0..N).all(|i| t(self, i, N) >= -EPS);
            let dual = (0..N).all(|j| t(self, N, j) <= EPS);
            let (row, col) = match (primal, dual) {
                (true, true) => return Some(iteration),
                (true, false) => {
                    // Enter the last largest reduced cost; leave on the first smallest ratio.
                    let col = (0..N)
                        .filter(|&j| t(self, N, j) > EPS)
                        .max_by(|&i, &j| t(self, N, i).total_cmp(&t(self, N, j)))?;
                    let ratio = |i: usize| t(self, i, N).max(0.0) / t(self, i, col);
                    let row = (0..N).filter(|&i| t(self, i, col) > EPS).min_by(|&i, &j| {
                        ratio(i)
                            .total_cmp(&ratio(j))
                            .then(self.basic[i].cmp(&self.basic[j]))
                    })?;
                    (row, col)
                }
                (false, true) => {
                    let row = (0..N)
                        .filter(|&i| t(self, i, N) < -EPS)
                        .min_by(|&i, &j| t(self, i, N).total_cmp(&t(self, j, N)))?;
                    let ratio = |j: usize| t(self, N, j) / t(self, row, j);
                    let col = (0..N)
                        .filter(|&j| t(self, row, j) < -EPS)
                        .min_by(|&i, &j| {
                            ratio(i)
                                .total_cmp(&ratio(j))
                                .then(self.nonbasic[i].cmp(&self.nonbasic[j]))
                        })?;
                    (row, col)
                }
                (false, false) => return None,
            };
            self.pivot(row, col);
        }
        None
    }

    /// Normalize the basis mixtures and check every row and column payoff of the square stage.
    fn certificate(&self, stage: &[f64; W]) -> Option<(f64, f64)> {
        let (s, f) = (&stage[..N], stage[N]);
        let mut p = [0.0; N];
        let mut q = [0.0; N];
        for i in (0..N).filter(|&i| self.basic[i] < N) {
            p[self.basic[i]] = self.t[i][N].max(0.0);
        }
        for j in (0..N).filter(|&j| self.nonbasic[j] >= N) {
            q[self.nonbasic[j] - N] = (-self.t[N][j]).max(0.0);
        }
        let ps: f64 = p.iter().sum();
        let qs: f64 = q.iter().sum();
        if !ps.is_finite() || !qs.is_finite() || ps <= 0.0 || qs <= 0.0 {
            return None;
        }
        for i in 0..N {
            p[i] /= ps;
            q[i] /= qs;
        }
        let (rows, cols) = payoffs(s, f, &p, &q);
        let upper = rows.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let lower = cols.into_iter().fold(f64::INFINITY, f64::min);
        let gap = upper - lower;
        (gap.is_finite() && (-1e-12..=1e-6).contains(&gap)).then_some(((upper + lower) * 0.5, gap))
    }

    fn basis_support(&self) -> (u64, u64) {
        let p = self
            .basic
            .iter()
            .filter(|&&b| b < N)
            .fold(0, |m, &b| m | 1 << b);
        let q = self
            .nonbasic
            .iter()
            .filter(|&&b| b >= N)
            .fold(0, |m, &b| m | 1 << (b - N));
        (p, q)
    }
}

/// Return the row payoffs `M q` and the column payoffs `p^T M` of the square stage.
///
/// The failure triangle is constant, so each row starts at `f` times the Checker mass before its
/// diagonal and each column at `f` times the Dropper mass after it.
pub(crate) fn payoffs(s: &[f64], f: f64, p: &[f64; N], q: &[f64; N]) -> ([f64; N], [f64; N]) {
    let mut rows = [0.0; N];
    let mut cols = [0.0; N];
    let (mut prefix, mut suffix) = (0.0, 0.0);
    for i in 0..N {
        rows[i] = f * prefix;
        prefix += q[i];
        cols[N - 1 - i] = f * suffix;
        suffix += p[N - 1 - i];
    }
    for i in 0..N {
        for j in i..N {
            rows[i] = s[j - i].mul_add(q[j], rows[i]);
            cols[j] = s[j - i].mul_add(p[i], cols[j]);
        }
    }
    (rows, cols)
}

/// Solve each stage from the slack basis, or with `warm` from the last certified basis of its
/// 2048-stage chunk while `groups` keeps its value. This is the warm-start benchmark of the
/// packing LP; the build uses crash starts.
///
/// # Panics
///
/// When `groups` and `stages` differ in length.
#[must_use]
pub fn runs(stages: &[[f64; W]], groups: &[f64], warm: bool) -> Vec<Run> {
    assert_eq!(stages.len(), groups.len(), "one group per stage");
    let start = if warm { Start::Warm } else { Start::Slack };
    stages
        .par_chunks(2048)
        .zip(groups.par_chunks(2048))
        .flat_map_iter(|(stages, groups)| {
            let mut packer = Packer::default();
            let mut group = f64::NAN;
            stages.iter().zip(groups).map(move |(stage, &g)| {
                if g != group {
                    packer.certified = None;
                    group = g;
                }
                packer.solve(stage, start)
            })
        })
        .collect()
}
