//! The packing LP of a residue stage, solved by a dense simplex and certified on the full matrix.
//!
//! For `d = f - s[0] > 0` let `A[i, j] = (f - s[j - i]) / d` for `j >= i` and 0 otherwise, so
//! the stage matrix is `M = f 11^T - d A`. The pair
//! `max {1^T y : A^T y <= 1, y >= 0} = min {1^T x : A x >= 1, x >= 0} = z`
//! gives the mixtures `p = y / z`, `q = x / z` and the value `f - d / z`. The tableau starts from
//! the slack basis or from the crash basis in which every packing variable is basic. That basis
//! matrix `A^T` is lower triangular Toeplitz, so its inverse is the Toeplitz matrix of the series
//! `c` with `c[0] = 1`, `c[k] = -sum_{m=1..k} a[m] c[k-m]`, which the stage kernel's recurrence
//! also computes. A crash start has wrong-signed reduced costs and basic values on a residue
//! stage; the solver zeroes the positive reduced costs, runs the dual simplex to primal
//! feasibility, restores the true costs and finishes with the primal simplex. A crash solve that
//! fails restarts from the slack basis. Every candidate passes only after both saddle bounds,
//! checked on all 60 rows and columns, lie within `1e-6`.

use crate::kernel::GAP;
use crate::rules::ACTIONS as N;

const W: usize = N + 1;
const EPS: f64 = 1e-10;
const PIVOTS: u32 = 512;

/// Where a packing solve starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// The slack basis.
    Slack,
    /// The recurrence basis in which every packing variable is basic.
    Crash,
}

/// A packing tableau, reused across solves.
pub struct Packer {
    t: Vec<[f64; W]>,
    basic: [usize; N],
    nonbasic: [usize; N],
}

impl Default for Packer {
    fn default() -> Self {
        Self {
            t: vec![[0.0; W]; W],
            basic: [0; N],
            nonbasic: [0; N],
        }
    }
}

impl Packer {
    /// Solve the packing LP of the square stage `(s, f)` and certify it. Return the stage value,
    /// the basic Dropper actions and the nonbasic Checker slacks (two masks with equal bit
    /// counts), or `None` when the stage has no packing form or no basis passes the saddle gate.
    pub fn solve(&mut self, s: &[f64; N], f: f64, start: Start) -> Option<(f64, u64, u64)> {
        let d = f - s[0];
        if d <= 1e-12 || !f.is_finite() || s.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let a: [f64; N] = std::array::from_fn(|i| (f - s[i]) / d);
        if start == Start::Crash && self.crash(&a) {
            for x in self.t[N][..N].iter_mut().filter(|x| **x > 0.0) {
                *x = 0.0;
            }
            if self.optimize().is_some() {
                self.objective();
                if let Some(found) = self.optimize().and_then(|()| self.certificate(s, f)) {
                    return Some(found);
                }
            }
        }
        self.cold(&a);
        self.optimize()?;
        self.certificate(s, f)
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

    /// Pivot to an optimum, or return `None` when the simplex stops without one.
    fn optimize(&mut self) -> Option<()> {
        let t = |s: &Self, i: usize, j: usize| s.t[i][j];
        for _ in 0..PIVOTS {
            let primal = (0..N).all(|i| t(self, i, N) >= -EPS);
            let dual = (0..N).all(|j| t(self, N, j) <= EPS);
            let (row, col) = match (primal, dual) {
                (true, true) => return Some(()),
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
    fn certificate(&self, s: &[f64; N], f: f64) -> Option<(f64, u64, u64)> {
        let (mut p, mut q, mut p_mask, mut q_mask) = ([0.0; N], [0.0; N], 0, 0);
        for (i, &b) in self.basic.iter().enumerate().filter(|&(_, &b)| b < N) {
            p[b] = self.t[i][N].max(0.0);
            p_mask |= 1 << b;
        }
        for (j, &b) in self.nonbasic.iter().enumerate().filter(|&(_, &b)| b >= N) {
            q[b - N] = (-self.t[N][j]).max(0.0);
            q_mask |= 1 << (b - N);
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
        (-1e-12..=GAP)
            .contains(&(upper - lower))
            .then_some(((upper + lower) * 0.5, p_mask, q_mask))
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
