//! Exact small zero-sum matrix games by Shapley-Snow kernel enumeration.
//!
//! Every square matrix game (row player maximizes, here at most 5 x 5; both teams hold the same
//! number of metal cards) has an optimal pair supported
//! on a square submatrix `S` with `1' adj(S) 1 != 0` (Shapley and Snow 1950):
//! `x = 1' adj(S) / (1' adj(S) 1)` and `y = adj(S) 1 / (1' adj(S) 1)`. The solver walks the
//! square submatrices, keeps the first pair that is nonnegative and optimal against every pure
//! reply within a tolerance, and reports the values each strategy guarantees exactly:
//! `min_j (x' M)_j` for the row player and `max_i (M y)_i` for the column player. Those bracket
//! the game value; their gap is the saddle gap.

/// The largest side of a game.
pub const MAX_SIZE: usize = 5;
const TOLERANCE: f64 = 1e-9;

/// A square payoff matrix for the row player, padded to 5 x 5.
pub type Matrix = [[f64; MAX_SIZE]; MAX_SIZE];

/// The solution of one matrix game.
#[derive(Clone, Copy, Debug)]
pub struct Saddle {
    /// The row player's mixed strategy.
    pub x: [f64; MAX_SIZE],
    /// The column player's mixed strategy.
    pub y: [f64; MAX_SIZE],
    /// `min_j (x' M)_j`, the value the row strategy guarantees.
    pub low: f64,
    /// `max_i (M y)_i`, the value the column strategy concedes at most.
    pub high: f64,
}

/// Determinant of the leading `k x k` block by elimination with partial pivoting.
fn determinant(a: &Matrix, k: usize) -> f64 {
    let mut m = *a;
    let mut det = 1.0;
    for col in 0..k {
        let pivot = (col + 1..k).fold(col, |p, row| {
            if m[row][col].abs() > m[p][col].abs() {
                row
            } else {
                p
            }
        });
        if m[pivot][col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            m.swap(pivot, col);
            det = -det;
        }
        det *= m[col][col];
        let pivot_row = m[col];
        for row in &mut m[col + 1..k] {
            let factor = row[col] / pivot_row[col];
            for (x, p) in row[col..k].iter_mut().zip(&pivot_row[col..k]) {
                *x -= factor * p;
            }
        }
    }
    det
}

/// The adjugate (transposed cofactor matrix) of the `k x k` matrix `s`.
fn adjugate(s: &Matrix, k: usize) -> Matrix {
    let mut out = [[0.0; MAX_SIZE]; MAX_SIZE];
    for (i, j) in (0..k).flat_map(|i| (0..k).map(move |j| (i, j))) {
        let mut minor = [[0.0; MAX_SIZE]; MAX_SIZE];
        for (r, a) in (0..k).filter(|&a| a != i).enumerate() {
            for (c, b) in (0..k).filter(|&b| b != j).enumerate() {
                minor[r][c] = s[a][b];
            }
        }
        let sign = if (i + j) % 2 == 1 { -1.0 } else { 1.0 };
        out[j][i] = sign * determinant(&minor, k - 1);
    }
    out
}

/// `(min_j (x' M)_j, max_i (M y)_i)`.
fn guarantees(m: &Matrix, n: usize, x: &[f64; MAX_SIZE], y: &[f64; MAX_SIZE]) -> (f64, f64) {
    let low = (0..n)
        .map(|j| (0..n).fold(0.0, |t, i| t + x[i] * m[i][j]))
        .fold(1e300, f64::min);
    let high = (0..n)
        .map(|i| (0..n).fold(0.0, |t, j| t + m[i][j] * y[j]))
        .fold(-1e300, f64::max);
    (low, high)
}

/// Solve the `n x n` game in the top-left block of `m`.
#[must_use]
pub fn solve(m: &Matrix, n: usize) -> Saddle {
    let mut best = Saddle {
        x: [0.0; MAX_SIZE],
        y: [0.0; MAX_SIZE],
        low: -1e300,
        high: 1e300,
    };
    let mut best_gap = 1e300;
    let scale = m[..n]
        .iter()
        .flat_map(|row| &row[..n])
        .fold(1.0, |s: f64, v| s.max(v.abs()));
    let members = |mask: usize| (0..n).filter(move |i| mask >> i & 1 == 1);
    for row_mask in 1..1usize << n {
        let k = row_mask.count_ones() as usize;
        for col_mask in (1..1usize << n).filter(|c: &usize| c.count_ones() as usize == k) {
            let mut sub = [[0.0; MAX_SIZE]; MAX_SIZE];
            for (a, i) in members(row_mask).enumerate() {
                for (b, j) in members(col_mask).enumerate() {
                    sub[a][b] = m[i][j];
                }
            }
            let adj = adjugate(&sub, k);
            let denominator = adj[..k]
                .iter()
                .flat_map(|row| &row[..k])
                .fold(0.0, |t, v| t + v);
            if denominator.abs() <= 1e-14 * scale.powi(k as i32 - 1) {
                continue;
            }
            let (mut x, mut y) = ([0.0; MAX_SIZE], [0.0; MAX_SIZE]);
            let feasible =
                members(row_mask)
                    .zip(members(col_mask))
                    .enumerate()
                    .all(|(b, (i, j))| {
                        let xv = (0..k).fold(0.0, |t, a| t + adj[a][b]) / denominator;
                        let yv = (0..k).fold(0.0, |t, a| t + adj[b][a]) / denominator;
                        x[i] = xv.max(0.0);
                        y[j] = yv.max(0.0);
                        xv >= -TOLERANCE && yv >= -TOLERANCE
                    });
            if !feasible {
                continue;
            }
            let (sx, sy) = (x[..n].iter().sum::<f64>(), y[..n].iter().sum::<f64>());
            x[..n].iter_mut().for_each(|v| *v /= sx);
            y[..n].iter_mut().for_each(|v| *v /= sy);
            let (low, high) = guarantees(m, n, &x, &y);
            if high - low < best_gap {
                best_gap = high - low;
                best = Saddle { x, y, low, high };
                if best_gap <= TOLERANCE * scale {
                    return best;
                }
            }
        }
    }
    best
}
