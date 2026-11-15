//! Reduced-support candidates for residue stages, and their payoff-kink seeds.
//!
//! Write `d = s[0] - f` and let `r` be the stage recurrence. For a Checker mixture `q` define
//! `g[i] = q[i] - sum_{j>i} b[j-i] q[j]`, so `q[j] = sum_{t>=j} r[t-j] g[t]` and
//! `(Mq)[i] - (Mq)[i+1] = d g[i]`. Given guessed supports `P` (Dropper rows) and `Q` (Checker
//! columns), adjacent supported rows force `g[i] = 0`, a hole between supported rows forces a zero
//! sum, and `q` vanishes outside `Q`. The same equations on the reversed supports give the Dropper
//! mixture. A candidate passes only when both mixtures are nonnegative and the saddle bounds over
//! all rows and columns lie within `1e-6`.
//!
//! A residue stage needs a negative step `s[k] - s[k-1]`; the most negative one is its kink. One
//! more second of Checker vial shifts the success payoffs by one action and the kink by -1, and
//! the holes of the support move with the nearer kink anchor. [`kink_seed`] moves a certified
//! support from one kink to the next.

use crate::kernel::GAP;
use crate::packing::payoffs;
use crate::rules::ACTIONS as A;

const ALL: u64 = (1 << A) - 1;

fn reverse(mask: u64) -> u64 {
    mask.reverse_bits() >> (64 - A)
}

/// Solve the augmented `n`-by-`n` system `a` (rows of `n + 1`) by scaled partial pivoting.
fn solve(a: &mut [f64], n: usize) -> Option<[f64; A]> {
    let w = n + 1;
    // Scale rows and pivot on magnitude; the full certificate still decides acceptance.
    for row in a.chunks_exact_mut(w) {
        let scale = row[..n].iter().fold(0.0_f64, |x, y| x.max(y.abs()));
        if !scale.is_finite() || scale == 0.0 {
            return None;
        }
        for x in row {
            *x /= scale;
        }
    }
    for col in 0..n {
        let magnitude = |row: usize| a[row * w + col].abs();
        let pivot = (col + 1..n).fold(col, |best, row| {
            if magnitude(row) > magnitude(best) {
                row
            } else {
                best
            }
        });
        if !a[pivot * w + col].is_finite() || magnitude(pivot) < 1e-14 {
            return None;
        }
        for k in 0..w {
            a.swap(col * w + k, pivot * w + k);
        }
        let (done, rest) = a.split_at_mut((col + 1) * w);
        let source = &done[col * w..];
        for row in rest.chunks_exact_mut(w) {
            let factor = row[col] / source[col];
            for k in col + 1..=n {
                row[k] = (-factor).mul_add(source[k], row[k]);
            }
        }
    }
    let mut g = [0.0; A];
    for j in (0..n).rev() {
        g[j] =
            (j + 1..n).fold(a[j * w + n], |x, k| (-a[j * w + k]).mul_add(g[k], x)) / a[j * w + j];
    }
    Some(g)
}

/// Solve the hole equations for `g` and return the normalized Checker mixture on `q_mask`.
fn reduced_weights(r: &[f64; A], p_mask: u64, q_mask: u64) -> Option<[f64; A]> {
    if p_mask == 0 || q_mask == 0 || p_mask.count_ones() != q_mask.count_ones() {
        return None;
    }
    let mut slots = [0; A];
    let mut n = 0;
    for t in (0..A - 1)
        .filter(|&t| (p_mask >> t) & 3 != 3)
        .chain([A - 1])
    {
        slots[n] = t;
        n += 1;
    }
    let positions = &slots[..n];
    // More zero columns than unknowns leaves the system overdetermined.
    if A - q_mask.count_ones() as usize >= n {
        return None;
    }
    // The augmented system, n rows of n unknowns and a right-hand side.
    let w = n + 1;
    let mut a = vec![0.0; n * w];
    let mut equation = 0;
    for j in (0..A).filter(|&j| q_mask & (1 << j) == 0) {
        for (x, &t) in a[equation * w..][..n].iter_mut().zip(positions) {
            if t >= j {
                *x = r[t - j];
            }
        }
        equation += 1;
    }
    let mut previous: Option<usize> = None;
    for c in (0..A).filter(|&c| p_mask & (1 << c) != 0) {
        if let Some(start) = previous.filter(|&start| c > start + 1) {
            if equation >= n {
                return None;
            }
            for (x, &t) in a[equation * w..][..n].iter_mut().zip(positions) {
                *x = f64::from(u8::from((start..c).contains(&t)));
            }
            equation += 1;
        }
        previous = Some(c);
    }
    if equation + 1 != n {
        return None;
    }
    let mut cumulative = [0.0; A];
    let mut total = 0.0;
    for (sum, x) in cumulative.iter_mut().zip(r) {
        total += x;
        *sum = total;
    }
    for (x, &t) in a[equation * w..][..n].iter_mut().zip(positions) {
        *x = cumulative[t];
    }
    a[equation * w + n] = 1.0;
    let g = solve(&mut a, n)?;
    let mut q = [0.0; A];
    let mut total = 0.0;
    for j in (0..A).filter(|&j| q_mask & (1 << j) != 0) {
        let x = positions
            .iter()
            .zip(&g)
            .filter(|&(&t, _)| t >= j)
            .fold(0.0, |x, (&t, &gk)| r[t - j].mul_add(gk, x));
        if !x.is_finite() || x < 0.0 {
            return None;
        }
        q[j] = x;
        total += x;
    }
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    Some(q.map(|x| x / total))
}

/// Certify the stage `(s, f)` on the supports `(p_mask, q_mask)` and return the square-stage value.
///
/// In a window stage both bounds are lifted through `max(., f)`, which certifies the leap row.
#[must_use]
pub fn attempt(
    s: &[f64; A],
    f: f64,
    r: &[f64; A],
    p_mask: u64,
    q_mask: u64,
    window: bool,
) -> Option<f64> {
    let q = reduced_weights(r, p_mask, q_mask)?;
    let reversed = reduced_weights(r, reverse(q_mask), reverse(p_mask))?;
    let p: [f64; A] = std::array::from_fn(|j| reversed[A - 1 - j]);
    let (prefix_q, suffix_p) = (q.iter().sum::<f64>(), p.iter().rev().sum::<f64>());
    if (prefix_q - 1.0).abs() > 1e-12 || (suffix_p - 1.0).abs() > 1e-12 {
        return None;
    }
    let (rows, cols) = payoffs(s, f, &p, &q);
    if rows.iter().chain(&cols).any(|x| !x.is_finite()) {
        return None;
    }
    let mut lower = cols.into_iter().fold(f64::INFINITY, f64::min);
    let mut upper = rows.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let value = (lower + upper) * 0.5;
    if window {
        lower = lower.max(f);
        upper = upper.max(f);
    }
    (-1e-12..=GAP).contains(&(upper - lower)).then_some(value)
}

/// Return the index `k` of the most negative step `s[k] - s[k-1]`, the first on a tie, or `None`
/// when `s` never decreases.
#[must_use]
pub fn kink_index(s: &[f64; A]) -> Option<usize> {
    let mut best = 0.0;
    let mut index = None;
    for k in 1..A {
        let step = s[k] - s[k - 1];
        if step < best {
            best = step;
            index = Some(k);
        }
    }
    index
}

/// Move each hole of `mask` with the nearer of the anchors `near` and `59 - old`: by `delta` at
/// the first and by `-delta` at the second. A hole that starts at action 0 stays; moved holes are
/// clipped to actions 0..59.
fn move_holes(mask: u64, near: i64, old: i64, delta: i64) -> u64 {
    let far = A as i64 - 1 - old;
    let mut out = ALL;
    let mut i = 0;
    while i < A {
        if (mask >> i) & 1 == 1 {
            i += 1;
            continue;
        }
        let start = i;
        while i + 1 < A && (mask >> (i + 1)) & 1 == 0 {
            i += 1;
        }
        let end = i as i64;
        let shift = match start {
            0 => 0,
            _ if (end - near).abs() <= (end - far).abs() => delta,
            _ => -delta,
        };
        for x in (start as i64 + shift).max(0)..=(end + shift).min(A as i64 - 1) {
            out &= !(1 << x);
        }
        i += 1;
    }
    out
}

/// Move the certified supports `(p, q)` of a stage with kink `old` to a stage with kink `new`.
///
/// Dropper holes use the anchors `old` and `59 - old`, Checker holes `old - 1` and `59 - old`.
/// The masks stay unchanged when either kink is absent or the kinks agree.
#[must_use]
pub fn kink_seed(p: u64, q: u64, old: Option<usize>, new: Option<usize>) -> (u64, u64) {
    match (old, new) {
        (Some(old), Some(new)) if old != new => {
            let (old, delta) = (old as i64, new as i64 - old as i64);
            (
                move_holes(p, old, old, delta),
                move_holes(q, old - 1, old, delta),
            )
        }
        _ => (p, q),
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)] // reference stages, copied digit for digit
mod tests {
    use super::*;

    #[rustfmt::skip]
    pub(crate) const S: [f64; A] = [
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
    pub(crate) const F: f64 = 0.813835962781682;
    const P: u64 = 1_152_921_500_328_656_895;
    const Q: u64 = 1_152_921_500_588_703_743;
    pub(crate) const VALUE: f64 = 0.1504047730058836;

    fn recurrence(s: &[f64; A], f: f64) -> [f64; A] {
        let d = s[0] - f;
        let mut r = [0.0; A];
        r[0] = 1.0;
        for k in 1..A {
            r[k] = (0..k).fold(0.0, |acc, j| {
                ((s[k - j - 1] - s[k - j]) / d).mul_add(r[j], acc)
            });
        }
        r
    }

    #[test]
    fn certifies_the_known_asymmetric_support() {
        let r = recurrence(&S, F);
        let square = attempt(&S, F, &r, P, Q, false).expect("a certified support");
        assert!((square - VALUE).abs() < 1e-9);
        // The window certificate lifts both bounds through max(., f) and keeps the square value.
        let window = attempt(&S, F, &r, P, Q, true).expect("a certified window support");
        assert_eq!(window, square);
    }

    #[test]
    fn rejects_invalid_supports() {
        let r = recurrence(&S, F);
        assert!(attempt(&S, F, &r, ALL, ALL, false).is_none());
        assert!(attempt(&S, F, &r, 0, Q, false).is_none());
        // A flat stage with f = s[0] has no recurrence.
        let flat = [1.0; A];
        assert!(attempt(&flat, 1.0, &recurrence(&flat, 1.0), P, Q, false).is_none());
    }

    #[test]
    fn kink_index_takes_the_first_most_negative_step() {
        let mut tie: [f64; A] = std::array::from_fn(|k| 0.3 * k as f64 / 59.0);
        for (k, x) in tie.iter_mut().enumerate() {
            *x -= if k >= 40 {
                0.04
            } else if k >= 10 {
                0.02
            } else {
                0.0
            };
        }
        assert_eq!(kink_index(&tie), Some(10));
        assert_eq!(kink_index(&[0.0; A]), None);
        assert_eq!(kink_index(&S), Some(28));
    }

    #[test]
    fn kink_seed_moves_holes_with_their_anchor() {
        // A Dropper hole at k* = 45 moves right; the mirrored hole at 59 - k* moves left.
        let p = ALL & !(0b111 << 12) & !(0b111 << 43);
        let q = ALL & !(0b111 << 12) & !(0b111 << 42);
        let (moved_p, moved_q) = kink_seed(p, q, Some(45), Some(46));
        assert_eq!(moved_p, ALL & !(0b111 << 11) & !(0b111 << 44));
        assert_eq!(moved_q, ALL & !(0b111 << 11) & !(0b111 << 43));
        assert_eq!(kink_seed(p, q, Some(45), None), (p, q));
        assert_eq!(kink_seed(p, q, Some(45), Some(45)), (p, q));
        // A hole at action 0 stays; the hole at 59 is nearer the anchor 59 - 1 and moves by -4.
        let edge = ALL & !1 & !(1 << 59);
        assert_eq!(
            kink_seed(edge, edge, Some(1), Some(5)).0,
            ALL & !1 & !(1 << 55)
        );
    }
}
