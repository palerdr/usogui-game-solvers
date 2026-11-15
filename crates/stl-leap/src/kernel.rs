//! The stage kernel: certified values of batches of half-round matrix games.
//!
//! With zero-based actions the Dropper's payoff is `M[i, j] = s[j - i]` for `j >= i` (a check at
//! or after the drop succeeds with squandered time `j - i + 1`) and `f` for `j < i` (a failed
//! check, including the revival chance). `s` and `f` are negated child values after the role swap.
//! The kernel never builds `M`. For each stage it tries, in order:
//!
//! 0. pure bounds: the Dropper can guarantee `max(min s, min(f, s[0]))` and the Checker can hold
//!    the Dropper to `min(max s, max(f, s[0]))`;
//! 1. the equalizer: setting adjacent row payoffs equal gives `r[0] = 1`,
//!    `r[k] = sum_{j<k} b[k-j] r[j]` with `b[m] = (s[m-1] - s[m]) / (s[0] - f)`. Nonnegative weights
//!    certify the value `f + d / sum(r)` in closed form (docs/LEAP.md); clipped weights need the
//!    full payoff check;
//! 2. in a leap window, row 61 pays `f` in every column, so the value is `max(v60, f)` and both
//!    bounds lift through `max(., f)`;
//! 3. the residue, by [`Residue`].
//!
//! Classes are solved in chunks of 1024 on the rayon pool and in lanes of 16 within a chunk, with
//! scratch laid out `[action][lane]` so the recurrence vectorizes. A chunk keeps the support of
//! its last residue stage as the next kink seed, so results depend on the chunking and the build
//! keeps the reference batch order.

use crate::Error;
use crate::packing::{Packer, Start};
use crate::rules::ACTIONS as A;
use crate::support;
use rayon::prelude::*;

/// The saddle gate: both bounds of every certificate lie within this distance.
pub const GAP: f64 = 1e-6;
const LANES: usize = 16;
const CHUNK: usize = 1024;

/// A kind of certificate, as the kernel reports it.
pub mod kind {
    /// The pure-action bounds meet.
    pub const PURE: u8 = 0;
    /// The recurrence equalizer.
    pub const EQUALIZER: u8 = 1;
    /// The leap row wins or ties the square stage.
    pub const LEAP_ROW: u8 = 2;
    /// The packing LP.
    pub const PACKING: u8 = 3;
    /// The kink-moved support of the previous residue stage.
    pub const KINK: u8 = 4;
    /// No kernel certificate; the stage needs `HiGHS`.
    pub const UNSOLVED: u8 = 255;
}

/// What the kernel does with a stage that the pure bounds and the equalizer leave uncertified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residue {
    /// Leave it unsolved for the caller.
    Defer,
    /// Solve the packing LP from the crash basis, on the stage as the reference's fallback
    /// gathers it. The build solves `REV` keys this way.
    Packing,
    /// Try the support of the chunk's previous residue stage, moved with the payoff kink, then
    /// the packing LP. The build solves `H1` and `H2` keys this way.
    Kink,
}

/// A child value table: `values[row * width + column]`.
#[derive(Clone, Copy)]
pub struct Table<'a> {
    /// The values in row-major order.
    pub values: &'a [f64],
    /// The row length.
    pub width: usize,
}

/// The child tables of a batch. After the role swap the parent's Dropper is the child's Checker,
/// so the kernel reads row `dropper` of both tables and negates it.
#[derive(Clone, Copy)]
pub struct Children<'a> {
    /// The child after a successful check.
    pub success: Table<'a>,
    /// The child after a survived failed check.
    pub failure: Table<'a>,
    /// For each Checker profile, the success column of each squandered time `1..=60`.
    pub success_columns: &'a [[u32; A]],
    /// For each Checker profile, the failure column.
    pub failure_columns: &'a [u32],
    /// For each Checker profile, the revival probability.
    pub revival: &'a [f64],
}

/// The certified value of one stage, from the Dropper's side, and the rung that certified it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solved {
    /// The stage value, or NaN when unsolved.
    pub value: f64,
    /// One of the [`kind`] constants.
    pub kind: u8,
}

impl Children<'_> {
    /// Return the success payoffs and the failure payoff of the stage `(checker, dropper)` as the
    /// reference's fallback gathers them: `f = rev * -child + 1 - rev`, left to right, which can
    /// differ from the kernel's `rev * -child + (1 - rev)` in the last bit.
    pub(crate) fn stage(&self, (pc, pd): (u32, u32)) -> ([f64; A], f64) {
        let (pc, pd) = (pc as usize, pd as usize);
        let s = std::array::from_fn(|k| {
            -self.success.values[pd * self.success.width + self.success_columns[pc][k] as usize]
        });
        let child =
            self.failure.values[pd * self.failure.width + self.failure_columns[pc] as usize];
        let rev = self.revival[pc];
        (s, rev * -child + 1.0 - rev)
    }

    fn check(&self, classes: &[(u32, u32)]) -> Result<(), Error> {
        let profiles = self.success_columns.len();
        let rows = |t: &Table| t.values.len() / t.width.max(1);
        let ok = self.failure_columns.len() == profiles
            && self.revival.len() == profiles
            && self
                .success_columns
                .iter()
                .flatten()
                .all(|&c| (c as usize) < self.success.width)
            && self
                .failure_columns
                .iter()
                .all(|&c| (c as usize) < self.failure.width)
            && self.revival.iter().all(|x| (0.0..=1.0).contains(x))
            && classes.iter().all(|&(pc, pd)| {
                (pc as usize) < profiles
                    && (pd as usize) < rows(&self.success)
                    && (pd as usize) < rows(&self.failure)
            });
        ok.then_some(()).ok_or_else(|| {
            Error::Invalid("profile, column, or revival probability out of range".into())
        })
    }
}

/// Certify one value per `(checker, dropper)` class, in input order.
///
/// # Errors
///
/// [`Error::Invalid`] for out-of-range profiles, columns or probabilities, a nonfinite child
/// value that a stage reads, or a certified value outside `[-1, 1]`.
pub fn sweep(
    classes: &[(u32, u32)],
    children: &Children,
    window: bool,
    residue: Residue,
) -> Result<Vec<Solved>, Error> {
    children.check(classes)?;
    let mut out = vec![
        Solved {
            value: f64::NAN,
            kind: kind::UNSOLVED
        };
        classes.len()
    ];
    out.par_chunks_mut(CHUNK)
        .zip(classes.par_chunks(CHUNK))
        .try_for_each(|(out, classes)| Chunk::new(children, window, residue).solve(classes, out))?;
    Ok(out)
}

/// The scratch and residue state of one chunk.
struct Chunk<'a> {
    children: &'a Children<'a>,
    window: bool,
    residue: Residue,
    packer: Packer,
    /// The certified supports and kink of the chunk's previous residue stage.
    seed: Option<(u64, u64, Option<usize>)>,
}

/// Sixteen stages laid out `[action][lane]`.
struct Batch {
    s: [[f64; LANES]; A],
    /// The recurrence weights `r`.
    r: [[f64; LANES]; A],
    /// First the recurrence coefficients `b`, then the clipped Checker candidate.
    q: [[f64; LANES]; A],
    f: [f64; LANES],
    d: [f64; LANES],
    /// The pure bounds.
    mn: [f64; LANES],
    mx: [f64; LANES],
    /// Whether all payoffs lie in `[-2, 2]`, a hypothesis of the closed-form certificate.
    bounded: [bool; LANES],
    sum: [f64; LANES],
    nonnegative: [bool; LANES],
}

impl Batch {
    fn new() -> Self {
        Self {
            s: [[0.0; LANES]; A],
            r: [[0.0; LANES]; A],
            q: [[0.0; LANES]; A],
            f: [0.0; LANES],
            d: [0.0; LANES],
            mn: [0.0; LANES],
            mx: [0.0; LANES],
            bounded: [false; LANES],
            sum: [0.0; LANES],
            nonnegative: [true; LANES],
        }
    }

    /// Gather the payoffs of up to 16 classes in place; spare lanes repeat the first class.
    fn gather(&mut self, c: &Children, classes: &[(u32, u32)]) -> Result<(), Error> {
        let b = self;
        b.sum = [0.0; LANES];
        b.nonnegative = [true; LANES];
        for lane in 0..LANES {
            let (pc, pd) = classes.get(lane).copied().unwrap_or(classes[0]);
            let (pc, pd) = (pc as usize, pd as usize);
            let row = &c.success.values[pd * c.success.width..][..c.success.width];
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for (k, &column) in c.success_columns[pc].iter().enumerate() {
                let x = -row[column as usize];
                if !x.is_finite() {
                    return Err(Error::Invalid("nonfinite success child".into()));
                }
                b.s[k][lane] = x;
                lo = lo.min(x);
                hi = hi.max(x);
            }
            let child = c.failure.values[pd * c.failure.width + c.failure_columns[pc] as usize];
            if !child.is_finite() {
                return Err(Error::Invalid("nonfinite failure child".into()));
            }
            // The Dropper wins outright when the Checker does not revive.
            let (rev, s0) = (c.revival[pc], b.s[0][lane]);
            let f = rev * -child + (1.0 - rev);
            b.f[lane] = f;
            b.d[lane] = s0 - f;
            b.mn[lane] = lo.max(f.min(s0));
            b.mx[lane] = hi.min(f.max(s0));
            b.bounded[lane] = lo >= -2.0 && hi <= 2.0 && f.abs() <= 2.0;
        }
        Ok(())
    }

    /// Run the recurrence in all lanes and form the clipped Checker candidates.
    fn weights(&mut self) {
        let (s, q, r, d) = (&self.s, &mut self.q, &mut self.r, &self.d);
        // A zero denominator reads 1 here; the pure test or the size guard on d decides.
        for k in 1..A {
            for lane in 0..LANES {
                q[k][lane] =
                    (s[k - 1][lane] - s[k][lane]) / if d[lane] == 0.0 { 1.0 } else { d[lane] };
            }
        }
        r[0] = [1.0; LANES];
        for k in 1..A {
            let mut acc = [0.0; LANES];
            for j in 0..k {
                for lane in 0..LANES {
                    acc[lane] = q[k - j][lane].mul_add(r[j][lane], acc[lane]);
                }
            }
            r[k] = acc;
        }
        // The reversed, clipped weights; a nonfinite weight poisons the sum.
        for k in 0..A {
            for lane in 0..LANES {
                let x = r[A - 1 - k][lane];
                self.nonnegative[lane] &= x >= 0.0;
                q[k][lane] = if x.is_finite() { x.max(0.0) } else { f64::NAN };
                self.sum[lane] += q[k][lane];
            }
        }
    }

    /// Certify one lane by the pure bounds or the equalizer.
    fn certify(&mut self, lane: usize, window: bool) -> Option<(f64, u8)> {
        let (f, d, total) = (self.f[lane], self.d[lane], self.sum[lane]);
        let lift = |lower: f64, upper: f64| {
            if window {
                (lower.max(f), upper.max(f))
            } else {
                (lower, upper)
            }
        };
        let (lower, upper) = lift(self.mn[lane], self.mx[lane]);
        if upper - lower <= GAP {
            return Some((0.5 * (self.mn[lane] + self.mx[lane]), kind::PURE));
        }
        if d.abs() < 1e-12 || !total.is_finite() || total <= 0.0 {
            return None;
        }
        if self.nonnegative[lane] && self.bounded[lane] && d.abs() >= GAP {
            // Under these hypotheses both saddle bounds lie within 1e-10 of this value.
            return Some((f + d / total, kind::EQUALIZER));
        }
        // With the Dropper mixing reverse(q), (p^T M)[j] = (Mq)[59-j], so the extremes of Mq
        // bound the saddle from both sides.
        for row in &mut self.q {
            row[lane] /= total;
        }
        let (s, q) = (&self.s, &self.q);
        let (mut cum, mut ok) = (0.0, true);
        let (mut lower, mut upper) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..A {
            let x = f.mul_add(
                cum,
                (0..A - k).fold(0.0, |x, j| s[j][lane].mul_add(q[k + j][lane], x)),
            );
            ok &= x.is_finite();
            lower = lower.min(x);
            upper = upper.max(x);
            cum += q[k][lane];
        }
        let value = 0.5 * (lower + upper);
        let (lower, upper) = lift(lower, upper);
        (ok && (cum - 1.0).abs() <= 1e-12 && upper - lower <= GAP)
            .then_some((value, kind::EQUALIZER))
    }

    fn lane(&self, lane: usize) -> ([f64; A], [f64; A]) {
        (
            std::array::from_fn(|k| self.s[k][lane]),
            std::array::from_fn(|k| self.r[k][lane]),
        )
    }
}

impl<'a> Chunk<'a> {
    fn new(children: &'a Children<'a>, window: bool, residue: Residue) -> Self {
        Self {
            children,
            window,
            residue,
            packer: Packer::default(),
            seed: None,
        }
    }

    // Kept out of line: inlined into rayon's recursive split helper, its ~100 KB of scratch
    // would sit in every split frame and overflow the 2 MiB worker stacks.
    #[inline(never)]
    fn solve(&mut self, classes: &[(u32, u32)], out: &mut [Solved]) -> Result<(), Error> {
        let mut batch = Batch::new();
        for (classes, out) in classes.chunks(LANES).zip(out.chunks_mut(LANES)) {
            batch.gather(self.children, classes)?;
            batch.weights();
            for ((lane, slot), &class) in out.iter_mut().enumerate().zip(classes) {
                let mut certified = batch
                    .certify(lane, self.window)
                    .map(|(value, rung)| (value, rung, batch.f[lane]));
                if certified.is_none() && self.residue != Residue::Defer {
                    certified = self.solve_residue(&batch, lane, class);
                }
                *slot = match certified {
                    Some((mut value, mut rung, f)) => {
                        if self.window {
                            // Ties go to the leap row; the residue rungs keep their kinds.
                            if f >= value && rung < kind::PACKING {
                                rung = kind::LEAP_ROW;
                            }
                            value = value.max(f);
                        }
                        if !value.is_finite() || value.abs() > 1.0 + 1e-9 {
                            return Err(Error::Invalid("stage value out of range".into()));
                        }
                        Solved { value, kind: rung }
                    }
                    None => Solved {
                        value: f64::NAN,
                        kind: kind::UNSOLVED,
                    },
                };
            }
        }
        Ok(())
    }

    /// Try the kink seed, then the packing LP; each certified support becomes the next seed.
    /// Return the square-stage value, its kind and the failure payoff of the stage solved.
    fn solve_residue(
        &mut self,
        batch: &Batch,
        lane: usize,
        class: (u32, u32),
    ) -> Option<(f64, u8, f64)> {
        let (s, r) = batch.lane(lane);
        let kink = support::kink_index(&s);
        let (s, f) = match self.residue {
            Residue::Packing => self.children.stage(class),
            _ => (s, batch.f[lane]),
        };
        if let (Residue::Kink, Some((p, q, old))) = (self.residue, self.seed) {
            let (p, q) = support::kink_seed(p, q, old, kink);
            if p != 0
                && q != 0
                && r.iter().all(|x| x.is_finite())
                && let Some(value) = support::attempt(&s, f, &r, p, q, self.window)
            {
                self.seed = Some((p, q, kink));
                return Some((value, kind::KINK, f));
            }
        }
        let run = self.packer.solve(
            &std::array::from_fn(|k| if k < A { s[k] } else { f }),
            Start::Crash,
        );
        let (p, q) = self.packer.support()?;
        self.seed = Some((p, q, kink));
        Some((run.value, kind::PACKING, f))
    }
}
