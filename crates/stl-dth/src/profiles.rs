//! Profiles, their rule tables and the class encoding (DTH.md 4.1-4.3).

use crate::{
    CAPACITY, Certificate, DEAD_PHI, DOSE, Error, LAGS, MAX_PHI, MAX_ST, MAX_TTD, Mix, N, N_ALIVE,
    N_DEAD, State, WIN, solve_class,
};

/// Whether a Checker at `(st, ttd)` can survive a failed check: the dose `st + 60` stays below
/// capacity and the TTD it leaves stays within 300.
#[must_use]
pub fn survives(st: usize, ttd: usize) -> bool {
    st + DOSE <= MAX_ST && st + DOSE + ttd <= MAX_TTD
}

/// The frozen revival probability `0.95 · (1 − st/240) · 0.75^(ttd/60)` of a failed check, or 0
/// when the Checker cannot survive it (RULES.md).
#[must_use]
pub fn revival(st: usize, ttd: usize) -> f64 {
    if survives(st, ttd) {
        0.95 * (1.0 - st as f64 / 240.0) * 0.75_f64.powf(ttd as f64 / 60.0)
    } else {
        0.0
    }
}

/// The 17,011 profiles in the normative order and every rule evaluated once: the sweep only
/// gathers.
pub struct Profiles {
    /// `alive_id[st * (MAX_TTD + 1) + ttd]`: the id of an alive profile.
    alive_id: Vec<Option<usize>>,
    /// Squandered time of each profile.
    pub st: Vec<usize>,
    /// Accrued toxin time of each profile; `DEAD_PHI` for the dead.
    pub ttd: Vec<usize>,
    /// Potential `st + ttd`; it grows on every move.
    pub phi: Vec<usize>,
    /// Revival probability of a failed check; 0 for the dead.
    pub rev: Vec<f64>,
    /// The profile a failed check revives into, or `WIN` for a dead Checker.
    pub fail: Vec<u32>,
    /// `succ[p][lag - 1]`: the profile a successful check of `lag` seconds grows into, or `WIN`.
    pub succ: Vec<[u32; LAGS]>,
    /// `bucket[v]`: the ascending ids with potential `v`, for `v` in `0..=MAX_PHI`.
    pub bucket: Vec<Vec<usize>>,
}

impl Profiles {
    /// Index the profiles (TTD ascending over 0 then 60..=300, ST ascending inside each TTD, then
    /// the dead sentinels by ST) and tabulate their transitions.
    ///
    /// # Panics
    ///
    /// When the census or the strictly increasing potential fails: the rules are broken.
    #[must_use]
    pub fn build() -> Self {
        let mut alive_id = vec![None; (MAX_ST + 1) * (MAX_TTD + 1)];
        let (mut st, mut ttd) = (Vec::with_capacity(N), Vec::with_capacity(N));
        for t in std::iter::once(0).chain(DOSE..=MAX_TTD) {
            for s in (0..=MAX_ST).filter(|&s| survives(s, t)) {
                alive_id[s * (MAX_TTD + 1) + t] = Some(st.len());
                st.push(s);
                ttd.push(t);
            }
        }
        assert_eq!(
            st.len(),
            N_ALIVE,
            "the normative order indexes every alive profile"
        );
        st.extend(0..N_DEAD);
        ttd.resize(N, DEAD_PHI);
        let id = |s: usize, t: usize| {
            alive_id[s * (MAX_TTD + 1) + t].expect("alive TTD is 0 or 60..=300") as u32
        };
        let succ: Vec<[u32; LAGS]> = (0..N)
            .map(|p| {
                std::array::from_fn(|k| match st[p] + k + 1 {
                    grown if grown >= CAPACITY => WIN as u32,
                    grown if p < N_ALIVE && survives(grown, ttd[p]) => id(grown, ttd[p]),
                    grown => (N_ALIVE + grown) as u32,
                })
            })
            .collect();
        let fail: Vec<u32> = (0..N)
            .map(|p| match st[p] + ttd[p] + DOSE {
                _ if p >= N_ALIVE => WIN as u32,
                t if survives(0, t) => id(0, t),
                _ => N_ALIVE as u32,
            })
            .collect();
        let phi: Vec<usize> = st.iter().zip(&ttd).map(|(s, t)| s + t).collect();
        let mut bucket = vec![Vec::new(); MAX_PHI + 1];
        for (p, &v) in phi.iter().enumerate() {
            bucket[v].push(p);
        }
        for p in 0..N {
            for &c in succ[p]
                .iter()
                .chain([&fail[p]])
                .filter(|&&c| c as usize != WIN)
            {
                assert!(
                    phi[c as usize] > phi[p],
                    "phi strictly increases on every move"
                );
            }
        }
        let rev = (0..N)
            .map(|p| {
                if p < N_ALIVE {
                    revival(st[p], ttd[p])
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            alive_id,
            st,
            ttd,
            phi,
            rev,
            fail,
            succ,
            bucket,
        }
    }

    /// The profile id of `(st, ttd)`; a dead profile collapses to the sentinel for its ST.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] outside ST 0..=299 and TTD 0..=300, and [`Error::OffDomain`] for an
    /// alive TTD in 1..=59.
    pub fn profile(&self, st: usize, ttd: usize) -> Result<usize, Error> {
        if st > MAX_ST || ttd > MAX_TTD {
            Err(Error::OutOfRange(st, ttd))
        } else if !survives(st, ttd) {
            Ok(N_ALIVE + st)
        } else {
            self.alive_id[st * (MAX_TTD + 1) + ttd].ok_or(Error::OffDomain(st, ttd))
        }
    }

    /// The (Checker, Dropper) profiles of a state: the class is `V[pc, pd]`.
    ///
    /// # Errors
    ///
    /// As [`Profiles::profile`].
    pub fn classes(&self, [sc, tc, sd, td]: State) -> Result<(usize, usize), Error> {
        Ok((self.profile(sc, tc)?, self.profile(sd, td)?))
    }

    /// The 60 successful-check values and the failed-check value of class `(pc, pd)`, from the
    /// Dropper's side (DTH.md 4.3). `value` reads the flat table `V[row * (N + 1) + col]`; every
    /// child swaps roles, so its value is negated.
    ///
    /// # Errors
    ///
    /// [`Error::Unsolved`] when a child is not a finished value.
    pub fn class_values(
        &self,
        pc: usize,
        pd: usize,
        value: impl Fn(usize) -> f64,
    ) -> Result<(Mix, f64), Error> {
        let row = pd * (N + 1);
        let s: Mix = std::array::from_fn(|k| -value(row + self.succ[pc][k] as usize));
        let p = self.rev[pc];
        let f = p * -value(row + self.fail[pc] as usize) + (1.0 - p);
        if s.iter().chain([&f]).all(|x| x.is_finite()) {
            Ok((s, f))
        } else {
            Err(Error::Unsolved(pc, pd))
        }
    }

    /// Gather class `(pc, pd)` and certify it with the ladder.
    ///
    /// # Errors
    ///
    /// [`Error::Unsolved`] or [`Error::Uncertified`].
    pub fn solve(
        &self,
        pc: usize,
        pd: usize,
        value: impl Fn(usize) -> f64,
    ) -> Result<Certificate, Error> {
        let (s, f) = self.class_values(pc, pd, value)?;
        solve_class(&s, f).ok_or(Error::Uncertified(pc, pd))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_and_normative_order() {
        let t = Profiles::build();
        assert_eq!((t.st.len(), t.alive_id.len()), (17_011, 300 * 301));
        let per_ttd = |v: usize| t.ttd[..N_ALIVE].iter().filter(|&&x| x == v).count();
        assert_eq!(per_ttd(0), 240);
        assert!((60..=240).all(|v| per_ttd(v) == 241 - v));
        assert!((241..=300).all(|v| per_ttd(v) == 0));
        let key = |p: usize| (t.ttd[p], t.st[p]);
        assert!(
            (1..N_ALIVE).all(|p| key(p - 1) < key(p)),
            "TTD ascending, ST ascending inside"
        );
    }

    #[test]
    fn spot_ids() {
        let t = Profiles::build();
        let id = |s, tt| t.profile(s, tt).expect("on the domain");
        assert_eq!(
            [id(0, 0), id(239, 0), id(0, 60), id(180, 60)],
            [0, 239, 240, 420]
        );
        assert_eq!(
            [id(240, 0), id(299, 0), id(240, 300)],
            [16_951, 17_010, 16_951]
        );
        assert!(matches!(t.profile(5, 30), Err(Error::OffDomain(5, 30))));
        assert_eq!(
            t.classes([240, 0, 240, 0]).expect("dead band"),
            (16_951, 16_951)
        );
        assert_eq!(
            t.classes([10, 60, 200, 0]).expect("alive"),
            (id(10, 60), id(200, 0))
        );
    }

    #[test]
    fn phi_and_buckets() {
        let t = Profiles::build();
        let alive = &t.phi[..N_ALIVE];
        assert_eq!(
            (alive.iter().min(), alive.iter().max()),
            (Some(&0), Some(&240))
        );
        assert!((0..N_DEAD).all(|s| t.phi[N_ALIVE + s] == s + 301));
        let sizes: Vec<usize> = t.bucket.iter().map(Vec::len).collect();
        assert_eq!((sizes.len(), sizes.iter().sum()), (601, N));
        assert!(sizes[241..301].iter().all(|&n| n == 0) && sizes[301..].iter().all(|&n| n == 1));
        for (v, b) in t.bucket.iter().enumerate() {
            assert!(b.iter().all(|&p| t.phi[p] == v) && b.is_sorted());
        }
    }

    #[test]
    fn successful_check_children() {
        let t = Profiles::build();
        let win = WIN as u32;
        assert!(
            t.succ[0].iter().copied().eq(1..=60),
            "(0, 0): every lag survives"
        );
        assert!(
            t.succ[239].iter().copied().eq(16_951..17_011),
            "(239, 0): every lag lands dead"
        );
        assert!(
            t.succ[17_010].iter().all(|&c| c == win),
            "(299, dead): every lag hits capacity"
        );
        for (p, row) in t.succ.iter().enumerate() {
            let kind = |c: u32| u8::from(c as usize >= N_ALIVE) + u8::from(c == win);
            for w in row.windows(2) {
                assert!(
                    kind(w[0]) <= kind(w[1]),
                    "alive ids, then dead ids, then WIN"
                );
                assert!(
                    kind(w[0]) != kind(w[1]) || w[1] == win || w[1] == w[0] + 1,
                    "runs are consecutive"
                );
            }
            assert!(
                p < N_ALIVE || row.iter().all(|&c| kind(c) >= 1),
                "the dead stay dead"
            );
            for (k, &c) in row.iter().enumerate() {
                assert_eq!(c == win, t.st[p] + k + 1 >= 300, "WIN exactly at capacity");
                if c != win {
                    assert_eq!(
                        t.st[c as usize],
                        t.st[p] + k + 1,
                        "the child grows by the lag"
                    );
                }
                if (c as usize) < N_ALIVE {
                    assert_eq!(t.ttd[c as usize], t.ttd[p], "an alive child keeps its TTD");
                }
            }
        }
    }

    #[test]
    fn failed_check_children_and_revival() {
        let t = Profiles::build();
        let p = t.profile(0, 0).expect("root");
        assert_eq!((t.fail[p], t.rev[p]), (240, 0.95));
        let p = t.profile(180, 60).expect("alive");
        assert_eq!(
            t.fail[p] as usize, N_ALIVE,
            "TTD 300 after the dose cannot survive again"
        );
        assert!((t.rev[p] - 0.178_125).abs() < 1e-15);
        assert!((N_ALIVE..N).all(|p| t.fail[p] as usize == WIN && t.rev[p] == 0.0));
        assert!(t.rev[..N_ALIVE].iter().all(|&r| r > 0.0 && r <= 0.95));
        for p in (0..N_ALIVE).filter(|&p| (t.fail[p] as usize) < N_ALIVE) {
            let c = t.fail[p] as usize;
            assert_eq!(
                (t.st[c], t.ttd[c]),
                (0, t.ttd[p] + t.st[p] + 60),
                "revived at ST 0"
            );
        }
    }

    #[test]
    fn potential_strictly_increases() {
        let t = Profiles::build();
        for p in 0..N {
            for &c in t.succ[p]
                .iter()
                .chain([&t.fail[p]])
                .filter(|&&c| c as usize != WIN)
            {
                assert!(t.phi[c as usize] > t.phi[p]);
            }
        }
    }

    #[test]
    fn frozen_revival_surface_matches_the_rule_text() {
        // RULES.md: eligible exactly when s <= 239 and s + t <= 240.
        for s in 0..300 {
            for t in 0..=300 {
                let eligible = s <= 239 && s + t <= 240;
                assert_eq!(survives(s, t), eligible);
                let p = 0.95 * (1.0 - s as f64 / 240.0) * 0.75_f64.powf(t as f64 / 60.0);
                assert_eq!(revival(s, t), if eligible { p } else { 0.0 });
            }
        }
        assert!((revival(0, 60) - 0.95 * 0.75).abs() < 1e-15);
        assert!((revival(239, 0) - 0.95 / 240.0).abs() < 1e-15);
    }

    #[test]
    fn dose_and_capacity_boundaries() {
        let t = Profiles::build();
        let id = |s, tt| t.profile(s, tt).expect("on the domain");
        let win = WIN as u32;
        assert_eq!(t.fail[id(240, 0)], win, "a failed check at ST 240 is fatal");
        let p = id(0, 240);
        assert!(
            t.rev[p] == revival(0, 240) && t.rev[p] > 0.0,
            "a resulting TTD of 300 stays eligible"
        );
        assert_eq!(
            (t.fail[p] as usize, id(0, 300)),
            (N_ALIVE, N_ALIVE),
            "revived at ST 0, TTD 300"
        );
        assert_eq!(
            (revival(0, 241), t.fail[id(0, 241)]),
            (0.0, win),
            "TTD above 300 is fatal"
        );
        assert_eq!(
            t.succ[id(298, 0)][..2],
            [id(299, 0) as u32, win],
            "ST 299 lives, 300 dumps"
        );
        assert!(t.succ[id(299, 0)].iter().all(|&c| c == win));
        assert_eq!(
            t.succ[id(240, 0)][59],
            win,
            "exactly 300 by a 60-second lag"
        );
    }

    #[test]
    fn rejects_states_off_the_domain() {
        let t = Profiles::build();
        for state in [
            [300, 0, 0, 0],
            [0, 301, 0, 0],
            [0, 0, 300, 0],
            [0, 0, 0, 301],
        ] {
            assert!(
                matches!(t.classes(state), Err(Error::OutOfRange(..))),
                "{state:?}"
            );
        }
        assert!(matches!(
            t.classes([5, 30, 0, 0]),
            Err(Error::OffDomain(..))
        ));
        assert_eq!(
            t.classes([250, 30, 0, 0]).expect("dead TTD collapses"),
            (N_ALIVE + 250, 0)
        );
    }
}
