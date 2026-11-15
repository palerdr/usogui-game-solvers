//! Player profiles, the STL clock and the table keys of the leap game.
//!
//! A profile is a vial content `st` (seconds, `0..300`) and a cumulative time-to-death `ttd`.
//! Profiles `0..N_ALIVE` can survive a failed check; they are ordered by `ttd` in
//! `[0, 60, 61, ..., 300]` and then by `st`. Profile `N_ALIVE + st` is a player with vial `st`
//! whom a failed check kills. Column `WIN` of a child table holds `-1`, the value of a state whose
//! Checker has just died, so the negated child reads `+1` for the Dropper.

use crate::Error;
use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;

/// The literal action seconds of the square stage, `1..=60`.
pub const ACTIONS: usize = 60;
/// The profiles that can survive a failed check.
pub const N_ALIVE: usize = 16_711;
/// All profiles: the survivors and one fatal profile per vial content.
pub const N: usize = 17_011;
/// The child-table column that means the Checker has died.
pub const WIN: usize = N;
/// The reset profiles that a revival can produce: vial 0 with each eligible TTD, and the fatal
/// profile with vial 0.
pub const RESETS: usize = 183;

const CAPACITY: u32 = 300;
const PENALTY: u32 = 60;
const LEAP_START: u32 = 3540;
const LEAP_END: u32 = 3600;

/// Return whether a half-round that starts at `clock` spans the leap second.
#[must_use]
pub fn is_leap_window(clock: u32) -> bool {
    (LEAP_START..=LEAP_END).contains(&clock)
}

/// Return the seconds a half-round lasts when it starts at `clock`.
#[must_use]
pub fn turn_duration(clock: u32) -> u32 {
    if is_leap_window(clock) { 61 } else { 60 }
}

/// Advance a round's end to the next whole minute; clock 3600 reads 8:59:60 and 3601 9:00:00.
#[must_use]
pub fn snap(clock: u32) -> u32 {
    match clock {
        ..3600 if clock / 60 == 59 => 3601,
        ..3600 => (clock / 60 + 1) * 60,
        3600 => 3601,
        _ => 3601 + ((clock - 3601) / 60 + 1) * 60,
    }
}

/// Return the start clock of the half that follows a half starting at `clock`.
///
/// `dose` is 0 after a successful check and the injected dose `st + 60` after a survived failed
/// check, which adds the dose and 120 seconds of procedure. The first half is followed by a
/// 60-second break; the second half ends the round at the next whole minute.
#[must_use]
pub fn child_clock(first_half: bool, clock: u32, dose: u32) -> u32 {
    let end = clock + turn_duration(clock) + if dose > 0 { dose + 120 } else { 0 };
    if first_half { end + 60 } else { snap(end) }
}

/// Return whether a player with vial `st` and TTD `ttd` survives a failed check with positive
/// probability.
#[must_use]
pub fn eligible(st: u32, ttd: u32) -> bool {
    st + PENALTY < CAPACITY && st + ttd + PENALTY <= CAPACITY
}

/// Return the probability that a Checker with vial `st` and TTD `ttd` survives a failed check:
/// `0.95 (1 - st/240) 0.75^(ttd/60)` when eligible, else 0.
#[must_use]
pub fn revival(st: u32, ttd: u32) -> f64 {
    if !eligible(st, ttd) {
        return 0.0;
    }
    let st_factor = 1.0 - f64::from(st) / 240.0;
    (0.95 * st_factor * 0.75_f64.powf(f64::from(ttd) / 60.0)).clamp(0.0, 1.0)
}

/// The profile tables that every key shares.
pub struct Profiles {
    /// Vial content of each profile.
    pub st: Vec<u16>,
    /// TTD of each profile; 301 marks a fatal profile.
    pub ttd: Vec<u16>,
    /// The Checker profile after a successful check with squandered time `k + 1`, or `WIN`.
    pub succ: Vec<[u32; ACTIONS]>,
    /// The Checker profile after a survived failed check, a reset profile; `WIN` when fatal.
    pub fail: Vec<u32>,
    /// The revival probability of each Checker profile.
    pub rev: Vec<f64>,
    /// The reset profiles in column order of a `REV` table.
    pub resets: [u32; RESETS],
    reset_column: Vec<u16>,
    /// Profile ranges that share one TTD, for the alive profiles.
    pub(crate) blocks: Vec<(usize, usize)>,
}

impl Profiles {
    /// Return the `REV` table column of a dropper profile: its reset index, or `RESETS` for `WIN`.
    #[must_use]
    pub fn reset_column(&self, profile: usize) -> Option<usize> {
        let column = self.reset_column[profile];
        (column != u16::MAX).then_some(usize::from(column))
    }
}

/// Return the profile tables, built on first use.
pub fn profiles() -> &'static Profiles {
    static PROFILES: OnceLock<Profiles> = OnceLock::new();
    PROFILES.get_or_init(build_profiles)
}

fn build_profiles() -> Profiles {
    let mut id = vec![[u32::MAX; 301]; 300];
    let (mut st, mut ttd, mut blocks) = (Vec::with_capacity(N), Vec::with_capacity(N), Vec::new());
    for t in std::iter::once(0).chain(60..=300) {
        let start = st.len();
        for s in (0..300).filter(|&s| eligible(s, t)) {
            id[s as usize][t as usize] = st.len() as u32;
            st.push(s as u16);
            ttd.push(t as u16);
        }
        if st.len() > start {
            blocks.push((start, st.len()));
        }
    }
    assert_eq!(st.len(), N_ALIVE, "the eligible profiles number N_ALIVE");
    st.extend(0..300);
    ttd.extend([301; 300]);
    let alive = |s: u32, t: u32| id[s as usize][t as usize];
    let succ = (0..N)
        .map(|pc| {
            std::array::from_fn(|k| {
                let grown = u32::from(st[pc]) + k as u32 + 1;
                match grown {
                    300.. => WIN as u32,
                    _ if pc < N_ALIVE && alive(grown, ttd[pc].into()) != u32::MAX => {
                        alive(grown, ttd[pc].into())
                    }
                    _ => (N_ALIVE as u32) + grown,
                }
            })
        })
        .collect();
    let reset = |t: u32| {
        if t <= 300 && alive(0, t) != u32::MAX {
            alive(0, t)
        } else {
            N_ALIVE as u32
        }
    };
    let fail = (0..N)
        .map(|pc| {
            if pc < N_ALIVE {
                reset(u32::from(st[pc] + ttd[pc]) + PENALTY)
            } else {
                WIN as u32
            }
        })
        .collect();
    let rev = (0..N)
        .map(|pc| {
            if pc < N_ALIVE {
                revival(st[pc].into(), ttd[pc].into())
            } else {
                0.0
            }
        })
        .collect();
    let mut resets = [N_ALIVE as u32; RESETS];
    for (slot, t) in resets.iter_mut().zip(std::iter::once(0).chain(60..=240)) {
        *slot = alive(0, t);
    }
    let mut reset_column = vec![u16::MAX; N + 1];
    for (column, &profile) in resets.iter().enumerate() {
        reset_column[profile as usize] = column as u16;
    }
    reset_column[WIN] = RESETS as u16;
    Profiles {
        st,
        ttd,
        succ,
        fail,
        rev,
        resets,
        reset_column,
        blocks,
    }
}

/// Return the profile index of a player with vial `st` and TTD `ttd` (301 for a fatal profile).
///
/// # Errors
///
/// [`Error::Invalid`] when no profile has these coordinates.
pub fn profile_index(st: u16, ttd: u16) -> Result<usize, Error> {
    let p = profiles();
    (0..N)
        .find(|&i| p.st[i] == st && p.ttd[i] == ttd)
        .ok_or_else(|| Error::Invalid(format!("no profile has st={st} and ttd={ttd}")))
}

/// A table of the leap game: one half-round start, holding a value per reachable state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    /// The first half of the round that starts at this minute; Hal drops.
    H1(u32),
    /// The second half of the round that starts at this minute; Baku drops.
    H2(u32),
    /// The half that resumes at this clock after a revival.
    Rev(u32),
}

impl Key {
    /// Return the clock, in seconds after 8:00, at which the half-round starts.
    #[must_use]
    pub fn clock(self) -> u32 {
        match self {
            Self::H1(minute) => 60 * minute,
            Self::H2(minute) => 60 * minute + 120,
            Self::Rev(clock) => clock,
        }
    }

    /// Return the dropper columns of the table: every profile, or the reset profiles after a
    /// revival.
    #[must_use]
    pub fn columns(self) -> usize {
        if matches!(self, Self::Rev(_)) {
            RESETS
        } else {
            N
        }
    }

    /// Return whether Baku may drop at second 61 in this half-round.
    #[must_use]
    pub fn is_window(self) -> bool {
        !matches!(self, Self::H1(_)) && is_leap_window(self.clock())
    }

    /// Return the key of the next half after a check with injected `dose` (0 for success), or
    /// `None` when that half starts after 3600 and the DTH table holds its values.
    #[must_use]
    pub fn child(self, dose: u32) -> Option<Self> {
        let first = matches!(self, Self::H1(_));
        let clock = child_clock(first, self.clock(), dose);
        (clock <= 3600).then_some(match (first, dose) {
            (true, 0) => Self::H2(self.minute()),
            (true, _) => Self::Rev(clock),
            (false, _) => Self::H1(clock / 60),
        })
    }

    /// Return the key type: `H1`, `H2` or `REV`.
    #[must_use]
    pub fn kind(self) -> &'static str {
        match self {
            Self::H1(_) => "H1",
            Self::H2(_) => "H2",
            Self::Rev(_) => "REV",
        }
    }

    fn minute(self) -> u32 {
        self.clock() / 60
    }

    /// Return every key of the game, ordered by clock.
    #[must_use]
    pub fn all() -> Vec<Self> {
        let mut keys: Vec<Self> = (12..60)
            .map(Self::H1)
            .chain((12..59).map(Self::H2))
            .chain((1020..=3600).map(Self::Rev))
            .collect();
        keys.sort_by_key(|key| key.clock());
        keys
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (Self::H1(index) | Self::H2(index) | Self::Rev(index)) = self;
        write!(f, "{}_{index}", self.kind())
    }
}

impl FromStr for Key {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Error> {
        let bad = || Error::Invalid(format!("{text} is not a leap key"));
        let (kind, index) = text.split_once('_').ok_or_else(bad)?;
        let index = index.parse().map_err(|_| bad())?;
        let key = match kind {
            "H1" => Self::H1(index),
            "H2" => Self::H2(index),
            "REV" => Self::Rev(index),
            _ => return Err(bad()),
        };
        Self::all().contains(&key).then_some(key).ok_or_else(bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks_match_the_referee() {
        // (first half, clock, dose, child clock) from stl.game.Game.resolve_half_round at 55e342e.
        let cases = [
            (true, 720, 0, 840),
            (false, 840, 0, 960),
            (true, 3420, 160, 3820),
            (false, 3420, 60, 3661),
            (true, 3540, 0, 3661),
            (false, 3540, 0, 3661),
            (false, 3539, 0, 3601),
            (false, 3600, 0, 3721),
            (false, 3601, 0, 3721),
            (true, 3599, 0, 3720),
            (true, 3479, 299, 4018),
            (false, 3480, 100, 3781),
        ];
        for (first, clock, dose, expected) in cases {
            assert_eq!(
                child_clock(first, clock, dose),
                expected,
                "{first} {clock} {dose}"
            );
        }
        assert_eq!(
            [snap(3479), snap(3480), snap(3539), snap(3600), snap(3601)],
            [3480, 3540, 3540, 3601, 3661]
        );
    }

    #[test]
    fn leap_window_is_closed() {
        let cases = [(3539, false), (3540, true), (3600, true), (3601, false)];
        for (clock, inside) in cases {
            assert_eq!(is_leap_window(clock), inside);
            assert_eq!(turn_duration(clock), if inside { 61 } else { 60 });
        }
        assert!(!Key::H1(59).is_window());
        assert!(
            Key::H2(57).is_window() && Key::Rev(3600).is_window() && !Key::Rev(3539).is_window()
        );
    }

    #[test]
    fn profiles_follow_the_rules() {
        let p = profiles();
        assert_eq!(p.resets.len(), RESETS);
        assert_eq!(p.rev[0], 0.95);
        let index = profile_index(60, 120).expect("an eligible profile");
        assert_eq!((p.st[index], p.ttd[index]), (60, 120));
        assert!(profile_index(299, 0).is_err());
        // The opening Checker grows to 1..=60 seconds or revives into REV_1020 at reset column 1.
        assert_eq!(p.succ[0].to_vec(), (1..=60).collect::<Vec<u32>>());
        assert_eq!(Key::H1(12).child(0), Some(Key::H2(12)));
        assert_eq!(Key::H1(12).child(60), Some(Key::Rev(1020)));
        assert_eq!(p.reset_column(p.fail[0] as usize), Some(1));
        assert_eq!(p.reset_column(WIN), Some(RESETS));
        assert_eq!(Key::all().len(), 48 + 47 + 2581);
        assert_eq!("REV_3600".parse::<Key>().ok(), Some(Key::Rev(3600)));
        assert!("H2_59".parse::<Key>().is_err());
    }
}
