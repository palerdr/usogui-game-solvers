//! The Air Poker engine: immutable states, legal moves, uniform upper chance and settlement.
//!
//! The engine is the oracle. The kernel sampler, the betting DAG and self-play are all checked
//! against its transitions. A transition returns a new state and never changes the old one.

use crate::cards::{FULL_DECK, evaluate, legal_hands, rank_sum};
use crate::{ANTES, Error, ROUNDS, TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS};

/// BIOS each lower player receives.
pub const STARTING_BIOS: u32 = 25;
/// Seconds of air in one BIOS.
const BIO_SECONDS: u32 = 300;

/// One lower player and its allied upper player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Team {
    /// Baku's side.
    Zero,
    /// Lalo's side.
    One,
}

impl Team {
    /// The team with index 0 or 1.
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        if index == 0 { Self::Zero } else { Self::One }
    }

    /// Index 0 or 1 into per-team arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The opposing team.
    #[must_use]
    pub const fn other(self) -> Self {
        Self::from_index(1 - self.index())
    }
}

/// The stage of the game, which fixes who acts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Team 0 picks a metal card in secret.
    NumberChoice0,
    /// Team 1 picks a metal card without seeing Team 0's.
    NumberChoice1,
    /// The upper players build their hands.
    UpperChance,
    /// The single betting street.
    Betting,
    /// The game is over.
    Terminal,
}

/// A betting action of the acting lower player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bet {
    /// Pass with nothing to call.
    Check,
    /// Pay the outstanding amount and go to judgement.
    Call,
    /// Concede the pot while facing a wager.
    Fold,
    /// Call, then raise by a positive number of BIOS.
    Raise(u32),
}

/// The two upper hands; `None` is a no-rank hand.
pub type HandPair = [Option<u64>; 2];

/// Any move: a lower player's metal card or bet, or an upper chance outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    /// A metal card chosen at a number-choice phase.
    Number(u8),
    /// Both upper hands at the chance phase.
    Hands(HandPair),
    /// A bet at the betting phase.
    Bet(Bet),
}

/// A lower player's resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lower {
    /// BIOS that can still be wagered.
    pub sealed_bios: u32,
    /// Seconds left in the opened BIOS, used only in the final comparison.
    pub air_seconds: u32,
    /// Metal cards not yet played.
    pub metal_cards: Vec<u8>,
}

/// The public state of the current betting street.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Betting {
    /// The team that bets first this round.
    pub first: Team,
    /// The team to act.
    pub actor: Team,
    /// Each team's total contribution to the pot, ante included.
    pub contributions: [u32; 2],
    /// Whether the previous action was a check.
    pub checked: bool,
    /// The actions so far.
    pub actions: Vec<(Team, Bet)>,
}

/// The current round: choices, hidden hands, retry flag and betting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Round {
    /// The chosen metal cards, `None` until chosen.
    pub chosen: [Option<u8>; 2],
    /// The upper hands once chance resolves.
    pub hands: Option<HandPair>,
    /// Whether a double no-rank sent the round to a retry.
    pub retry: bool,
    /// The betting street once chance resolves.
    pub betting: Option<Betting>,
}

/// The public facts of a completed round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// Both metal cards.
    pub chosen: [u8; 2],
    /// The first bettor.
    pub first: Team,
    /// The betting actions.
    pub actions: Vec<(Team, Bet)>,
    /// The judgement winner; `None` is a round with no judgement.
    pub winner: Option<Team>,
    /// Whether the judged hands shared a card.
    pub calamity: bool,
}

/// The complete hidden-truth state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameState {
    /// The stage of the game.
    pub phase: Phase,
    /// Both lower players.
    pub lower: [Lower; 2],
    /// The cards left in the shared deck.
    pub deck: u64,
    /// The current round.
    pub round: Round,
    /// Completed rounds.
    pub history: Vec<Record>,
    /// The winner once terminal; `None` is a tie.
    pub winner: Option<Team>,
}

impl GameState {
    /// The state after both players open one BIOS and pay the round-1 ante.
    ///
    /// # Errors
    /// [`Error::Setup`] unless the ten metal cards are distinct values in `6..=64` and each
    /// residual air value is at most 300.
    pub fn new(metal_cards: [[u8; 5]; 2], air_seconds: [u32; 2]) -> Result<Self, Error> {
        let mut all = metal_cards.concat();
        if all.iter().any(|card| !(6..=64).contains(card)) {
            return Err(Error::Setup("metal card values lie in 6..=64"));
        }
        all.sort_unstable();
        all.dedup();
        if all.len() != 10 {
            return Err(Error::Setup("the ten metal cards are distinct"));
        }
        if air_seconds.iter().any(|&air| air > BIO_SECONDS) {
            return Err(Error::Setup("residual air lies in 0..=300"));
        }
        Ok(Self {
            phase: Phase::NumberChoice0,
            lower: [0, 1].map(|i| Lower {
                sealed_bios: STARTING_BIOS - 1 - ANTES[0],
                air_seconds: air_seconds[i],
                metal_cards: metal_cards[i].to_vec(),
            }),
            deck: FULL_DECK,
            round: Round::default(),
            history: Vec::new(),
            winner: None,
        })
    }

    /// The canonical root: Baku against Lalo with full residual air.
    #[must_use]
    pub fn canonical() -> Self {
        Self::new([TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS], [BIO_SECONDS; 2])
            .expect("the canonical root is valid")
    }

    fn betting(&self) -> &Betting {
        self.round
            .betting
            .as_ref()
            .expect("the betting phase has a betting state")
    }

    /// The acting lower team, or `None` at chance and terminal phases.
    #[must_use]
    pub fn actor(&self) -> Option<Team> {
        match self.phase {
            Phase::NumberChoice0 => Some(Team::Zero),
            Phase::NumberChoice1 => Some(Team::One),
            Phase::Betting => Some(self.betting().actor),
            Phase::UpperChance | Phase::Terminal => None,
        }
    }

    /// Every legal lower-player move: unused metal cards, or check or call and fold, then raises.
    #[must_use]
    pub fn legal_moves(&self) -> Vec<Move> {
        match self.phase {
            Phase::NumberChoice0 | Phase::NumberChoice1 => {
                let team = self.actor().expect("a choice phase has an actor").index();
                self.lower[team]
                    .metal_cards
                    .iter()
                    .map(|&card| Move::Number(card))
                    .collect()
            }
            Phase::Betting => {
                let betting = self.betting();
                let (i, j) = (betting.actor.index(), betting.actor.other().index());
                let to_call = betting.contributions[j].saturating_sub(betting.contributions[i]);
                let mut bets = vec![if to_call == 0 { Bet::Check } else { Bet::Fold }];
                if self.lower[i].sealed_bios >= to_call {
                    if to_call > 0 {
                        bets.insert(0, Bet::Call);
                    }
                    let pot_after_call =
                        betting.contributions[0] + betting.contributions[1] + to_call;
                    let max_extra = (self.lower[i].sealed_bios - to_call)
                        .min(self.lower[j].sealed_bios)
                        .min(pot_after_call / 2);
                    bets.extend((1..=max_extra).map(Bet::Raise));
                }
                bets.into_iter().map(Move::Bet).collect()
            }
            Phase::UpperChance | Phase::Terminal => Vec::new(),
        }
    }

    /// Both upper hands with their probabilities: each upper player draws uniformly from the
    /// legal hands of the deck for its metal card, or holds a no-rank hand when there are none.
    ///
    /// A uniform draw gives a double no-rank only when both targets are impossible, so a retry
    /// has that single outcome again, which settles the round with no judgement.
    ///
    /// # Errors
    /// [`Error::Illegal`] outside the chance phase.
    pub fn chance_outcomes(&self) -> Result<Vec<(HandPair, f64)>, Error> {
        let (Phase::UpperChance, [Some(m0), Some(m1)]) = (self.phase, self.round.chosen) else {
            return Err(Error::Illegal(
                "chance outcomes exist only at the upper chance phase",
            ));
        };
        let uniform = |target: u8| -> Vec<(Option<u64>, f64)> {
            let hands = legal_hands(u32::from(target), self.deck);
            if hands.is_empty() {
                return vec![(None, 1.0)];
            }
            let p = 1.0 / hands.len() as f64;
            hands.into_iter().map(|hand| (Some(hand), p)).collect()
        };
        let o1 = uniform(m1);
        Ok(uniform(m0)
            .into_iter()
            .flat_map(|(h0, p0)| o1.iter().map(move |&(h1, p1)| ([h0, h1], p0 * p1)))
            .collect())
    }

    /// The state after one legal move.
    ///
    /// At the chance phase the hands are checked for card count, sum and availability.
    ///
    /// # Errors
    /// [`Error::Illegal`] for a move the state does not allow.
    pub fn apply(&self, mv: Move) -> Result<Self, Error> {
        match (self.phase, mv) {
            (Phase::NumberChoice0 | Phase::NumberChoice1, Move::Number(card)) => {
                let team = self.actor().expect("a choice phase has an actor");
                let mut next = self.clone();
                let cards = &mut next.lower[team.index()].metal_cards;
                let position = cards
                    .iter()
                    .position(|&c| c == card)
                    .ok_or(Error::Illegal("the metal card is unused"))?;
                cards.remove(position);
                next.round.chosen[team.index()] = Some(card);
                next.phase = if team == Team::Zero {
                    Phase::NumberChoice1
                } else {
                    Phase::UpperChance
                };
                Ok(next)
            }
            (Phase::UpperChance, Move::Hands(hands)) => self.apply_hands(hands),
            (Phase::Betting, Move::Bet(bet)) => self.apply_bet(bet),
            _ => Err(Error::Illegal("the move does not fit the phase")),
        }
    }

    fn apply_hands(&self, hands: HandPair) -> Result<Self, Error> {
        let [Some(m0), Some(m1)] = self.round.chosen else {
            return Err(Error::Illegal("chance follows both choices"));
        };
        validate_hand(hands[0], self.deck, m0)?;
        validate_hand(hands[1], self.deck, m1)?;
        let first = match self.history.last() {
            Some(record) => record.first.other(),
            None if m0 < m1 => Team::Zero,
            None => Team::One,
        };
        let mut next = self.clone();
        if hands == [None, None] && !self.round.retry {
            next.round.retry = true;
            return Ok(next);
        }
        let ante = ANTES[self.history.len()];
        next.round.betting = Some(Betting {
            first,
            actor: first,
            contributions: [ante; 2],
            checked: false,
            actions: Vec::new(),
        });
        next.round.hands = Some(hands);
        if hands == [None, None] {
            // A double no-rank on a retry: the round ends with no judgement.
            return Ok(next.settle(None, false));
        }
        next.deck &= !(hands[0].unwrap_or(0) | hands[1].unwrap_or(0));
        next.phase = Phase::Betting;
        Ok(next)
    }

    fn apply_bet(&self, bet: Bet) -> Result<Self, Error> {
        if !self.legal_moves().contains(&Move::Bet(bet)) {
            return Err(Error::Illegal("the bet is not legal here"));
        }
        let mut next = self.clone();
        let hands = self.round.hands.expect("betting follows chance");
        let betting = next
            .round
            .betting
            .as_mut()
            .expect("the betting phase has a betting state");
        let actor = betting.actor;
        let (i, j) = (actor.index(), actor.other().index());
        let was_checked = betting.checked;
        betting.actions.push((actor, bet));
        match bet {
            Bet::Check => {
                betting.actor = actor.other();
                betting.checked = true;
                Ok(if was_checked {
                    next.settle(poker_winner(hands), true)
                } else {
                    next
                })
            }
            Bet::Fold => Ok(next.settle(Some(actor.other()), false)),
            Bet::Call | Bet::Raise(_) => {
                let extra = if let Bet::Raise(extra) = bet {
                    extra
                } else {
                    0
                };
                let payment =
                    betting.contributions[j].saturating_sub(betting.contributions[i]) + extra;
                betting.contributions[i] += payment;
                betting.actor = actor.other();
                betting.checked = false;
                next.lower[i].sealed_bios -= payment;
                Ok(if bet == Bet::Call {
                    next.settle(poker_winner(hands), true)
                } else {
                    next
                })
            }
        }
    }

    fn settle(mut self, winner: Option<Team>, showdown: bool) -> Self {
        let betting = self
            .round
            .betting
            .take()
            .expect("settlement follows betting");
        let hands = self.round.hands.expect("settlement follows chance");
        let c = betting.contributions;
        match winner {
            None => (0..2).for_each(|i| self.lower[i].sealed_bios += c[i]),
            Some(team) => self.lower[team.index()].sealed_bios += c[0] + c[1],
        }
        let calamity =
            showdown && winner.is_some() && matches!(hands, [Some(h0), Some(h1)] if h0 & h1 != 0);
        let mut bankrupt = false;
        if calamity && let Some(team) = winner {
            let loser = &mut self.lower[team.other().index()];
            let penalty = c[team.other().index()];
            bankrupt = loser.sealed_bios < penalty;
            loser.sealed_bios = loser.sealed_bios.saturating_sub(penalty);
        }
        let chosen = self
            .round
            .chosen
            .map(|card| card.expect("a settled round has both choices"));
        self.history.push(Record {
            chosen,
            first: betting.first,
            actions: betting.actions,
            winner,
            calamity,
        });
        self.round = Round::default();
        self.phase = Phase::Terminal;
        if bankrupt {
            self.winner = winner;
            return self;
        }
        if self.history.len() == ROUNDS {
            let totals = self
                .lower
                .each_ref()
                .map(|p| p.sealed_bios * BIO_SECONDS + p.air_seconds);
            self.winner = (totals[0] != totals[1])
                .then(|| Team::from_index(usize::from(totals[1] > totals[0])));
            return self;
        }
        let ante = ANTES[self.history.len()];
        let can_pay = self.lower.each_ref().map(|p| p.sealed_bios >= ante);
        if can_pay != [true, true] {
            self.winner =
                (can_pay[0] != can_pay[1]).then(|| Team::from_index(usize::from(can_pay[1])));
            return self;
        }
        self.lower.iter_mut().for_each(|p| p.sealed_bios -= ante);
        self.phase = Phase::NumberChoice0;
        self
    }

    /// Zero-sum returns `(1, -1)`, `(-1, 1)` or `(0, 0)` of a terminal state.
    ///
    /// # Errors
    /// [`Error::Illegal`] before the game ends.
    pub fn returns(&self) -> Result<[f64; 2], Error> {
        match (self.phase, self.winner) {
            (Phase::Terminal, Some(Team::Zero)) => Ok([1.0, -1.0]),
            (Phase::Terminal, Some(Team::One)) => Ok([-1.0, 1.0]),
            (Phase::Terminal, None) => Ok([0.0, 0.0]),
            _ => Err(Error::Illegal("returns exist only at a terminal state")),
        }
    }
}

fn validate_hand(hand: Option<u64>, deck: u64, target: u8) -> Result<(), Error> {
    match hand {
        Some(mask) if mask.count_ones() != 5 => {
            Err(Error::Illegal("an upper hand holds five cards"))
        }
        Some(mask) if mask & !deck != 0 => {
            Err(Error::Illegal("an upper hand uses available cards"))
        }
        Some(mask) if rank_sum(mask) != u32::from(target) => {
            Err(Error::Illegal("an upper hand sums to its metal card"))
        }
        _ => Ok(()),
    }
}

/// The judgement winner; equal strength would be `None` but needs equal metal cards.
fn poker_winner(hands: HandPair) -> Option<Team> {
    match hands {
        [None, _] => Some(Team::One),
        [_, None] => Some(Team::Zero),
        [Some(h0), Some(h1)] => {
            let [s0, s1] =
                [h0, h1].map(|hand| evaluate(hand).expect("validated hands hold five cards"));
            (s0 != s1).then(|| Team::from_index(usize::from(s1 > s0)))
        }
    }
}
