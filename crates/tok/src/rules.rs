//! The rules engine: states, legal actions, transitions and payoffs of `D_K`.
//!
//! The certificate depends on this module alone, so it stays a literal reading of the rules and
//! shares no code with the public-tree arrays.

use crate::Error;

/// A player. P0 picks beads first and announces first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Player {
    /// The first player.
    P0,
    /// The second player.
    P1,
}

/// A move: pick a bead count, or announce a total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Pick this many beads.
    Choose(u8),
    /// Announce this total.
    Guess(u8),
}

/// A public announcement that missed the true total.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Miss {
    /// The player who announced.
    pub player: Player,
    /// The announced total.
    pub total: u8,
}

/// Whose move it is and what kind of move, or the end of the game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// The player picks its beads.
    Choose(Player),
    /// The player announces a total.
    Guess(Player),
    /// The game is over.
    Terminal,
}

/// A full game state. Only [`GameState::new`] and [`GameState::apply`] build one, so every state
/// is reachable.
#[derive(Clone, Debug)]
pub struct GameState {
    phase: Phase,
    horizon: usize,
    turn: usize,
    beads: [Option<u8>; 2],
    history: Vec<Miss>,
    winner: Option<Player>,
}

/// What one player knows: its own beads and the public history.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InformationState {
    /// The observing player.
    pub(crate) player: Player,
    /// The phase of the observed state.
    pub(crate) phase: Phase,
    /// The number of turns per player.
    pub(crate) horizon: usize,
    /// The current turn, 0 before the announcements.
    pub(crate) turn: usize,
    /// The observer's beads, once picked.
    pub(crate) own_beads: Option<u8>,
    /// All misses so far, in order.
    pub(crate) history: Vec<Miss>,
}

impl GameState {
    /// The start of `D_horizon`: P0 is about to pick its beads.
    ///
    /// # Errors
    ///
    /// [`Error::Horizon`] when `horizon` is 0.
    pub fn new(horizon: usize) -> Result<Self, Error> {
        if horizon == 0 {
            return Err(Error::Horizon);
        }
        Ok(Self {
            phase: Phase::Choose(Player::P0),
            horizon,
            turn: 0,
            beads: [None, None],
            history: Vec::new(),
            winner: None,
        })
    }

    /// The phase of the state.
    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// The bead counts picked so far, indexed by player.
    #[must_use]
    pub fn beads(&self) -> [Option<u8>; 2] {
        self.beads
    }

    /// All misses so far, in order.
    #[must_use]
    pub fn history(&self) -> &[Miss] {
        &self.history
    }

    /// The player to move, or `None` at the end.
    #[must_use]
    pub(crate) fn actor(&self) -> Option<Player> {
        match self.phase {
            Phase::Choose(p) | Phase::Guess(p) => Some(p),
            Phase::Terminal => None,
        }
    }

    /// The legal actions in order: bead counts 1..=10, or own beads + 1..=own beads + 10.
    ///
    /// # Panics
    ///
    /// Never: [`GameState::apply`] fills both bead counts before the announcements.
    pub fn legal_actions(&self) -> impl Iterator<Item = Action> + use<> {
        let (make, first, count): (fn(u8) -> Action, u8, u8) = match self.phase {
            Phase::Choose(_) => (Action::Choose, 1, 10),
            Phase::Guess(p) => {
                let own = self.beads[p as usize].expect("an announcer has picked beads");
                (Action::Guess, own + 1, 10)
            }
            Phase::Terminal => (Action::Guess, 0, 0),
        };
        (first..first + count).map(make)
    }

    /// The state after `action`.
    ///
    /// # Errors
    ///
    /// [`Error::IllegalAction`] when `action` is not among [`GameState::legal_actions`].
    pub fn apply(&self, action: Action) -> Result<Self, Error> {
        if !self.legal_actions().any(|a| a == action) {
            return Err(Error::IllegalAction(action));
        }
        let mut next = self.clone();
        match (self.phase, action) {
            (Phase::Choose(p), Action::Choose(count)) => {
                next.beads[p as usize] = Some(count);
                next.phase = match p {
                    Player::P0 => Phase::Choose(Player::P1),
                    Player::P1 => {
                        next.turn = 1;
                        Phase::Guess(Player::P0)
                    }
                };
            }
            (Phase::Guess(p), Action::Guess(total)) => {
                let [Some(x0), Some(x1)] = self.beads else {
                    unreachable!("both players pick beads before the announcements")
                };
                if total == x0 + x1 {
                    next.phase = Phase::Terminal;
                    next.winner = Some(p);
                } else {
                    next.history.push(Miss { player: p, total });
                    next.phase = match p {
                        Player::P0 => Phase::Guess(Player::P1),
                        Player::P1 if self.turn == self.horizon => Phase::Terminal,
                        Player::P1 => {
                            next.turn += 1;
                            Phase::Guess(Player::P0)
                        }
                    };
                }
            }
            _ => unreachable!("legal_actions matches the action kind to the phase"),
        }
        Ok(next)
    }

    /// What `player` observes in this state.
    #[must_use]
    pub(crate) fn information_state(&self, player: Player) -> InformationState {
        InformationState {
            player,
            phase: self.phase,
            horizon: self.horizon,
            turn: self.turn,
            own_beads: self.beads[player as usize],
            history: self.history.clone(),
        }
    }

    /// The payoffs (P0, P1) at the end of the game: 1 and -1 for a hit, 0 for a draw. `None`
    /// while the game goes on.
    #[must_use]
    pub fn returns(&self) -> Option<[i32; 2]> {
        match (self.phase, self.winner) {
            (Phase::Terminal, None) => Some([0, 0]),
            (Phase::Terminal, Some(Player::P0)) => Some([1, -1]),
            (Phase::Terminal, Some(Player::P1)) => Some([-1, 1]),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Action, GameState, Miss, Phase, Player};
    use crate::Error;
    use std::collections::HashSet;

    fn play(horizon: usize, actions: &[Action]) -> GameState {
        let mut state = GameState::new(horizon).expect("positive horizon");
        for &action in actions {
            state = state.apply(action).expect("legal action");
        }
        state
    }

    fn after_beads(x0: u8, x1: u8, horizon: usize) -> GameState {
        play(horizon, &[Action::Choose(x0), Action::Choose(x1)])
    }

    fn guesses(state: &GameState) -> Vec<Action> {
        state.legal_actions().collect()
    }

    #[test]
    fn initial_state() {
        let state = GameState::new(3).expect("positive horizon");
        assert_eq!(state.phase(), Phase::Choose(Player::P0));
        assert_eq!(
            (state.horizon, state.turn, state.beads()),
            (3, 0, [None, None])
        );
        assert!(state.history().is_empty() && state.returns().is_none());
        assert!(matches!(GameState::new(0), Err(Error::Horizon)));
    }

    #[test]
    fn actor_follows_phase() {
        let mut state = GameState::new(1).expect("positive horizon");
        for (action, actor) in [
            (Action::Choose(4), Some(Player::P1)),
            (Action::Choose(7), Some(Player::P0)),
            (Action::Guess(5), Some(Player::P1)),
            (Action::Guess(8), None),
        ] {
            state = state.apply(action).expect("legal action");
            assert_eq!(state.actor(), actor);
        }
    }

    #[test]
    fn legal_actions_by_phase() {
        let picks: Vec<_> = (1..=10).map(Action::Choose).collect();
        let state = GameState::new(1).expect("positive horizon");
        assert_eq!(guesses(&state), picks);
        assert_eq!(guesses(&play(1, &[Action::Choose(4)])), picks);
        let state = after_beads(4, 7, 1);
        assert_eq!(
            guesses(&state),
            (5..15).map(Action::Guess).collect::<Vec<_>>()
        );
        let state = state.apply(Action::Guess(5)).expect("legal action");
        assert_eq!(
            guesses(&state),
            (8..18).map(Action::Guess).collect::<Vec<_>>()
        );
        assert!(
            guesses(&play(
                1,
                &[Action::Choose(1), Action::Choose(1), Action::Guess(2)]
            ))
            .is_empty()
        );
    }

    #[test]
    fn p1_does_not_observe_p0_beads_while_picking() {
        let seen: HashSet<_> = (1..=10)
            .map(|x0| play(1, &[Action::Choose(x0)]).information_state(Player::P1))
            .collect();
        assert_eq!(seen.len(), 1);
        let info = seen.into_iter().next().expect("one observation");
        assert_eq!(
            (info.phase, info.own_beads),
            (Phase::Choose(Player::P1), None)
        );
    }

    #[test]
    fn announcer_observes_own_beads_and_public_history() {
        let state = after_beads(4, 7, 2)
            .apply(Action::Guess(5))
            .expect("legal action");
        let p0 = state.information_state(Player::P0);
        let p1 = state.information_state(Player::P1);
        assert_eq!((p0.own_beads, p1.own_beads), (Some(4), Some(7)));
        assert_eq!(
            p0.history,
            [Miss {
                player: Player::P0,
                total: 5
            }]
        );
        assert_eq!(p1.history, p0.history);
        let other = after_beads(4, 8, 2)
            .apply(Action::Guess(5))
            .expect("legal action");
        assert_eq!(p0, other.information_state(Player::P0));
        assert_eq!(
            after_beads(4, 7, 1).information_state(Player::P1),
            after_beads(5, 7, 1).information_state(Player::P1)
        );
    }

    #[test]
    fn bead_picks_are_hidden_then_announcements_start() {
        let after_p0 = play(2, &[Action::Choose(9)]);
        assert_eq!(
            (after_p0.phase(), after_p0.beads()),
            (Phase::Choose(Player::P1), [Some(9), None])
        );
        let after_p1 = after_p0.apply(Action::Choose(8)).expect("legal action");
        assert_eq!(after_p1.phase(), Phase::Guess(Player::P0));
        assert_eq!((after_p1.turn, after_p1.beads()), (1, [Some(9), Some(8)]));
    }

    #[test]
    fn misses_pass_the_turn() {
        let state = after_beads(9, 9, 2)
            .apply(Action::Guess(11))
            .expect("legal action");
        assert_eq!((state.phase(), state.turn), (Phase::Guess(Player::P1), 1));
        assert_eq!(
            state.history(),
            [Miss {
                player: Player::P0,
                total: 11
            }]
        );
        let state = state.apply(Action::Guess(10)).expect("legal action");
        assert_eq!((state.phase(), state.turn), (Phase::Guess(Player::P0), 2));
        assert_eq!(
            state.history()[1],
            Miss {
                player: Player::P1,
                total: 10
            }
        );
    }

    #[test]
    fn hits_end_the_game() {
        let p0_win = after_beads(4, 7, 1)
            .apply(Action::Guess(11))
            .expect("legal action");
        assert_eq!(
            (p0_win.phase(), p0_win.returns()),
            (Phase::Terminal, Some([1, -1]))
        );
        assert!(p0_win.history().is_empty());
        let missed = after_beads(4, 7, 1)
            .apply(Action::Guess(5))
            .expect("legal action");
        let p1_win = missed.apply(Action::Guess(11)).expect("legal action");
        assert_eq!(p1_win.returns(), Some([-1, 1]));
        assert_eq!(
            p1_win.history(),
            [Miss {
                player: Player::P0,
                total: 5
            }]
        );
        assert_eq!(
            missed
                .apply(Action::Guess(8))
                .expect("legal action")
                .returns(),
            Some([0, 0])
        );
    }

    #[test]
    fn missed_totals_stay_legal() {
        let state = after_beads(9, 9, 2)
            .apply(Action::Guess(10))
            .expect("legal action");
        assert!(state.legal_actions().any(|a| a == Action::Guess(10)));
        let state = state.apply(Action::Guess(10)).expect("legal action");
        assert!(state.legal_actions().any(|a| a == Action::Guess(10)));
    }

    #[test]
    fn three_turns_of_misses_draw() {
        let line = [11, 10, 12, 14, 13, 15];
        let mut state = after_beads(9, 9, 3);
        for total in line {
            state = state.apply(Action::Guess(total)).expect("legal action");
        }
        assert_eq!(
            (state.phase(), state.returns(), state.turn),
            (Phase::Terminal, Some([0, 0]), 3)
        );
        let players = [Player::P0, Player::P1].into_iter().cycle();
        let expected: Vec<_> = players
            .zip(line)
            .map(|(player, total)| Miss { player, total })
            .collect();
        assert_eq!(state.history(), expected);
    }

    #[test]
    fn every_bead_pair_and_announcement() {
        for x0 in 1..=10 {
            for x1 in 1..=10 {
                let p0_state = after_beads(x0, x1, 1);
                let p0_actions = guesses(&p0_state);
                assert_eq!(p0_actions.len(), 10);
                for &action in &p0_actions {
                    let next = p0_state.apply(action).expect("legal action");
                    if action == Action::Guess(x0 + x1) {
                        assert_eq!(next.returns(), Some([1, -1]));
                    } else {
                        assert_eq!(next.phase(), Phase::Guess(Player::P1));
                    }
                }
                let miss = p0_actions
                    .into_iter()
                    .find(|&a| a != Action::Guess(x0 + x1));
                let p1_state = p0_state
                    .apply(miss.expect("a miss exists"))
                    .expect("legal action");
                for action in p1_state.legal_actions() {
                    let hit = action == Action::Guess(x0 + x1);
                    let returns = p1_state.apply(action).expect("legal action").returns();
                    assert_eq!(returns, Some(if hit { [-1, 1] } else { [0, 0] }));
                }
            }
        }
    }

    #[test]
    fn illegal_actions_are_rejected() {
        let start = GameState::new(1).expect("positive horizon");
        for action in [Action::Choose(0), Action::Choose(11), Action::Guess(2)] {
            assert!(matches!(start.apply(action), Err(Error::IllegalAction(_))));
        }
        let state = after_beads(4, 7, 1);
        for action in [Action::Guess(4), Action::Guess(15), Action::Choose(5)] {
            assert!(matches!(state.apply(action), Err(Error::IllegalAction(_))));
        }
        let terminal = after_beads(1, 1, 1)
            .apply(Action::Guess(2))
            .expect("legal action");
        assert!(terminal.apply(Action::Guess(2)).is_err());
    }
}
