//! Card arithmetic, the exhaustive evaluator check, and the engine's rules and information
//! boundary (ports of `test_poker_math.py` and `test_game.py`).

use ap::cards::{FULL_DECK, evaluate, legal_hands, parse_card, rank_sum};
use ap::engine::{Bet, GameState, HandPair, Move, Phase, Team, UpperPolicy, uniform_upper_policy};
use ap::{TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS};

fn hand(names: &[&str]) -> u64 {
    let mask = names.iter().fold(0u64, |mask, name| {
        mask | 1 << parse_card(name).expect("valid card")
    });
    assert_eq!(mask.count_ones() as usize, names.len(), "distinct cards");
    mask
}

fn score(names: &[&str]) -> u32 {
    evaluate(hand(names)).expect("five cards")
}

#[test]
fn every_card_parses_to_its_index_and_bad_text_is_rejected() {
    for (suit, s) in "cdhs".chars().enumerate() {
        for (rank, r) in "23456789TJQKA".chars().enumerate() {
            assert_eq!(
                parse_card(&format!("{r}{s}")),
                Ok((suit * 13 + rank) as u32)
            );
        }
    }
    assert_eq!(
        [("2c", 0), ("Ac", 12), ("2d", 13), ("Th", 34), ("As", 51)].map(|(t, _)| parse_card(t)),
        [0, 12, 13, 34, 51].map(Ok)
    );
    assert_eq!(parse_card("aC"), parse_card("Ac"));
    for bad in ["10h", "A", "Ace", "1c", "Tx", ""] {
        assert!(parse_card(bad).is_err(), "{bad}");
    }
}

#[test]
fn air_poker_values_run_from_ace_one_to_king_thirteen_whatever_the_suit() {
    for (rank, value) in (2..=13).chain([1]).enumerate() {
        assert_eq!(
            rank_sum((0..4).fold(0, |m, suit| m | 1 << (suit * 13 + rank))),
            4 * value
        );
    }
    assert_eq!(rank_sum(hand(&["Ac", "2c", "3c", "4c", "5c"])), 15);
    assert_eq!(rank_sum(hand(&["As", "2d", "3h", "4s", "5d"])), 15);
}

#[test]
fn legal_hands_span_six_to_sixty_four_and_match_brute_force() {
    assert_eq!(legal_hands(6, FULL_DECK).len(), 4);
    assert_eq!(legal_hands(64, FULL_DECK).len(), 4);
    assert!(legal_hands(5, FULL_DECK).is_empty() && legal_hands(65, FULL_DECK).is_empty());
    let deck = hand(&["Ac", "2c", "3c", "4c", "5c", "6d", "7d", "8h", "Qs", "Ks"]);
    let cards: Vec<u32> = ap::cards::cards(deck).collect();
    for target in 6..=64 {
        let mut expected = Vec::new();
        for a in 0..10 {
            for b in a + 1..10 {
                for c in b + 1..10 {
                    for d in c + 1..10 {
                        for e in d + 1..10 {
                            let mask = [a, b, c, d, e].iter().fold(0u64, |m, &i| m | 1 << cards[i]);
                            if rank_sum(mask) == target {
                                expected.push(mask);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(legal_hands(target, deck), expected, "{target}");
    }
    let only = hand(&["Ac", "2d", "3h", "4s", "5c"]);
    assert_eq!(legal_hands(15, only), vec![only]);
    assert!(legal_hands(15, only & !(1 << 12)).is_empty());
}

#[test]
fn categories_and_tie_breakers_follow_poker_order() {
    let ladder = [
        ["Ac", "Kd", "Qh", "9s", "3c"],
        ["Ac", "As", "Kd", "Qh", "Jh"],
        ["Ac", "As", "Kd", "Kc", "Jh"],
        ["Ac", "As", "Ad", "Kd", "Qh"],
        ["Ac", "Kc", "Qd", "Jh", "Td"],
        ["Ac", "Kc", "Qc", "9c", "2c"],
        ["Ac", "Ad", "Ah", "Kc", "Kd"],
        ["Ac", "Ad", "Ah", "As", "Kd"],
        ["Ac", "Kc", "Qc", "Jc", "Tc"],
    ]
    .map(|h| score(&h));
    assert!(ladder.is_sorted());
    assert_eq!(ladder.map(|s| s >> 24), std::array::from_fn(|i| i as u32));
    let stronger = [
        (
            ["Ac", "Kc", "Qc", "Jc", "Tc"],
            ["Kc", "Qc", "Jc", "Tc", "9c"],
        ),
        (
            ["Ac", "Ad", "Ah", "As", "Kd"],
            ["Kc", "Kd", "Kh", "Ks", "Ad"],
        ),
        (
            ["Ac", "Ad", "Ah", "Kc", "Kd"],
            ["Kc", "Kd", "Kh", "As", "Ad"],
        ),
        (
            ["Ac", "Ad", "Ah", "Kc", "Qd"],
            ["As", "Ah", "Ac", "Kd", "Jc"],
        ),
        (
            ["Ac", "Ad", "Kc", "Kd", "Qh"],
            ["As", "Ah", "Qc", "Qd", "Kh"],
        ),
        (
            ["Ac", "Ad", "Kc", "Qd", "Jh"],
            ["As", "Ah", "Kd", "Qc", "Th"],
        ),
        (
            ["Ac", "Qc", "9c", "6c", "3c"],
            ["Ad", "Qd", "9d", "5d", "4d"],
        ),
        (
            ["Ac", "Qd", "9h", "6s", "3c"],
            ["As", "Qh", "9d", "5s", "4c"],
        ),
        (
            ["6h", "5d", "4c", "3s", "2d"],
            ["5h", "4d", "3c", "2s", "As"],
        ),
    ];
    for (a, b) in stronger {
        assert!(score(&a) > score(&b), "{a:?} beats {b:?}");
    }
    assert_eq!(
        score(&["Ac", "Qh", "9d", "6s", "3c"]),
        score(&["As", "Qd", "9h", "6c", "3s"])
    );
    for size in [0, 1, 4, 6, 52] {
        assert!(evaluate((1u64 << size) - 1).is_err(), "{size}");
    }
    assert!(evaluate(1 << 52 | 0b1111).is_err());
}

#[test]
fn all_2_598_960_hands_give_the_known_category_counts_and_7_462_strengths() {
    let mut counts = [0u32; 9];
    let mut strengths = std::collections::HashSet::new();
    for a in 0..52 {
        for b in a + 1..52 {
            for c in b + 1..52 {
                for d in c + 1..52 {
                    for e in d + 1..52 {
                        let s = evaluate(1 << a | 1 << b | 1 << c | 1 << d | 1u64 << e)
                            .expect("five cards");
                        counts[(s >> 24) as usize] += 1;
                        strengths.insert(s);
                    }
                }
            }
        }
    }
    assert_eq!(
        counts,
        [
            1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40
        ]
    );
    assert_eq!(strengths.len(), 7_462);
}

const HAND_36: [&str; 5] = ["8c", "8d", "8h", "8s", "4d"];
const HAND_36_OVERLAP: [&str; 5] = ["8c", "8d", "8h", "8s", "4c"];
const HAND_15: [&str; 5] = ["Ac", "2c", "3c", "4c", "5c"];
const HAND_25: [&str; 5] = ["3h", "4h", "5h", "6h", "7h"];

fn play(state: &GameState, moves: &[Move]) -> GameState {
    moves
        .iter()
        .fold(state.clone(), |s, &mv| s.apply(mv).expect("legal move"))
}

fn upper_chance() -> GameState {
    play(
        &GameState::canonical(),
        &[Move::Number(36), Move::Number(15)],
    )
}

fn betting(hands: HandPair) -> GameState {
    upper_chance()
        .apply(Move::Hands(hands))
        .expect("legal hands")
}

const CHECK_CHECK: [Move; 2] = [Move::Bet(Bet::Check), Move::Bet(Bet::Check)];

fn sealed(state: &GameState) -> [u32; 2] {
    state.lower.each_ref().map(|p| p.sealed_bios)
}

#[test]
fn the_root_has_23_sealed_bios_full_air_the_canonical_cards_and_the_full_deck() {
    let state = GameState::canonical();
    assert_eq!(sealed(&state), [23, 23]);
    assert_eq!(state.lower.each_ref().map(|p| p.air_seconds), [300, 300]);
    assert_eq!(state.lower[0].metal_cards, TEAM_0_METAL_CARDS);
    assert_eq!(state.lower[1].metal_cards, TEAM_1_METAL_CARDS);
    assert_eq!(
        (
            state.deck,
            state.round.chosen,
            state.history.len(),
            state.winner
        ),
        (FULL_DECK, [None, None], 0, None)
    );
    for (team_0, team_1, air) in [
        ([25, 25, 36, 39, 45], TEAM_1_METAL_CARDS, [300, 300]),
        ([5, 25, 36, 39, 45], TEAM_1_METAL_CARDS, [300, 300]),
        (TEAM_0_METAL_CARDS, [25, 15, 44, 47, 63], [300, 300]),
        (TEAM_0_METAL_CARDS, TEAM_1_METAL_CARDS, [301, 300]),
    ] {
        assert!(GameState::new([team_0, team_1], air).is_err());
    }
}

#[test]
fn numbers_stay_private_until_both_are_chosen() {
    let chosen = GameState::canonical()
        .apply(Move::Number(36))
        .expect("legal");
    assert_eq!(chosen.information(Team::Zero).visible, [Some(36), None]);
    assert_eq!(chosen.information(Team::One).visible, [None, None]);
    assert_eq!(chosen.actor(), Some(Team::One));
    assert!(chosen.apply(Move::Number(36)).is_err());
    assert_eq!(chosen.lower[0].metal_cards.len(), 4);
    let state = upper_chance();
    for team in [Team::Zero, Team::One] {
        assert_eq!(state.information(team).visible, [Some(36), Some(15)]);
    }
    assert_eq!((state.actor(), state.legal_moves()), (None, vec![]));
    let info = betting([Some(hand(&HAND_36)), Some(hand(&HAND_15))]).information(Team::Zero);
    assert_eq!(
        (info.opponent_metal_card_count, info.visible),
        (4, [Some(36), Some(15)])
    );
}

#[test]
fn hands_are_checked_against_the_pre_round_deck_and_remove_their_union() {
    let state = upper_chance();
    assert!(
        state
            .apply(Move::Hands([Some(hand(&HAND_15)), Some(hand(&HAND_15))]))
            .is_err()
    );
    assert!(
        state
            .apply(Move::Hands([
                Some(hand(&HAND_36) & !hand(&["4d"])),
                Some(hand(&HAND_15))
            ]))
            .is_err()
    );
    let mut depleted = state.clone();
    depleted.deck &= !1;
    assert!(
        depleted
            .apply(Move::Hands([Some(hand(&HAND_36)), Some(hand(&HAND_15))]))
            .is_err()
    );
    let union = hand(&HAND_36_OVERLAP) | hand(&HAND_15);
    assert_eq!(
        betting([Some(hand(&HAND_36_OVERLAP)), Some(hand(&HAND_15))]).deck,
        FULL_DECK & !union
    );
    let no_rank = betting([Some(hand(&HAND_36)), None]);
    assert_eq!(no_rank.deck, FULL_DECK & !hand(&HAND_36));
    let settled = play(&no_rank, &CHECK_CHECK);
    assert_eq!(
        (settled.history[0].winner, settled.history[0].calamity),
        (Some(Team::Zero), false)
    );
}

#[test]
fn upper_policies_see_only_their_own_target_until_a_retry() {
    let only = hand(&HAND_15);
    assert_eq!(
        uniform_upper_policy(only, 15, None),
        vec![(Some(only), 1.0)]
    );
    assert_eq!(uniform_upper_policy(only, 64, None), vec![(None, 1.0)]);
    let seen = std::cell::RefCell::new(Vec::new());
    let p0 = |deck: u64, target: u8, opposing: Option<u8>| {
        assert_eq!(deck, FULL_DECK);
        seen.borrow_mut().push((target, opposing));
        vec![(Some(hand(&HAND_36)), 1.0)]
    };
    let p1 = |_: u64, target: u8, opposing: Option<u8>| {
        seen.borrow_mut().push((target, opposing));
        vec![(Some(hand(&HAND_15)), 1.0)]
    };
    let outcomes = upper_chance()
        .chance_outcomes([&p0, &p1])
        .expect("valid policies");
    assert_eq!(
        outcomes,
        vec![([Some(hand(&HAND_36)), Some(hand(&HAND_15))], 1.0)]
    );
    assert_eq!(seen.take(), vec![(36, None), (15, None)]);

    let retry = upper_chance()
        .apply(Move::Hands([None, None]))
        .expect("legal");
    assert_eq!(
        (retry.phase, retry.round.retry, retry.deck),
        (Phase::UpperChance, true, FULL_DECK)
    );
    let q0 = |_: u64, target: u8, opposing: Option<u8>| {
        seen.borrow_mut().push((target, opposing));
        vec![(None, 0.25), (Some(hand(&HAND_36)), 0.75)]
    };
    let q1 = |_: u64, target: u8, opposing: Option<u8>| {
        seen.borrow_mut().push((target, opposing));
        vec![(None, 0.40), (Some(hand(&HAND_15)), 0.60)]
    };
    let outcomes = retry.chance_outcomes([&q0, &q1]).expect("valid policies");
    assert_eq!(seen.take(), vec![(36, Some(15)), (15, Some(36))]);
    assert_eq!(outcomes.len(), 3);
    assert!(outcomes.iter().all(|(h, _)| *h != [None, None]));
    assert!((outcomes.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn a_retry_without_resolving_mass_ends_with_no_judgement() {
    let retry = play(
        &GameState::canonical(),
        &[Move::Number(25), Move::Number(8), Move::Hands([None, None])],
    );
    let impossible = |_: u64, _: u8, _: Option<u8>| vec![(None, 1.0)];
    assert_eq!(
        retry.chance_outcomes([&impossible, &impossible]),
        Ok(vec![([None, None], 1.0)])
    );
    let settled = retry.apply(Move::Hands([None, None])).expect("legal");
    assert_eq!(
        (settled.phase, settled.deck),
        (Phase::NumberChoice0, FULL_DECK)
    );
    let record = &settled.history[0];
    assert_eq!(
        (
            record.winner,
            record.calamity,
            record.actions.len(),
            record.chosen
        ),
        (None, false, 0, [25, 8])
    );
    assert_eq!(sealed(&settled), [22, 22]);
    assert_eq!(settled.lower[0].metal_cards, [26, 36, 39, 45]);
    assert_eq!(settled.lower[1].metal_cards, [15, 44, 47, 63]);

    let mut state = GameState::canonical();
    for (a, b) in TEAM_0_METAL_CARDS.into_iter().zip(TEAM_1_METAL_CARDS) {
        state = play(
            &state,
            &[
                Move::Number(a),
                Move::Number(b),
                Move::Hands([None, None]),
                Move::Hands([None, None]),
            ],
        );
    }
    assert_eq!(
        (
            state.phase,
            state.winner,
            state.returns(),
            state.history.len()
        ),
        (Phase::Terminal, None, Ok([0.0, 0.0]), 5)
    );
}

#[test]
fn invalid_upper_policies_are_rejected() {
    let state = upper_chance();
    let good = |_: u64, _: u8, _: Option<u8>| vec![(Some(hand(&HAND_15)), 1.0)];
    let bad: [&UpperPolicy<'_>; 5] = [
        &|_, _, _| vec![],
        &|_, _, _| vec![(Some(hand(&HAND_36)), 0.5)],
        &|_, _, _| vec![(Some(hand(&HAND_36)), 0.0), (None, 1.0)],
        &|_, _, _| vec![(Some(hand(&HAND_36)), 0.5), (Some(hand(&HAND_36)), 0.5)],
        &|_, _, _| vec![(Some(hand(&HAND_15)), 1.0)],
    ];
    for policy in bad {
        assert!(state.chance_outcomes([policy, &good]).is_err());
    }
}

#[test]
fn betting_follows_the_first_bettor_parity_and_half_pot_rules() {
    let state = betting([Some(hand(&HAND_36)), Some(hand(&HAND_15))]);
    assert_eq!(
        (state.round.betting.as_ref().map(|b| b.first), state.actor()),
        (Some(Team::One), Some(Team::One))
    );
    assert_eq!(
        state.legal_moves(),
        vec![Move::Bet(Bet::Check), Move::Bet(Bet::Raise(1))]
    );
    let raised = state.apply(Move::Bet(Bet::Raise(1))).expect("legal");
    let moves = raised.legal_moves();
    assert_eq!(moves[..2], [Move::Bet(Bet::Call), Move::Bet(Bet::Fold)]);
    assert_eq!(moves.last(), Some(&Move::Bet(Bet::Raise(2))));
    assert!(!moves.contains(&Move::Bet(Bet::Check)));
    for bet in [Bet::Call, Bet::Fold, Bet::Raise(0), Bet::Raise(2)] {
        assert!(state.apply(Move::Bet(bet)).is_err(), "{bet:?}");
    }
}

#[test]
fn the_canonical_raise_sequences_reach_pots_of_sixteen_and_thirty() {
    let total = |s: &GameState| {
        sealed(s).iter().sum::<u32>()
            + s.round
                .betting
                .as_ref()
                .map_or(0, |b| b.contributions.iter().sum())
    };
    let mut state = betting([Some(hand(&HAND_36)), Some(hand(&HAND_15))]);
    assert_eq!(total(&state), 48);
    for extra in [1, 2, 4] {
        state = state.apply(Move::Bet(Bet::Raise(extra))).expect("legal");
        assert_eq!(total(&state), 48);
    }
    assert_eq!(
        state.round.betting.as_ref().map(|b| b.contributions),
        Some([4, 8])
    );
    state = state.apply(Move::Bet(Bet::Call)).expect("legal");
    assert_eq!(sealed(&state), [14, 30]);
    assert_eq!(
        state.history[0]
            .actions
            .iter()
            .map(|(_, b)| *b)
            .collect::<Vec<_>>(),
        [Bet::Raise(1), Bet::Raise(2), Bet::Raise(4), Bet::Call]
    );
    state = play(
        &state,
        &[
            Move::Number(25),
            Move::Number(8),
            Move::Hands([Some(hand(&HAND_25)), None]),
        ],
    );
    assert_eq!(
        state.round.betting.as_ref().map(|b| b.first),
        Some(Team::Zero)
    );
    for extra in [2, 1, 5, 1, 4] {
        state = state.apply(Move::Bet(Bet::Raise(extra))).expect("legal");
    }
    assert_eq!(
        state.round.betting.as_ref().map(|b| b.contributions),
        Some([15, 11])
    );
    state = state.apply(Move::Bet(Bet::Call)).expect("legal");
    assert_eq!(
        (state.history.len(), state.history[1].actions.len()),
        (2, 6)
    );
}

#[test]
fn calamity_destroys_the_losers_contribution_and_folds_skip_it() {
    let overlap = [Some(hand(&HAND_36_OVERLAP)), Some(hand(&HAND_15))];
    let state = play(&betting(overlap), &CHECK_CHECK);
    assert_eq!(
        (state.history[0].winner, state.history[0].calamity),
        (Some(Team::One), true)
    );
    assert_eq!(sealed(&state), [20, 23]);
    let start = betting(overlap);
    let folded = play(&start, &[Move::Bet(Bet::Raise(1)), Move::Bet(Bet::Fold)]);
    assert!(!folded.history[0].calamity);
    assert_eq!(folded.deck, start.deck);
    assert_eq!(sealed(&folded).iter().sum::<u32>(), 44);
    let mut broke = betting(overlap);
    broke.lower[0].sealed_bios = 0;
    let state = play(&broke, &CHECK_CHECK);
    assert_eq!(
        (state.phase, state.winner, state.returns()),
        (Phase::Terminal, Some(Team::One), Ok([-1.0, 1.0]))
    );
    let mut short = betting([Some(hand(&HAND_36)), Some(hand(&HAND_15))]);
    short.lower[0].sealed_bios = 1;
    let state = play(&short, &CHECK_CHECK);
    assert_eq!(
        (state.phase, state.winner),
        (Phase::Terminal, Some(Team::One))
    );
}

#[test]
fn five_rounds_with_rising_antes_end_in_the_air_comparison() {
    let mut state = GameState::canonical();
    let mut before = Vec::new();
    for round in 0..5 {
        before.push(sealed(&state)[1]);
        let (a, b) = (state.lower[0].metal_cards[0], state.lower[1].metal_cards[0]);
        state = play(&state, &[Move::Number(a), Move::Number(b)]);
        let first = legal_hands(a.into(), state.deck)[0];
        state = play(
            &state,
            &[
                Move::Hands([Some(first), None]),
                CHECK_CHECK[0],
                CHECK_CHECK[1],
            ],
        );
        if round == 3 {
            // Team 0 takes the ten-BIOS pot next; Team 1's one residual second decides.
            let mut rigged = play(&state, &[Move::Number(45), Move::Number(63)]);
            let first = legal_hands(45, rigged.deck)[0];
            rigged = rigged
                .apply(Move::Hands([Some(first), None]))
                .expect("legal");
            rigged.lower[0].sealed_bios = 0;
            rigged.lower[0].air_seconds = 0;
            rigged.lower[1].sealed_bios = 10;
            rigged.lower[1].air_seconds = 1;
            let end = play(&rigged, &CHECK_CHECK);
            assert_eq!(
                (end.phase, end.winner, end.returns()),
                (Phase::Terminal, Some(Team::One), Ok([-1.0, 1.0]))
            );
        }
    }
    assert_eq!(before, [23, 21, 18, 14, 9]);
    assert_eq!(
        (
            state.phase,
            state.history.len(),
            state.returns(),
            state.actor()
        ),
        (Phase::Terminal, 5, Ok([1.0, -1.0]), None)
    );
    assert!(state.lower.iter().all(|p| p.metal_cards.is_empty()) && state.legal_moves().is_empty());
    assert!(state.apply(Move::Number(25)).is_err());
    assert!(GameState::canonical().returns().is_err());
}
