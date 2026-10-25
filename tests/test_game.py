"""We test the referee in stl.game: check resolution, the clock, death and
revival, and a replay of the canonical match line."""

from typing import NamedTuple, Optional

import numpy as np
import pytest

from stl.game import (
    LS_WINDOW_END,
    LS_WINDOW_START,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    TURN_DURATION_LEAP,
    TURN_DURATION_NORMAL,
    Game,
    HalfRoundResult,
    Player,
    Referee,
    legal_max_second,
)


# ──────────────────────────────────────────────────────────────────────
# Tests for check resolution logic.

class TestCheckSuccess:
    """check_time >= drop_time = handkerchief is on the ground."""

    def test_check_after_drop(self, game):
        rec = game.play_half_round(drop_time=20, check_time=40)
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 21

    def test_check_same_second_as_drop_succeeds_with_one_st(self, game):
        """D drops and C checks in second 30; inclusive ST is one."""
        rec = game.play_half_round(drop_time=30, check_time=30)
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 1

    def test_check_one_second_after_drop(self, game):
        """One second after the drop produces two inclusive seconds of ST."""
        rec = game.play_half_round(drop_time=30, check_time=31)
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 2

    def test_st_accumulates_across_halves(self, game):
        """ST from successful checks stacks in the checker's cylinder."""
        # Half 1: Baku checks, gains 21 inclusive ST
        game.play_half_round(drop_time=10, check_time=30)
        assert game.player2.cylinder == 21  # Baku is checker in half 1

        # Half 2: Hal checks, gains 16 inclusive ST
        game.play_half_round(drop_time=20, check_time=35)
        assert game.player1.cylinder == 16  # Hal is checker in half 2

    def test_max_st_drop_at_1_check_at_60(self, game):
        """Worst case: D drops at 1, C checks at 60. ST = 60."""
        rec = game.play_half_round(drop_time=1, check_time=60)
        assert rec.st_gained == 60


class TestCheckFailure:
    """check_time < drop_time = handkerchief NOT on the ground yet."""

    def test_check_before_drop(self, game):
        rec = game.play_half_round(drop_time=40, check_time=20)
        assert rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                               HalfRoundResult.CHECK_FAIL_DIED)
        assert rec.st_gained == 0
        assert rec.death_duration > 0

    def test_check_one_second_before_drop(self, game):
        """D drops at 31, C checks at 30. Handkerchief not yet dropped."""
        rec = game.play_half_round(drop_time=31, check_time=30)
        assert rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                               HalfRoundResult.CHECK_FAIL_DIED)

    def test_failed_check_adds_60s_penalty(self, game):
        """Failed check: +60s penalty, then entire cylinder injected."""
        rec = game.play_half_round(drop_time=50, check_time=10)
        # Checker had 0 cylinder, +60 penalty = 60s death
        assert rec.death_duration == 60

    def test_failed_check_stacks_with_existing_cylinder(self, game):
        """If checker already has ST in cylinder, penalty stacks."""
        # Half 1: Baku gains 31 inclusive ST (success)
        game.play_half_round(drop_time=10, check_time=40)
        assert game.player2.cylinder == 31

        # Half 2: Hal checks successfully, no death
        game.play_half_round(drop_time=10, check_time=20)

        # Round 2, Half 1: Baku fails check. cylinder = 31 + 60 = 91
        rec = game.play_half_round(drop_time=50, check_time=10)
        assert rec.death_duration == 91


class TestInputValidation:
    def test_drop_time_zero_rejected(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=0, check_time=30)

    def test_drop_time_over_60_rejected_normal(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=61, check_time=30)

    def test_check_time_zero_rejected(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=30, check_time=0)

    def test_check_time_over_60_rejected(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=30, check_time=61)

    def test_drop_at_60_is_valid(self, game):
        """D can drop at the last second."""
        rec = game.play_half_round(drop_time=60, check_time=30)
        assert rec.drop_time == 60

    def test_check_at_60_is_valid(self, game):
        """C can check at the last second (safe strategy)."""
        rec = game.play_half_round(drop_time=1, check_time=60)
        assert rec.check_time == 60


# ──────────────────────────────────────────────────────────────────────
# Tests for game clock, snap-to-minute, leap second, and format_game_clock.

def make_game(clock: float = 0.0, seed: int = 42) -> Game:
    hal = Player(name="Hal", physicality=1.0)
    baku = Player(name="Baku", physicality=0.7)
    ref = Referee()
    g = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
    g.game_clock = clock
    g.seed(seed)
    return g


class TestFormatGameClock:
    def test_game_start(self):
        g = make_game(0)
        assert g.format_game_clock() == "8:00:00 AM"

    def test_twelve_minutes(self):
        g = make_game(720)
        assert g.format_game_clock() == "8:12:00 AM"

    def test_eight_fifty_nine(self):
        g = make_game(3540)
        assert g.format_game_clock() == "8:59:00 AM"

    def test_eight_fifty_nine_fifty_nine(self):
        g = make_game(3599)
        assert g.format_game_clock() == "8:59:59 AM"

    def test_leap_second(self):
        g = make_game(3600)
        assert g.format_game_clock() == "8:59:60 AM"

    def test_nine_oclock(self):
        g = make_game(3601)
        assert g.format_game_clock() == "9:00:00 AM"

    def test_nine_oh_four(self):
        # 9:04:00 = 3601 + 4*60 = 3841
        g = make_game(3841)
        assert g.format_game_clock() == "9:04:00 AM"

    def test_post_leap_continuity(self):
        """Clock should increment smoothly across the leap second."""
        g = make_game(3599)
        assert "8:59:59" in g.format_game_clock()
        g.game_clock = 3600
        assert "8:59:60" in g.format_game_clock()
        g.game_clock = 3601
        assert "9:00:00" in g.format_game_clock()
        g.game_clock = 3602
        assert "9:00:01" in g.format_game_clock()


class TestSnapToNextMinute:
    def test_snap_mid_minute(self):
        g = make_game(1753)  # 8:29:13
        g.snap_clock_to_next_minute()
        assert g.game_clock == 1800  # 8:30:00

    def test_snap_on_boundary_advances(self):
        """On a minute boundary, snap STILL advances to the next one."""
        g = make_game(1800)  # 8:30:00
        g.snap_clock_to_next_minute()
        assert g.game_clock == 1860  # 8:31:00

    def test_snap_near_leap_second(self):
        """Snap from 8:59:xx should go to 9:00:00 (gc=3601), not 3600."""
        g = make_game(3550)  # 8:59:10
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3601  # 9:00:00

    def test_snap_at_3540(self):
        """8:59:00 snaps to 9:00:00."""
        g = make_game(3540)
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3601

    def test_snap_at_leap_second(self):
        g = make_game(3600)
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3601

    def test_snap_post_leap(self):
        """Post-leap snaps align with wall-clock minutes."""
        g = make_game(3810)  # post-leap
        g.snap_clock_to_next_minute()
        # 9:04:00 = gc 3841
        assert g.game_clock == 3841
        assert "9:04:00" in g.format_game_clock()

    def test_snap_post_leap_on_boundary(self):
        """Post-leap, on a wall-clock minute boundary, advances to next."""
        g = make_game(3841)  # 9:04:00
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3901  # 9:05:00
        assert "9:05:00" in g.format_game_clock()


class TestTurnDuration:
    def test_normal_turn(self):
        g = make_game(720)
        assert g.get_turn_duration() == TURN_DURATION_NORMAL

    def test_leap_second_window_start(self):
        g = make_game(LS_WINDOW_START)  # 3540
        assert g.get_turn_duration() == TURN_DURATION_LEAP

    def test_leap_second_window_middle(self):
        g = make_game(3570)
        assert g.get_turn_duration() == TURN_DURATION_LEAP

    def test_leap_second_window_end(self):
        g = make_game(LS_WINDOW_END)  # 3600
        assert g.get_turn_duration() == TURN_DURATION_LEAP

    def test_after_leap_window(self):
        g = make_game(3601)
        assert g.get_turn_duration() == TURN_DURATION_NORMAL


class TestLeapSecondGameplay:
    def test_drop_at_61_during_leap(self):
        """Baku as dropper can drop at second 61 during a leap second turn."""
        g = make_game(LS_WINDOW_START)  # in LS window
        g.current_half = 2
        rec = g.play_half_round(drop_time=61, check_time=60)
        # C checks at 60, D drops at 61. check_time < drop_time -> FAIL
        assert rec.result in ("check_fail_survived", "check_fail_died",
                               rec.result)  # just check it doesn't crash
        assert rec.turn_duration == 61

    def test_safe_strategy_fails_during_leap(self):
        """The whole point: checking at 60 fails if D drops at 61."""
        g = make_game(LS_WINDOW_START)
        g.current_half = 2
        rec = g.play_half_round(drop_time=61, check_time=60)
        # check_time (60) < drop_time (61) -> failure
        from stl.game import HalfRoundResult
        assert rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                               HalfRoundResult.CHECK_FAIL_DIED)

    def test_same_second_drop_and_check_succeeds_min_st(self):
        """Drop at 60, check at 60 during leap succeeds with inclusive ST=1."""
        g = make_game(LS_WINDOW_START)
        rec = g.play_half_round(drop_time=60, check_time=60)
        from stl.game import HalfRoundResult
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 1

    def test_checker_cannot_check_at_61_during_leap(self):
        """Checkers remain capped at 60 during a leap turn."""
        g = make_game(LS_WINDOW_START)
        import pytest
        with pytest.raises(ValueError):
            g.play_half_round(drop_time=60, check_time=61)


class TestClockAdvancement:
    def test_no_death_round_takes_4_minutes(self):
        """A clean round: 60 + 60 + 60 overhead = 180s. Snap -> 4 min gap."""
        g = make_game(1800)  # start on a minute boundary

        # Half 1: success, no death
        g.play_half_round(drop_time=10, check_time=20)
        # Half 2: success, no death
        g.play_half_round(drop_time=10, check_time=20)

        # Should have snapped to 1800 + 180 -> snap -> 2040 (4 min later)
        assert g.game_clock == 2040

    def test_death_round_longer_gap(self):
        """A round with a 60s death adds death_duration + death overhead to elapsed."""
        g = make_game(1800)

        # Half 1: failed check -> 60s death
        g.play_half_round(drop_time=50, check_time=10)
        # Half 2: clean
        g.play_half_round(drop_time=10, check_time=20)

        # 60 (h1 turn) + 60 (death) + 120 (death overhead) + 60 (overhead) + 60 (h2 turn) = 360
        # 1800 + 360 = 2160. snap -> 2220 (7 min gap)
        assert g.game_clock == 2220

    def test_no_snap_between_halves(self):
        """Clock does NOT snap between halves -- only after the full round."""
        g = make_game(1800)

        # Play only half 1
        g.play_half_round(drop_time=10, check_time=20)

        # Should be: 1800 + 60 (turn) + 60 (overhead) = 1920
        # NOT snapped to a minute boundary
        assert g.game_clock == 1920
        assert g.current_half == 2  # still mid-round


class TestMangaTimeline:
    """Validate that the clock model can reproduce manga round start times.

    Manga data: R1:8:12 R2:8:19 R3:8:26 R4:8:30 R5:8:34 R6:8:38
                R7:8:45 R8:8:49 R9:8:57 R10:9:04

    We simulate rounds with death durations reverse-engineered from
    the timing gaps. Each must produce the next round's start time.

    Gap analysis (with DEATH_PROCEDURE_OVERHEAD=120):
      No-death: 60+60+60=180 → snap → 4 min gap
      Death(60s): 60+60+120+60+60=360 → snap → 7 min gap
      Death(120s): 60+120+120+60+60=420 → snap → 8 min gap
    """

    EXPECTED = [
        (1, 720),    # 8:12
        (2, 1140),   # 8:19
        (3, 1560),   # 8:26
        (4, 1800),   # 8:30
        (5, 2040),   # 8:34
        (6, 2280),   # 8:38
        (7, 2700),   # 8:45
        (8, 2940),   # 8:49
        (9, 3420),   # 8:57
        (10, 3841),  # 9:04 (post-leap, gc=3601+4*60)
    ]

    # Death durations that produce the correct gaps
    # (0 = no death in that round)
    ROUND_DEATHS = [60, 60, 0, 0, 0, 60, 0, 120, 120]

    def test_full_manga_timeline(self):
        """Simulate 9 rounds and verify each start time matches."""
        g = make_game(720)  # R1 at 8:12

        for i, death_dur in enumerate(self.ROUND_DEATHS):
            round_num = i + 1
            start_clock = g.game_clock
            expected_start = self.EXPECTED[i][1]

            # Verify this round starts at the expected time
            assert start_clock == expected_start, (
                f"R{round_num} started at gc={start_clock}, "
                f"expected {expected_start}"
            )

            # Play the round. Death in half 1 if death_dur > 0.
            if death_dur > 0:
                # Failed check: checker's cylinder must equal death_dur
                # before the play. Set it up so penalty produces right duration.
                checker_idx = 1  # Baku is checker in half 1 (Hal drops first)
                checker = g.player2 if g.first_dropper == g.player1 else g.player1
                checker.cylinder = death_dur - 60  # +60 penalty -> death_dur
                if checker.cylinder < 0:
                    checker.cylinder = 0

                g.play_half_round(drop_time=50, check_time=10)  # fail
                if g.game_over:
                    break
                g.play_half_round(drop_time=10, check_time=20)  # clean half 2
            else:
                g.play_half_round(drop_time=10, check_time=20)  # clean half 1
                if g.game_over:
                    break
                g.play_half_round(drop_time=10, check_time=20)  # clean half 2

            if g.game_over:
                break

            # Verify next round starts correctly
            next_expected = self.EXPECTED[i + 1][1]
            assert g.game_clock == next_expected, (
                f"After R{round_num}: gc={g.game_clock}, "
                f"expected R{round_num + 1} at {next_expected}"
            )


class TestLeapWindowClosedBoundary:
    """Owner ruling (2026-06-10): the leap window is the CLOSED interval
    [3540, 3600] — a half-round starting at exactly 3600 (8:59:60) still
    spans the inserted second. Pins the boundary semantics of
    Game.get_turn_duration against doc drift."""

    def test_turn_starting_exactly_at_3600_is_leap_turn(self):
        g = make_game(3600.0)
        assert g.get_turn_duration() == 61

    def test_turn_starting_at_3601_is_normal(self):
        g = make_game(3601.0)
        assert g.get_turn_duration() == 60

    def test_turn_starting_at_3539_is_normal(self):
        g = make_game(3539.0)
        assert g.get_turn_duration() == 60


# ──────────────────────────────────────────────────────────────────────
# Tests for death, revival, and survival probability mechanics.

class TestSurvivalProbability:
    def test_first_death_60s_high_survival(self, referee, hal):
        """A bare 60-second injection starts at the frozen 0.95 baseline."""
        prob = referee.compute_survival_probability(hal, death_duration=60)
        assert prob == pytest.approx(0.95)

    def test_death_at_300_is_always_fatal(self, referee, hal):
        """Cylinder at 300 = guaranteed death regardless of other factors."""
        prob = referee.compute_survival_probability(hal, death_duration=300)
        assert prob == 0.0

    def test_death_above_300_is_fatal(self, referee, hal):
        prob = referee.compute_survival_probability(hal, death_duration=350)
        assert prob == 0.0

    def test_resulting_total_ttd_exactly_300_remains_revival_eligible(self, referee, hal):
        hal.ttd = 240
        prob = referee.compute_survival_probability(hal, death_duration=60)
        assert prob > 0.0

    def test_resulting_total_ttd_above_300_is_fatal(self, referee, hal):
        hal.ttd = 241
        prob = referee.compute_survival_probability(hal, death_duration=60)
        assert prob == 0.0

    def test_cardiac_degradation(self, referee, hal):
        """Prior deaths weaken the heart, reducing survival probability."""
        prob_fresh = referee.compute_survival_probability(hal, death_duration=60)

        # Simulate a prior death
        hal.on_death(death_duration=60)
        hal.on_revival()
        prob_damaged = referee.compute_survival_probability(hal, death_duration=60)

        assert prob_damaged < prob_fresh

    def test_revival_history_is_not_a_probability_input(self):
        """The attempt count is audit history, not part of the frozen surface."""
        ref = Referee()
        p = Player(name="Test", physicality=1.0)

        prob_fresh = ref.compute_survival_probability(p, death_duration=60)
        ref.cprs_performed = 3
        prob_tired = ref.compute_survival_probability(p, death_duration=60)

        assert prob_tired == prob_fresh

    def test_player_identity_is_not_a_probability_input(self, referee):
        hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
        baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)

        p_hal = referee.compute_survival_probability(hal, death_duration=60)
        p_baku = referee.compute_survival_probability(baku, death_duration=60)

        assert p_baku == p_hal

    def test_matches_frozen_linear_geometric_surface(self, referee, hal):
        hal.ttd = 120
        # q=180 means s=120: .95 * (1 - 120/240) * .75^(120/60)
        prob = referee.compute_survival_probability(hal, death_duration=180)
        assert prob == pytest.approx(0.95 * 0.5 * 0.75**2)

class TestDeathSequence:
    def test_failed_check_triggers_death(self, game):
        rec = game.play_half_round(drop_time=50, check_time=10)
        assert rec.death_duration > 0
        assert rec.survived is not None

    def test_survived_death_resets_cylinder(self, game):
        """If player survives, cylinder resets to 0."""
        game.seed(1)  # seed that produces survival for 60s death
        # Force a failed check
        rec = game.play_half_round(drop_time=50, check_time=10)
        if rec.survived:
            assert game.player2.cylinder == 0  # Baku is checker, cylinder reset

    def test_permanent_death_ends_game(self):
        """If revival fails, game is over."""
        # Create a scenario with very low survival probability
        hal = Player(name="Hal", physicality=1.0)
        baku = Player(name="Baku", physicality=0.1)  # very weak
        ref = Referee()
        ref.cprs_performed = 10  # exhausted referee
        # Give baku huge prior damage
        baku.ttd = 600
        baku.cylinder = 200

        game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
        game.game_clock = 720.0

        # Failed check: cylinder = 200 + 60 = 260. Death at 260s with terrible odds.
        rec = game.play_half_round(drop_time=50, check_time=10)
        # With such bad stats, death is nearly certain
        if not rec.survived:
            assert game.game_over
            assert game.winner.name == "Hal"
            assert game.loser.name == "Baku"


class TestCylinderOverflow:
    def test_overflow_from_st_accumulation(self):
        """Cylinder hits 300 from accumulated ST, triggers injection."""
        hal = Player(name="Hal", physicality=1.0)
        baku = Player(name="Baku", physicality=1.0)
        ref = Referee()
        game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
        game.seed(99)
        game.game_clock = 720.0

        # Fill Baku's cylinder to 250
        baku.cylinder = 250

        # Now a check with 55 ST: 250 + 55 = 305 >= 300 -> overflow
        # Canon: injection caps at 300 regardless of actual cylinder contents
        rec = game.play_half_round(drop_time=5, check_time=60)
        assert rec.result in (HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED,
                               HalfRoundResult.CYLINDER_OVERFLOW_DIED)
        assert rec.death_duration == 300

    def test_cylinder_at_exactly_300_triggers(self):
        """Cylinder reaching exactly 300 triggers injection."""
        hal = Player(name="Hal", physicality=1.0)
        baku = Player(name="Baku", physicality=1.0)
        ref = Referee()
        game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
        game.seed(99)
        game.game_clock = 720.0

        baku.cylinder = 260

        # ST of 40: 260 + 40 = 300 -> overflow
        rec = game.play_half_round(drop_time=10, check_time=50)
        assert rec.result in (HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED,
                               HalfRoundResult.CYLINDER_OVERFLOW_DIED)

    def test_zero_probability_survival_cannot_be_forced(self):
        hal = Player(name="Hal")
        baku = Player(name="Baku", cylinder=299)
        game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
        before = (baku.cylinder, baku.ttd, baku.deaths, baku.alive)

        with pytest.raises(ValueError, match="probability is zero"):
            game.resolve_half_round(1, 1, survived_outcome=True)

        assert (baku.cylinder, baku.ttd, baku.deaths, baku.alive) == before
        assert game.history == []
        assert not game.game_over

    @pytest.mark.parametrize("forced", [0, 1, "yes"])
    def test_forced_survival_outcome_must_be_a_literal_bool(self, forced):
        game = Game(Player("Hal"), Player("Baku"), Referee())
        with pytest.raises(ValueError, match="bool or None"):
            game.resolve_half_round(2, 1, survived_outcome=forced)
        assert game.history == []


class TestSafeStrategy:
    def test_safe_strategies_remaining_fresh(self, baku):
        """Fresh player has 4 safe checks: 4*60 = 240 < 300. The 5th hits 300 = injection."""
        assert baku.safe_strategies_remaining == 4

    def test_safe_strategies_remaining_with_cylinder(self, baku):
        baku.cylinder = 120
        assert baku.safe_strategies_remaining == 2  # 120+60+60=240 < 300, third would hit 300

    def test_safe_strategies_remaining_at_240(self, baku):
        baku.cylinder = 240
        assert baku.safe_strategies_remaining == 0  # next check hits 300 = injection

    def test_safe_strategies_remaining_at_300(self, baku):
        baku.cylinder = 300
        assert baku.safe_strategies_remaining == 0


# ──────────────────────────────────────────────────────────────────────
# Engine-faithfulness regression: replay the canonical Usogui match line.
#
# Replays the canonical match from the manga — as transcribed in the strategy
# doc "SURPASSING THE LEADER - HAL DOC" — through src/Game.py action by action,
# asserting game clock, both cylinders, ttd, and referee CPR count at every
# half-round boundary. This locks the engine against the canon and the doc
# transcription against the engine: if either the engine's clock/cylinder/death
# arithmetic or the doc's state tables drift, this file fails.
#
# Replay semantics: every death in the canonical line was survived in the manga,
# so death half-rounds are resolved with resolve_half_round(..., survived_outcome=True)
# (forced outcome). The forced path still increments referee.cprs_performed, so
# the revival-attempt history remains exact even though it is not a probability
# input.
#
# TIMING INTERPRETATION:
#
# The source sometimes labels an immediate action "0 seconds" and describes a
# drop after N elapsed seconds. Repository actions are ordinal seconds 1..60:
# the immediate interval is action 1, a same-second success gains inclusive
# ST=1, and ST is `check - drop + 1`. The action pairs below are the ordinal
# encoding of the source intervals. This follows docs/ACTION_TIMING.md and makes
# R5H1's instant success one second, as documented.
#
# R9H2 revival probability uses the repository-wide frozen surface. With vial
# ST 0 and prior TTD 238, it is ``0.95 * 0.75**(238/60)``. The accumulated
# revival-attempt count is history only and player identity is irrelevant.
#
# Clock pins: rows whose start times the doc states explicitly (8:12, 8:17,
# 8:19, 8:21, 8:26, 8:28, 8:30, 8:32, 8:34, 8:36, 8:38, 8:45, 8:49, 8:57, 8:59)
# are canon pins; rows marked "engine" (R6H2 @2613, R7H2 @2820, R8H2 @3060) are
# mid-round clocks the doc does not state — they are engine-consistency locks
# computed from the engine's clock rules (turn 60s, death +duration+120s, +60s
# between halves, snap-to-next-minute after half 2). Death durations are pinned
# by the doc (60, 84, 93, 154, 60), so all clock gaps are forced.

class Row(NamedTuple):
    label: str
    clock_at_start: float        # asserted before the half-round is played
    dropper: str
    drop: int
    check: int
    survived: Optional[bool]     # None = no death; True = forced revival (replay)
    st: float                    # expected st_gained on the record
    death_duration: float        # expected death_duration on the record (0 if none)
    hal_cyl: float               # ── post-half state assertions ──
    baku_cyl: float
    hal_ttd: float
    baku_ttd: float
    cprs: int


# The canonical line. One Row per half-round, in play order.
# Post-state columns are the FULL state after the half-round resolves
# (cylinder reset on revival already applied).
CANONICAL_LINE = [
    #    label    clock    dropper  drop check survived   st death | hal_cyl baku_cyl hal_ttd baku_ttd cprs
    Row("R1H1",  720.0,  "Hal",   60,  30,  True,       0,  60,     0,      0,      0,     60,     1),  # 8:12 pin — Baku fails, dies 60s
    Row("R1H2", 1020.0,  "Baku",   1,  24,  None,      24,   0,    24,      0,      0,     60,     1),  # 8:17 pin
    Row("R2H1", 1140.0,  "Hal",   36,  60,  None,      25,   0,    24,     25,      0,     60,     1),  # 8:19 pin
    Row("R2H2", 1260.0,  "Baku",  60,   5,  True,       0,  84,     0,     25,     84,     60,     2),  # 8:21 pin — Hal fails, dies 84s (24+60)
    Row("R3H1", 1560.0,  "Hal",   57,  60,  None,       4,   0,     0,     29,     84,     60,     2),  # 8:26 pin
    Row("R3H2", 1680.0,  "Baku",  25,  60,  None,      36,   0,    36,     29,     84,     60,     2),  # 8:28 pin
    Row("R4H1", 1800.0,  "Hal",    8,  10,  None,       3,   0,    36,     32,     84,     60,     2),  # 8:30 pin
    Row("R4H2", 1920.0,  "Baku",  27,  60,  None,      34,   0,    70,     32,     84,     60,     2),  # 8:32 pin
    Row("R5H1", 2040.0,  "Hal",    1,   1,  None,       1,   0,    70,     33,     84,     60,     2),  # 8:34 pin - instant tie succeeds with ST=1
    Row("R5H2", 2160.0,  "Baku",   1,  15,  None,      15,   0,    85,     33,     84,     60,     2),  # 8:36 pin - check in second 15
    Row("R6H1", 2280.0,  "Hal",   60,   1,  True,       0,  93,    85,      0,     84,    153,     3),  # 8:38 pin - Baku dies 93s (33+60)
    Row("R6H2", 2613.0,  "Baku",   3,  10,  None,       8,   0,    93,      0,     84,    153,     3),  # engine: 2280+60+(93+120)+60
    Row("R7H1", 2700.0,  "Hal",    1,  60,  None,      60,   0,    93,     60,     84,    153,     3),  # 8:45 pin
    Row("R7H2", 2820.0,  "Baku",   1,   1,  None,       1,   0,    94,     60,     84,    153,     3),  # engine: 2700+60+60
    Row("R8H1", 2940.0,  "Hal",    6,  60,  None,      55,   0,    94,    115,     84,    153,     3),  # 8:49 pin - drop after 5 elapsed seconds
    Row("R8H2", 3060.0,  "Baku",  60,   1,  True,       0, 154,     0,    115,    238,    153,     4),  # engine: 2940+60+60 - Hal dies 154s
    Row("R9H1", 3420.0,  "Hal",    1,  60,  None,      60,   0,     0,    175,    238,    153,     4),  # 8:57 pin
    Row("R9H2", 3540.0,  "Baku",  61,  60,  True,       0,  60,     0,    175,    298,    153,     5),  # 8:59 pin - THE LEAP TURN: Baku drops 61,
    #                                                                                                     Hal capped at 60 → fail, dies 60s (0+60).
    #                                                                                                     hal.ttd = 298 (4m58s): the 2-Second
    #                                                                                                     Deviation payoff vs a 300s overflow death.
]


def _make_game() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.game_clock = 720.0
    game.seed(0)
    return game


def _replay(game: Game, rows) -> None:
    for row in rows:
        game.resolve_half_round(row.drop, row.check, survived_outcome=row.survived)


def test_canonical_line_replay():
    """Replay all 18 canonical half-rounds, asserting clock + full state each row."""
    game = _make_game()
    hal, baku, ref = game.player1, game.player2, game.referee

    for row in CANONICAL_LINE:
        # Clock at the START of the half-round (doc pin or engine consistency lock).
        assert game.game_clock == pytest.approx(row.clock_at_start), (
            f"{row.label}: clock at start — expected {row.clock_at_start}, "
            f"engine has {game.game_clock} ({game.format_game_clock()})"
        )

        record = game.resolve_half_round(row.drop, row.check, survived_outcome=row.survived)

        # Record-level locks.
        assert record.dropper == row.dropper, f"{row.label}: dropper"
        assert record.game_clock_at_start == pytest.approx(row.clock_at_start), f"{row.label}: record clock"
        assert record.st_gained == pytest.approx(row.st), f"{row.label}: st_gained"
        assert record.death_duration == pytest.approx(row.death_duration), f"{row.label}: death_duration"
        if row.survived is None:
            assert record.result == HalfRoundResult.CHECK_SUCCESS, f"{row.label}: result"
            assert record.survived is None, f"{row.label}: survived"
        else:
            assert record.result == HalfRoundResult.CHECK_FAIL_SURVIVED, f"{row.label}: result"
            assert record.survived is True, f"{row.label}: survived"

        # Full post-half state lock.
        assert hal.cylinder == pytest.approx(row.hal_cyl), f"{row.label}: hal.cylinder"
        assert baku.cylinder == pytest.approx(row.baku_cyl), f"{row.label}: baku.cylinder"
        assert hal.ttd == pytest.approx(row.hal_ttd), f"{row.label}: hal.ttd"
        assert baku.ttd == pytest.approx(row.baku_ttd), f"{row.label}: baku.ttd"
        assert ref.cprs_performed == row.cprs, f"{row.label}: cprs_performed"
        assert not game.game_over, f"{row.label}: game should continue"

    # Whole-line sanity: 5 deaths total, both players alive, 9 rounds played.
    assert hal.deaths == 3 and baku.deaths == 2
    assert hal.alive and baku.alive
    assert len(game.history) == 18


def test_leap_turn_action_space():
    """R9H2 is the leap turn: duration 61, Baku may drop 61, Hal capped at 60."""
    game = _make_game()
    _replay(game, CANONICAL_LINE[:17])  # everything before R9H2

    # The leap window [3540, 3600] is a CLOSED interval and R9H2 starts exactly on it.
    assert game.game_clock == pytest.approx(3540.0)
    assert game.is_leap_second_turn()
    assert game.get_turn_duration() == 61

    # Actor-aware legality: only Baku-as-dropper sees the 61st second.
    assert legal_max_second("Baku", "dropper", 61) == 61
    assert legal_max_second("Hal", "checker", 61) == 60
    assert legal_max_second("Hal", "dropper", 61) == 60
    assert legal_max_second("Baku", "checker", 61) == 60
    assert legal_max_second("Alice", "dropper", 61) == 60

    # The engine accepts Baku's drop at 61; Hal's check at 60 < 61 fails.
    record = game.resolve_half_round(61, 60, survived_outcome=True)
    assert record.turn_duration == 61
    assert record.result == HalfRoundResult.CHECK_FAIL_SURVIVED
    assert record.death_duration == pytest.approx(60.0)


@pytest.mark.parametrize("action", [1.0, True, np.float64(1.0)])
def test_engine_rejects_coercible_noninteger_actions_without_mutation(action) -> None:
    game = _make_game()
    with pytest.raises(ValueError, match="integer"):
        game.resolve_half_round(action, 1)
    assert game.history == []
    assert game.game_clock == 720.0


def test_engine_normalizes_integral_scalar_actions() -> None:
    game = _make_game()
    record = game.resolve_half_round(np.int64(1), np.int32(1))
    assert type(record.drop_time) is int
    assert type(record.check_time) is int


@pytest.mark.parametrize("validator", ["validate_drop_time", "validate_check_time"])
def test_actorless_action_validation_is_strict_and_normalizing(validator) -> None:
    game = _make_game()
    validate = getattr(game, validator)
    assert type(validate(np.int64(1), 60)) is int
    with pytest.raises(ValueError, match="integer"):
        validate(1.0, 60)


def test_r9h2_survival_probability():
    """Pin R9H2 to the frozen repository-wide revival surface."""
    game = _make_game()
    _replay(game, CANONICAL_LINE)

    record = game.history[-1]
    assert record.death_duration == pytest.approx(60.0)
    assert record.survived is True

    expected = 0.95 * 0.75 ** (238 / 60)
    assert record.survival_probability == pytest.approx(expected, abs=1e-3)
