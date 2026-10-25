"""Exact Hal, the Perfect Hal base model, translated Hal and its golden trace.

The golden trace section at the end pins translated Hal's decisions.

A scripted Baku plays 8 games of 50 half-rounds against
TranslatedHal on the StageTable fake of tests/conftest.py. Three games hold a
leap pair (turn 61), so the trace covers the DTH fallback and the sequence
break. Hal's sample comes from a fixed golden-ratio sequence, so the trace uses
no random generator. One run keeps one provider. The other run builds a fresh
provider per game and restores the dumped opponent memory, as the hosted app
does. Both runs must match hal_trace.json, which play() wrote at 938fb25.
Rewrite that file only for an intended change to Hal's behavior.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stl.dth import Stage
from stl.hal import (
    ACTION_COUNT,
    CanonicalDecision,
    ExactHal,
    PerfectHalConfig,
    PerfectHalOpponentModel,
    PerfectHalPolicyProvider,
    PublicGameOutcome,
    TranslatedHal,
    TranslatedHalConfig,
    TranslatedHalOpponentModel,
    project_to_dth_state,
)
from tests.conftest import (
    StageTable as _StageTable,
    make_decision as _decision,
    make_forecast as _forecast,
    make_reveal as _reveal,
)


def test_config_fixes_pure_dth_and_hard_response_defaults() -> None:
    config = PerfectHalConfig()
    assert config.action_count == 60
    assert config.response_temperature == 0.0
    assert config.expert_names[:5] == (
        "uniform",
        "global",
        "slow_recency",
        "fast_recency",
        "flash_recency",
    )

    with pytest.raises(ValueError, match="60"):
        PerfectHalConfig(action_count=61)
    with pytest.raises(ValueError, match="slow to flash"):
        PerfectHalConfig(recency_retentions=(0.3, 0.8, 0.9))
    with pytest.raises(ValueError, match="nonnegative"):
        PerfectHalConfig(response_temperature=-0.1)


def test_online_model_learns_a_deterministic_role_without_cross_role_leakage() -> None:
    model = PerfectHalOpponentModel()
    initial_checker = _forecast(model, "checker")
    initial_dropper = _forecast(model, "dropper")
    assert initial_checker.policy == pytest.approx(
        np.full(ACTION_COUNT, 1.0 / ACTION_COUNT)
    )
    assert initial_dropper.policy == pytest.approx(initial_checker.policy)

    for index in range(8):
        prediction = _forecast(model, "checker", decision_index=index)
        model.observe(prediction, opponent_action=7, self_action=3)

    learned = _forecast(model, "checker", decision_index=8)
    untouched = _forecast(model, "dropper", decision_index=8)
    assert learned.policy[6] > 0.90
    assert learned.confidence > 0.70
    assert untouched.policy == pytest.approx(initial_dropper.policy)
    assert model.observations("checker") == 8
    assert model.observations("dropper") == 0


def test_online_model_retargets_after_a_public_change_point() -> None:
    model = PerfectHalOpponentModel()
    decision_index = 0
    for action, repeats in ((7, 20), (49, 8)):
        for _ in range(repeats):
            prediction = _forecast(
                model,
                "checker",
                decision_index=decision_index,
            )
            model.observe(prediction, opponent_action=action, self_action=11)
            decision_index += 1

    switched = _forecast(model, "checker", decision_index=decision_index)
    weights = dict(zip(switched.expert_names, switched.expert_weights, strict=True))
    assert switched.policy[48] > switched.policy[6]
    assert weights["flash_recency"] > weights["global"]


def test_self_response_expert_uses_only_the_previous_revealed_self_action() -> None:
    model = PerfectHalOpponentModel()
    previous_self = 10
    for index in range(20):
        prediction = _forecast(model, "dropper", decision_index=index)
        current_self = 20 if previous_self == 10 else 10
        model.observe(
            prediction,
            opponent_action=previous_self,
            self_action=current_self,
        )
        previous_self = current_self

    prediction = _forecast(model, "dropper", decision_index=20)
    response_index = prediction.expert_names.index("self_response")
    expected_action = previous_self
    assert prediction.expert_policies[response_index, expected_action - 1] > 0.95
    assert prediction.policy[expected_action - 1] > 0.70


def test_forecast_tokens_are_single_use_and_causally_ordered() -> None:
    model = PerfectHalOpponentModel()
    prediction = _forecast(model, "checker")
    model.observe(prediction, opponent_action=5, self_action=9)
    with pytest.raises(RuntimeError, match="stale or already observed"):
        model.observe(prediction, opponent_action=5, self_action=9)


def test_noninteger_actions_fail_without_consuming_the_forecast() -> None:
    model = PerfectHalOpponentModel()
    prediction = _forecast(model, "checker")
    with pytest.raises(ValueError, match="integer second"):
        model.observe(prediction, opponent_action=True, self_action=9)
    with pytest.raises(ValueError, match="integer second"):
        model.observe(prediction, opponent_action=5, self_action=9.0)

    model.observe(prediction, opponent_action=5, self_action=9)
    assert model.observations("checker") == 1


def test_provider_places_all_mass_on_the_forecast_best_response() -> None:
    table = _StageTable()
    provider = PerfectHalPolicyProvider(table)

    first = provider.policy(_decision(role="dropper"))
    assert first == pytest.approx(
        {action: 1.0 / ACTION_COUNT for action in range(1, ACTION_COUNT + 1)}
    )
    provider.observe(_reveal(opponent_action=7, self_action=3))

    second = provider.policy(_decision(role="dropper"))
    assert second == {7: pytest.approx(1.0)}
    assert provider.last_decision is not None
    assert provider.last_decision.opponent_policy[6] > 0.5
    assert provider.last_decision.expected_exploit_gain > 0.0
    assert provider.last_decision.best_response_actions == (7,)
    assert table.states == [(12, 60, 24, 120), (12, 60, 24, 120)]


def test_provider_preserves_session_memory_across_games_and_seat_names() -> None:
    provider = PerfectHalPolicyProvider(_StageTable())
    provider.reset_game()
    provider.policy(_decision(role="dropper", actor_name="Hal"))
    provider.observe(
        _reveal(
            self_role="dropper",
            self_name="Hal",
            opponent_name="Baku",
            opponent_action=13,
            game_over=True,
        )
    )
    provider.end_game(PublicGameOutcome(0, "Hal", 1))
    provider.reset_game()

    next_game = provider.policy(_decision(role="dropper", actor_name="Baku"))
    assert next_game == {13: pytest.approx(1.0)}
    assert provider.has_session_memory
    provider.observe(
        _reveal(
            self_role="dropper",
            self_name="Baku",
            opponent_name="Hal",
            opponent_action=13,
            game_index=1,
        )
    )

    provider.reset_session()
    assert not provider.has_session_memory
    assert provider.last_decision is None


def test_provider_fails_closed_before_stage_lookup_outside_pure_dth() -> None:
    table = _StageTable()
    provider = PerfectHalPolicyProvider(table)
    with pytest.raises(ValueError, match="pure DTH only"):
        provider.policy(_decision(legal_seconds=tuple(range(1, ACTION_COUNT + 2))))
    assert table.states == []


def test_provider_requires_one_public_reveal_per_decision() -> None:
    provider = PerfectHalPolicyProvider(_StageTable())
    provider.policy(_decision())
    with pytest.raises(RuntimeError, match="twice before a reveal"):
        provider.policy(_decision())
    with pytest.raises(RuntimeError, match="unrevealed action"):
        provider.reset_game()


def test_diagnostics_are_json_serializable_and_explicitly_unrestricted() -> None:
    provider = PerfectHalPolicyProvider(_StageTable())
    provider.policy(_decision())
    diagnostics = provider.experiment_diagnostics()
    assert diagnostics["pure_dth_only"] is True
    assert diagnostics["public_history_only"] is True
    assert diagnostics["unrestricted_best_response"] is True
    assert diagnostics["equilibrium_safety_blend"] is False
    assert "Perfect Hal" in provider.match_summary()
    json.dumps(diagnostics)


@pytest.mark.parametrize("role", ["dropper", "checker"])
def test_offsets_transfer_to_an_unseen_hal_action(role):
    model = TranslatedHalOpponentModel()
    last = 10
    for own in (13, 22, 8, 37, 15, 28, 42, 19, 31, 47, 25, 51):
        forecast = _forecast(model, role)
        model.observe(forecast, opponent_action=last + 2, self_action=own)
        last = own
    forecast = _forecast(model, role)
    index = forecast.expert_names.index("offset_last_copy_global")
    assert forecast.expert_policies[index, 52] > 0.95
    assert forecast.policy[52] > 0.8
    assert model.observations("checker" if role == "dropper" else "dropper") == 0


def test_offsets_use_the_last_reveal_across_roles_and_track_a_switch():
    model = TranslatedHalOpponentModel(TranslatedHalConfig(offset_retention=0.7))
    rng = np.random.default_rng(917)
    last = 20
    for index in range(90):
        role = "dropper" if index % 2 else "checker"
        forecast = _forecast(model, role)
        action = last + (3 if index < 40 else -3)
        own = int(rng.integers(5, 55))
        model.observe(forecast, opponent_action=action, self_action=own)
        last = own
    forecast = _forecast(model, "checker")
    recent = forecast.expert_names.index("offset_last_copy_recent")
    assert forecast.expert_policies[recent, last - 4] > 0.95
    assert forecast.policy[last - 4] > 0.8


def test_no_offset_ablation_matches_old_bit_for_bit():
    old = PerfectHalOpponentModel()
    candidate = TranslatedHalOpponentModel(TranslatedHalConfig(translated=False))
    rng = np.random.default_rng(81)
    for index in range(30):
        role = "dropper" if index % 2 else "checker"
        before = [_forecast(model, role) for model in (old, candidate)]
        np.testing.assert_array_equal(before[0].policy, before[1].policy)
        own, action = rng.integers(1, 61, 2)
        for model, forecast in zip((old, candidate), before):
            model.observe(forecast, opponent_action=int(action), self_action=int(own))


def test_reset_and_excluded_turn_clear_translation_references():
    model = TranslatedHalOpponentModel()
    forecast = _forecast(model, "dropper")
    model.observe(forecast, opponent_action=12, self_action=10)
    with pytest.raises(RuntimeError, match="stale"):
        model.observe(forecast, opponent_action=13, self_action=11)
    model.break_sequence()
    forecast = _forecast(model, "dropper")
    assert forecast.context.last_self_action is None
    assert forecast.context.previous_self_action is None
    assert model.total_observations == 1
    model.reset()
    assert model.total_observations == 0
    np.testing.assert_allclose(_forecast(model, "dropper").policy, np.ones(60) / 60)


def test_offset_projection_clips_and_preserves_probability_mass():
    model = TranslatedHalOpponentModel()
    for own in (10, 20, 30, 40, 60):
        before = _forecast(model, "dropper")
        last = before.context.last_self_action
        model.observe(before, opponent_action=20 if last is None else last + 4, self_action=own)
    forecast = _forecast(model, "dropper")
    index = forecast.expert_names.index("offset_last_copy_global")
    assert forecast.expert_policies[index, 59] > 0.95
    np.testing.assert_allclose(forecast.expert_policies.sum(axis=1), 1)


def _literal_decision(*, legal: tuple[int, ...] = (1, 2, 3)) -> CanonicalDecision:
    return CanonicalDecision(
        role="dropper",
        actor_name="Hal",
        turn_duration=60,
        legal_seconds=legal,
        checker_cylinder_seconds=19,
        checker_ttd_seconds=120,
        dropper_cylinder_seconds=29,
        dropper_ttd_seconds=70,
        native_state=object(),
    )


def test_dth_projection_is_the_exact_literal_second_identity() -> None:
    assert project_to_dth_state(_literal_decision()) == (19, 120, 29, 70)
    wide = CanonicalDecision(
        role="checker",
        actor_name="Hal",
        turn_duration=60,
        legal_seconds=tuple(range(1, 61)),
        checker_cylinder_seconds=305.0,
        checker_ttd_seconds=301.0,
        dropper_cylinder_seconds=-2.0,
        dropper_ttd_seconds=299.6,
        native_state=object(),
    )
    with pytest.raises(ValueError, match="literal-second"):
        project_to_dth_state(wide)


def test_dth_provider_serves_only_complete_exact_policies() -> None:
    class _Table:
        def stage(self, state):
            return Stage(
                state=state,
                value=0.1,
                matrix=np.zeros((60, 60)),
                drop=np.array((1.0,) + (0.0,) * 59),
                check=np.array((0.0,) * 59 + (1.0,)),
                gap=1e-7,
            )

    provider = ExactHal(_Table())
    assert provider.policy(_literal_decision()) == {1: 1.0}
    assert "1 exact moves" in provider.match_summary()


def provider():
    return TranslatedHal(_StageTable())


def test_checkpoint_preserves_forecasts_without_sharing_mutable_evidence():
    first = provider()
    rng = np.random.default_rng(2291)
    for index in range(100):
        role = "checker" if index % 2 else "dropper"
        model = first.opponent_model
        model.observe(_forecast(model, role), opponent_action=int(rng.integers(1, 61)), self_action=int(rng.integers(1, 61)))
    snapshot = first.dump_state()
    restored = provider()
    restored.restore_state(snapshot)
    for role in ("dropper", "checker"):
        np.testing.assert_array_equal(_forecast(first.opponent_model, role).policy, _forecast(restored.opponent_model, role).policy)
    restored.opponent_model.observe(_forecast(restored.opponent_model, "dropper"), opponent_action=12, self_action=3)
    assert first.opponent_model.total_observations == 100
    assert restored.opponent_model.total_observations == 101
    assert first.dump_state() == snapshot


@pytest.mark.parametrize("damage", ["schema", "config", "nan", "shape", "action", "compressed"])
def test_checkpoint_rejects_corruption_without_replacing_the_model(damage):
    first = provider()
    encoded = first.dump_state()
    data = json.loads(zlib.decompress(base64.b64decode(encoded)))
    if damage == "schema":
        data["schema"] = "unknown"
    elif damage == "config":
        data["config"]["offset_retention"] = 0.01
    elif damage == "nan":
        data["roles"]["checker"]["global_counts"][0] = float("nan")
    elif damage == "shape":
        data["offsets"]["dropper"] = []
    elif damage == "action":
        data["last_self"] = 61
    encoded = base64.b64encode(zlib.compress(json.dumps(data).encode())).decode()
    if damage == "compressed":
        encoded = encoded[:12]
    original = first.opponent_model
    with pytest.raises(ValueError):
        first.restore_state(encoded)
    assert first.opponent_model is original


def test_leap_fallback_keeps_evidence_and_clears_sequence_references():
    p = provider()
    p.policy(_decision())
    p.observe(_reveal())
    leap = replace(_decision(role="checker"), turn_duration=61)
    assert set(p.policy(leap)) == set(range(1, 61))
    reveal = _reveal(self_role="checker", opponent_action=61, half_round_index=1)
    reveal = replace(reveal, pre_decision_state=replace(reveal.pre_decision_state, turn_duration=61))
    p.observe(reveal)
    assert p.opponent_model.total_observations == 1
    assert p.opponent_model._last_self is None
    p.policy(_decision())
    p.observe(_reveal(half_round_index=2))
    assert p.opponent_model.total_observations == 2


# The golden trace of translated Hal.

TRACE = Path(__file__).with_name("hal_trace.json")
GAMES, HALF_ROUNDS = 8, 50
LEAP_GAMES = (2, 5, 7)
LEAP_HALF_ROUNDS = (24, 25)


def _matrix():
    """The dropper's payoff: 1 for a failed check, the gap over 60 for a caught drop.

    No action dominates, so each best reply follows the forecast: the dropper
    drops one second after the predicted check, and the checker checks at the
    predicted drop.
    """
    drop = np.arange(1, 61)[:, None]
    check = np.arange(1, 61)[None, :]
    return np.where(check >= drop, (check - drop) / 60, 1.0)


def _baku(n, role, hal_last, baku_last):
    """Copy, then cycle, then repeat with drift, so each expert family gets a phase."""
    if n < 150:
        second = hal_last + 2 if role == "checker" else 61 - hal_last
    elif n < 300:
        second = (12, 25, 37, 48, 55)[n % 5] if role == "checker" else 20 + (7 * n) % 30
    else:
        second = baku_last + (5 if n % 4 == 0 else 0)
    return min(60, max(1, second))


def _provider():
    return TranslatedHal(_StageTable(_matrix()))


def play(*, checkpoint):
    """Return the trace rows and the final observation counts by Baku's role."""
    provider, rows = _provider(), []
    n, hal_last, baku_last = 0, 30, 30
    for game in range(GAMES):
        if checkpoint and game:
            snapshot = provider.dump_state()
            provider = _provider()
            provider.restore_state(snapshot)
        provider.reset_game()
        for half_round in range(HALF_ROUNDS):
            role = "dropper" if (half_round + game) % 2 == 0 else "checker"
            leap = game in LEAP_GAMES and half_round in LEAP_HALF_ROUNDS
            hal = ((37 * n + 11 * game) % 300, 60 * ((n // 10) % 5))
            baku = ((53 * n + 7 * game) % 300, 60 * ((n // 7) % 4))
            checker, dropper = (baku, hal) if role == "dropper" else (hal, baku)
            decision = _decision(role=role, checker_cylinder=checker[0], checker_ttd=checker[1],
                                 dropper_cylinder=dropper[0], dropper_ttd=dropper[1])
            if leap:
                decision = replace(decision, turn_duration=61)
            policy = provider.policy(decision)
            support = sorted(policy)
            total = sum(policy[a] for a in support)
            draw, cumulative = (n * 0.6180339887498949) % 1.0, 0.0
            hal_action = support[-1]
            for action in support:
                cumulative += policy[action] / total
                if draw < cumulative:
                    hal_action = action
                    break
            baku_role = "checker" if role == "dropper" else "dropper"
            baku_action = _baku(n, baku_role, hal_last, baku_last)
            if leap and baku_role == "dropper":
                baku_action = 61
            reveal = _reveal(self_role=role, self_action=hal_action, opponent_action=baku_action,
                             game_index=game, half_round_index=half_round)
            if leap:
                reveal = replace(reveal, pre_decision_state=replace(reveal.pre_decision_state, turn_duration=61))
            gain = None if leap else round(provider.inner.decisions[-1].expected_exploit_gain, 9)
            rows.append([n, game, half_round, role, decision.turn_duration, hal_action, baku_action, support, gain])
            provider.observe(reveal)
            n, hal_last, baku_last = n + 1, hal_action, min(baku_action, 60)
        provider.end_game(PublicGameOutcome(game_index=game, winner_name=None, half_rounds=HALF_ROUNDS))
    model = provider.opponent_model
    return {"rows": rows, "observations": {role: model.observations(role) for role in ("dropper", "checker")}}


def dump(trace):
    """One row per line, so a diff names the first decision that moved."""
    rows = ",\n".join(json.dumps(row, separators=(",", ":")) for row in trace["rows"])
    return f'{{"observations":{json.dumps(trace["observations"])},\n"rows":[\n{rows}\n]}}\n'


@pytest.mark.parametrize("checkpoint", [False, True], ids=["one-provider", "memory-per-game"])
def test_translated_hal_reproduces_the_golden_trace(checkpoint):
    trace = play(checkpoint=checkpoint)
    golden = json.loads(TRACE.read_text())
    leaps = [row for row in trace["rows"] if row[4] == 61]
    assert len(trace["rows"]) == GAMES * HALF_ROUNDS and len(leaps) == 2 * len(LEAP_GAMES)
    assert sum(trace["observations"].values()) == len(trace["rows"]) - len(leaps)
    for index, (row, expected) in enumerate(zip(trace["rows"], golden["rows"], strict=True)):
        assert row == expected, f"decision {index} left the golden trace"
    assert trace["observations"] == golden["observations"]
