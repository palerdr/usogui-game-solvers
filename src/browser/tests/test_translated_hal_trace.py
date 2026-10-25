"""Pin the production translated Hal's decisions to a committed golden trace.

A scripted Baku plays 8 games of 50 half-rounds against
TranslatedHalPolicyProvider on arena.testing's fake stage. Three games hold a
leap pair (turn 61), so the trace covers the DTH fallback and the sequence
break. Hal's sample comes from a fixed golden-ratio sequence, so the trace uses
no random generator. One run keeps one provider. The other run builds a fresh
provider per game and restores OpponentMemory, as the hosted app does. Both
runs must match translated_hal_trace.json, which play() wrote at 938fb25.
Rewrite that file only for an intended change to Hal's behavior.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arena.contracts import PublicGameOutcome
from arena.testing import StageAgent, make_decision, make_reveal
from arena.translated_hal_adapter import TranslatedHalPolicyProvider
from browser.opponent_memory import OpponentMemory

TRACE = Path(__file__).with_name("translated_hal_trace.json")
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
    return TranslatedHalPolicyProvider("unused", agent=StageAgent(_matrix()))


def _memory_app(provider):
    return SimpleNamespace(state=SimpleNamespace(hal_provider=provider))


def play(*, checkpoint):
    """Return the trace rows and the final observation counts by Baku's role."""
    provider, memory, rows = _provider(), OpponentMemory(), []
    n, hal_last, baku_last = 0, 30, 30
    for game in range(GAMES):
        if checkpoint and game:
            snapshot = memory.dump(_memory_app(provider))
            provider = _provider()
            memory.restore(_memory_app(provider), snapshot)
        provider.reset_game()
        for half_round in range(HALF_ROUNDS):
            role = "dropper" if (half_round + game) % 2 == 0 else "checker"
            leap = game in LEAP_GAMES and half_round in LEAP_HALF_ROUNDS
            hal = ((37 * n + 11 * game) % 300, 60 * ((n // 10) % 5))
            baku = ((53 * n + 7 * game) % 300, 60 * ((n // 7) % 4))
            checker, dropper = (baku, hal) if role == "dropper" else (hal, baku)
            decision = make_decision(role=role, checker_cylinder=checker[0], checker_ttd=checker[1],
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
            reveal = make_reveal(self_role=role, self_action=hal_action, opponent_action=baku_action,
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
