from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from arena.agent import PolicyDrivenAgent, decision_from_game, normalize_legal_policy
from arena.contracts import CanonicalDecision
from stl.game import Game, Player, Referee


def _decision(*, legal: tuple[int, ...] = (1, 2, 3)) -> CanonicalDecision:
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


def test_policy_normalization_discards_illegal_zero_mass_entries() -> None:
    actions, probabilities = normalize_legal_policy(
        {0: 4.0, 1: 1.0, 2: 3.0, 61: 2.0}, (1, 2)
    )
    assert actions.tolist() == [1, 2]
    assert probabilities.tolist() == pytest.approx([0.25, 0.75])


@dataclass
class _Provider:
    def policy(self, decision: CanonicalDecision) -> dict[int, float]:
        return {0: 100.0, 1: 1.0, 60: 5.0}


def test_algorithm_agnostic_agent_samples_only_a_legal_engine_action() -> None:
    game = Game(Player("Hal"), Player("Baku"), Referee())
    agent = PolicyDrivenAgent(_Provider(), player_name="Hal", seed=4)
    action = agent.choose_action(game, "dropper", game.get_turn_duration())
    assert 1 <= action <= 60
    assert action != 0


def test_decision_uses_engine_role_relative_state() -> None:
    game = Game(
        Player("Hal", cylinder=20, ttd=30),
        Player("Baku", cylinder=40, ttd=50),
        Referee(),
    )
    game.first_dropper = game.player1
    decision = decision_from_game(game, role="dropper", turn_duration=60)
    assert decision.actor_name == "Hal"
    assert decision.dropper_cylinder_seconds == 20
    assert decision.checker_cylinder_seconds == 40


def test_stl_and_dth_revival_surfaces_match_over_the_full_domain() -> None:
    from stl.dth import revival_probability

    referee = Referee(cprs_performed=17)
    player = Player(name="Either", physicality=0.01)
    for st_in_vial in range(300):
        for ttd in range(301):
            player.ttd = ttd
            actual = referee.compute_survival_probability(
                player, death_duration=st_in_vial + 60
            )
            assert actual == revival_probability(st_in_vial, ttd)


def test_dth_projection_is_the_exact_literal_second_identity() -> None:
    from arena.dth_adapter import project_to_dth_state

    assert project_to_dth_state(_decision()) == (19, 120, 29, 70)
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
    import arena.dth_adapter as adapter
    from stl.dth import Stage

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

    provider = adapter.DTHCompletePolicyProvider(_Table())
    assert provider.policy(_decision()) == {1: 1.0}
    assert "1 exact moves" in provider.match_summary()
