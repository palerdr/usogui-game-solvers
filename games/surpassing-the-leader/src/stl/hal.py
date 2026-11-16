# Copyright 2026 The STL authors
"""Hal, the opponent: exact Hal and translated Hal.

Exact Hal plays the certified DTH equilibrium from ``stl.dth.Table``.
Translated Hal, the production opponent, forecasts the human from public
reveals and plays a best response in the exact stage matrix. In the leap
window it plays the exact policy. Its opponent evidence dumps to a bounded
JSON checkpoint, which the hosted server stores between games.

``make_hal(kind, table)`` builds either one. A provider returns a policy over
literal seconds; ``stl.server.session.PolicyDrivenAgent`` masks and samples it.
The other opponent models of the Hal study are in git history; docs/HAL.md
lists them.
"""

from __future__ import annotations

import base64
import json
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from operator import index as integer_index
from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np

from stl.dth import MAX_SADDLE_GAP, Stage
from stl.game import legal_seconds, validate_action

KINDS = ("translated", "exact")
MEMORY_SCHEMA = "arena-translated-hal-memory-v1"
MAX_MEMORY_BYTES = 4_000_000


@dataclass(frozen=True, slots=True)
class CanonicalDecision:
    """Public canonical decision context passed to a policy provider.

    The numeric fields are literal engine seconds.
    """

    role: str
    actor_name: str
    turn_duration: int
    legal_seconds: tuple[int, ...]
    checker_cylinder_seconds: float
    checker_ttd_seconds: float
    dropper_cylinder_seconds: float
    dropper_ttd_seconds: float


@dataclass(frozen=True, slots=True)
class PublicPlayerState:
    """One player's public load state before a simultaneous decision."""

    name: str
    cylinder_seconds: float
    ttd_seconds: float


@dataclass(frozen=True, slots=True)
class PublicDecisionState:
    """Public state captured before either half-round action is requested."""

    game_clock_seconds: float
    round_index: int
    half_index: int
    turn_duration: int
    players: tuple[PublicPlayerState, PublicPlayerState]


@dataclass(frozen=True, slots=True)
class PublicHalfRound:
    """Revealed public result delivered once to every session provider."""

    game_index: int
    half_round_index: int
    pre_decision_state: PublicDecisionState
    dropper_name: str
    checker_name: str
    drop_time: int
    check_time: int
    outcome: str
    game_over: bool
    winner_name: str | None


@dataclass(frozen=True, slots=True)
class PublicGameOutcome:
    """End-of-game notification, including capped games without a winner."""

    game_index: int
    winner_name: str | None
    half_rounds: int


@runtime_checkable
class CanonicalPolicyProvider(Protocol):
    """Provide an unnormalized literal-second policy for one decision."""

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]: ...


def reset_provider_game(provider: object) -> None:
    """Start a fresh per-game lifecycle when a provider implements the hook."""

    hook = getattr(provider, "reset_game", None)
    if callable(hook):
        hook()


def observe_provider(provider: object, record: PublicHalfRound) -> None:
    """Deliver one revealed public half-round to an interested provider."""

    hook = getattr(provider, "observe", None)
    if callable(hook):
        hook(record)


def end_provider_game(provider: object, outcome: PublicGameOutcome) -> None:
    """Finish one game without requiring hooks on exact/stateless providers."""

    hook = getattr(provider, "end_game", None)
    if callable(hook):
        hook(outcome)


def project_to_dth_state(decision: CanonicalDecision) -> tuple[int, int, int, int]:
    """Project the shared literal-second coordinates without approximation."""

    coordinates = (
        decision.checker_cylinder_seconds,
        decision.checker_ttd_seconds,
        decision.dropper_cylinder_seconds,
        decision.dropper_ttd_seconds,
    )
    if any(value != int(value) for value in coordinates):
        message = "canonical STL state is not on the literal-second DTH grid"
        raise ValueError(message)
    return tuple(int(value) for value in coordinates)  # type: ignore[return-value]


class StageReader(Protocol):
    def stage(self, state: tuple[int, int, int, int]) -> Stage: ...


@dataclass
class ExactHal:
    """Serve exact equilibrium policies from the certified DTH table.

    ``table`` is one opened ``stl.dth.Table``; the hosted runtime shares it
    between every player. Each decision records its stage and latency. With
    ``record_decisions`` false the provider keeps no per-move record, so a
    long-lived provider does not grow.
    """

    table: StageReader = field(repr=False)
    decisions: list[tuple[Stage, float]] = field(default_factory=list, repr=False)
    record_decisions: bool = True

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        started = time.monotonic()
        stage = self.table.stage(project_to_dth_state(decision))
        if self.record_decisions:
            self.decisions.append((stage, time.monotonic() - started))
        row = stage.drop if decision.role == "dropper" else stage.check
        return {
            second: float(probability)
            for second, probability in enumerate(row, start=1)
            if probability > 0.0
        }

    def match_summary(self) -> str:
        if not self.decisions:
            return "dth complete: no moves played"
        latencies = np.asarray(
            [elapsed for _stage, elapsed in self.decisions], dtype=np.float64
        )
        worst_gap = max(stage.gap for stage, _elapsed in self.decisions)
        return (
            f"dth complete: {len(self.decisions)} exact moves; "
            f"worst saddle gap {worst_gap:.3g}; "
            f"latency p95 {float(np.quantile(latencies, 0.95)):.3f}s "
            f"max {float(np.max(latencies)):.3f}s"
        )


ACTION_COUNT = 60
ACTIONS = tuple(range(1, ACTION_COUNT + 1))
PERFECT_HAL_DIAGNOSTICS_SCHEMA = "arena-perfect-hal-diagnostics-v1"

_BASE_EXPERT_NAMES = (
    "uniform",
    "global",
    "slow_recency",
    "fast_recency",
    "flash_recency",
    "repeat",
    "opponent_markov",
    "self_response",
    "action_delta",
    "state_regime",
)


def _finite(value: float, *, label: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        message = f"{label} must be finite"
        raise ValueError(message)
    return resolved


def _normalize(raw: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if (
        values.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
    ):
        message = f"{label} must be finite nonnegative length 60 with mass"
        raise ValueError(message)
    result = values / float(values.sum())
    result.setflags(write=False)
    return result


def _readonly(raw: np.ndarray) -> np.ndarray:
    result = np.asarray(raw, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _point(action: int) -> np.ndarray:
    resolved = _literal_action(action, label="pure-DTH action")
    result = np.zeros(ACTION_COUNT, dtype=np.float64)
    result[resolved - 1] = 1.0
    return result


def _literal_action(raw: object, *, label: str) -> int:
    if isinstance(raw, (bool, np.bool_)):
        message = f"{label} must be an integer second in 1..60"
        raise ValueError(message)
    try:
        resolved = int(integer_index(cast(SupportsIndex, raw)))
    except TypeError as error:
        message = f"{label} must be an integer second in 1..60"
        raise ValueError(message) from error
    if not 1 <= resolved <= ACTION_COUNT:
        message = f"{label} must be an integer second in 1..60"
        raise ValueError(message)
    return resolved


def _band(value: float) -> int:
    resolved = _finite(float(value), label="public DTH state coordinate")
    if resolved < 0.0:
        message = "public DTH state coordinates must be nonnegative"
        raise ValueError(message)
    return min(5, int(resolved // 60.0))


@dataclass(frozen=True, slots=True)
class PerfectHalConfig:
    """Online forecast and unrestricted response controls.

    The zero response temperature is intentional: the default policy puts all
    mass on the current forecast's exact best-response set and never blends in
    equilibrium for safety.  Positive temperatures remain available for
    experiments that need a softer response.
    """

    action_count: int = ACTION_COUNT
    prior_strength: float = 0.02
    conditional_prior_strength: float = 0.10
    recency_retentions: tuple[float, float, float] = (0.995, 0.82, 0.30)
    expert_learning_rate: float = 1.25
    expert_weight_retention: float = 0.92
    expert_weight_floor: float = 0.002
    log_probability_floor: float = 1e-9
    periodicities: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
    response_temperature: float = 0.0
    best_response_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.action_count != ACTION_COUNT:
            message = "Perfect Hal v1 requires exactly 60 pure-DTH actions"
            raise ValueError(message)
        if _finite(self.prior_strength, label="prior_strength") <= 0.0:
            message = "prior_strength must be positive"
            raise ValueError(message)
        if (
            _finite(
                self.conditional_prior_strength,
                label="conditional_prior_strength",
            )
            <= 0.0
        ):
            message = "conditional_prior_strength must be positive"
            raise ValueError(message)
        if len(self.recency_retentions) != 3 or not all(
            0.0 < _finite(value, label="recency retention") < 1.0
            for value in self.recency_retentions
        ):
            message = "recency_retentions must contain three values in (0, 1)"
            raise ValueError(message)
        if not (
            self.recency_retentions[0]
            > self.recency_retentions[1]
            > self.recency_retentions[2]
        ):
            message = "recency_retentions must be ordered slow to flash"
            raise ValueError(message)
        if _finite(self.expert_learning_rate, label="expert_learning_rate") <= 0.0:
            message = "expert_learning_rate must be positive"
            raise ValueError(message)
        if not (
            0.0
            <= _finite(
                self.expert_weight_retention,
                label="expert_weight_retention",
            )
            <= 1.0
        ):
            message = "expert_weight_retention must lie in [0, 1]"
            raise ValueError(message)
        if not self.periodicities or any(
            isinstance(period, bool) or not isinstance(period, int) or period < 2
            for period in self.periodicities
        ):
            message = "periodicities must be nonempty integers at least two"
            raise ValueError(message)
        if len(set(self.periodicities)) != len(self.periodicities):
            message = "periodicities must be unique"
            raise ValueError(message)
        expert_count = len(_BASE_EXPERT_NAMES) + len(self.periodicities)
        floor = _finite(self.expert_weight_floor, label="expert_weight_floor")
        if floor < 0.0 or floor * expert_count >= 1.0:
            message = "expert_weight_floor leaves no probability mass"
            raise ValueError(message)
        probability_floor = _finite(
            self.log_probability_floor,
            label="log_probability_floor",
        )
        if not 0.0 < probability_floor < 1.0:
            message = "log_probability_floor must lie in (0, 1)"
            raise ValueError(message)
        if _finite(self.response_temperature, label="response_temperature") < 0.0:
            message = "response_temperature must be nonnegative"
            raise ValueError(message)
        if (
            _finite(
                self.best_response_tolerance,
                label="best_response_tolerance",
            )
            < 0.0
        ):
            message = "best_response_tolerance must be nonnegative"
            raise ValueError(message)

    @property
    def expert_names(self) -> tuple[str, ...]:
        return _BASE_EXPERT_NAMES + tuple(
            f"period_{period}" for period in self.periodicities
        )


@dataclass(frozen=True, slots=True)
class PerfectHalContext:
    """Causal context fixed before the simultaneous action is revealed."""

    opponent_role: str
    observation_index: int
    game_index: int
    game_decision_index: int
    state_regime: tuple[int, int, int, int]
    previous_opponent_action: int | None
    previous_self_action: int | None


@dataclass(frozen=True, slots=True)
class PerfectHalForecast:
    """One immutable pre-decision opponent forecast and its expert evidence."""

    context: PerfectHalContext
    policy: np.ndarray
    expert_names: tuple[str, ...]
    expert_weights: np.ndarray
    expert_policies: np.ndarray
    entropy: float
    confidence: float


@dataclass(slots=True)
class _RoleState:
    observations: int
    global_counts: np.ndarray
    recency_counts: tuple[np.ndarray, np.ndarray, np.ndarray]
    transition_counts: np.ndarray
    response_counts: np.ndarray
    delta_counts: np.ndarray
    state_counts: dict[tuple[int, int, int, int], np.ndarray]
    phase_counts: dict[tuple[int, int], np.ndarray]
    log_weights: np.ndarray
    previous_opponent_action: int | None = None
    previous_self_action: int | None = None


def _new_role_state(config: PerfectHalConfig) -> _RoleState:
    recency_counts = (
        np.zeros(ACTION_COUNT, dtype=np.float64),
        np.zeros(ACTION_COUNT, dtype=np.float64),
        np.zeros(ACTION_COUNT, dtype=np.float64),
    )
    return _RoleState(
        observations=0,
        global_counts=np.zeros(ACTION_COUNT, dtype=np.float64),
        recency_counts=recency_counts,
        transition_counts=np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.float64),
        response_counts=np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.float64),
        delta_counts=np.zeros(2 * ACTION_COUNT - 1, dtype=np.float64),
        state_counts={},
        phase_counts={},
        log_weights=np.zeros(len(config.expert_names), dtype=np.float64),
    )


class PerfectHalOpponentModel:
    """Role-separated online expert mixture trained only by public reveals."""

    def __init__(self, config: PerfectHalConfig = PerfectHalConfig()) -> None:  # noqa: B008 - frozen dataclass default
        self.config = config
        self._roles = {
            "dropper": _new_role_state(config),
            "checker": _new_role_state(config),
        }

    def reset(self) -> None:
        """Forget the opponent and restore an independent session prior."""

        self._roles = {
            "dropper": _new_role_state(self.config),
            "checker": _new_role_state(self.config),
        }

    @property
    def total_observations(self) -> int:
        return sum(state.observations for state in self._roles.values())

    def observations(self, role: str) -> int:
        return self._state(role).observations

    def _state(self, role: str) -> _RoleState:
        if role not in self._roles:
            message = "opponent role must be 'dropper' or 'checker'"
            raise ValueError(message)
        return self._roles[role]

    @staticmethod
    def state_regime(decision: CanonicalDecision) -> tuple[int, int, int, int]:
        """Coarsen the public load state without using native engine internals."""

        if decision.role == "dropper":
            own = (
                decision.dropper_cylinder_seconds,
                decision.dropper_ttd_seconds,
            )
            opponent = (
                decision.checker_cylinder_seconds,
                decision.checker_ttd_seconds,
            )
        elif decision.role == "checker":
            own = (
                decision.checker_cylinder_seconds,
                decision.checker_ttd_seconds,
            )
            opponent = (
                decision.dropper_cylinder_seconds,
                decision.dropper_ttd_seconds,
            )
        else:
            message = "decision role must be 'dropper' or 'checker'"
            raise ValueError(message)
        return (_band(own[0]), _band(own[1]), _band(opponent[0]), _band(opponent[1]))

    def _predictive(
        self,
        counts: np.ndarray,
        prior: np.ndarray,
        *,
        strength: float,
    ) -> np.ndarray:
        values = np.asarray(counts, dtype=np.float64)
        if (
            values.shape != (ACTION_COUNT,)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            message = "Perfect Hal count state is malformed"
            raise RuntimeError(message)
        return _normalize(
            values + float(strength) * np.asarray(prior, dtype=np.float64),
            label="Perfect Hal predictive policy",
        )

    def _delta_policy(
        self,
        state: _RoleState,
        global_policy: np.ndarray,
    ) -> np.ndarray:
        previous = state.previous_opponent_action
        if previous is None or float(state.delta_counts.sum()) <= 0.0:
            return global_policy
        projected = np.zeros(ACTION_COUNT, dtype=np.float64)
        for index, count in enumerate(state.delta_counts):
            if count <= 0.0:
                continue
            delta = index - (ACTION_COUNT - 1)
            action = previous + delta
            if 1 <= action <= ACTION_COUNT:
                projected[action - 1] += count
        return self._predictive(
            projected,
            global_policy,
            strength=self.config.conditional_prior_strength,
        )

    def _expert_policies(
        self,
        state: _RoleState,
        context: PerfectHalContext,
    ) -> np.ndarray:
        uniform = np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float64)
        global_policy = self._predictive(
            state.global_counts,
            uniform,
            strength=self.config.prior_strength,
        )
        policies: list[np.ndarray] = [uniform, global_policy]
        policies.extend(
            self._predictive(
                counts,
                uniform,
                strength=self.config.prior_strength,
            )
            for counts in state.recency_counts
        )
        policies.append(
            _point(state.previous_opponent_action)
            if state.previous_opponent_action is not None
            else global_policy
        )
        policies.append(
            self._predictive(
                state.transition_counts[state.previous_opponent_action - 1],
                global_policy,
                strength=self.config.conditional_prior_strength,
            )
            if state.previous_opponent_action is not None
            else global_policy
        )
        policies.append(
            self._predictive(
                state.response_counts[state.previous_self_action - 1],
                global_policy,
                strength=self.config.conditional_prior_strength,
            )
            if state.previous_self_action is not None
            else global_policy
        )
        policies.append(self._delta_policy(state, global_policy))
        policies.append(
            self._predictive(
                state.state_counts.get(
                    context.state_regime,
                    np.zeros(ACTION_COUNT, dtype=np.float64),
                ),
                global_policy,
                strength=self.config.conditional_prior_strength,
            )
        )
        for period in self.config.periodicities:
            phase = context.observation_index % period
            policies.append(
                self._predictive(
                    state.phase_counts.get(
                        (period, phase),
                        np.zeros(ACTION_COUNT, dtype=np.float64),
                    ),
                    global_policy,
                    strength=self.config.conditional_prior_strength,
                )
            )
        matrix = np.stack(policies, axis=0)
        if matrix.shape != (len(self.config.expert_names), ACTION_COUNT):
            message = "Perfect Hal expert matrix shape drifted"
            raise RuntimeError(message)
        return matrix

    def predict(
        self,
        opponent_role: str,
        *,
        state_regime: tuple[int, int, int, int],
        game_index: int,
        game_decision_index: int,
    ) -> PerfectHalForecast:
        """Forecast one opponent action before the simultaneous reveal exists."""

        state = self._state(opponent_role)
        if game_index < 0 or game_decision_index < 0:
            message = "game and decision indices must be nonnegative"
            raise ValueError(message)
        if len(state_regime) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in state_regime
        ):
            message = "state_regime must contain four nonnegative integers"
            raise ValueError(message)
        context = PerfectHalContext(
            opponent_role=opponent_role,
            observation_index=state.observations,
            game_index=int(game_index),
            game_decision_index=int(game_decision_index),
            state_regime=state_regime,
            previous_opponent_action=state.previous_opponent_action,
            previous_self_action=state.previous_self_action,
        )
        expert_policies = self._expert_policies(state, context)
        centered = state.log_weights - float(np.max(state.log_weights))
        weights = np.exp(np.clip(centered, -700.0, 0.0))
        weights /= float(weights.sum())
        floor = self.config.expert_weight_floor
        if floor > 0.0:
            weights = (1.0 - floor * weights.size) * weights + floor
        policy = _normalize(weights @ expert_policies, label="Perfect Hal forecast")
        clipped = np.clip(
            policy,
            self.config.log_probability_floor,
            1.0,
        )
        entropy = -float(np.sum(policy * np.log(clipped)))
        evidence = state.observations / (
            state.observations + self.config.prior_strength
        )
        confidence = evidence * max(0.0, 1.0 - entropy / np.log(ACTION_COUNT))
        expert_matrix = np.asarray(expert_policies, dtype=np.float64).copy()
        expert_matrix.setflags(write=False)
        return PerfectHalForecast(
            context=context,
            policy=policy,
            expert_names=self.config.expert_names,
            expert_weights=_readonly(weights),
            expert_policies=expert_matrix,
            entropy=entropy,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def observe(
        self,
        forecast: PerfectHalForecast,
        *,
        opponent_action: int,
        self_action: int,
    ) -> None:
        """Score the frozen forecast, then learn from one public reveal."""

        resolved_opponent_action = _literal_action(
            opponent_action,
            label="opponent action",
        )
        resolved_self_action = _literal_action(
            self_action,
            label="self action",
        )
        state = self._state(forecast.context.opponent_role)
        if forecast.context.observation_index != state.observations:
            message = "Perfect Hal forecast token is stale or already observed"
            raise RuntimeError(message)
        if forecast.expert_names != self.config.expert_names or (
            forecast.expert_policies.shape
            != (len(self.config.expert_names), ACTION_COUNT)
        ):
            message = "Perfect Hal forecast token is incompatible"
            raise ValueError(message)

        action_index = resolved_opponent_action - 1
        realized = np.clip(
            forecast.expert_policies[:, action_index],
            self.config.log_probability_floor,
            1.0,
        )
        state.log_weights *= self.config.expert_weight_retention
        state.log_weights += self.config.expert_learning_rate * np.log(realized)
        state.log_weights -= float(np.max(state.log_weights))

        state.global_counts[action_index] += 1.0
        for retention, counts in zip(
            self.config.recency_retentions,
            state.recency_counts,
            strict=True,
        ):
            counts[...] *= retention
            counts[action_index] += 1.0
        previous_opponent = forecast.context.previous_opponent_action
        previous_self = forecast.context.previous_self_action
        if previous_opponent is not None:
            state.transition_counts[previous_opponent - 1, action_index] += 1.0
            delta_index = (
                resolved_opponent_action - previous_opponent + (ACTION_COUNT - 1)
            )
            state.delta_counts[delta_index] += 1.0
        if previous_self is not None:
            state.response_counts[previous_self - 1, action_index] += 1.0
        state.state_counts.setdefault(
            forecast.context.state_regime,
            np.zeros(ACTION_COUNT, dtype=np.float64),
        )[action_index] += 1.0
        for period in self.config.periodicities:
            phase = forecast.context.observation_index % period
            state.phase_counts.setdefault(
                (period, phase),
                np.zeros(ACTION_COUNT, dtype=np.float64),
            )[action_index] += 1.0
        state.previous_opponent_action = resolved_opponent_action
        state.previous_self_action = resolved_self_action
        state.observations += 1


def _validate_stage(stage: Stage) -> None:
    matrix = np.asarray(stage.matrix, dtype=np.float64)
    if matrix.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(np.isfinite(matrix)):
        message = "certified DTH stage matrix must be finite 60x60"
        raise ValueError(message)
    for label, raw in (
        ("dropper", stage.drop),
        ("checker", stage.check),
    ):
        policy = np.asarray(raw, dtype=np.float64)
        if (
            policy.shape != (ACTION_COUNT,)
            or not np.all(np.isfinite(policy))
            or np.any(policy < 0.0)
            or abs(float(policy.sum()) - 1.0) > 1e-8
        ):
            message = f"certified DTH {label} policy is malformed"
            raise ValueError(message)
    if (
        not np.isfinite(stage.value)
        or not np.isfinite(stage.gap)
        or stage.gap < 0.0
        or stage.gap > MAX_SADDLE_GAP
    ):
        message = "certified DTH value or saddle gap is malformed"
        raise ValueError(message)


def _best_response_policy(
    action_values: np.ndarray,
    config: PerfectHalConfig,
) -> np.ndarray:
    values = np.asarray(action_values, dtype=np.float64)
    if values.shape != (ACTION_COUNT,) or not np.all(np.isfinite(values)):
        message = "Perfect Hal action values must be finite length 60"
        raise ValueError(message)
    best = float(np.max(values))
    if config.response_temperature == 0.0:
        winners = values >= best - config.best_response_tolerance
        policy = winners.astype(np.float64)
    else:
        logits = (values - best) / config.response_temperature
        policy = np.exp(np.clip(logits, -700.0, 0.0))
    return _normalize(policy, label="Perfect Hal best response")


@dataclass(frozen=True, slots=True)
class PerfectHalDecision:
    """Public-history and exact-matrix diagnostics for one decision."""

    role: str
    opponent_role: str
    policy: tuple[float, ...]
    opponent_policy: tuple[float, ...]
    action_values: tuple[float, ...]
    expert_weights: tuple[tuple[str, float], ...]
    evidence_count: int
    forecast_entropy: float
    forecast_confidence: float
    expected_payoff: float
    expected_exploit_gain: float
    best_response_actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    self_name: str
    self_role: str
    forecast: PerfectHalForecast


class PerfectHalPolicyProvider(CanonicalPolicyProvider):
    """Causal, session-persistent, unrestricted pure-DTH policy provider."""

    def __init__(
        self,
        table: StageReader,
        config: PerfectHalConfig = PerfectHalConfig(),  # noqa: B008 - frozen dataclass default
        *,
        opponent_model: PerfectHalOpponentModel | None = None,
    ) -> None:
        self.table = table
        self.config = config
        if opponent_model is None:
            opponent_model = PerfectHalOpponentModel(config)
        self.opponent_model = opponent_model
        if self.opponent_model.config != config:
            message = "Perfect Hal provider and opponent-model configs differ"
            raise ValueError(message)
        self.decisions: list[PerfectHalDecision] = []
        self._pending: _PendingDecision | None = None
        self._seen_reveals: set[tuple[int, int, int]] = set()
        self._current_actor_name: str | None = None
        self._game_started = False
        self._game_epoch = -1
        self._game_decision_index = 0

    @property
    def last_decision(self) -> PerfectHalDecision | None:
        return self.decisions[-1] if self.decisions else None

    def reset_session(self) -> None:
        """Forget the opponent and restore an independent repeated session."""

        if self._pending is not None:
            message = "Perfect Hal session reset with an unrevealed action"
            raise RuntimeError(message)
        self.opponent_model.reset()
        self.decisions.clear()
        self._seen_reveals.clear()
        self._current_actor_name = None
        self._game_started = False
        self._game_epoch = -1
        self._game_decision_index = 0

    def reset_game(self) -> None:
        """Start a fresh game while retaining the repeated-opponent model."""

        if self._pending is not None:
            message = "Perfect Hal game reset with an unrevealed action"
            raise RuntimeError(message)
        if self._game_started:
            self._game_epoch += 1
        else:
            self._game_started = True
            self._game_epoch = 0
        self._current_actor_name = None
        self._game_decision_index = 0

    @staticmethod
    def _validate_decision(decision: CanonicalDecision) -> None:
        if decision.role not in {"dropper", "checker"}:
            message = "Perfect Hal role must be dropper or checker"
            raise ValueError(message)
        if (
            decision.turn_duration != ACTION_COUNT
            or tuple(decision.legal_seconds) != ACTIONS
        ):
            message = (
                "Perfect Hal supports pure DTH only: turn duration and legal "
                "actions must be exactly literal seconds 1..60"
            )
            raise ValueError(message)

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        """Return an unrestricted exact-matrix response to the causal forecast."""

        self._validate_decision(decision)
        if self._pending is not None:
            message = "Perfect Hal was asked to act twice before a reveal"
            raise RuntimeError(message)
        if not self._game_started:
            self.reset_game()
        if self._current_actor_name is None:
            self._current_actor_name = decision.actor_name
        elif self._current_actor_name.casefold() != decision.actor_name.casefold():
            message = "Perfect Hal cannot switch player identity mid-game"
            raise RuntimeError(message)

        stage = self.table.stage(project_to_dth_state(decision))
        _validate_stage(stage)
        opponent_role = "checker" if decision.role == "dropper" else "dropper"
        forecast = self.opponent_model.predict(
            opponent_role,
            state_regime=self.opponent_model.state_regime(decision),
            game_index=self._game_epoch,
            game_decision_index=self._game_decision_index,
        )
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        if decision.role == "dropper":
            action_values = matrix @ forecast.policy
            exact_policy = np.asarray(stage.drop, dtype=np.float64)
        else:
            action_values = -(matrix.T @ forecast.policy)
            exact_policy = np.asarray(stage.check, dtype=np.float64)
        policy = _best_response_policy(action_values, self.config)
        expected_payoff = float(policy @ action_values)
        exact_expected = float(exact_policy @ action_values)
        best = float(np.max(action_values))
        best_actions = tuple(
            index + 1
            for index, value in enumerate(action_values)
            if value >= best - self.config.best_response_tolerance
        )
        diagnostic = PerfectHalDecision(
            role=decision.role,
            opponent_role=opponent_role,
            policy=tuple(float(value) for value in policy),
            opponent_policy=tuple(float(value) for value in forecast.policy),
            action_values=tuple(float(value) for value in action_values),
            expert_weights=tuple(
                (name, float(weight))
                for name, weight in zip(
                    forecast.expert_names,
                    forecast.expert_weights,
                    strict=True,
                )
            ),
            evidence_count=forecast.context.observation_index,
            forecast_entropy=forecast.entropy,
            forecast_confidence=forecast.confidence,
            expected_payoff=expected_payoff,
            expected_exploit_gain=expected_payoff - exact_expected,
            best_response_actions=best_actions,
        )
        self.decisions.append(diagnostic)
        self._pending = _PendingDecision(
            self_name=decision.actor_name,
            self_role=decision.role,
            forecast=forecast,
        )
        self._game_decision_index += 1
        return {
            action: float(policy[action - 1])
            for action in ACTIONS
            if policy[action - 1] > 0.0
        }

    def observe(self, record: PublicHalfRound) -> None:
        """Learn from one revealed result after the pending policy was fixed."""

        pending = self._pending
        if pending is None:
            message = "Perfect Hal received a reveal without a pending action"
            raise RuntimeError(message)
        if record.game_index < 0 or record.half_round_index < 0:
            message = "public reveal indices must be nonnegative"
            raise ValueError(message)
        key = (self._game_epoch, record.game_index, record.half_round_index)
        if key in self._seen_reveals:
            message = "public reveal was delivered more than once"
            raise RuntimeError(message)
        self_name = pending.self_name.casefold()
        if record.dropper_name.casefold() == self_name:
            self_role = "dropper"
            self_action = _literal_action(record.drop_time, label="self action")
            opponent_action = _literal_action(
                record.check_time,
                label="opponent action",
            )
        elif record.checker_name.casefold() == self_name:
            self_role = "checker"
            self_action = _literal_action(record.check_time, label="self action")
            opponent_action = _literal_action(
                record.drop_time,
                label="opponent action",
            )
        else:
            message = "public reveal does not include the Perfect Hal player"
            raise ValueError(message)
        if self_role != pending.self_role:
            message = "public reveal role disagrees with the pending decision"
            raise RuntimeError(message)
        self.opponent_model.observe(
            pending.forecast,
            opponent_action=opponent_action,
            self_action=self_action,
        )
        self._seen_reveals.add(key)
        self._pending = None

    def end_game(self, outcome: PublicGameOutcome) -> None:
        """Record a terminal public result without erasing opponent memory."""

        if self._pending is not None:
            message = "Perfect Hal game ended with an unrevealed action"
            raise RuntimeError(message)
        if outcome.game_index < 0 or outcome.half_rounds < 0:
            message = "public game outcome indices must be nonnegative"
            raise ValueError(message)

    def match_summary(self) -> str:
        if not self.decisions:
            return "Perfect Hal: no moves played"
        gains = np.asarray(
            [decision.expected_exploit_gain for decision in self.decisions],
            dtype=np.float64,
        )
        latest = self.decisions[-1]
        dominant = max(latest.expert_weights, key=lambda item: item[1])
        return (
            f"Perfect Hal: {len(self.decisions)} unrestricted pure-DTH moves; "
            f"{self.opponent_model.total_observations} public reveals; "
            f"mean forecasted gain over exact {float(gains.mean()):+.4f}; "
            f"latest dominant expert {dominant[0]} ({dominant[1]:.1%})"
        )

    def experiment_diagnostics(self) -> dict[str, object]:
        latest = self.last_decision
        return {
            "schema_version": PERFECT_HAL_DIAGNOSTICS_SCHEMA,
            "model_schema": getattr(
                self.opponent_model, "schema", "arena-perfect-hal-online-adaptive-v1"
            ),
            "opponent_model": None,
            "pure_dth_only": True,
            "public_history_only": True,
            "unrestricted_best_response": True,
            "equilibrium_safety_blend": False,
            "config": asdict(self.config),
            "decision_count": len(self.decisions),
            "observations_by_opponent_role": {
                role: self.opponent_model.observations(role)
                for role in ("dropper", "checker")
            },
            "latest": None
            if latest is None
            else {
                "role": latest.role,
                "opponent_role": latest.opponent_role,
                "evidence_count": latest.evidence_count,
                "forecast_entropy": latest.forecast_entropy,
                "forecast_confidence": latest.forecast_confidence,
                "expected_exploit_gain": latest.expected_exploit_gain,
                "best_response_actions": list(latest.best_response_actions),
                "expert_weights": dict(latest.expert_weights),
            },
        }


@dataclass(frozen=True, slots=True)
class TranslatedHalConfig(PerfectHalConfig):
    offset_retention: float = 0.9
    translated: bool = True

    def __post_init__(self) -> None:
        super(TranslatedHalConfig, self).__post_init__()
        if not np.isfinite(self.offset_retention) or not 0 < self.offset_retention <= 1:
            message = "offset_retention must be in (0, 1]"
            raise ValueError(message)
        if not isinstance(self.translated, bool):
            message = "translated must be boolean"
            raise ValueError(message)
        if self.expert_weight_floor * len(self.expert_names) >= 1:
            message = "expert floor leaves no probability mass"
            raise ValueError(message)

    @property
    def expert_names(self) -> tuple[str, ...]:
        base = super(TranslatedHalConfig, self).expert_names
        return base + (
            tuple(
                f"offset_{anchor}_{direction}_{rate}"
                for anchor in ("role", "last")
                for direction in ("copy", "mirror")
                for rate in ("global", "recent")
            )
            if self.translated
            else ()
        )


@dataclass(frozen=True, slots=True)
class TranslatedContext(PerfectHalContext):
    last_self_action: int | None = None


class TranslatedHalOpponentModel(PerfectHalOpponentModel):
    config: TranslatedHalConfig
    """Score Old's experts and offset experts before each public reveal."""

    schema = "arena-translated-hal-v1"

    def __init__(self, config: TranslatedHalConfig = TranslatedHalConfig()) -> None:  # noqa: B008 - frozen dataclass default
        super().__init__(config)
        base = {
            key: value
            for key, value in asdict(config).items()
            if key in PerfectHalConfig.__dataclass_fields__
        }
        self._base = PerfectHalOpponentModel(PerfectHalConfig(**base))
        self._offsets = {role: np.zeros((2, 2, 2, 119)) for role in self._roles}
        self._last_self = None

    def reset(self) -> None:
        super().reset()
        self._offsets = {role: np.zeros((2, 2, 2, 119)) for role in self._roles}
        self._last_self = None

    def break_sequence(self) -> None:
        """Keep learned counts; clear references after an excluded leap reveal."""
        self._last_self = None
        for state in self._roles.values():
            state.previous_self_action = None
            state.previous_opponent_action = None

    def _expert_policies(
        self, state: _RoleState, context: PerfectHalContext
    ) -> np.ndarray:
        base = self._base._expert_policies(state, context)
        if not self.config.translated:
            return base
        policies = list(base)
        prior = base[1]
        offsets = self._offsets[context.opponent_role]
        for anchor_index, anchor in enumerate(
            (context.previous_self_action, self._last_self)
        ):
            for mirrored in range(2):
                for rate in range(2):
                    if anchor is None:
                        policies.append(prior)
                        continue
                    reference = 61 - anchor if mirrored else anchor
                    actions = np.clip(reference + np.arange(-59, 60), 1, 60) - 1
                    counts = np.bincount(
                        actions,
                        weights=offsets[anchor_index, mirrored, rate],
                        minlength=60,
                    )
                    policies.append(
                        self._predictive(
                            counts,
                            prior,
                            strength=self.config.conditional_prior_strength,
                        )
                    )
        return np.stack(policies)

    def predict(
        self,
        opponent_role: str,
        *,
        state_regime: tuple[int, int, int, int],
        game_index: int,
        game_decision_index: int,
    ) -> PerfectHalForecast:
        forecast = super().predict(
            opponent_role,
            state_regime=state_regime,
            game_index=game_index,
            game_decision_index=game_decision_index,
        )
        context = TranslatedContext(
            **asdict(forecast.context), last_self_action=self._last_self
        )
        return replace(forecast, context=context)

    def observe(
        self, forecast: PerfectHalForecast, *, opponent_action: int, self_action: int
    ) -> None:
        super().observe(
            forecast, opponent_action=opponent_action, self_action=self_action
        )
        if self.config.translated:
            offsets = self._offsets[forecast.context.opponent_role]
            anchors = (
                forecast.context.previous_self_action,
                cast("TranslatedContext", forecast.context).last_self_action,
            )
            for anchor_index, anchor in enumerate(anchors):
                if anchor is None:
                    continue
                for mirrored in range(2):
                    reference = 61 - anchor if mirrored else anchor
                    offset = opponent_action - reference + 59
                    offsets[anchor_index, mirrored, 1] *= self.config.offset_retention
                    offsets[anchor_index, mirrored, :, offset] += 1
        self._last_self = int(self_action)


# The 13 values that the translated-Hal study selected (candidate
# "offset_slow"). Before this module they lived in
# src/arena/config/translated_hal_v1_selection.json.
TRANSLATED_HAL_V1 = TranslatedHalConfig(
    action_count=60,
    prior_strength=0.02,
    conditional_prior_strength=0.1,
    recency_retentions=(0.995, 0.82, 0.3),
    expert_learning_rate=1.25,
    expert_weight_retention=0.97,
    expert_weight_floor=0.002,
    log_probability_floor=1e-09,
    periodicities=(2, 3, 4, 5, 6, 8),
    response_temperature=0.0,
    best_response_tolerance=1e-12,
    offset_retention=0.97,
    translated=True,
)


class TranslatedHal:
    """Exploit ordinary turns; use DTH equilibrium during the leap window."""

    def __init__(self, table: StageReader) -> None:
        config = TRANSLATED_HAL_V1
        self.inner = PerfectHalPolicyProvider(
            table, config, opponent_model=TranslatedHalOpponentModel(config)
        )
        self._leap_pending = None
        self.leap_fallbacks = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        if self._leap_pending is not None:
            message = "Hal was asked to act twice before a leap reveal"
            raise RuntimeError(message)
        if decision.turn_duration == ACTION_COUNT:
            return self.inner.policy(decision)
        if (
            decision.turn_duration != ACTION_COUNT + 1
            or decision.role not in ("dropper", "checker")
            or tuple(decision.legal_seconds)
            != legal_seconds(decision.actor_name, decision.role, 61)
        ):
            message = "invalid canonical leap decision"
            raise ValueError(message)
        if self.inner._pending is not None:
            message = "Hal has an unrevealed ordinary action"
            raise RuntimeError(message)
        stage = self.inner.table.stage(project_to_dth_state(decision))
        _validate_stage(stage)
        policy = stage.drop if decision.role == "dropper" else stage.check
        self._leap_pending = decision
        self.leap_fallbacks += 1
        return {i + 1: float(p) for i, p in enumerate(policy) if p > 0}

    def observe(self, record: PublicHalfRound) -> None:
        pending = self._leap_pending
        if pending is None:
            self.inner.observe(record)
            return
        name = record.dropper_name if pending.role == "dropper" else record.checker_name
        if (
            name != pending.actor_name
            or record.pre_decision_state.turn_duration != ACTION_COUNT + 1
        ):
            message = "leap reveal disagrees with the pending decision"
            raise ValueError(message)
        if record.game_index < 0 or record.half_round_index < 0:
            message = "invalid leap reveal indices"
            raise ValueError(message)
        validate_action(
            record.drop_time,
            actor=record.dropper_name,
            role="dropper",
            turn_duration=61,
        )
        validate_action(
            record.check_time,
            actor=record.checker_name,
            role="checker",
            turn_duration=61,
        )
        cast("TranslatedHalOpponentModel", self.inner.opponent_model).break_sequence()
        self._leap_pending = None

    def reset_game(self) -> None:
        if self._leap_pending is not None:
            message = "game reset with an unrevealed leap action"
            raise RuntimeError(message)
        self.inner.reset_game()
        self.inner.decisions.clear()
        self.inner._seen_reveals.clear()
        self.leap_fallbacks = 0

    def reset_session(self) -> None:
        if self._leap_pending is not None:
            message = "session reset with an unrevealed leap action"
            raise RuntimeError(message)
        self.inner.reset_session()
        self.leap_fallbacks = 0

    def end_game(self, outcome: PublicGameOutcome) -> None:
        if self._leap_pending is not None:
            message = "game ended with an unrevealed leap action"
            raise RuntimeError(message)
        self.inner.end_game(outcome)

    def experiment_diagnostics(self) -> dict[str, object]:
        return {
            **self.inner.experiment_diagnostics(),
            "pure_dth_only": False,
            "leap_fallbacks": self.leap_fallbacks,
            "leap_policy": (
                "DTH equilibrium projection; no action-61 optimization claim"
            ),
        }

    def dump_state(self) -> str:
        """Return the opponent evidence as a bounded JSON checkpoint."""
        if self.inner._pending is not None or self._leap_pending is not None:
            message = "cannot checkpoint an unrevealed action"
            raise RuntimeError(message)
        model = self.opponent_model
        roles = {}
        for role, state in model._roles.items():
            fields = asdict(state)
            for key, value in fields.items():
                if isinstance(value, np.ndarray):
                    fields[key] = value.tolist()
            fields["recency_counts"] = [a.tolist() for a in state.recency_counts]
            for key in ("state_counts", "phase_counts"):
                fields[key] = [
                    [list(k), v.tolist()] for k, v in getattr(state, key).items()
                ]
            roles[role] = fields
        data = {
            "schema": MEMORY_SCHEMA,
            "config": asdict(model.config),
            "roles": roles,
            "offsets": {r: a.tolist() for r, a in model._offsets.items()},
            "last_self": model._last_self,
        }
        raw = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > MAX_MEMORY_BYTES:
            message = "opponent checkpoint exceeds size limit"
            raise ValueError(message)
        return base64.b64encode(zlib.compress(raw)).decode("ascii")

    def restore_state(self, memory: str) -> None:
        """Replace the opponent evidence with a checkpoint, or raise ValueError."""
        if not isinstance(memory, str) or len(memory) > MAX_MEMORY_BYTES:
            message = "invalid opponent checkpoint"
            raise ValueError(message)
        try:
            packed = base64.b64decode(memory, validate=True)
            inflater = zlib.decompressobj()
            raw = inflater.decompress(packed, MAX_MEMORY_BYTES + 1)
            if len(raw) > MAX_MEMORY_BYTES or not inflater.eof or inflater.unused_data:
                message = "invalid compressed opponent checkpoint"
                raise ValueError(message)
            data = json.loads(raw)
            model = TranslatedHalOpponentModel(self.config)
            expected_config = json.loads(json.dumps(asdict(model.config)))
            if data["schema"] != MEMORY_SCHEMA or data["config"] != expected_config:
                message = "opponent checkpoint version mismatch"
                raise ValueError(message)

            def array(
                value: object, shape: tuple[int, ...], *, negative: bool = False
            ) -> np.ndarray:
                result = np.asarray(value, dtype=np.float64)
                if (
                    result.shape != shape
                    or not np.all(np.isfinite(result))
                    or (not negative and np.any(result < 0))
                ):
                    message = "invalid opponent evidence array"
                    raise ValueError(message)
                return result.copy()

            def action(value: object) -> int | None:
                if value is not None and (
                    type(value) is not int or not 1 <= value <= ACTION_COUNT
                ):
                    message = "invalid opponent action reference"
                    raise ValueError(message)
                return value

            for role, state in model._roles.items():
                fields = data["roles"][role]
                count = fields["observations"]
                if type(count) is not int or not 0 <= count <= 10**12:
                    message = "invalid opponent observation count"
                    raise ValueError(message)
                state.observations = count
                for key in (
                    "global_counts",
                    "transition_counts",
                    "response_counts",
                    "delta_counts",
                    "log_weights",
                ):
                    setattr(
                        state,
                        key,
                        array(
                            fields[key],
                            getattr(state, key).shape,
                            negative=key == "log_weights",
                        ),
                    )
                recency = fields["recency_counts"]
                if len(recency) != 3:
                    message = "invalid recency evidence"
                    raise ValueError(message)
                state.recency_counts = (
                    array(recency[0], (60,)),
                    array(recency[1], (60,)),
                    array(recency[2], (60,)),
                )
                for key, width in (("state_counts", 4), ("phase_counts", 2)):
                    counts = {}
                    for k, v in fields[key]:
                        if len(k) != width or any(
                            type(i) is not int or i < 0 for i in k
                        ):
                            message = "invalid opponent context"
                            raise ValueError(message)
                        if key == "state_counts" and any(i > 5 for i in k):
                            message = "invalid state band"
                            raise ValueError(message)
                        if key == "phase_counts" and (
                            k[0] not in model.config.periodicities or k[1] >= k[0]
                        ):
                            message = "invalid periodic phase"
                            raise ValueError(message)
                        if tuple(k) in counts:
                            message = "duplicate opponent context"
                            raise ValueError(message)
                        counts[tuple(k)] = array(v, (60,))
                    setattr(state, key, counts)
                for key in ("previous_self_action", "previous_opponent_action"):
                    setattr(state, key, action(fields[key]))
                model._offsets[role] = array(data["offsets"][role], (2, 2, 2, 119))
            model._last_self = action(data["last_self"])
        except (KeyError, TypeError, zlib.error, UnicodeError) as error:
            message = "invalid opponent checkpoint"
            raise ValueError(message) from error
        self.inner.opponent_model = model


def make_hal(kind: str, table: StageReader) -> ExactHal | TranslatedHal:
    """Build Hal of one kind in ``KINDS`` over one opened DTH table."""

    if kind == "exact":
        return ExactHal(table)
    if kind == "translated":
        return TranslatedHal(table)
    message = f"unknown Hal kind {kind!r}; choose one of {', '.join(KINDS)}"
    raise ValueError(message)
