"""Certified DTH table adapter for the canonical STL arena.

The four load/TTD coordinates and every ordinary action are identical between
pure DTH and canonical STL. The sole prospective rules difference is the
public leap window: Baku as Dropper may play 61 there, while a DTH policy still
supplies actions 1..60. Arena owns legality and sampling around that one extra
canonical action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from arena.contracts import CanonicalDecision
from stl.dth import Stage, Table


def project_to_dth_state(decision: CanonicalDecision) -> tuple[int, int, int, int]:
    """Project the shared literal-second coordinates without approximation."""

    coordinates = (
        decision.checker_cylinder_seconds,
        decision.checker_ttd_seconds,
        decision.dropper_cylinder_seconds,
        decision.dropper_ttd_seconds,
    )
    if any(value != int(value) for value in coordinates):
        raise ValueError("canonical STL state is not on the literal-second DTH grid")
    return tuple(int(value) for value in coordinates)  # type: ignore[return-value]


@dataclass
class DTHCompletePolicyProvider:
    """Serve exact equilibrium policies from the certified DTH table.

    ``table`` is one opened ``stl.dth.Table``; the hosted runtime shares it
    between every player. Each decision records its stage and latency. With
    ``record_decisions`` false the provider keeps no per-move record, so a
    long-lived provider does not grow.
    """

    table: Table = field(repr=False)
    decisions: list[tuple[Stage, float]] = field(default_factory=list, repr=False)
    record_decisions: bool = True

    def close(self) -> None:
        """Match the provider lifecycle contract; memmaps need no explicit close."""

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
