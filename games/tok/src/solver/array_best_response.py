"""Float64 best responses and saddle gap on the public-tree arrays.

This is the CFR stopping test. It is not the certificate: engine_certificate
computes best responses through the rules engine without these arrays.
"""

import numpy as np

from src.solver.public_traversal import Behavior, Choose, reach, values
from src.solver.public_tree import Layer


def best_response_value(layers: list[Layer], i: int, behavior: Behavior, choose: Choose) -> float:
    """Player i's best-response value against the opponent's part of behavior."""
    opp = reach(layers, behavior, choose, 1 - i)
    root = values(layers, i, opp, behavior, lambda d, cfv: cfv.max(axis=1))
    return float(root.max())


def saddle_gap(layers: list[Layer], behavior: Behavior, choose: Choose) -> tuple[float, float]:
    """Return (P0's guarantee, P0's cap) of the profile."""
    lower = -best_response_value(layers, 1, behavior, choose)
    upper = best_response_value(layers, 0, behavior, choose)
    return lower, upper


def profile_value(layers: list[Layer], behavior: Behavior, choose: Choose) -> float:
    """P0's expected payoff when both players follow behavior."""
    opp = reach(layers, behavior, choose, 1)
    root = values(layers, 0, opp, behavior, lambda d, cfv: np.einsum("ij,ij->i", behavior[d], cfv))
    return float(choose[0] @ root)
