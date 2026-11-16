"""Save and load solved profiles under src/artifacts."""

import json

import numpy as np

from src.solver.array_lp import build_sequence_form, plan_behavior
from src.solver.cfr_plus import CFRPlus, normalize
from src.solver.public_traversal import Behavior, Choose
from src.solver.public_tree import Layer

ROOT = "src/artifacts"


def save_cfr(cfr: CFRPlus, path: str) -> None:
    arrays = {f"regret{d}": r for d, r in enumerate(cfr.regret)}
    arrays |= {f"weight{d}": w for d, w in enumerate(cfr.weight)}
    np.savez(path, iterations=cfr.iterations, choose_regret=np.array(cfr.choose_regret),
             choose_weight=np.array(cfr.choose_weight), **arrays)


def load_profile(layers: list[Layer], horizon: int, source: str) -> tuple[Behavior, Choose]:
    """The LP plans or the CFR+ average as behavior on the arrays."""
    data = np.load(f"{ROOT}/{source}_{horizon}.npz")
    if source == "lp":
        form = build_sequence_form(layers)
        behavior, choose = [None] * len(layers), [None, None]
        plan_behavior(form, layers, 0, data["p0"], behavior, choose)
        plan_behavior(form, layers, 1, data["p1"], behavior, choose)
        return behavior, (choose[0], choose[1])
    behavior = [normalize(data[f"weight{d}"]) for d in range(len(layers))]
    return behavior, (normalize(data["choose_weight"][0]), normalize(data["choose_weight"][1]))


def save_certificate(horizon: int, source: str, record: dict) -> None:
    with open(f"{ROOT}/{source}_{horizon}_certificate.json", "w") as f:
        json.dump(record, f, indent=1)


def load_certificate(horizon: int, source: str) -> dict:
    with open(f"{ROOT}/{source}_{horizon}_certificate.json") as f:
        return json.load(f)
