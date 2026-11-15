"""Write reference.json from the Python solver. See README.md for the command."""
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.solver.array_best_response import profile_value, saddle_gap
from src.solver.array_lp import build_sequence_form, solve_lp
from src.solver.cfr_plus import CFRPlus
from src.solver.engine_certificate import certify
from src.solver.public_abstraction import slot_keys
from src.solver.public_tree import build_public_tree, sizes
from src.solver.strategy_table import export_table

ROW_STRIDE = 101


def cfr(horizon, iterations):
    layers = build_public_tree(horizon)
    solver = CFRPlus(layers)
    for _ in range(iterations):
        solver.iterate()
    return layers, solver.average()


out = {"sizes": {k: asdict(sizes(build_public_tree(k))) for k in (1, 2, 3)}, "lp_value": {}}
for k in (1, 2):
    layers = build_public_tree(k)
    out["lp_value"][k] = solve_lp(build_sequence_form(layers), layers).value

layers, (behavior, choose) = cfr(2, 100)
lower, upper = saddle_gap(layers, behavior, choose)
out["cfr"] = {
    "horizon": 2, "iterations": 100, "lower": lower, "upper": upper,
    "value": profile_value(layers, behavior, choose),
    "choose": [c.tolist() for c in choose],
    "rows": [[d, s, b[s].tolist()] for d, b in enumerate(behavior) for s in range(0, len(b), ROW_STRIDE)],
}
if len(sys.argv) > 1:  # full average behavior for a tolerance study, as raw float64
    for d, b in enumerate(behavior):
        b.astype("<f8").tofile(f"{sys.argv[1]}/cfr2_layer{d}.f64")

out["certificate"] = []
for k, iterations in ((1, 50), (2, 40)):
    layers, (behavior, choose) = cfr(k, iterations)
    with tempfile.TemporaryDirectory() as tmp:
        paths = [str(Path(tmp) / f"p{p}.npz") for p in (0, 1)]
        for p in (0, 1):
            export_table(layers, behavior, choose, p).save(paths[p])
        lower, upper = certify(k, *paths)
    out["certificate"].append({"horizon": k, "iterations": iterations, "lower": lower, "upper": upper})

layers = build_public_tree(2)
out["slot_keys_2"] = [
    {name: slot_keys(layers, p, name)[1] for name in ("full", "support", "hard", "opening+hard")} for p in (0, 1)
]
out["slot_keys_3"] = [{}, {}]  # recorded by order_blind_k3.py in the abstraction_3_* artifacts
for path in sorted(Path("src/artifacts").glob("abstraction_3_*.json")):
    record = json.loads(path.read_text())
    out["slot_keys_3"][record["player"]] |= {"full": record["infosets"], record["key"]: record["keys"]}
certificate = json.loads(Path("src/artifacts/cfr_3_certificate.json").read_text())
out["cfr_3_certificate"] = {"lower": certificate["lower"], "upper": certificate["upper"]}
print(json.dumps(out))
