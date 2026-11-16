"""Write the stl-dth golden fixtures from the Python reference (stl at 55e342e).

cd games/surpassing-the-leader && uv run python <this file> <output directory>
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

from stl import dth as m
from stl import dth_kernel as k

out = Path(sys.argv[1])
t = m.build_table()
arrays = {"st": t.st, "ttd": t.ttd, "rev": t.rev, "fail": t.fail_child, "succ": t.success_children}
(out / "profiles.sha256").write_text(
    "".join(f"{hashlib.sha256(a.tobytes()).hexdigest()}  {n}\n" for n, a in arrays.items())
)

rng = np.random.default_rng(5)
uniform = np.hstack([rng.uniform(-1, 1, (300, 60)), rng.uniform(-1, 1, (300, 1))])
sorted_ = np.hstack([np.sort(rng.uniform(-1, 1, (300, 60)), axis=1), rng.uniform(-1, 1, (300, 1))])
special = np.array([[0.5] * 61, [1.0] * 61, [*np.linspace(-1, 1, 60), -1.0]])
V = np.load(m.DEFAULT_PATH, mmap_mode="r")
real = np.array([
    np.append(*m.class_values(*m.decode_class(c), V, t))
    for c in np.linspace(0, m.CLASSES - 1, 1000).astype(np.int64)
])
rows = []
for x in np.vstack([uniform, sorted_, special, real]):
    value, kind = k._ladder(x[:60].copy(), float(x[60]), np.empty(60), np.empty(60))
    lp = m.try_rung3(x[:60], float(x[60]))
    rows.append([*x, value, kind, lp.value if lp.certified else np.nan])
np.asarray(rows, "<f8").tofile(out / "ladder.bin")

values, _, _ = k.build(t, stop_at=1082, log=lambda _: None)
core = values[:, : m.N]
core[~np.isnan(core)].tofile(out / "dead_band.bin")
np.save(out / "wrong_shape.npy", np.zeros((3, 4)))
