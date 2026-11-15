"""Convert Python StrategyTable .npz files into the table format of tok::table.

Run: uv run python npz_to_table.py IN.npz OUT.tbl [IN.npz OUT.tbl ...]
"""
import sys

import numpy as np

for source, target in zip(sys.argv[1::2], sys.argv[2::2]):
    data = np.load(source)
    keys = data["keys"].astype("<u8")
    with open(target, "wb") as f:
        f.write(b"TOKTABLE")
        f.write(np.uint64(len(keys)).astype("<u8").tobytes())
        for name, array in (("choose", data["choose"]), ("keys", keys), ("probs", data["probs"])):
            f.write(np.ascontiguousarray(array, dtype="<u8" if name == "keys" else "<f8").tobytes())
