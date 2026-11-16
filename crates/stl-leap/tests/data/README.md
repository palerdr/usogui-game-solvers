# Test data

## `leap_kernel_golden.json` and `leap_kernel_golden.bin`

The golden fixture of the leap stage kernel. Commit `13aa898` of the stl history
(`games/stl`) captured it from the retired Rust kernel as
`tests/leap_kernel_golden.npz`; the numba kernel at `55e342e` still reproduces it bit for bit.
It holds seeded stage families and child tables with the values, kinds, failure counts and basis
masks of the square and window sweeps (rungs only, and the production residue options), of the
packing LP from the slack and crash bases, and of the warm and cold packing runs. No test reads
the packing-run arrays: the crate dropped the warm-start path, which the build never used. They
stay so the converted fixture remains a byte copy of the npz.

The `.bin` file concatenates the arrays as raw little-endian bytes, each padded to 8 bytes. The
`.json` file holds the fixture's `meta` record and, per array, its numpy dtype, shape and byte
offset. Command, run in `games/stl` at `55e342e` (npz SHA-256
`29ff0c4d48ace51d7a364f48e26f3f84a0fba10de2364fa683cdb6c26c6a843c`):

```sh
uv run python - crates/stl-leap/tests/data <<'EOF'
import json, sys, numpy as np
out = sys.argv[1]
data = np.load("tests/leap_kernel_golden.npz")
arrays, blob = {}, bytearray()
for name in sorted(data.files):
    if name == "meta":
        continue
    a = np.ascontiguousarray(data[name])
    arrays[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "offset": len(blob)}
    blob += a.astype(a.dtype.newbyteorder("<")).tobytes()
    blob += bytes(-len(blob) % 8)
header = {"meta": json.loads(str(data["meta"])), "arrays": arrays}
open(f"{out}/leap_kernel_golden.json", "w").write(json.dumps(header, indent=1, sort_keys=True) + "\n")
open(f"{out}/leap_kernel_golden.bin", "wb").write(blob)
EOF
```

Replace `crates/stl-leap/tests/data` with the absolute path of this directory.

## Values written into the tests

- `src/rules.rs` `clocks_match_the_referee`: child clocks from `stl.game.Game.resolve_half_round`
  at `55e342e`.
- `src/lp.rs`, `src/support.rs`, `tests/kernel.rs`: the archived stages `REV[2933]`, `H2[46]` and
  the asymmetric-support stage `S`, `F`, `VALUE`, copied from `tests/test_leap.py` and
  `tests/test_leap_kernel.py` at `55e342e`.
- `src/reach.rs` `forward_counts_match_the_reference`: the counts of `test_forward_counts`.
