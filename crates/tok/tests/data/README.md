# Golden fixtures

Source: the Python TOK solver in `games/tok`, commit `f185b24`
("Solve and certify D_3 with array CFR+; test coarse last-layer keys at K=3").

## `reference.json`

Generator: `make_reference.py` in this directory. Command, from `games/tok`:

```sh
PYTHONPATH=. uv run python <this directory>/make_reference.py > <this directory>/reference.json
```

Contents:

| Key | Python source | Rust test |
| --- | --- | --- |
| `sizes` | `public_tree.sizes` for K = 1, 2, 3 | `public_tree.rs` |
| `lp_value` | `array_lp.solve_lp` (method `highs`) for K = 1, 2 | `solvers.rs` |
| `cfr` | `CFRPlus` average after 100 iterations at K = 2: saddle gap, profile value, bead picks, every 101st row of each layer | `solvers.rs` |
| `certificate` | `engine_certificate.certify` of the `CFRPlus` average after 50 (K = 1) and 40 (K = 2) iterations | `certificate.rs` |
| `slot_keys_2` | `public_abstraction.slot_keys` key counts at K = 2 | `abstraction.rs` |
| `slot_keys_3` | key counts recorded in `src/artifacts/abstraction_3_*.json` | `abstraction.rs` |
| `cfr_3_certificate` | lower and upper of `src/artifacts/cfr_3_certificate.json` | `certificate.rs` (ignored) |

## Stored `D_3` profile

`stored_d3_profile_matches_its_certificate` is `#[ignore]`: it reads the stored CFR+ profile
(31,500 iterations, 21 MB and 258 MB) from `artifacts/tok/` at the workspace root. Convert the
Python strategy tables with `npz_to_table.py`, from `games/tok`:

```sh
A=src/artifacts; T=<workspace>/artifacts/tok; mkdir -p $T
uv run python <this directory>/npz_to_table.py $A/cfr_3_p0.npz $T/cfr_3_p0.tbl $A/cfr_3_p1.npz $T/cfr_3_p1.tbl
cargo test -p tok --release --test certificate -- --ignored
```
