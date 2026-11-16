# Golden fixtures

Source: the Python reference in `games/surpassing-the-leader` at commit `55e342e`,
modules `stl.dth` and `stl.dth_kernel`. `generate.py` in this directory writes every fixture:

```sh
cd games/surpassing-the-leader && uv run python <this directory>/generate.py <this directory>
```

It reads the reference table `artifacts/dth/V.npy` (SHA-256 `e43977b7...29b2`) for the real-class
ladder inputs.

| File | Contents | Checked by |
| --- | --- | --- |
| `profiles.sha256` | SHA-256 of `build_table()`'s `st`, `ttd`, `rev`, `fail_child` and `success_children` as little-endian int32 or float64 | `profiles_match_the_reference_hashes` |
| `ladder.bin` | 1,603 records of 64 little-endian f64: `s[0..60]`, `f`, the numba `_ladder` value and rung, and scipy's `try_rung3` value. Rows 0..300 are uniform in [-1, 1], 300..600 sorted, 600..603 the special stages, and 603.. the stages of 1,000 strided real classes | `ladder_matches_the_numba_kernel_bit_for_bit` (bitwise), `lp_agrees_with_scipy_and_every_rung_brackets_it` (within the 1e-6 gate), `rungs_return_certifying_strategies` |
| `dead_band.bin` | The 7,140 values of `dth_kernel.build(stop_at=1082)`, row-major over `(pc, pd)` | `dead_band_build_matches_the_reference` (bitwise) |
| `wrong_shape.npy` | `numpy.save` of `zeros((3, 4))` | `open_fails_closed_on_the_wrong_shape` |

The LP values agree within the gate and not bitwise: scipy and the `highs` crate bundle different
HiGHS builds, and each LP value is the midpoint of a certificate at most 1e-6 wide that contains
the game value, so two midpoints differ by at most 1e-6. Every other comparison is bitwise.
