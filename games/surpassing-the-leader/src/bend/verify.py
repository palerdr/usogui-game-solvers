"""Compare a Bend DTH build with the frozen reference, including signed zero."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROFILES = 17_011
DEAD = 16_711
TABLE_SHAPE = (PROFILES, PROFILES + 1)
DEAD_BAND_SIZE = 7_140
DEAD_ANCHOR = DEAD + 240


def load_table(path: Path) -> np.ndarray:
    """Map a table with the reference shape and byte layout."""
    table = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(table, np.ndarray)
        or table.shape != TABLE_SHAPE
        or table.dtype != np.dtype("<f8")
        or not table.flags.c_contiguous
    ):
        raise ValueError(
            f"{path}: expected C-order little-endian float64 shape {TABLE_SHAPE}"
        )
    return table


def compare(
    candidate: np.ndarray, reference: np.ndarray, dead_keys: np.ndarray | None
) -> bool:
    """Scan all cells in bounded chunks and report exact and numerical differences."""
    mismatches = 0
    invalid = 0
    max_delta = 0.0
    first_mismatch = None
    first_invalid = None
    for start in range(0, len(candidate), 128):
        got = candidate[start : start + 128]
        want = reference[start : start + 128]
        changed = got.view("<u8") != want.view("<u8")
        bad = (
            ~np.isfinite(got)
            | ~np.isfinite(want)
            | (np.abs(got) > 1.0 + 1e-9)
            | (np.abs(want) > 1.0 + 1e-9)
        )
        if dead_keys is None:
            bad[:, -1] |= (got[:, -1] != -1.0) | (want[:, -1] != -1.0)
        bad_count = int(np.count_nonzero(bad))
        changed_count = int(np.count_nonzero(changed))
        invalid += bad_count
        mismatches += changed_count
        if first_invalid is None and bad_count:
            first_invalid = np.unravel_index(int(np.argmax(bad)), bad.shape)
            first_invalid = (start + first_invalid[0], *first_invalid[1:])
        if changed_count:
            if first_mismatch is None:
                key = np.unravel_index(int(np.argmax(changed)), changed.shape)
                first_mismatch = (start + key[0], *key[1:])
            with np.errstate(invalid="ignore", over="ignore"):
                delta = np.abs(got[changed] - want[changed])
            max_delta = max(
                max_delta, float(np.max(np.where(np.isfinite(delta), delta, np.inf)))
            )

    print(
        f"checked={candidate.size:,} bitwise_mismatches={mismatches:,} "
        f"max_delta={max_delta:.17g}"
    )
    for label, key in (
        ("first_mismatch", first_mismatch),
        ("first_invalid", first_invalid),
    ):
        if key is not None:
            profile_ids = tuple(dead_keys[key[0]]) if dead_keys is not None else key
            pc, pd = map(int, profile_ids)
            print(
                f"{label} profiles=({pc},{pd}) candidate={candidate[key]} "
                f"reference={reference[key]}"
            )
    if invalid:
        print(f"invalid_cells={invalid:,}")
    return mismatches == 0 and invalid == 0


def main() -> int:
    """Verify a complete value table or a dead-band build from CLI paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Bend table in NumPy format")
    parser.add_argument("reference", type=Path, help="reference .npy or dead_band.bin")
    parser.add_argument(
        "--dead-band", action="store_true", help="verify just layers 1082 and above"
    )
    args = parser.parse_args()

    try:
        candidate = load_table(args.candidate)
        dead_keys = None
        if args.dead_band:
            if (
                args.reference.stat().st_size
                != DEAD_BAND_SIZE * np.dtype("<f8").itemsize
            ):
                raise ValueError(
                    f"{args.reference}: expected {DEAD_BAND_SIZE:,} float64 values"
                )
            reference = np.fromfile(args.reference, dtype="<f8")
            sc, sd = np.nonzero(np.add.outer(np.arange(300), np.arange(300)) >= 480)
            dead_keys = np.column_stack((sc + DEAD, sd + DEAD))
            candidate = candidate[dead_keys[:, 0], dead_keys[:, 1]]
            index = int(np.flatnonzero((dead_keys == DEAD_ANCHOR).all(axis=1))[0])
            print(
                f"dead_band_anchor={candidate[index]:.17g} "
                f"reference={reference[index]:.17g}"
            )
        else:
            reference = load_table(args.reference)
            print(f"root={candidate[0, 0]:.17g} reference={reference[0, 0]:.17g}")
        return 0 if compare(candidate, reference, dead_keys) else 1
    except (EOFError, OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
