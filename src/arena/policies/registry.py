"""Build Hal for the local browser server: exact Hal or translated Hal.

``exact`` plays the certified DTH equilibrium. ``translated`` is the
production opponent model, with the exact policy as its leap-window fallback.
Each factory imports its provider when it runs, so importing the registry
loads no solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arena.agent import PolicyDrivenAgent

KINDS = ("exact", "translated")


def open_table(args: argparse.Namespace):
    from stl.dth import Table

    path = Path(args.dth)
    if not path.is_file():
        raise FileNotFoundError(
            f"the certified DTH table is required at {path}; "
            "build it with: uv run python -m stl.dth_kernel"
        )
    return Table.open(path)


def make_provider(kind: str, args: argparse.Namespace):
    if kind == "exact":
        from arena.dth_adapter import DTHCompletePolicyProvider

        return DTHCompletePolicyProvider(open_table(args))
    if kind == "translated":
        from arena.translated_hal_adapter import TranslatedHalPolicyProvider

        return TranslatedHalPolicyProvider(open_table(args))
    raise ValueError(f"unknown Hal kind {kind!r}; choose one of {', '.join(KINDS)}")


def make_hal(args: argparse.Namespace) -> PolicyDrivenAgent:
    return PolicyDrivenAgent(
        make_provider(args.hal_agent, args), player_name="Hal", seed=args.seed
    )


def add_play_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the DTH table option that both Hal kinds read."""
    from stl.dth import DEFAULT_PATH

    parser.add_argument(
        "--dth",
        default=str(DEFAULT_PATH),
        help="the certified DTH table V.npy that python -m stl.dth_kernel writes",
    )
