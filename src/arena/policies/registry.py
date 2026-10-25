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

DEFAULT_DTH_COMPLETE_TABLEBASE = "src/dth/artifacts/complete_full_v1"
KINDS = ("exact", "translated")


def dth_artifact_dir(args: argparse.Namespace) -> Path:
    artifact_dir = Path(args.dth_complete_tablebase)
    manifest = artifact_dir / "tablebase.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"complete DTH tablebase is required at {artifact_dir}; "
            "build it with: uv run python -m dth complete"
        )
    return artifact_dir


def make_provider(kind: str, args: argparse.Namespace):
    if kind == "exact":
        from arena.dth_adapter import DTHCompletePolicyProvider

        return DTHCompletePolicyProvider(artifact_dir=dth_artifact_dir(args))
    if kind == "translated":
        from arena.translated_hal_adapter import TranslatedHalPolicyProvider

        return TranslatedHalPolicyProvider(dth_artifact_dir(args))
    raise ValueError(f"unknown Hal kind {kind!r}; choose one of {', '.join(KINDS)}")


def make_hal(args: argparse.Namespace) -> PolicyDrivenAgent:
    return PolicyDrivenAgent(
        make_provider(args.hal_agent, args), player_name="Hal", seed=args.seed
    )


def add_play_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the tablebase option that both Hal kinds read."""

    parser.add_argument(
        "--dth-complete-tablebase",
        default=DEFAULT_DTH_COMPLETE_TABLEBASE,
        metavar="DTH_COMPLETE_TABLEBASE",
        help="completed exact DTH quotient tablebase directory",
    )
