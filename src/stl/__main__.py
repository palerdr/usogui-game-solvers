# Copyright 2026 The STL authors
"""The four commands of python -m stl: dth, leap, serve and deploy.

Each command imports its modules when it runs, so serve never loads numba and
--help loads nothing heavy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def dth(output: Path | None) -> None:
    """Build the certified DTH table, check the six anchors, and write V.npy."""
    from stl import dth_kernel  # noqa: PLC0415 - load only the selected CLI runtime
    from stl.dth import DEFAULT_PATH, build_table, verify_anchors  # noqa: PLC0415 - load only the selected CLI runtime

    table = build_table()
    values, kinds, counts = dth_kernel.build(table)
    dth_kernel.finalize(values, kinds, table)
    for state, expected, got, ok in verify_anchors(values, table):
        print(  # noqa: T201 - CLI progress output
            f"anchor {state}: expected {expected:.16f} got {got:.16f} "
            f"{'ok' if ok else 'MISMATCH'}"
        )
    print(f"pure={counts[0]} support={counts[1]} lp={counts[2]} total={counts.sum()}")  # noqa: T201 - CLI progress output
    dth_kernel.write(values, output or DEFAULT_PATH)


def leap(dth_path: Path | None, output: Path | None) -> None:
    """Build the full certified leap sweep in an empty output directory."""
    from stl.dth import DEFAULT_PATH  # noqa: PLC0415 - load only the selected CLI runtime
    from stl.leap import DEFAULT_ARTIFACT, run_full  # noqa: PLC0415 - load only the selected CLI runtime

    run_full(dth_path or DEFAULT_PATH, output or DEFAULT_ARTIFACT)


def serve(hal: str, host: str, port: int, seed: int | None) -> None:
    """Serve the built browser client and play against Hal on this machine."""
    import uvicorn  # noqa: PLC0415 - load only the selected CLI runtime

    from stl.dth import DEFAULT_PATH, Table  # noqa: PLC0415 - load only the selected CLI runtime
    from stl.hal import make_hal  # noqa: PLC0415 - load only the selected CLI runtime
    from stl.server.app import SeriesConfig, SessionConfig, create_app  # noqa: PLC0415 - load only the selected CLI runtime
    from stl.server.session import PolicyDrivenAgent  # noqa: PLC0415 - load only the selected CLI runtime

    if not DEFAULT_PATH.is_file():
        message = (
            f"the certified DTH table is required at {DEFAULT_PATH}; "
            "build it with: uv run python -m stl dth"
        )
        raise SystemExit(message)
    provider = make_hal(hal, Table.open(DEFAULT_PATH))
    app = create_app(
        hal_factory=lambda: PolicyDrivenAgent(provider, player_name="Hal", seed=seed),
        config=SessionConfig(seed=seed),
        series=SeriesConfig(hal_agent=hal),
    )
    print(f"Surpassing The Leader: http://{host}:{port}")  # noqa: T201 - CLI progress output
    uvicorn.run(app, host=host, port=port, log_level="info")


def deploy(artifact: Path | None) -> None:
    """Build the Vercel bundle, add 3.13 bytecode, and ship a preview."""
    from stl.dth import DEFAULT_PATH  # noqa: PLC0415 - load only the selected CLI runtime
    from stl.server import deploy as pipeline  # noqa: PLC0415 - load only the selected CLI runtime

    if sys.version_info[:2] != pipeline.RUNTIME:
        message = (
            "deploy needs Python 3.13: uv run --isolated --python 3.13 "
            "python -m stl deploy"
        )
        raise SystemExit(message)
    version = pipeline.prepare(artifact or DEFAULT_PATH)
    url = pipeline.ship()
    print(f"Preview {url} serves version {version}.")  # noqa: T201 - CLI progress output
    print(  # noqa: T201 - CLI progress output
        "Run the preview smoke checks, then follow README.md's "
        "staged-production procedure and repeat the checks before promotion."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m stl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("dth", help=dth.__doc__, description=dth.__doc__)
    command.add_argument("--output", type=Path, help="default: artifacts/dth/V.npy")

    command = commands.add_parser("leap", help=leap.__doc__, description=leap.__doc__)
    command.add_argument("--dth", type=Path, help="default: artifacts/dth/V.npy")
    command.add_argument("--output", type=Path, help="default: artifacts/leap")

    command = commands.add_parser(
        "serve", help=serve.__doc__, description=serve.__doc__
    )
    command.add_argument("--hal", choices=("translated", "exact"), default="translated")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--seed", type=int, help="game N of the series uses seed + N")

    command = commands.add_parser(
        "deploy", help=deploy.__doc__, description=deploy.__doc__
    )
    command.add_argument("--artifact", type=Path, help="default: artifacts/dth/V.npy")

    args = parser.parse_args(argv)
    if args.command == "dth":
        dth(args.output)
    elif args.command == "leap":
        leap(args.dth, args.output)
    elif args.command == "serve":
        serve(args.hal, args.host, args.port, args.seed)
    else:
        deploy(args.artifact)


if __name__ == "__main__":
    main()
