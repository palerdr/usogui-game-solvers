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
    from stl import dth_kernel
    from stl.dth import DEFAULT_PATH, build_table, verify_anchors

    table = build_table()
    V, K, counts = dth_kernel.build(table)
    dth_kernel.finalize(V, K, table)
    for state, expected, got, ok in verify_anchors(V, table):
        print(f"anchor {state}: expected {expected:.16f} got {got:.16f} {'ok' if ok else 'MISMATCH'}")
    print(f"pure={counts[0]} support={counts[1]} lp={counts[2]} total={counts.sum()}")
    dth_kernel.write(V, output or DEFAULT_PATH)


def leap(dth_path: Path | None, output: Path | None) -> None:
    """Build, or resume, the full certified leap sweep to the clock-720 opening."""
    from stl.dth import DEFAULT_PATH
    from stl.leap import DEFAULT_ARTIFACT, run_full

    run_full(dth_path or DEFAULT_PATH, output or DEFAULT_ARTIFACT)


def serve(hal: str, host: str, port: int, seed: int | None) -> None:
    """Serve the built browser client and play against Hal on this machine."""
    import uvicorn

    from stl.dth import DEFAULT_PATH, Table
    from stl.hal import make_hal
    from stl.server.app import SeriesConfig, SessionConfig, create_app
    from stl.server.session import PolicyDrivenAgent

    if not DEFAULT_PATH.is_file():
        raise SystemExit(
            f"the certified DTH table is required at {DEFAULT_PATH}; "
            "build it with: uv run python -m stl dth"
        )
    provider = make_hal(hal, Table.open(DEFAULT_PATH))
    app = create_app(
        hal_factory=lambda: PolicyDrivenAgent(provider, player_name="Hal", seed=seed),
        config=SessionConfig(seed=seed),
        series=SeriesConfig(hal_agent=hal),
    )
    print(f"Surpassing The Leader: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def deploy(artifact: Path | None) -> None:
    """Build the Vercel bundle, add 3.13 bytecode, and ship a preview."""
    from stl.dth import DEFAULT_PATH
    from stl.server import deploy as pipeline

    if sys.version_info[:2] != pipeline.RUNTIME:
        raise SystemExit(
            "deploy needs Python 3.13: uv run --isolated --python 3.13 python -m stl deploy"
        )
    version = pipeline.prepare(artifact or DEFAULT_PATH)
    url = pipeline.ship()
    print(f"Preview {url} serves version {version}.")
    print(f"After /api/health and one game there: npx vercel promote {url} --yes")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m stl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("dth", help=dth.__doc__, description=dth.__doc__)
    command.add_argument("--output", type=Path, help="default: artifacts/dth/V.npy")

    command = commands.add_parser("leap", help=leap.__doc__, description=leap.__doc__)
    command.add_argument("--dth", type=Path, help="default: artifacts/dth/V.npy")
    command.add_argument("--output", type=Path, help="default: artifacts/leap")

    command = commands.add_parser("serve", help=serve.__doc__, description=serve.__doc__)
    command.add_argument("--hal", choices=("translated", "exact"), default="translated")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--seed", type=int, help="game N of the series uses seed + N")

    command = commands.add_parser("deploy", help=deploy.__doc__, description=deploy.__doc__)
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
