"""Build the Bend DTH solver with the pinned CPU float64 backend."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from backend import lower_literals, patch

HERE = Path(__file__).resolve().parent
RELEASE = "bend-2.0.40"
URL = "https://codeload.github.com/bendlang/bend/tar.gz/refs/tags/v2.0.40"
SHA256 = "69461d93f626bf73e0e90bb4d85f6117a4ee2cdb70b2c9c0957e5000ca541cc6"


def build(output: Path, archive: Path | None) -> None:
    """Check and emit the solver using Bend, then link its native table driver."""
    bun = os.environ.get("BUN") or shutil.which("bun")
    runner = [bun] if bun else ["npx", "--yes", "bun@1.4.5"]
    cc = os.environ.get("CC", "clang")
    with tempfile.TemporaryDirectory(prefix="stl-bend-") as directory:
        work = Path(directory)
        if archive is None:
            archive = work / "bend.tar.gz"
            with (
                urllib.request.urlopen(URL, timeout=60) as response,
                archive.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        with archive.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != SHA256:
                raise ValueError("Bend compiler archive failed SHA-256 verification")
            source.seek(0)
            with tarfile.open(fileobj=source, mode="r:gz") as tar:
                members = (
                    entry for entry in tar if entry.name.startswith(f"{RELEASE}/bend2/")
                )
                tar.extractall(work, members=members, filter="data")
        compiler = work / RELEASE / "bend2"
        patch(compiler)
        for name in ("profiles.bend", "stage.bend"):
            (work / name).write_text(lower_literals((HERE / name).read_text()))
        for name in ("link.bend", "bridge.c", "bridge.h"):
            shutil.copyfile(HERE / name, work / name)
        subprocess.run(
            [*runner, str(compiler / "main.ts"), "link.bend", "-o", "dth.c"],
            cwd=work,
            env=os.environ | {"BEND_NO_TELEMETRY": "1"},
            check=True,
        )
        options = ["-O3", "-ffp-contract=off", "-pthread"]
        subprocess.run(
            [
                cc,
                *options,
                "-Dmain=bend_dth_emitted_main",
                "-c",
                "dth.c",
                "-o",
                "dth.o",
            ],
            cwd=work,
            check=True,
        )
        subprocess.run(
            [cc, *options, str(HERE / "sweep.c"), "dth.o", "-lm", "-o", str(output)],
            cwd=work,
            check=True,
        )


def main() -> None:
    """Build an executable at the path named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source", type=Path, help="local copy of the pinned Bend release archive"
    )
    args = parser.parse_args()
    try:
        build(args.output.resolve(), args.source)
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
