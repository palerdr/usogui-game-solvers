# Copyright 2026 The STL authors
"""Build the prebuilt Vercel bundle and ship it as a preview.

``prepare`` rechecks the DTH table, builds the browser client, and writes the
bundle: the client under ``public/``, every ``src/stl`` Python file and the
table under ``runtime/``, an ``app.py`` that calls
``stl.server.hosted.create_production_app``, a ``pyproject.toml`` with the
package pins of the root ``uv.lock``, ``vercel.json`` and the Vercel project
link. Each bundle gets a new session version. Hosted replay discards records
from an older deployment so it cannot replay old games through changed rules.

``ship`` runs ``vercel build``, adds Python 3.13 bytecode to the built
function, and uploads it as a preview. Promote a preview by hand after one
game on it.
"""

from __future__ import annotations

import json
import py_compile
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from uuid import uuid4

from stl.dth import RECHECK_SAMPLES, Table

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "artifacts" / "vercel"
WEB = ROOT / "web"
VERCEL_LINK = ROOT / ".vercel" / "project.json"
# The packages the function installs. Their versions come from the root lock,
# which pins the environment that the server tests run in.
BUNDLE_PACKAGES = ("fastapi", "numpy", "scipy", "httpx", "uvicorn")
# Vercel's Python runtime. The bytecode must come from the same version.
RUNTIME = (3, 13)


def copy_sources(root: Path, runtime: Path) -> list[Path]:
    """Copy each Python file of ``root/src/stl`` to ``runtime/src/stl``.

    Return the copied paths, relative to ``runtime``, in sorted order.
    """
    copied = []
    for source in sorted((root / "src" / "stl").rglob("*.py")):
        relative = source.relative_to(root)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative)
    return copied


def locked_versions(
    lock: Path, names: tuple[str, ...] = BUNDLE_PACKAGES
) -> dict[str, str]:
    """Return the one version that ``lock`` pins for each package in ``names``."""
    found: dict[str, set[str]] = {name: set() for name in names}
    for package in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
        if package.get("name") in found:
            found[package["name"]].add(str(package["version"]))
    for name, versions in found.items():
        if len(versions) != 1:
            message = (
                f"{lock} must pin one version of {name}; it pins {sorted(versions)}"
            )
            raise SystemExit(message)
    return {name: versions.pop() for name, versions in found.items()}


def prepare(artifact: Path, bundle: Path = BUNDLE) -> str:
    """Write the prebuilt bundle to ``bundle`` and return the hosted code version."""
    table = Table.open(artifact, rechecks=RECHECK_SAMPLES)
    opening = table.stage((0, 0, 0, 0))
    print(  # noqa: T201 - CLI progress output
        f"Certified table: opening value={opening.value}, gap={opening.gap}"
    )
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    subprocess.run(["npm", "--prefix", str(WEB), "run", "build"], check=True)  # noqa: S603, S607 - fixed deployment command, no shell; use the installed npm/npx command
    shutil.copytree(WEB / "dist", bundle / "public")
    runtime = bundle / "runtime"
    copy_sources(ROOT, runtime)
    version = uuid4().hex
    table_copy = runtime / "artifacts" / "dth" / "V.npy"
    table_copy.parent.mkdir(parents=True)
    shutil.copy2(artifact, table_copy)
    (bundle / "app.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(root / "runtime/src"))\n'
        "from stl.server.hosted import create_production_app\n"
        'app = create_production_app(root / "runtime/artifacts/dth/V.npy", '
        f'"{version}")\n'
    )
    pins = locked_versions(ROOT / "uv.lock")
    dependencies = ", ".join(f'"{name}=={pin}"' for name, pin in pins.items())
    (bundle / "pyproject.toml").write_text(
        '[project]\nname = "stl-browser"\nversion = "0.1.0"\n'
        f'requires-python = ">={RUNTIME[0]}.{RUNTIME[1]},'
        f'<{RUNTIME[0]}.{RUNTIME[1] + 1}"\n'
        f"dependencies = [{dependencies}]\n"
    )
    config = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": "fastapi",
        "functions": {"app.py": {"maxDuration": 60, "includeFiles": "runtime/**"}},
    }
    (bundle / "vercel.json").write_text(json.dumps(config, indent=2) + "\n")
    if VERCEL_LINK.exists():
        (bundle / ".vercel").mkdir()
        shutil.copy2(VERCEL_LINK, bundle / ".vercel" / "project.json")
    print(f"Prepared {bundle} at version {version}.")  # noqa: T201 - CLI progress output
    return version


def add_bytecode(bundle: Path) -> int:
    """Compile every Python file the built function maps and map the results.

    Vercel ships no bytecode and forbids writing it, so each new process
    compiled numpy, scipy, and FastAPI from source before its first answer.
    `unchecked-hash` files load whatever modification time the bundle gives
    the sources. Run this after `vercel build` and before `vercel deploy`.
    """
    if sys.version_info[:2] != RUNTIME:
        message = f"bytecode must come from Python {RUNTIME[0]}.{RUNTIME[1]}"
        raise SystemExit(message)
    written = 0
    for config in (bundle / ".vercel/output/functions").glob("*.func/.vc-config.json"):
        settings = json.loads(config.read_text())
        mapped = settings["filePathMap"]
        for name, source in list(mapped.items()):
            if not name.endswith(".py") or name.startswith("_vendor/pip/"):
                continue
            tag = f"{Path(name).stem}.{sys.implementation.cache_tag}.pyc"
            compiled = (bundle / source).parent / "__pycache__" / tag
            try:
                py_compile.compile(
                    str(bundle / source),
                    cfile=str(compiled),
                    dfile=f"/var/task/{name}",
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
            except py_compile.PyCompileError:
                # Vendored packages carry template and Python 2 files that never import.
                continue
            mapped[str(Path(name).parent / "__pycache__" / tag)] = str(
                compiled.relative_to(bundle)
            )
            written += 1
        config.write_text(json.dumps(settings, indent=2))
    return written


def ship(bundle: Path = BUNDLE) -> str:
    """Build the bundle with Vercel, add bytecode, deploy a preview, return its URL.

    The prebuilt archive upload is required: a direct upload of the 2.3 GB
    table exceeds the upload request limit.
    """
    subprocess.run(  # noqa: S603 - fixed deployment command, no shell
        ["npx", "vercel", "build", "--cwd", str(bundle), "--yes"],  # noqa: S607 - installed CLI
        check=True,
    )
    print(f"Added {add_bytecode(bundle)} bytecode files to the built function.")  # noqa: T201 - CLI progress output
    deployed = subprocess.run(  # noqa: S603 - fixed deployment command, no shell
        [  # noqa: S607 - use the installed npm/npx command
            "npx",
            "vercel",
            "deploy",
            "--cwd",
            str(bundle),
            "--prebuilt",
            "--archive=tgz",
            "--target",
            "preview",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return deployed.stdout.strip().splitlines()[-1]
