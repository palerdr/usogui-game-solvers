"""Prepare the browser and certified artifact for a prebuilt Vercel deployment."""

from __future__ import annotations

import json
import argparse
import hashlib
import py_compile
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from browser.deploy.manifest import RUNTIME_FILES
from stl.dth import DEFAULT_PATH, RECHECK_SAMPLES, Table

# The bundle, the client, and the Vercel project link, by repository path.
BUNDLE = "src/browser/build/vercel"
WEBCLIENT = "src/browser/webclient"
VERCEL_LINK = "src/browser/.vercel/project.json"
# The packages the function installs. Their versions come from the browser
# project's lock, which pins the environment that the browser tests run in.
BROWSER_LOCK = "src/browser/uv.lock"
BUNDLE_PACKAGES = ("fastapi", "numpy", "scipy", "httpx", "uvicorn")


def add_bytecode(target: Path) -> int:
    """Compile every Python file the built function maps and map the results.

    Vercel ships no bytecode and forbids writing it, so each new process
    compiled numpy, scipy, and FastAPI from source before its first answer.
    `unchecked-hash` files load whatever modification time the bundle gives
    the sources. Run this after `vercel build` and before `vercel deploy`.
    """
    runtime = (3, 13)
    if sys.version_info[:2] != runtime:
        raise SystemExit(f"bytecode must come from Python {runtime[0]}.{runtime[1]}")
    written = 0
    for config in (target / ".vercel/output/functions").glob("*.func/.vc-config.json"):
        settings = json.loads(config.read_text())
        mapped = settings["filePathMap"]
        for name, source in list(mapped.items()):
            if not name.endswith(".py") or name.startswith("_vendor/pip/"):
                continue
            tag = f"{Path(name).stem}.{sys.implementation.cache_tag}.pyc"
            compiled = (target / source).parent / "__pycache__" / tag
            try:
                py_compile.compile(
                    str(target / source),
                    cfile=str(compiled),
                    dfile=f"/var/task/{name}",
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
            except py_compile.PyCompileError:
                # Vendored packages carry template and Python 2 files that never import.
                continue
            mapped[str(Path(name).parent / "__pycache__" / tag)] = str(
                compiled.relative_to(target)
            )
            written += 1
        config.write_text(json.dumps(settings, indent=2))
    return written


def copy_runtime_sources(root: Path, target: Path) -> None:
    """Copy each file of the runtime manifest to ``runtime/`` in the bundle.

    The copy keeps each file's path from the repository root.
    """
    for entry in RUNTIME_FILES:
        destination = target / "runtime" / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / entry.path, destination)


def locked_versions(lock: Path, names: tuple[str, ...] = BUNDLE_PACKAGES) -> dict[str, str]:
    """Return the one version that ``lock`` pins for each package in ``names``."""
    found: dict[str, set[str]] = {name: set() for name in names}
    for package in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
        if package.get("name") in found:
            found[package["name"]].add(str(package["version"]))
    for name, versions in found.items():
        if len(versions) != 1:
            raise SystemExit(f"{lock} must pin one version of {name}; it pins {sorted(versions)}")
    return {name: versions.pop() for name, versions in found.items()}


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--bytecode",
        action="store_true",
        help="after `vercel build`: add compiled bytecode to the built function",
    )
    options = parser.parse_args()
    if options.bytecode:
        count = add_bytecode(root / BUNDLE)
        print(f"Added {count} bytecode files to the built function.")
        return
    table = Table.open(options.artifact, rechecks=RECHECK_SAMPLES)
    opening = table.stage((0, 0, 0, 0))
    with options.artifact.open("rb") as file:
        dth_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    print(
        f"Certified table: opening value={opening.value}, gap={opening.gap}, "
        f"sha256={dth_sha256}"
    )
    target = root / BUNDLE
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(
        ["npm", "--prefix", str(root / WEBCLIENT), "run", "build"],
        check=True,
    )
    shutil.copytree(root / WEBCLIENT / "dist", target / "public")
    copy_runtime_sources(root, target)
    artifact = target / "runtime/artifacts/dth/V.npy"
    artifact.parent.mkdir(parents=True)
    shutil.copy2(options.artifact, artifact)
    (target / "app.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(root / "runtime/src"))\n'
        "from browser.deploy.production import create_production_app\n"
        f'app = create_production_app(root / "runtime/artifacts/dth/V.npy", "{dth_sha256}")\n'
    )
    pins = locked_versions(root / BROWSER_LOCK)
    dependencies = ", ".join(f'"{name}=={version}"' for name, version in pins.items())
    (target / "pyproject.toml").write_text(
        '[project]\nname = "stl-browser"\nversion = "0.1.0"\n'
        'requires-python = ">=3.13,<3.14"\n'
        f"dependencies = [{dependencies}]\n"
    )
    config = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": "fastapi",
        "functions": {"app.py": {"maxDuration": 60, "includeFiles": "runtime/**"}},
    }
    (target / "vercel.json").write_text(json.dumps(config, indent=2) + "\n")
    link = root / VERCEL_LINK
    if link.exists():
        (target / ".vercel").mkdir()
        shutil.copy2(link, target / ".vercel/project.json")
    print(
        f"Prepared {target}. Use vercel build, then this command with --bytecode, "
        "then vercel deploy --prebuilt --archive=tgz."
    )


if __name__ == "__main__":
    main()
