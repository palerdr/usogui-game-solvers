"""The Vercel bundle: its copied runtime, its code version and its package pins.

The runtime check copies the sources as ``prepare`` does and runs the
production entry from that copy in a fresh interpreter, because this test
process already holds numba from other tests. The run builds the hosted app
with an in-memory store and a stand-in table, then plays one half-round
through a restart, so a module that a request imports counts too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from stl.server.deploy import BUNDLE_PACKAGES, ROOT, code_version, copy_sources, locked_versions

# Modules the hosted function must never load: the numba kernels and the
# training stack.
FORBIDDEN_MODULES = ("numba", "llvmlite", "torch", "gymnasium", "stable_baselines3", "sb3_contrib")

_RUN_PRODUCTION = """
import json, sys, types
from pathlib import Path

runtime = Path(sys.argv[1])
sys.path.insert(0, str(runtime / "src"))

import numpy as np
from fastapi.testclient import TestClient

from stl.dth import Stage
from stl.hal import TranslatedHal
from stl.server import hosted as production


class Table:
    def stage(self, state):
        uniform = np.full(60, 1.0 / 60)
        return Stage(
            state=tuple(int(value) for value in state), value=1.0 / 60, matrix=np.eye(60),
            drop=uniform, check=uniform.copy(), gap=0.0,
        )


class Store:
    def __init__(self):
        self.rows = {}

    async def get(self, key):
        return self.rows.get(key)

    async def compare_set(self, key, old, new):
        if self.rows.get(key) != old:
            return False
        self.rows[key] = new
        return True


assert TranslatedHal(object()).config.offset_retention == 0.97
production.Table = types.SimpleNamespace(open=lambda *args, **kwargs: Table())
production.RedisSessionStore = lambda *args: Store()
app = production.create_production_app(Path("unused"), "closure")
with TestClient(app, base_url="https://testserver") as client:
    def post(path, state, **values):
        response = client.post(path, json={"sequence": state["sequence"], **values})
        assert response.status_code == 200, response.text
        return response.json()

    health = client.get("/api/health").json()
    assert health == {"status": "ok", "policy": "translated-hal-v1", "version": "closure"}
    assert client.get("/api/rules").status_code == 200
    state = post("/api/session/begin", client.get("/api/session").json())
    state = post("/api/session/action", state, second=state["legal_seconds"][0])
    state = post("/api/session/ack", state)
    post("/api/session/restart", state)
    assert client.get("/api/transcript").status_code == 200

files = sorted(
    str(Path(module.__file__).resolve())
    for name, module in list(sys.modules.items())
    if (name == "stl" or name.startswith("stl.")) and getattr(module, "__file__", None)
)
print(json.dumps({"stl_files": files, "modules": sorted(sys.modules)}))
"""


@pytest.fixture(scope="module")
def production_run(tmp_path_factory) -> dict[str, object]:
    runtime = tmp_path_factory.mktemp("bundle") / "runtime"
    copy_sources(ROOT, runtime)
    environment = dict(os.environ)
    for name in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", _RUN_PRODUCTION, str(runtime)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return {"runtime": runtime, **json.loads(result.stdout.strip().splitlines()[-1])}


def test_the_production_entry_runs_from_the_copied_sources(production_run) -> None:
    runtime = production_run["runtime"].resolve()
    outside = [path for path in production_run["stl_files"] if not path.startswith(str(runtime))]
    assert production_run["stl_files"] and outside == []


def test_the_production_entry_loads_no_numba_and_no_training_stack(production_run) -> None:
    loaded = {name.split(".", 1)[0] for name in production_run["modules"]}
    assert sorted(loaded & set(FORBIDDEN_MODULES)) == []


def test_the_copy_holds_every_package_source_and_nothing_else(tmp_path) -> None:
    copied = copy_sources(ROOT, tmp_path)
    expected = sorted(path.relative_to(ROOT) for path in (ROOT / "src" / "stl").rglob("*.py"))
    assert copied == expected
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()) == expected


def test_the_version_covers_each_source_and_the_table(tmp_path) -> None:
    sources = copy_sources(ROOT, tmp_path)
    version = code_version(tmp_path, sources, "table-a")
    assert code_version(tmp_path, sources, "table-a") == version
    assert code_version(tmp_path, sources, "table-b") != version
    app = tmp_path / "src" / "stl" / "server" / "app.py"
    app.write_text(app.read_text() + "\n# a word-list change\n")
    assert code_version(tmp_path, sources, "table-a") != version


def test_the_bundle_pins_one_locked_version_per_package(tmp_path) -> None:
    pins = locked_versions(ROOT / "uv.lock")
    assert sorted(pins) == sorted(BUNDLE_PACKAGES)
    assert all(pins.values())
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "numpy"\nversion = "2.0.0"\n'
        '[[package]]\nname = "numpy"\nversion = "2.1.0"\n'
    )
    with pytest.raises(SystemExit, match="one version of numpy"):
        locked_versions(lock, ("numpy",))
