"""The hosted browser runtime imports without the training stack.

The check runs in a fresh interpreter, because this test process may already
hold torch from another test module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]

# The hosted providers, the translated opponent memory, and the Vercel entry.
RUNTIME_MODULES = (
    "arena.policies.perfect_hal",
    "arena.policies.translated_hal",
    "arena.translated_hal_adapter",
    "browser.opponent_memory",
    "browser.deploy.production",
)
TRAINING_MODULES = ("torch", "gymnasium", "stable_baselines3", "sb3_contrib")


def _loaded_modules(candidates: tuple[str, ...]) -> list[str]:
    code = (
        "import importlib, json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "for name in json.loads(sys.argv[2]):\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted(set(json.loads(sys.argv[3])) & set(sys.modules))))\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(SOURCE_ROOT),
            json.dumps(RUNTIME_MODULES),
            json.dumps(candidates),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_hosted_runtime_imports_without_training_dependencies() -> None:
    assert _loaded_modules(TRAINING_MODULES) == []
