"""The four commands of python -m stl and their lazy imports."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stl.__main__ import main


@pytest.mark.parametrize("command", ["dth", "leap", "serve", "deploy"])
def test_each_command_has_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        main([command, "--help"])
    assert stop.value.code == 0
    assert f"python -m stl {command}" in capsys.readouterr().out


def test_serve_accepts_only_the_two_hal_kinds() -> None:
    with pytest.raises(SystemExit) as stop:
        main(["serve", "--hal", "perfect"])
    assert stop.value.code == 2


def test_help_lists_the_commands_and_imports_no_solver() -> None:
    code = (
        "import json, sys, contextlib, io\n"
        "from stl.__main__ import main\n"
        "out = io.StringIO()\n"
        "with contextlib.redirect_stdout(out):\n"
        "    try:\n"
        "        main(['--help'])\n"
        "    except SystemExit:\n"
        "        pass\n"
        "print(json.dumps({'help': out.getvalue(), 'modules': sorted(sys.modules)}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    run = json.loads(result.stdout.strip().splitlines()[-1])
    assert "{dth,leap,serve,deploy}" in run["help"]
    loaded = {name.split(".", 1)[0] for name in run["modules"]}
    assert sorted(loaded & {"numpy", "numba", "scipy", "fastapi"}) == []
