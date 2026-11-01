import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sigop.sandbox import Sandbox, default_sandbox, python_prefixes

NEEDS_BWRAP = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")


def sandbox_for(tmp_path: Path) -> Sandbox:
    run_root = tmp_path / "runs" / "r1"
    run_root.mkdir(parents=True)
    return Sandbox(run_root=run_root, package=Path(__file__).resolve().parent.parent / "sigop",
                   environments=python_prefixes(), binary=Path(sys.executable).resolve(), home=Path.home())


@NEEDS_BWRAP
def test_command_mounts_only_what_is_listed(tmp_path):
    sandbox = sandbox_for(tmp_path)
    command = sandbox.command(["claude", "-p", "x"])
    assert command[0].endswith("bwrap") and command[-3:] == ["claude", "-p", "x"]
    assert command[command.index("--bind") + 1] == str(sandbox.run_root)
    assert "--unshare-pid" in command and "--unshare-net" not in command
    assert str(tmp_path / "runs" / "r1.heldout") not in command


def run_inside(sandbox: Sandbox, code: str) -> subprocess.CompletedProcess:
    command = sandbox.command([sys.executable, "-c", code])
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=120,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(sandbox.package.parent)})


@NEEDS_BWRAP
def test_heldout_and_objectives_are_invisible_inside(tmp_path):
    sandbox = sandbox_for(tmp_path)
    heldout = tmp_path / "runs" / "r1.heldout"
    heldout.mkdir()
    (heldout / "secret.json").write_text("{}", encoding="utf-8")
    objectives = sandbox.package.parent / "objectives"
    result = run_inside(sandbox, f"import os; print(os.path.exists({str(heldout)!r}), os.path.exists({str(objectives)!r}))")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


@NEEDS_BWRAP
def test_run_folder_is_writable_and_sigop_importable_inside(tmp_path):
    sandbox = sandbox_for(tmp_path)
    result = run_inside(sandbox, "import os, sigop.harness; open('note.txt', 'w').write('hi'); print(os.getcwd())")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(sandbox.run_root)
    assert (sandbox.run_root / "note.txt").read_text(encoding="utf-8") == "hi"


@NEEDS_BWRAP
def test_home_and_tmp_are_fresh_inside(tmp_path):
    """Files of the real home and /tmp are absent; only the mount points' parent folders
    exist there (the venv and the uv Python live under the home, so their paths do)."""
    sandbox = sandbox_for(tmp_path)
    host_home_file = next(p for p in Path.home().iterdir() if p.is_file())
    host_tmp_file = Path("/tmp") / f"sigop-sandbox-marker-{os.getpid()}"
    host_tmp_file.write_text("host", encoding="utf-8")
    try:
        result = run_inside(sandbox, f"import os; print(os.path.exists({str(host_home_file)!r}), "
                                     f"os.path.exists({str(host_tmp_file)!r}), os.path.exists(os.path.expanduser('~/.claude')))")
    finally:
        host_tmp_file.unlink()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False False"


def test_default_sandbox_binds_this_interpreter_and_the_package(tmp_path):
    sandbox = default_sandbox(tmp_path, binary="python3")
    assert sandbox.package.name == "sigop"
    assert Path(sys.prefix) in sandbox.environments
    assert sandbox.run_root == tmp_path.resolve()


def test_anthropic_sandbox_binds_the_login_file_and_dns(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    sandbox = default_sandbox(tmp_path / "run", binary="python3", anthropic=True)
    command = sandbox.command(["x"])
    assert command[command.index("--bind") + 1] == str(tmp_path / ".claude" / ".credentials.json")
    assert sandbox.dns is True
    plain = default_sandbox(tmp_path / "run", binary="python3")
    assert plain.credentials is None and plain.dns is False
