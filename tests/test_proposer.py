import json
from pathlib import Path

from sigop.proposer import Candidate, read_pending, skill_text, task_prompt
from sigop.rundir import RunDir


def test_task_prompt_gives_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = RunDir(Path("runs") / "r1")
    prompt = task_prompt(run, 3)
    assert "Run iteration 3" in prompt
    assert f"Write pending.json to: `{tmp_path / 'runs' / 'r1' / 'pending.json'}`" in prompt
    assert "`runs/r1`" not in prompt


def test_read_pending_tolerates_missing_junk_and_paths(tmp_path):
    run = RunDir(tmp_path)
    assert read_pending(run) == []
    run.pending.write_text("not json", encoding="utf-8")
    assert read_pending(run) == []
    run.pending.write_text(json.dumps({"candidates": [
        {"name": "harnesses/knn_v2.py", "hypothesis": "h", "axis": "exploration", "components": ["a", 1]},
        {"name": ""},
    ]}), encoding="utf-8")
    assert read_pending(run) == [Candidate("knn_v2", hypothesis="h", axis="exploration", components=("a", "1"))]


def test_skill_text_states_the_candidate_count():
    assert "You MUST implement 3 new harnesses" in skill_text(3)
    assert "exactly 2 candidates" in skill_text(2)


def test_sandboxed_session_runs_the_mounted_binary_by_full_path(tmp_path, monkeypatch):
    import subprocess

    from sigop.proposer import ClaudeCode, ClaudeCodeConfig
    from sigop.sandbox import Sandbox

    seen = {}

    class FakeProcess:
        returncode = 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, **options):
        seen["command"] = command
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/bwrap")
    run = RunDir(tmp_path)
    run.create()
    sandbox = Sandbox(run_root=tmp_path, package=tmp_path / "sigop", environments=(), binary=Path("/opt/claude/claude.exe"),
                      home=Path("/home/me"))
    ClaudeCode(ClaudeCodeConfig(base_url="http://127.0.0.1:8030"), sandbox)(run, 1)
    command = seen["command"]
    assert command[0] == "/usr/bin/bwrap"
    assert command[command.index("--") + 1] == "/opt/claude/claude.exe"


def test_anthropic_backend_keeps_the_users_key_and_sets_no_proxy(tmp_path, monkeypatch):
    import subprocess

    from sigop.proposer import ClaudeCode, ClaudeCodeConfig

    seen = {}

    class FakeProcess:
        returncode = 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda command, **options: seen.update(command=command, env=options["env"]) or FakeProcess())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-user")
    run = RunDir(tmp_path)
    run.create()
    ClaudeCode(ClaudeCodeConfig(base_url=None, model="claude-opus-4-8"), None)(run, 1)
    assert seen["env"]["ANTHROPIC_API_KEY"] == "sk-user" and "ANTHROPIC_BASE_URL" not in seen["env"]
    assert seen["command"][seen["command"].index("--model") + 1] == "claude-opus-4-8"

    ClaudeCode(ClaudeCodeConfig(base_url="http://127.0.0.1:8030"), None)(run, 1)
    assert "ANTHROPIC_API_KEY" not in seen["env"] and seen["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8030"


def test_skill_shows_the_seed_import_lines():
    from sigop.harness import SEED_HEADER

    text = skill_text(3)
    assert SEED_HEADER.rstrip() in text and "{seed_header}" not in text

