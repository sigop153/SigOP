"""The proposer: a Claude Code session, pointed at an open-weight model, that reads the run
folder and writes candidate harnesses.

The session is launched the way the paper's runs launched it: `claude -p` with the skill
text appended to the system prompt, the file and shell tools, permissions skipped, and a
wall-clock limit enforced by killing the process. With an open-weight proposer,
ANTHROPIC_BASE_URL points it at a proxy that speaks the Anthropic API in front of the
proposer's vLLM server (serving.py). With an Anthropic model id as the proposer the CLI talks
to Anthropic with the user's own login or ANTHROPIC_API_KEY, the paper's Opus reference setting.

A proposal ends with pending.json in the run folder; its candidates name files under
harnesses/. A session stopped at the limit, or one that exits non-zero, without pending.json
still yields candidates: the harness files it wrote are evaluated as they are, marked
timed_out or crashed in history.jsonl. Before each session the loop writes a time-budget
note, and a PostToolUse hook tells the session after every tool call how much of its time
is used (elapsed_hook.py).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sigop.harness import SEED_HEADER
from sigop.rundir import RunDir
from sigop.sandbox import CLI_HOME, Sandbox
from sigop.signals import Budget

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ["Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash"]


@dataclass(frozen=True)
class Candidate:
    name: str
    hypothesis: str = ""
    axis: str = ""
    base_system: str = ""
    components: tuple[str, ...] = ()

    @property
    def file(self) -> str:
        return f"harnesses/{self.name}.py"


Proposer = Callable[[RunDir, int], list[Candidate]]


@dataclass(frozen=True)
class ClaudeCodeConfig:
    base_url: str | None                 # the proxy, e.g. http://127.0.0.1:8030; None uses the CLI's own Anthropic access
    auth_token: str = "local"
    model: str = "opus"                  # an alias the proxy maps to the proposer, or an Anthropic model for the CLI
    timeout_seconds: int = 2400
    effort: str = "max"
    candidates: int = 3
    context_length: int | None = None    # the served context; None leaves the CLI's own window sizing
    max_output_tokens: int = 64000
    compact_margin: float = 1.1          # compact at context - margin * output tokens


class ClaudeCode:
    def __init__(self, config: ClaudeCodeConfig, sandbox: Sandbox | None, binary: str = "claude"):
        self.config = config
        self.sandbox = sandbox            # None runs the session unconfined
        self.binary = binary

    def __call__(self, run: RunDir, iteration: int) -> list[Candidate]:
        run.pending.unlink(missing_ok=True)
        before = {path.name for path in run.harnesses.glob("*.py")}
        # Inside the sandbox only the mounted binary exists, so it is run by its full path
        executable = str(self.sandbox.binary) if self.sandbox else self.binary
        command = [
            executable, "--dangerously-skip-permissions", "-p", task_prompt(run, iteration),
            "--output-format", "stream-json", "--verbose", "--model", self.config.model,
            "--setting-sources", "", "--tools", ",".join(TOOLS), "--allowedTools", *TOOLS,
            "--disable-slash-commands", "--strict-mcp-config",
            "--append-system-prompt", skill_text(self.config.candidates), "--effort", self.config.effort,
        ]
        environment = {key: value for key, value in os.environ.items() if key != "CLAUDECODE"}
        if self.config.base_url:
            # Through the proxy to the open-weight proposer; the user's own key must not win
            environment.pop("ANTHROPIC_API_KEY", None)
            environment["ANTHROPIC_BASE_URL"] = self.config.base_url
            environment["ANTHROPIC_AUTH_TOKEN"] = self.config.auth_token
        # sigop.harness is importable from the run folder, for the validation step, and
        # `python` is this interpreter's environment
        environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        if self.config.context_length:
            environment.update(compaction_environment(self.config.context_length, self.config.max_output_tokens,
                                                      self.config.compact_margin))
        config_dir = None
        if self.hook_allowed(environment):
            config_dir = make_config_dir(run, self.config.timeout_seconds, sandboxed=self.sandbox is not None)
            settings = Path(CLI_HOME if self.sandbox else str(config_dir)) / "settings.json"
            command += ["--settings", str(settings)]
            if not self.sandbox:
                environment["CLAUDE_CONFIG_DIR"] = str(config_dir)
        if self.sandbox:
            command = self.sandbox.command(command, config_dir)
        started = time.time()
        timed_out = False
        log = run.sessions / f"iter{iteration:02d}.jsonl"
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=run.root, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=self.config.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group(process)
        seconds = time.time() - started
        candidates = read_pending(run)
        wrote_pending = bool(candidates)
        minutes = round(seconds / 60)
        if not wrote_pending and (timed_out or process.returncode != 0):
            reason = (f"timed out at {minutes} min" if timed_out else f"exited with code {process.returncode}")
            candidates = salvage(run, before, reason, "timed_out" if timed_out else "crashed")
            run.event({"iteration": iteration, "event": "proposal_timed_out" if timed_out else "proposal_crashed",
                       "minutes": minutes, "exit_code": process.returncode, "candidates": [c.name for c in candidates]})
        run.session_meta(iteration).write_text(json.dumps({
            "seconds": round(seconds, 1), "wrote_pending": wrote_pending, "timed_out": timed_out,
            "exit_code": process.returncode, "candidates": [c.name for c in candidates]}) + "\n", encoding="utf-8")
        return candidates

    def hook_allowed(self, environment: dict) -> bool:
        """The elapsed-time hook needs its own CLAUDE_CONFIG_DIR, which would also move a
        claude.ai login's credentials; so an Anthropic proposer gets it only with an API key."""
        return bool(self.config.base_url) or "ANTHROPIC_API_KEY" in environment


def compaction_environment(context: int, output: int, margin: float) -> dict[str, str]:
    """Size the CLI's window from the served context, as the paper's launcher did. Unset,
    the CLI sizes it from the alias "opus". It compacts at
    min(effective * pct / 100, effective - 13000) with effective = window - min(output, 20000),
    so pct is solved for a threshold of context - margin * output."""
    effective = context - min(output, 20000)
    pct = 100 * (context - margin * output) / effective
    return {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(output), "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(context),
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": f"{pct:.4f}"}


def kill_group(process: subprocess.Popen) -> None:
    """The session and everything it spawned (start_new_session puts them in one group)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def make_config_dir(run: RunDir, limit_seconds: int, sandboxed: bool) -> Path:
    """A fresh Claude Code config home for one session: the hook, its clock, and the settings
    that install it. Homes older than two days are removed."""
    run.cli_homes.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 2 * 24 * 3600
    for old in run.cli_homes.iterdir():
        if old.is_dir() and old.stat().st_mtime < cutoff:
            shutil.rmtree(old, ignore_errors=True)
    config_dir = Path(tempfile.mkdtemp(prefix="session.", dir=run.cli_homes))
    (config_dir / "session_clock").write_text(f"{int(time.time())} {limit_seconds}\n", encoding="utf-8")
    shutil.copy(ROOT / "sigop" / "elapsed_hook.py", config_dir / "elapsed_hook.py")
    home = CLI_HOME if sandboxed else str(config_dir)
    python = "/usr/bin/python3" if sandboxed else sys.executable
    settings = {"hooks": {"PostToolUse": [{"matcher": "*", "hooks": [
        {"type": "command", "command": f"{python} {home}/elapsed_hook.py"}]}]}}
    (config_dir / "settings.json").write_text(json.dumps(settings) + "\n", encoding="utf-8")
    return config_dir


def salvage(run: RunDir, before: set[str], reason: str, marker: str) -> list[Candidate]:
    """The harness files a stopped or crashed session wrote, as this iteration's candidates.
    No import check: a file that fails to import is scored as a failed candidate, which is
    a real outcome of the session."""
    names = sorted(path.stem for path in run.harnesses.glob("*.py") if path.name not in before)
    return [Candidate(name, hypothesis=f"({reason}: the loop evaluated the harness files the session had written)",
                      components=(marker,)) for name in names]


def read_budget(run: RunDir, minutes: int) -> Budget:
    """This run's proposals so far, from the sessions' own records: how long the ones that
    delivered took, and how many the limit stopped. A session that never started is in
    neither count."""
    finished: list[float] = []
    timed_out = attempted = 0
    for path in sorted(run.sessions.glob("iter*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        attempted += 1
        if meta.get("timed_out"):
            timed_out += 1
        elif meta.get("wrote_pending"):
            finished.append(float(meta.get("seconds", 0)) / 60)
    finished.sort()
    median = round(finished[len(finished) // 2]) if finished else minutes
    return Budget(minutes=minutes, typical_minutes=min(median, minutes - 1), timed_out=timed_out, attempted=attempted)


def read_pending(run: RunDir) -> list[Candidate]:
    if not run.pending.exists():
        return []
    try:
        rows = json.loads(run.pending.read_text(encoding="utf-8")).get("candidates", [])
    except ValueError:
        return []
    candidates = []
    for row in rows:
        # The file is the candidate's identity: its stem names the harness, the scores and
        # the history row, whatever the session put under "name"
        name = Path(str(row.get("file") or row.get("name", ""))).stem
        if name:
            candidates.append(Candidate(name=name, hypothesis=str(row.get("hypothesis", "")),
                                        axis=str(row.get("axis", "")), base_system=str(row.get("base_system", "")),
                                        components=tuple(str(c) for c in row.get("components", []))))
    return candidates


def task_prompt(run: RunDir, iteration: int) -> str:
    """Paths are absolute: a relative one is resolved by the session against its own
    working directory, and pending.json then lands in a nested copy of the run path."""
    root = run.root.resolve()
    return f"""Run iteration {iteration} of the harness evolution loop.

## Run directory
Everything for this run is in the current directory (`{root}`):
- `TASK.md`: the task
- `history.jsonl`: every candidate so far, with its train score and outcome
- `incumbent.json`: the current best harness
- `harnesses/`: the seed harness and every candidate's source
- `scores/<name>/train.json` and `scores/<name>/train_trace.jsonl`: each candidate's train score and per-item predictions
- `reports/`: post-eval reports and any SIGNAL file for this iteration
- Write pending.json to: `{root / 'pending.json'}`
"""


def skill_text(candidates: int) -> str:
    return SKILL.replace("{candidates}", str(candidates)).replace("{seed_header}", SEED_HEADER.rstrip())


# Adapted from Meta-Harness's meta-harness skill (MIT, Yoonho Lee): the same workflow and
# rules, against this package's run layout and harness interface.
SKILL = """Follow these skill instructions:

## Skill: harness-evolution

Run ONE iteration of harness evolution. Do all work in the main session; do NOT delegate to subagents.

**You do NOT run the evaluation.** You analyse results and prediction traces, prototype changes, and implement new harnesses. The outer loop scores them.

### CRITICAL CONSTRAINTS

- You MUST implement {candidates} new harnesses this iteration.
- Do NOT write "the incumbent is optimal" or "stop iterating", or abort early.
- ALWAYS complete all steps including prototyping.
- Design exactly {candidates} candidates: a mix of exploitation and exploration.

### Anti-parameter-tuning rules

The most common failure is a candidate that is a parameter variant of an existing harness. Check `history.jsonl` for what has been tried. Parameter sweeps (pool sizes, retrieval counts, context budgets, similarity metrics) almost always regress or tie.

Good candidates change a mechanism: a new retrieval algorithm, a new prompt architecture, a new learning strategy, a new memory structure. Combining mechanisms from two harnesses is valid.

Exploitation axes: A=prompt template, B=memory content, C=selection algorithm, D=memory sizing, E=learning trigger, F=LLM usage in learning. If the last 3 iterations explored the same axis, pick different ones.

### Anti-overfitting rules

- No task-specific hints. Do not hard-code knowledge about the task's items or labels.
- General patterns are fine: "prioritise recent errors", "balance label coverage".

### SIGNALS

`reports/` may hold a file `iterNN_SIGNAL_OVERFIT.md` or `iterNN_SIGNAL_PLATEAU.md` addressed to this iteration. Read it first and follow it: OVERFIT names candidates that improved train but failed on a held-out split you cannot see, so change mechanism rather than refine them; PLATEAU means the incumbent has stalled, so prefer exploration.

### WORKFLOW

Step 0, post-eval reports: for each past iteration in `history.jsonl` with no `reports/iterNN_report.md`, write one (30 lines or fewer): what changed, what improved or regressed and why, a takeaway.

Step 1, analyse: read `TASK.md`, `history.jsonl`, `incumbent.json`, and recent `scores/<name>/train_trace.jsonl` traces. Form {candidates} falsifiable hypotheses, each targeting a different mechanism.

Step 2, prototype (mandatory): for each candidate write a test script in `/tmp/` that exercises the core retrieval or learning logic on real items pulled from a trace; try 2 or 3 variants; delete the scripts.

Step 3, implement: copy a top harness from `harnesses/` to `harnesses/<name>.py` and make targeted changes. Self-critique: if `predict()` and `learn_from_batch()` are identical to the base except for numbers, rewrite. Validate with:

    python -c "from pathlib import Path; from sigop.harness import load_harness; load_harness(Path('harnesses/<name>.py'), lambda prompt: ''); print('OK')"

Step 4, write pending.json to the path given in the task prompt:

```json
{"iteration": N, "candidates": [{"name": "<snake_case_name>", "file": "harnesses/<name>.py", "hypothesis": "<falsifiable claim>", "axis": "exploitation|exploration", "base_system": "<what it builds on>", "components": ["tag1", "tag2"]}]}
```

### Harness interface

Start every candidate with the seed file's import lines, exactly as they are there:

```python
{seed_header}


class MyHarness(Harness):
    def learn_from_batch(self, results: list[LearnItem]) -> None: ...   # LearnItem: input, raw_question, ground_truth, prediction, was_correct
    def predict(self, input: str) -> Prediction: ...                     # Prediction(answer, full_response, extra)
    def get_state(self) -> str: ...                                      # JSON
```

- One Harness subclass per file. Call the executor with `self.call_llm(prompt)`, never `self._llm`.
- Keep the import lines above even when you do not use every name; a candidate that fails to import scores nothing.
- Use `extract_json_field(response, "final_answer")` to pull the answer out of a JSON reply.
- `predict` must work before any learning (cold start). `learn_from_batch` is called once per proposal item, with the target as the prediction.

### Files

- `history.jsonl`: one JSON per line: `{"iteration", "name", "train", "delta", "axis", "hypothesis", "components", "outcome"}`
- `incumbent.json`: `{"name", "train"}`
- `scores/<name>/train.json`: `{"accuracy", "correct", "total"}`; `train_trace.jsonl`: one record per item with input, target, prediction, correct, full_response
"""
