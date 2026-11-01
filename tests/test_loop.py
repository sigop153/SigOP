"""The loop end to end, with a stub executor whose accuracy per harness and item is scripted
and a stub proposer that writes candidate files. No model, no server."""
import json
import re
from pathlib import Path

import pytest

from sigop.loop import BackendsDown, LoopConfig, run_search
from sigop.objective import load_objective
from sigop.proposer import Candidate
from sigop.rundir import RunDir

CANDIDATE_SOURCE = '''from sigop.harness import Harness, LearnItem, Prediction


class Candidate(Harness):
    def learn_from_batch(self, results: list[LearnItem]) -> None:
        pass

    def predict(self, input: str) -> Prediction:
        response = self.call_llm("HARNESS={name}\\n" + input)
        return Prediction(answer=response, full_response=response)

    def get_state(self) -> str:
        return "{{}}"
'''

# Which question numbers each harness answers correctly. Items 0-9 proposal, 10-19 train,
# 20-29 val, 30-39 test (one CSV per split, below). The seed has no marker in its prompt.
CORRECT = {
    "seed": set(range(10, 15)) | set(range(20, 25)) | set(range(30, 35)),      # 5/10 on every split
    "better": set(range(10, 18)) | set(range(20, 28)) | set(range(30, 38)),    # 8/10 everywhere
    "cheat": set(range(10, 19)) | set(range(20, 22)) | set(range(30, 32)),     # 9/10 train, 2/10 val
    "late": set(range(40)),                                                    # right everywhere (salvage tests)
}


def target_of(number: int) -> str:
    return "yes" if number % 2 else "no"


def stub_executor(prompt: str) -> str:
    match = re.match(r"HARNESS=(\w+)", prompt)
    harness = match.group(1) if match else "seed"
    number = int(re.findall(r"question (\d+)", prompt)[-1])
    answer = target_of(number) if number in CORRECT[harness] else ("no" if target_of(number) == "yes" else "yes")
    return f'{{"final_answer": "[ANSWER]{answer}[/ANSWER]"}}'


def objective_at(tmp_path: Path):
    folder = tmp_path / "toy"
    folder.mkdir()
    for index, name in enumerate(("proposal", "train", "val", "test")):
        rows = ["input,target"] + [f"question {i},{target_of(i)}" for i in range(10 * index, 10 * index + 10)]
        (folder / f"{name}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_objective(str(folder))


class ScriptedProposer:
    """Iteration 1 proposes two candidates; later iterations propose nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, run: RunDir, iteration: int) -> list[Candidate]:
        self.calls.append(iteration)
        if iteration != 1:
            return []
        for name in ("better", "cheat"):
            (run.harnesses / f"{name}.py").write_text(CANDIDATE_SOURCE.format(name=name), encoding="utf-8")
        return [Candidate("better", hypothesis="more examples help", axis="exploitation"),
                Candidate("cheat", hypothesis="memorise train", axis="exploration")]


def test_loop_adopts_rejects_overfit_and_signals_plateau(tmp_path):
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "toy")
    proposer = ScriptedProposer()
    incumbent = run_search(objective, run, stub_executor, proposer,
                           LoopConfig(iterations=6, patience=5, tolerance=0.1, concurrency=4))

    # The seed scored 5/10 on every split and the better candidate took over
    assert proposer.calls == [1, 2, 3, 4, 5, 6]
    assert (incumbent.name, incumbent.train, incumbent.val, incumbent.test) == ("better", 0.8, 0.8, 0.8)
    assert json.loads(run.incumbent.read_text(encoding="utf-8")) == {"name": "better", "train": 0.8}

    history = run.read_rows(run.history)
    by_name = {row["name"]: row for row in history}
    assert by_name["fewshot_all"]["iteration"] == 0 and by_name["fewshot_all"]["train"] == 0.5
    assert by_name["better"]["outcome"].startswith("ADOPTED: 80.0% (+30.0%)")
    assert by_name["cheat"]["outcome"].startswith("NOT ADOPTED (OVERFIT): train 90.0% (+40.0%)")
    assert "val" not in by_name["cheat"]["outcome"]

    # OVERFIT note for iteration 2, retired to the archive once iteration 3 began; then the
    # incumbent stalls for five iterations and PLATEAU is addressed to iteration 7
    assert (run.reports / ".archive" / "iter02_SIGNAL_OVERFIT.md").exists()
    assert sorted(p.name for p in run.reports.glob("*.md")) == ["iter07_SIGNAL_PLATEAU.md"]
    plateau = (run.reports / "iter07_SIGNAL_PLATEAU.md").read_text(encoding="utf-8")
    assert "not improved for 5 iterations" in plateau and "`better`" in plateau

    # Train traces where the proposer reads; val and test only in the heldout folder
    assert run.trace_path("better").exists() and run.score_path("better", "train").exists()
    assert run.score_path("better", "val").exists() and run.score_path("better", "test").exists()
    assert not (run.root / "scores" / "better" / "val.json").exists()
    assert run.heldout.name == "toy.heldout" and run.heldout.parent == run.root.parent
    assert run.score_path("cheat", "val").exists() and not run.score_path("cheat", "test").exists()

    events = [row["event"] for row in run.read_rows(run.detector)]
    assert events[:2] == ["seed", "candidate"]
    assert events.count("iteration_end") == 6 and events.count("no_candidates") == 5
    ends = [row for row in run.read_rows(run.detector) if row["event"] == "iteration_end"]
    assert [row["stale"] for row in ends] == [0, 1, 2, 3, 4, 0]
    assert [row["signal_out"] for row in ends] == ["OVERFIT", None, None, None, None, "PLATEAU"]


def test_candidate_that_crashes_is_recorded_not_fatal(tmp_path):
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "crash")

    def proposer(run: RunDir, iteration: int) -> list[Candidate]:
        (run.harnesses / "broken.py").write_text("raise RuntimeError('boom at import')\n", encoding="utf-8")
        return [Candidate("broken")]

    incumbent = run_search(objective, run, stub_executor, proposer, LoopConfig(iterations=1, concurrency=2))
    assert incumbent.name == "fewshot_all"
    rows = run.read_rows(run.history)
    assert rows[-1]["name"] == "broken" and rows[-1]["outcome"].startswith("failed: RuntimeError: boom")


def test_resume_continues_the_count_and_carries_the_state(tmp_path):
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "resumable")
    config = LoopConfig(iterations=3, patience=5, tolerance=0.1, concurrency=4)
    run_search(objective, run, stub_executor, ScriptedProposer(), config)
    rows_before = len(run.read_rows(run.detector))

    # "Broken" after iteration 3; continue to 6 with a proposer that proposes nothing
    later = ScriptedProposer()
    later.calls.append(1)                                   # so it proposes nothing from now on
    incumbent = run_search(objective, run, stub_executor, later, LoopConfig(iterations=6, patience=5, tolerance=0.1,
                                                                            concurrency=4), resume=True)
    rows = run.read_rows(run.detector)
    assert [row["event"] for row in rows].count("seed") == 1
    assert rows[rows_before]["event"] == "resume" and rows[rows_before]["iteration"] == 3
    assert later.calls[1:] == [4, 5, 6]
    assert (incumbent.name, incumbent.test) == ("better", 0.8)        # test score read back from heldout
    ends = [row for row in rows if row["event"] == "iteration_end"]
    assert [row["iteration"] for row in ends] == [1, 2, 3, 4, 5, 6]
    assert [row["stale"] for row in ends] == [0, 1, 2, 3, 4, 0]       # the counter carried over 3 -> 4
    assert ends[-1]["signal_out"] == "PLATEAU"


def test_resume_of_a_run_that_only_seeded_starts_at_iteration_one(tmp_path):
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "seeded")
    run_search(objective, run, stub_executor, ScriptedProposer(), LoopConfig(iterations=0, concurrency=4))
    proposer = ScriptedProposer()
    run_search(objective, run, stub_executor, proposer, LoopConfig(iterations=1, tolerance=0.1, concurrency=4), resume=True)
    assert proposer.calls == [1]
    assert [row["event"] for row in run.read_rows(run.detector)].count("seed") == 1


def test_resume_with_nothing_logged_seeds_normally(tmp_path):
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "fresh")
    run_search(objective, run, stub_executor, ScriptedProposer(), LoopConfig(iterations=0, concurrency=4), resume=True)
    assert [row["event"] for row in run.read_rows(run.detector)] == ["seed"]


class FlakyBackend:
    """Down for the first `down_checks` health checks, then up (or down for good)."""

    def __init__(self, down_checks: int):
        self.role = "proxy"
        self.down_checks = down_checks
        self.checks = 0

    def up(self) -> bool:
        self.checks += 1
        return self.checks > self.down_checks


class EmptyFirst:
    """The first session proposes nothing (it spoke to a dead server); the second proposes."""

    def __init__(self):
        self.inner = ScriptedProposer()
        self.calls = 0

    def __call__(self, run: RunDir, iteration: int) -> list[Candidate]:
        self.calls += 1
        return [] if self.calls == 1 else self.inner(run, iteration)


def test_backend_that_recovers_reruns_the_empty_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr("sigop.loop.time.sleep", lambda seconds: None)
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "flaky")
    backend = FlakyBackend(down_checks=0)
    proposer = EmptyFirst()
    # Up for the first check, down for the two after the empty proposal, then up again
    checks = iter([True, False, False, True] + [True] * 20)
    backend.up = lambda: next(checks)
    incumbent = run_search(objective, run, stub_executor, proposer,
                           LoopConfig(iterations=1, tolerance=0.1, concurrency=4, backend_retries=3), backends=[backend])
    assert proposer.calls == 2 and incumbent.name == "better"
    events = [json.loads(line) for line in run.detector.read_text(encoding="utf-8").splitlines()]
    assert [row["attempt"] for row in events if row["event"] == "backend_down"] == [0]


def test_backend_that_stays_down_stops_the_run(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("sigop.loop.time.sleep", waits.append)
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "dead")
    backend = FlakyBackend(down_checks=10**6)
    proposer = ScriptedProposer()
    with pytest.raises(BackendsDown, match="--resume dead"):
        run_search(objective, run, stub_executor, proposer,
                   LoopConfig(iterations=2, concurrency=4, backend_retries=3, backend_wait_seconds=60), backends=[backend])
    assert proposer.calls == [] and waits == [60, 60, 60]
    events = [json.loads(line) for line in run.detector.read_text(encoding="utf-8").splitlines()]
    assert [row["attempt"] for row in events if row["event"] == "backend_down"] == [0, 1, 2, 3]
