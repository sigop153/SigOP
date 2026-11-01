"""The evaluation child process: results come back, a stalled harness is killed, an
overlong one is killed, a crashing one reports its error. Harness files sleep to simulate
slow code; the executor is a module-level function so the spawned child can unpickle it."""
from pathlib import Path

import pytest

from sigop.evaluation import Evaluation, EvaluationError, Job, Limits
from sigop.objective import SPLIT_NAMES
from sigop.rundir import RunDir

HARNESS = '''import time
from sigop.harness import Harness, LearnItem, Prediction


class Slow(Harness):
    def learn_from_batch(self, results: list[LearnItem]) -> None:
        pass

    def predict(self, input: str) -> Prediction:
        time.sleep({sleep})
        return Prediction(answer=self.call_llm(input), full_response="")

    def get_state(self) -> str:
        return "{{}}"
'''


def yes(prompt: str) -> str:
    return "[ANSWER]yes[/ANSWER]"


def objective_at(tmp_path: Path, rows: int = 4) -> str:
    folder = tmp_path / "toy"
    folder.mkdir()
    for name in SPLIT_NAMES:
        (folder / f"{name}.csv").write_text("input,target\n" + "".join(f"q{i},yes\n" for i in range(rows)), encoding="utf-8")
    return str(folder)


def job_for(tmp_path: Path, sleep: float) -> Job:
    run = RunDir(tmp_path / "run")
    run.create()
    harness = run.harnesses / "slow.py"
    harness.write_text(HARNESS.format(sleep=sleep), encoding="utf-8")
    return Job(objective=objective_at(tmp_path), seed=1, executor=yes, harness=harness, run_root=run.root, concurrency=4)


def test_scores_come_back_and_files_are_written(tmp_path):
    job = job_for(tmp_path, sleep=0)
    with Evaluation(job, Limits(stall_seconds=30, cap_seconds=60)) as evaluation:
        train = evaluation.score("train")
        val = evaluation.score("val")
    assert (train.accuracy, train.total, val.accuracy) == (1.0, 4, 1.0)
    run = RunDir(job.run_root)
    assert run.score_path("slow", "train").exists() and run.trace_path("slow").exists()
    assert run.score_path("slow", "val").exists() and not (run.root / "scores" / "slow" / "val.json").exists()


def test_a_stalled_harness_is_killed(tmp_path):
    job = job_for(tmp_path, sleep=60)                     # no item ever completes
    with Evaluation(job, Limits(stall_seconds=2, cap_seconds=60)) as evaluation, \
            pytest.raises(EvaluationError, match="stalled: no item scored for 2 s"):
        evaluation.score("train")
    assert not evaluation.process.is_alive()


def test_a_harness_past_the_cap_is_killed_even_while_progressing(tmp_path):
    job = job_for(tmp_path, sleep=0.6)                    # items keep completing, one at a time
    job = Job(**{**job.__dict__, "concurrency": 1})
    with Evaluation(job, Limits(stall_seconds=30, cap_seconds=2)) as evaluation, \
            pytest.raises(EvaluationError, match="timed out: 2 s cap"):
        evaluation.score("train")
    assert not evaluation.process.is_alive()


def test_a_crashing_harness_reports_the_error(tmp_path):
    run = RunDir(tmp_path / "run")
    run.create()
    harness = run.harnesses / "broken.py"
    harness.write_text("raise RuntimeError('boom at import')\n", encoding="utf-8")
    job = Job(objective=objective_at(tmp_path), seed=1, executor=yes, harness=harness, run_root=run.root, concurrency=2)
    with pytest.raises(EvaluationError, match="RuntimeError: boom at import"):
        Evaluation(job, Limits(stall_seconds=30, cap_seconds=60))
