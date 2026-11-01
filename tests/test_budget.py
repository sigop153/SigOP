import json

from sigop.proposer import read_budget
from sigop.rundir import RunDir
from sigop.signals import Budget, retire_signals, write_budget, write_plateau


def test_note_states_the_limit_the_typical_time_and_the_salvage_rule(tmp_path):
    path = write_budget(tmp_path / "reports", 4, Budget(minutes=40, typical_minutes=25, timed_out=0, attempted=3),
                        "harnesses/cand_04_1.py")
    text = path.read_text(encoding="utf-8")
    assert path.name == "iter04_TIME_BUDGET.md"
    assert "stopped after 40 minutes" in text and "harness files it has written are evaluated as they are" in text
    assert "typically take about 25 minutes" in text and "write `harnesses/cand_04_1.py`" in text
    assert "already been stopped" not in text


def test_note_counts_the_timed_out_proposals(tmp_path):
    text = write_budget(tmp_path, 9, Budget(40, 30, timed_out=2, attempted=8), "x.py").read_text(encoding="utf-8")
    assert "**2 of 8 proposals have already been stopped**" in text and "took about 30 minutes" in text


def test_budget_notes_are_not_signals(tmp_path):
    reports = tmp_path / "reports"
    write_budget(reports, 1, Budget(40, 39, 0, 0), "x.py")
    write_plateau(reports, 1, 5, 0.3, "a")
    assert retire_signals(reports, keep_iteration=5) == 1
    assert (reports / "iter01_TIME_BUDGET.md").exists()


def test_read_budget_takes_the_median_of_delivered_sessions_and_clamps(tmp_path):
    run = RunDir(tmp_path)
    run.create()
    for iteration, meta in enumerate([
        {"seconds": 600, "wrote_pending": True, "timed_out": False},
        {"seconds": 2400, "wrote_pending": False, "timed_out": True},
        {"seconds": 3000, "wrote_pending": True, "timed_out": False},      # 50 min, above the limit
        {"seconds": 900, "wrote_pending": True, "timed_out": False},
    ], start=1):
        run.session_meta(iteration).write_text(json.dumps(meta), encoding="utf-8")
    budget = read_budget(run, minutes=40)
    assert (budget.attempted, budget.timed_out) == (4, 1)
    assert budget.typical_minutes == 15                     # median of 10, 15, 50
    assert read_budget(RunDir(tmp_path / "empty"), 40).typical_minutes == 39      # nothing yet: limit - 1
