from sigop.loop import LoopConfig, run_search
from sigop.rundir import RunDir
from sigop.summary import HEADER, rows, summary
from tests.test_loop import ScriptedProposer, objective_at, stub_executor


def test_summary_table_from_the_toy_run(tmp_path):
    objective = objective_at(tmp_path)
    run = RunDir(tmp_path / "runs" / "toy")
    run_search(objective, run, stub_executor, ScriptedProposer(), LoopConfig(iterations=6, patience=5, tolerance=0.1, concurrency=4))
    table = summary(run)
    lines = table.splitlines()
    assert lines[0] == "tr: train score" and HEADER.splitlines()[0] in lines
    body = lines[lines.index(HEADER.splitlines()[1]) + 1:]
    assert body[0] == "|  0 | .500 | .500 | .500 |   n | .500 | n | n |  - |    - |"
    # iteration 1: `better` adopted (train .8, val .8, test .8); `cheat` rejected -> OVERFIT fired
    assert body[1].startswith("|  1 | .800 | .800 | .800 |   y | .800 | y | n |")
    # iterations 2-5 nothing proposed; PLATEAU fires at 6; h* stays .800
    assert body[2].startswith("|  2 |    - |    - |    - |   n | .800 | n | n |")
    assert body[6].startswith("|  6 |    - |    - |    - |   n | .800 | n | y |")
    assert len(rows(run)) == 7
    assert (run.heldout / "summary.md").read_text(encoding="utf-8") == table      # kept current by the loop
