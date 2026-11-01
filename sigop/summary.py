"""One table per run, one row per iteration, built from the heldout decision log.

    python -m sigop.summary runs/<name>

Each row is the adopted candidate of that iteration, else its best candidate by train
score. Written to runs/<name>.heldout/summary.md at the end of a run and printed.
"""
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sigop.rundir import RunDir

LEGEND = """tr: train score
val: validation score
tst: test score
fh*: a new h* (best harness) was found this iteration
h*: test score of the current h*
O: OVERFIT signal fired
P: PLATEAU signal fired
t: time taken for this iteration, in minutes
cu_t: cumulative time taken so far, in minutes
"""
HEADER = "| it |   tr |  val |  tst | fh* |   h* | O | P |  t | cu_t |\n|---:|-----:|-----:|-----:|----:|-----:|--:|--:|---:|-----:|"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(summary(RunDir(Path(sys.argv[1]).resolve())))


@dataclass(frozen=True)
class Row:
    iteration: int
    train: float | None
    val: float | None
    test: float | None
    adopted: bool
    incumbent_test: float | None
    overfit: bool
    plateau: bool
    minutes: float | None
    cumulative: float | None

    def line(self) -> str:
        return (f"| {self.iteration:>2} | {score(self.train)} | {score(self.val)} | {score(self.test)} | "
                f"{flag(self.adopted):>3} | {score(self.incumbent_test)} | {flag(self.overfit)} | {flag(self.plateau)} | "
                f"{minutes(self.minutes):>2} | {minutes(self.cumulative):>4} |")


def summary(run: RunDir) -> str:
    return LEGEND + "\n" + HEADER + "\n" + "\n".join(row.line() for row in rows(run)) + "\n"


def rows(run: RunDir) -> list[Row]:
    events = run.read_rows(run.detector)
    history = run.read_rows(run.history)
    out: list[Row] = []
    incumbent_test: float | None = None
    previous_time: datetime | None = None
    cumulative = 0.0
    for event in events:
        if event.get("event") == "seed":
            incumbent_test = event["test"]
            previous_time = stamp(event)
            out.append(Row(0, event["train"], event["val"], event["test"], False, incumbent_test, False, False, None, None))
        elif event.get("event") == "iteration_end":
            iteration = event["iteration"]
            candidates = [e for e in events if e.get("event") == "candidate" and e.get("iteration") == iteration]
            adopted = next((c for c in candidates if c.get("adopt")), None)
            scored = [h for h in history if h.get("iteration") == iteration and h.get("train") is not None]
            best = max(scored, key=lambda h: h["train"], default=None)
            shown = adopted or next((c for c in candidates if best and c["name"] == best["name"]), None)
            train = (adopted or best or {}).get("train")
            val = shown["val"] if shown else None
            test = None
            if adopted:
                test = incumbent_test = read_test(run, adopted["name"])
            now = stamp(event)
            elapsed = (now - previous_time).total_seconds() / 60 if previous_time and now else None
            previous_time = now
            if elapsed is not None:
                cumulative += elapsed
            out.append(Row(iteration, train, val, test, adopted is not None, incumbent_test,
                           event.get("signal_out") == "OVERFIT", event.get("signal_out") == "PLATEAU",
                           elapsed, cumulative if elapsed is not None else None))
    return out


def read_test(run: RunDir, name: str) -> float | None:
    path = run.score_path(name, "test")
    return json.loads(path.read_text(encoding="utf-8"))["accuracy"] if path.exists() else None


def stamp(event: dict) -> datetime | None:
    try:
        return datetime.strptime(event["t"], "%Y-%m-%dT%H:%M:%S")
    except (KeyError, ValueError):
        return None


def score(value: float | None) -> str:
    return "   -" if value is None else f"{value:.3f}"[1:] if value < 1 else "1.00"


def flag(value: bool) -> str:
    return "y" if value else "n"


def minutes(value: float | None) -> str:
    return "-" if value is None else f"{round(value)}"


if __name__ == "__main__":
    main()
