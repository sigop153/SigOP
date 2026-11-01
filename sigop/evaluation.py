"""Train and score one harness in a child process the loop can kill.

A harness is the proposer's code: it can loop forever, allocate without bound, or call the
executor one item at a time. Scoring it in the loop's own process would hang the run, so
each harness runs in a child that the parent watches:

    stall     no item scored for `stall_seconds`      -> killed, "stalled"
    hard cap  `cap_seconds` since the child started   -> killed, "timed out"

The child trains once, then scores whichever splits the parent asks for, in order, so the
loop can score train first and val or test only when needed. It writes the score files
itself and sends back the summary. A killed harness is a failed candidate, not a dead run.
"""
import multiprocessing
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Self

from sigop.evaluate import score, train, write_result
from sigop.executor import Executor, ExecutorConfig
from sigop.harness import load_harness
from sigop.objective import load_objective
from sigop.rundir import RunDir

STALL_SECONDS = 600
CAP_SECONDS = 5400
POLL_SECONDS = 1.0


class EvaluationError(Exception):
    """The child could not deliver a score: it crashed, stalled or ran past the cap."""


@dataclass(frozen=True)
class Job:
    """Everything the child needs, all picklable."""
    objective: str                          # as given to load_objective
    seed: int
    executor: ExecutorConfig | Callable[[str], str]   # a config, or a picklable callable for tests
    harness: Path
    run_root: Path
    concurrency: int


@dataclass(frozen=True)
class Scores:
    accuracy: float
    correct: float
    total: int


@dataclass(frozen=True)
class Limits:
    stall_seconds: int = STALL_SECONDS
    cap_seconds: int = CAP_SECONDS


class Evaluation:
    """One harness in its child process. Use as a context manager; score() per split."""

    def __init__(self, job: Job, limits: Limits):
        self.job = job
        self.limits = limits
        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=child_main, args=(child_connection, job), daemon=True)
        self.started = time.time()
        self.last_progress = self.started
        self.process.start()
        child_connection.close()
        self.wait_for("trained")

    def score(self, split: str) -> Scores:
        self.connection.send(split)
        message = self.wait_for("result")
        return Scores(**message[1])

    def wait_for(self, kind: str) -> tuple:
        """Pump messages until one of `kind` arrives, enforcing the stall and the cap."""
        while True:
            if self.connection.poll(POLL_SECONDS):
                try:
                    message = self.connection.recv()
                except EOFError:
                    self.kill()
                    raise EvaluationError("the harness process exited without reporting") from None
                if message[0] == "progress":
                    self.last_progress = time.time()
                elif message[0] == "error":
                    self.kill()
                    raise EvaluationError(message[1])
                elif message[0] == kind:
                    self.last_progress = time.time()
                    return message
            now = time.time()
            if now - self.last_progress > self.limits.stall_seconds:
                self.kill()
                raise EvaluationError(f"stalled: no item scored for {self.limits.stall_seconds} s")
            if now - self.started > self.limits.cap_seconds:
                self.kill()
                raise EvaluationError(f"timed out: {self.limits.cap_seconds} s cap")
            if not self.process.is_alive() and not self.connection.poll():
                raise EvaluationError(f"the harness process died with exit code {self.process.exitcode}")

    def kill(self) -> None:
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.connection.send(None)
                self.process.join(10)
            except (OSError, ValueError):
                pass
        self.kill()
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exception) -> None:
        self.close()


def child_main(connection: Connection, job: Job) -> None:
    try:
        objective = load_objective(job.objective, job.seed)
        executor = Executor(job.executor) if isinstance(job.executor, ExecutorConfig) else job.executor
        harness = load_harness(job.harness, executor)
        run = RunDir(job.run_root)
        train(harness, objective)
        connection.send(("trained",))
        while True:
            split = connection.recv()
            if split is None:
                return
            result = score(harness, objective, objective.splits.of(split), job.concurrency,
                           on_item=lambda: connection.send(("progress",)))
            write_result(result, run.score_path(job.harness.stem, split),
                         run.trace_path(job.harness.stem) if split == "train" else None)
            connection.send(("result", {"accuracy": result.accuracy, "correct": result.correct, "total": result.total}))
    except Exception as error:  # noqa: BLE001  reported to the parent, which records the failure
        connection.send(("error", f"{type(error).__name__}: {error}"[:300]))
