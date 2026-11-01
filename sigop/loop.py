"""The search.

    seed      train the seed harness, score it on train, val and test; it is the incumbent
    each iteration, T times, never stopping early:
      propose   the proposer reads the run folder (and any SIGNAL file) and writes candidates
      score     every candidate that imports is trained and scored on train
      decide    best train score first; a candidate that beats the incumbent on train is scored
                on val, and signals.classify says adopt or OVERFIT
      test      a new incumbent is scored on test
      plateau   signals.plateau on the incumbent's train score
      signal    the note for the next proposal is written to reports/

Every harness is trained and scored in a child process the loop can kill (evaluation.py):
one that stalls or overruns is a failed candidate, not a dead run.

A dead server is a dead run. Before each proposal and before scoring, every backend (executor,
proposer, proxy) must answer its health URL; one that does not is checked again every
backend_wait_seconds, backend_retries times, and then the run stops with BackendsDown. A
proposal that comes back empty while a backend is down is run again once the backend is back,
since the session spoke to a dead server.

Train scores and traces go where the proposer reads; val and test scores and every
decision go to the heldout folder (rundir.py).

Resuming: every value the loop carries between iterations is in the decision log's
iteration_end rows, so a broken run continues from the iteration after the last complete
one. An iteration that was cut off part-way is redone: its proposer session runs again,
and any rows it had already written to history.jsonl stay as a record.
"""
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Protocol

from sigop import signals
from sigop.evaluation import Evaluation, EvaluationError, Job, Limits, Scores
from sigop.executor import ExecutorConfig
from sigop.harness import Executor, write_seed
from sigop.objective import Objective
from sigop.proposer import Candidate, Proposer, read_budget
from sigop.rundir import RunDir
from sigop.summary import summary


@dataclass(frozen=True)
class LoopConfig:
    iterations: int = 50
    patience: int = 5
    tolerance: float = 0.01
    concurrency: int = 32
    seed_harness: str = "fewshot_all"
    stall_seconds: int = Limits.stall_seconds     # kill a harness that scores no item for this long
    cap_seconds: int = Limits.cap_seconds         # kill a harness that runs longer than this in total
    propose_minutes: int = 0                      # the proposer's limit, for the time-budget note; 0 writes none
    backend_retries: int = 10                     # health checks after the first failure before the run stops
    backend_wait_seconds: int = 60                # between those checks

    @property
    def limits(self) -> Limits:
        return Limits(self.stall_seconds, self.cap_seconds)


class Backend(Protocol):
    """A server the loop needs: serving.HttpBackend, or a stand-in in tests. `role` is
    read-only here so a frozen dataclass satisfies the protocol."""
    @property
    def role(self) -> str: ...

    def up(self) -> bool: ...


class BackendsDown(SystemExit):
    """A backend stayed down through every retry; the run stops and can be resumed."""


@dataclass(frozen=True)
class Incumbent:
    name: str
    train: float
    val: float
    test: float


@dataclass(frozen=True)
class Resumed:
    incumbent: Incumbent
    state: signals.PlateauState
    next_iteration: int


def run_search(objective: Objective, run: RunDir, executor: ExecutorConfig | Executor, proposer: Proposer,
               config: LoopConfig, resume: bool = False, backends: Sequence[Backend] = ()) -> Incumbent:
    """executor: the executor endpoint's config, or a picklable callable for tests.
    backends: the servers checked before each proposal and each scoring; none in tests."""
    run.create()
    run.task.write_text(objective.task, encoding="utf-8")
    resumed = resume_state(run) if resume else None
    if resumed:
        incumbent, state, first = resumed.incumbent, resumed.state, resumed.next_iteration
        emit(run, {"iteration": first - 1, "event": "resume", "incumbent": incumbent.name, "train": incumbent.train,
                   "best_train": state.best_train, "stale": state.stale, "signal_out": state.signal})
        print(f"resuming at iteration {first}: incumbent {incumbent.name} train {incumbent.train:.3f} "
              f"stale {state.stale}/{config.patience} pending signal {state.signal or 'none'}", flush=True)
    else:
        incumbent = seed(objective, run, executor, config)
        state = signals.PlateauState(best_train=incumbent.train, stale=0, signal=None)
        first = 1
    for iteration in range(first, config.iterations + 1):
        signals.retire_signals(run.reports, keep_iteration=iteration)
        state = replace(state, signal=None)
        if config.propose_minutes:
            signals.write_budget(run.reports, iteration, read_budget(run, config.propose_minutes),
                                 f"harnesses/cand_{iteration:02d}_1.py")
        await_backends(run, iteration, backends, config)
        candidates = proposer(run, iteration)
        if not candidates and any(not backend.up() for backend in backends):
            await_backends(run, iteration, backends, config)      # stops the run if they stay down
            candidates = proposer(run, iteration)                 # the first session spoke to a dead server
        if not candidates:
            emit(run, {"iteration": iteration, "event": "no_candidates"})
        await_backends(run, iteration, backends, config)
        scored = score_candidates(objective, run, executor, candidates, iteration, config)
        incumbent, rejected = decide(run, incumbent, scored, iteration, config)
        if rejected:                      # OVERFIT is pending before plateau runs, as in the paper's loop
            state = replace(state, signal=signals.OVERFIT)
        state = signals.plateau(incumbent.train, state, config.patience)
        write_incumbent(run, incumbent)
        emit(run, {"iteration": iteration, "event": "iteration_end", "incumbent": incumbent.name,
                   "train": incumbent.train, "val": incumbent.val, "best_train": state.best_train,
                   "stale": state.stale, "signal_out": state.signal})
        if state.signal == signals.OVERFIT:
            signals.write_overfit(run.reports, iteration + 1, rejected, len(objective.splits.train))
        elif state.signal == signals.PLATEAU:
            signals.write_plateau(run.reports, iteration + 1, config.patience, state.best_train, incumbent.name)
        write_summary(run)
        print(f"iteration {iteration}: incumbent {incumbent.name} train {incumbent.train:.3f} "
              f"stale {state.stale}/{config.patience} signal {state.signal or 'none'}", flush=True)
    return incumbent


def await_backends(run: RunDir, iteration: int, backends: Sequence[Backend], config: LoopConfig) -> None:
    """Return once every backend answers; raise BackendsDown after backend_retries more checks."""
    down: list[str] = []
    for attempt in range(config.backend_retries + 1):
        down = [backend.role for backend in backends if not backend.up()]
        if not down:
            return
        emit(run, {"iteration": iteration, "event": "backend_down", "down": down, "attempt": attempt})
        if attempt < config.backend_retries:
            print(f"iteration {iteration}: {', '.join(down)} not answering; check {attempt + 1} of "
                  f"{config.backend_retries} in {config.backend_wait_seconds} s", flush=True)
            time.sleep(config.backend_wait_seconds)
    raise BackendsDown(f"{', '.join(down)} still down after {config.backend_retries} checks "
                       f"{config.backend_wait_seconds} s apart; the run stopped in iteration {iteration}. "
                       f"Fix the server and continue with --resume {run.root.name}")


def resume_state(run: RunDir) -> Resumed | None:
    """The loop's state after the last complete iteration, from the decision log; None when
    the run never got past the seed."""
    rows = run.read_rows(run.detector)
    ends = [row for row in rows if row.get("event") == "iteration_end"]
    seeds = [row for row in rows if row.get("event") == "seed"]
    if not seeds:
        return None
    if not ends:
        row = seeds[-1]
        incumbent = Incumbent(row["name"], row["train"], row["val"], row["test"])
        return Resumed(incumbent, signals.PlateauState(best_train=incumbent.train, stale=0, signal=None), 1)
    last = ends[-1]
    test_path = run.score_path(last["incumbent"], "test")
    test = json.loads(test_path.read_text(encoding="utf-8"))["accuracy"] if test_path.exists() else 0.0
    incumbent = Incumbent(last["incumbent"], last["train"], last["val"], test)
    state = signals.PlateauState(best_train=last["best_train"], stale=last["stale"], signal=last.get("signal_out"))
    return Resumed(incumbent, state, last["iteration"] + 1)


def seed(objective: Objective, run: RunDir, executor: ExecutorConfig | Executor, config: LoopConfig) -> Incumbent:
    write_seed(config.seed_harness, run.harnesses)
    with Evaluation(job(objective, run, executor, config.seed_harness, config), config.limits) as evaluation:
        results = {split: evaluation.score(split) for split in ("train", "val", "test")}
    incumbent = Incumbent(config.seed_harness, results["train"].accuracy, results["val"].accuracy, results["test"].accuracy)
    run.append(run.history, history_row(0, Candidate(config.seed_harness), results["train"].accuracy, 0.0,
                                        f"{results['train'].accuracy:.1%} (seed)"))
    emit(run, {"iteration": 0, "event": "seed", **asdict(incumbent)})
    write_incumbent(run, incumbent)
    write_summary(run)
    print(f"seed {config.seed_harness}: train {incumbent.train:.3f} val {incumbent.val:.3f} test {incumbent.test:.3f}", flush=True)
    return incumbent


def job(objective: Objective, run: RunDir, executor: ExecutorConfig | Executor, name: str, config: LoopConfig) -> Job:
    return Job(objective=objective.source, seed=objective.seed, executor=executor,
               harness=run.harnesses / f"{name}.py", run_root=run.root, concurrency=config.concurrency)


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    train: Scores
    evaluation: Evaluation           # the trained harness's process, kept open for val and test


def score_candidates(objective: Objective, run: RunDir, executor: ExecutorConfig | Executor,
                     candidates: list[Candidate], iteration: int, config: LoopConfig) -> list[Scored]:
    scored = []
    for candidate in candidates:
        try:
            evaluation = Evaluation(job(objective, run, executor, candidate.name, config), config.limits)
            train_scores = evaluation.score("train")
        except EvaluationError as error:     # a crashing, stalling or overlong harness is a failed candidate
            record_failure(run, iteration, candidate, str(error))
            continue
        scored.append(Scored(candidate, train_scores, evaluation))
    return scored


def record_failure(run: RunDir, iteration: int, candidate: Candidate, reason: str) -> None:
    run.append(run.history, history_row(iteration, candidate, None, None, f"failed: {reason}"))
    emit(run, {"iteration": iteration, "event": "candidate_failed", "name": candidate.name, "error": reason})
    print(f"  failed {candidate.name}: {reason}", flush=True)


def decide(run: RunDir, incumbent: Incumbent, scored: list[Scored], iteration: int,
           config: LoopConfig) -> tuple[Incumbent, list[signals.Rejected]]:
    rejected: list[signals.Rejected] = []
    try:
        for entry in sorted(scored, key=lambda s: -s.train.accuracy):
            incumbent = decide_one(run, incumbent, entry, iteration, config, rejected)
    finally:
        for entry in scored:
            entry.evaluation.close()
    return incumbent, rejected


def decide_one(run: RunDir, incumbent: Incumbent, entry: Scored, iteration: int, config: LoopConfig,
               rejected: list[signals.Rejected]) -> Incumbent:
    candidate, train_score = entry.candidate, entry.train.accuracy
    delta_train = train_score - incumbent.train
    if train_score <= incumbent.train:
        run.append(run.history, history_row(iteration, candidate, train_score, delta_train,
                                            f"{train_score:.1%} ({delta_train:+.1%})"))
        return incumbent
    try:
        val = entry.evaluation.score("val")
    except EvaluationError as error:
        record_failure(run, iteration, candidate, str(error))
        return incumbent
    delta_val = val.accuracy - incumbent.val
    decision = signals.classify(delta_train, delta_val, config.tolerance)
    emit(run, {"iteration": iteration, "event": "candidate", "name": candidate.name, "train": train_score,
               "val": val.accuracy, "delta_train": delta_train, "delta_val": delta_val,
               "adopt": decision.adopt, "signal": decision.signal, "incumbent": incumbent.name})
    if decision.adopt:
        try:
            test = entry.evaluation.score("test")
        except EvaluationError as error:
            record_failure(run, iteration, candidate, str(error))
            return incumbent
        outcome = f"ADOPTED: {train_score:.1%} ({delta_train:+.1%})"
        print(f"  adopt {candidate.name}: train {train_score:.3f} ({delta_train:+.3f}) "
              f"val {val.accuracy:.3f} ({delta_val:+.3f})", flush=True)
        run.append(run.history, history_row(iteration, candidate, train_score, delta_train, outcome))
        return Incumbent(candidate.name, train_score, val.accuracy, test.accuracy)
    rejected.append(signals.Rejected(candidate.name, delta_train))
    # Categorical only: the val score and its change stay out of the proposer's view
    outcome = (f"NOT ADOPTED (OVERFIT): train {train_score:.1%} ({delta_train:+.1%}) "
               "but did not generalise to the held-out split")
    print(f"  OVERFIT {candidate.name}: train {train_score:.3f} ({delta_train:+.3f}) "
          f"val {val.accuracy:.3f} ({delta_val:+.3f}); not adopted", flush=True)
    run.append(run.history, history_row(iteration, candidate, train_score, delta_train, outcome))
    return incumbent


def history_row(iteration: int, candidate: Candidate, train_score: float | None, delta: float | None,
                outcome: str) -> dict:
    return {"iteration": iteration, "name": candidate.name,
            "train": None if train_score is None else round(train_score, 4),
            "delta": None if delta is None else round(delta, 4),
            "axis": candidate.axis, "hypothesis": candidate.hypothesis,
            "components": list(candidate.components), "outcome": outcome}


def write_summary(run: RunDir) -> None:
    """The run's table so far, rewritten after the seed and after every iteration."""
    (run.heldout / "summary.md").write_text(summary(run), encoding="utf-8")


def write_incumbent(run: RunDir, incumbent: Incumbent) -> None:
    """Name and train score only; val and test stay in the heldout folder."""
    run.incumbent.write_text(json.dumps({"name": incumbent.name, "train": round(incumbent.train, 4)}) + "\n",
                             encoding="utf-8")


def emit(run: RunDir, row: dict) -> None:
    run.event(row)
