"""The two rules, and the notes they send to the proposer.

    train up, val down past the tolerance  -> OVERFIT: do not adopt, tell the proposer
    train up, otherwise                    -> adopt
    train flat or down                     -> do not adopt (Meta-Harness's own rule)

    train of the incumbent not improved for `patience` iterations -> PLATEAU: tell the proposer

OVERFIT outranks PLATEAU: one signal is carried at a time, and PLATEAU is only raised when
none is pending. The plateau counter watches the incumbent's train score, not val, and
resets both when train improves and after PLATEAU fires. Neither signal stops the loop.

The tolerance is a dead band: below it a val change is noise. It must also clear the
split's resolution; on 50 items one item is 2pp.

A signal is one file in the run's reports/ folder, which the proposer reads every
iteration. The val score and its change are never written: the proposer gets the verdict
only, so it cannot steer by the held-out split.
"""
from dataclasses import dataclass
from pathlib import Path

OVERFIT = "OVERFIT"
PLATEAU = "PLATEAU"

OVERFIT_TEXT = """# Signal for iteration {iteration}: OVERFIT

{count} candidate{plural} from the previous iteration **improved the train score but did not
generalise to a held-out split you cannot see**:

{rows}

That is the signature of fitting the {train_size}-item train set rather than learning something
that generalises. **{those} not adopted**: the incumbent is unchanged, and the file{plural}
{is_are} still in `harnesses/` if you want to read {it_them}.

For this iteration: do not refine {those_lower} or tune {their} parameters. Change mechanism:
a different retrieval rule, a different memory representation, a different way of organising
what goes in the prompt. A variant that scores a little higher on train the same way will be
rejected the same way.
"""

PLATEAU_TEXT = """# Signal for iteration {iteration}: PLATEAU

The incumbent's train score has not improved for {stale} iterations (currently {best:.1%} with
`{incumbent}`). The directions tried recently are not paying off.

For this iteration: do not make another variant of the incumbent. Pick a mechanism that has
not been tried in this run (`history.jsonl` lists what has been) and prefer an exploratory
candidate over an incremental one.
"""


BUDGET_TEXT = """# Time budget for iteration {iteration}

**This session is stopped after {minutes} minutes.** At the limit the session is stopped and the
harness files it has written are evaluated as they are, so write the file early and refine it in
place rather than prototype until the stop.

{history}

Spend the first part of the session diagnosing and the rest writing. Prototyping several
mechanisms and measuring each one does not fit in {minutes} minutes. Pick the most promising
mechanism early, write `{target}`, and validate it. A simple harness that is written and
validated beats a better one that was still being prototyped when the session was stopped.
"""


@dataclass(frozen=True)
class Budget:
    """The wall-clock a proposal gets, and what the previous ones did with it."""
    minutes: int                 # the limit
    typical_minutes: int         # median wall-clock of this run's proposals that wrote pending.json
    timed_out: int               # proposals this run stopped at the limit so far
    attempted: int               # proposals so far


@dataclass(frozen=True)
class Decision:
    """One candidate that was scored on val against the incumbent."""
    adopt: bool
    signal: str | None            # OVERFIT or None


@dataclass(frozen=True)
class PlateauState:
    """Carried between iterations: the best train score seen and how long since it moved."""
    best_train: float
    stale: int
    signal: str | None            # the signal pending for the next proposal


@dataclass(frozen=True)
class Rejected:
    """A candidate OVERFIT turned away, as the note describes it."""
    name: str
    delta_train: float


def classify(delta_train: float, delta_val: float, tolerance: float) -> Decision:
    improves = delta_train > 0
    if improves and delta_val <= -tolerance:
        return Decision(adopt=False, signal=OVERFIT)
    return Decision(adopt=improves, signal=None)


def plateau(incumbent_train: float, state: PlateauState, patience: int) -> PlateauState:
    """Once per iteration, after adoption."""
    if incumbent_train > state.best_train:
        return PlateauState(best_train=incumbent_train, stale=0, signal=state.signal)
    stale = state.stale + 1
    if stale < patience:
        return PlateauState(best_train=state.best_train, stale=stale, signal=state.signal)
    # Fires, or yields to a pending OVERFIT; the counter resets either way
    return PlateauState(best_train=state.best_train, stale=0, signal=state.signal or PLATEAU)


def write_overfit(reports: Path, iteration: int, rejected: list[Rejected], train_size: int) -> Path:
    many = len(rejected) > 1
    body = OVERFIT_TEXT.format(
        iteration=iteration, count=len(rejected), plural="s" if many else "",
        rows="\n".join(f"- `{r.name}`: train score {r.delta_train:+.1%}" for r in rejected),
        train_size=train_size, those="They were" if many else "It was",
        those_lower="them" if many else "it", is_are="are" if many else "is",
        it_them="them" if many else "it", their="their" if many else "its")
    return write_note(reports, iteration, OVERFIT, body)


def write_plateau(reports: Path, iteration: int, stale: int, best: float, incumbent: str) -> Path:
    body = PLATEAU_TEXT.format(iteration=iteration, stale=stale, best=best, incumbent=incumbent)
    return write_note(reports, iteration, PLATEAU, body)


def write_budget(reports: Path, iteration: int, budget: Budget, target: str) -> Path:
    """The time-budget note for the next proposal. Not a signal: it is written every
    iteration, never retired, and leaves the OVERFIT/PLATEAU state alone."""
    if budget.timed_out:
        history = (f"In this run **{budget.timed_out} of {budget.attempted} proposals have already been stopped**"
                   f" this way. The ones that finished took about {budget.typical_minutes} minutes, so the budget"
                   " is tight rather than generous.")
    else:
        history = (f"Proposals that finish typically take about {budget.typical_minutes} minutes, so the budget"
                   " is tight rather than generous.")
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"iter{iteration:02d}_TIME_BUDGET.md"
    path.write_text(BUDGET_TEXT.format(iteration=iteration, minutes=budget.minutes, history=history, target=target),
                    encoding="utf-8")
    return path


def write_note(reports: Path, iteration: int, signal: str, body: str) -> Path:
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"iter{iteration:02d}_SIGNAL_{signal}.md"
    path.write_text(body, encoding="utf-8")
    return path


def retire_signals(reports: Path, keep_iteration: int) -> int:
    """A signal is addressed to one proposal. Older ones move to reports/.archive so the
    proposer never reads a spent flag beside the current state of the run."""
    if not reports.is_dir():
        return 0
    archive = reports / ".archive"
    moved = 0
    for path in reports.glob("iter*_SIGNAL_*.md"):
        if int(path.name[4:6]) < keep_iteration:
            archive.mkdir(exist_ok=True)
            path.rename(archive / path.name)
            moved += 1
    return moved
