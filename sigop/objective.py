"""A objective is a folder:

    objectives/<name>/
        objective.toml   name, scorer, answer tags, executor sampling, prompt around each input
        TASK.md          what the proposer is told about the task
        proposal.jsonl   read by the proposer (for a retrieval task, the corpus)
        train.jsonl      the objective; h* is selected on it
        val.jsonl        the monitor; detector only, never shown to the proposer
        test.jsonl       held out, scored once per adopted h*

Split lines are {"input": "...", "target": "..."}. Split names follow the paper; Meta-Harness
calls train the search set and its code calls it val. A split may instead be a CSV with input
and target columns (proposal.csv, ...). Without objective.toml the folder is graded by exact
match on [ANSWER]...[/ANSWER].

A single CSV with input,target columns is shuffled with a fixed seed, split into four equal
parts and graded the same way.
"""
import csv
import json
import random
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from sigop.scorers import DEFAULT_SEPARATORS, SCORERS, AnswerTags, Scorer

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "objectives"
SPLIT_NAMES = ("proposal", "train", "val", "test")
SEED = 42
DEFAULT_ANSWER_TAGS = AnswerTags(open="[ANSWER]", close="[/ANSWER]")
DEFAULT_PROMPT = """Answer the question below. Think step by step, then give your final answer in the format:
[ANSWER]your answer[/ANSWER]

## Question
{input}"""


@dataclass(frozen=True)
class Item:
    input: str
    target: str




@dataclass(frozen=True)
class Splits:
    proposal: list[Item]
    train: list[Item]
    val: list[Item]
    test: list[Item]

    def of(self, name: str) -> list[Item]:
        return getattr(self, name)


@dataclass(frozen=True)
class Sampling:
    """How the executor is run on every item. Classification runs once at temperature 0,
    so a harness scores the same each time; a maths task samples several times at
    temperature 1 and averages pass@1."""
    temperature: float = 0.0
    samples: int = 1


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    scorer: str                       # a key of scorers.SCORERS
    tags: AnswerTags
    prompt: str                       # wraps each input; must contain {input}
    sampling: Sampling = Sampling()
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    labels: tuple[str, ...] = ()      # the label set, when the task has one (for the proposer)


@dataclass(frozen=True)
class Objective:
    spec: ObjectiveSpec
    splits: Splits
    scorer: Scorer
    task: str                         # TASK.md, or a one-line stand-in without one
    root: Path | None                 # the folder, None for a single CSV
    source: str = ""                  # what load_objective was given, so a child process can reload it
    seed: int = SEED

    def sizes(self) -> str:
        return ", ".join(f"{name} {len(self.splits.of(name))}" for name in SPLIT_NAMES)


def builtin_names() -> list[str]:
    return sorted(path.name for path in BUILTIN_DIR.iterdir() if (path / "objective.toml").exists())


def load_objective(name_or_path: str, seed: int = SEED) -> Objective:
    """A built-in name, an objective folder, or a CSV with input,target columns."""
    return replace(resolve_objective(name_or_path, seed), source=name_or_path, seed=seed)


def resolve_objective(name_or_path: str, seed: int) -> Objective:
    path = Path(name_or_path)
    if (BUILTIN_DIR / name_or_path / "objective.toml").exists():
        return load_folder(BUILTIN_DIR / name_or_path)
    if path.is_dir():
        return load_folder(path)
    if path.suffix.lower() == ".csv" and path.is_file():
        return load_csv(path, seed)
    raise FileNotFoundError(
        f"{name_or_path!r} is not a built-in objective ({', '.join(builtin_names())}), "
        "a objective folder, or a CSV file")


def load_folder(root: Path) -> Objective:
    """A folder with the four splits, as proposal/train/val/test .jsonl or .csv files.
    objective.toml names the scorer and prompt; without it, exact match on [ANSWER] tags."""
    toml = root / "objective.toml"
    spec = read_spec(toml) if toml.exists() else ObjectiveSpec(name=root.name, scorer="exact",
                                                                 tags=DEFAULT_ANSWER_TAGS, prompt=DEFAULT_PROMPT)
    files = []
    for name in SPLIT_NAMES:
        file = next((root / f"{name}{suffix}" for suffix in (".jsonl", ".csv") if (root / f"{name}{suffix}").exists()), None)
        if file is None:
            raise FileNotFoundError(f"{root} has neither {name}.jsonl nor {name}.csv")
        files.append(file)
    splits = Splits(*(read_items(file) for file in files))
    task_path = root / "TASK.md"
    task = task_path.read_text(encoding="utf-8") if task_path.exists() else f"# {spec.name}\n"
    return Objective(spec=spec, splits=splits, scorer=make_scorer(spec), task=task, root=root)


def read_items(path: Path) -> list[Item]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows and ("input" not in rows[0] or "target" not in rows[0]):
            raise ValueError(f"{path} needs input and target columns")
        return [Item(input=row["input"], target=row["target"]) for row in rows]
    return read_jsonl(path)


def load_csv(path: Path, seed: int) -> Objective:
    """A single CSV becomes an exact-match objective named after the file."""
    spec = exact_match_spec(path.stem)
    return Objective(spec=spec, splits=equal_splits(read_csv(path), seed), scorer=make_scorer(spec),
                     task=stand_in_task(spec.name), root=None)


def exact_match_spec(name: str) -> ObjectiveSpec:
    return ObjectiveSpec(name=name, scorer="exact", tags=DEFAULT_ANSWER_TAGS, prompt=DEFAULT_PROMPT)


def stand_in_task(name: str) -> str:
    return f"# {name}\n\nMap each input to its target label.\n"


def equal_splits(items: list[Item], seed: int) -> Splits:
    """Four equal parts of the shuffled rows, in SPLIT_NAMES order; sizes differ by at most one row."""
    if len(items) < len(SPLIT_NAMES):
        raise ValueError(f"{len(items)} rows cannot fill the four splits {', '.join(SPLIT_NAMES)}")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cuts = [len(shuffled) * part // len(SPLIT_NAMES) for part in range(len(SPLIT_NAMES) + 1)]
    return Splits(**{name: shuffled[cuts[index]:cuts[index + 1]] for index, name in enumerate(SPLIT_NAMES)})


def read_csv(path: Path) -> list[Item]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "input" not in rows[0] or "target" not in rows[0]:
        raise ValueError(f"{path} needs input and target columns")
    return [Item(input=row["input"], target=row["target"]) for row in rows]


def read_spec(path: Path) -> ObjectiveSpec:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    scorer = raw.get("scorer", "exact")
    if scorer not in SCORERS:
        raise ValueError(f"{path}: scorer {scorer!r} is not one of {', '.join(SCORERS)}")
    prompt = raw.get("prompt", DEFAULT_PROMPT)
    if "{input}" not in prompt:
        raise ValueError(f"{path}: prompt must contain {{input}}")
    tags = AnswerTags(open=raw.get("answer_open", DEFAULT_ANSWER_TAGS.open),
                      close=raw.get("answer_close", DEFAULT_ANSWER_TAGS.close),
                      fallback_pattern=raw.get("answer_fallback_pattern", ""))
    sampling = Sampling(temperature=float(raw.get("temperature", 0.0)), samples=int(raw.get("samples", 1)))
    return ObjectiveSpec(name=raw.get("name", path.parent.name), scorer=scorer, tags=tags, prompt=prompt,
                         sampling=sampling, separators=tuple(raw.get("separators", DEFAULT_SEPARATORS)),
                         labels=tuple(raw.get("labels", ())))


def read_jsonl(path: Path) -> list[Item]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "input" not in row or "target" not in row:
                raise ValueError(f"{path}:{line_number} needs input and target fields")
            items.append(Item(input=str(row["input"]), target=str(row["target"])))
    return items


def make_scorer(spec: ObjectiveSpec) -> Scorer:
    if spec.scorer == "set":
        return SCORERS["set"](spec.tags, spec.separators)
    return SCORERS["exact"](spec.tags)
