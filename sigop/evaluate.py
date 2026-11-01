"""Train a harness on the proposal split, then score it on a split.

Training is Meta-Harness's offline mode: one epoch, one item per learn_from_batch call, the
target visible as the prediction. Scoring runs predict over the split concurrently and
grades each answer with the objective's scorer. With samples > 1 (IMO-style tasks) every
item is predicted that many times and its correctness is the mean, i.e. averaged pass@1.
"""
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from sigop.harness import Harness, LearnItem
from sigop.objective import Item, Objective


@dataclass(frozen=True)
class Record:
    """One scored item, as written to the trace the proposer reads."""
    input: str
    target: str
    prediction: str
    correct: float               # 1.0 or 0.0, or the mean over samples
    f1: float
    full_response: str
    prompt_length: int


@dataclass(frozen=True)
class Result:
    accuracy: float
    correct: float
    total: int
    seconds: float
    records: list[Record]

    def summary(self) -> dict:
        return {"accuracy": self.accuracy, "correct": self.correct, "total": self.total,
                "seconds": round(self.seconds, 1)}


def train(harness: Harness, objective: Objective) -> None:
    for item in objective.splits.proposal:
        harness.learn_from_batch([learn_item(item, objective)])


def learn_item(item: Item, objective: Objective) -> LearnItem:
    return LearnItem(input=objective.spec.prompt.format(input=item.input), raw_question=item.input,
                     ground_truth=item.target, prediction=item.target)


def score(harness: Harness, objective: Objective, items: list[Item], concurrency: int = 32,
          on_item: Callable[[], None] | None = None) -> Result:
    """on_item is called after every scored item: the loop's stall detector listens to it."""
    started = time.time()
    samples = objective.spec.sampling.samples

    def one(item: Item) -> Record:
        prompt = objective.spec.prompt.format(input=item.input)
        predictions = [harness.predict(prompt) for _ in range(samples)]
        scores = [objective.scorer(prediction.answer, item.target) for prediction in predictions]
        if on_item:
            on_item()
        return Record(input=item.input, target=item.target, prediction=predictions[0].answer,
                      correct=sum(s.correct for s in scores) / samples, f1=sum(s.f1 for s in scores) / samples,
                      full_response=predictions[0].full_response, prompt_length=len(harness.last_prompt()))

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(items)))) as pool:
        records = list(pool.map(one, items))
    correct = sum(record.correct for record in records)
    return Result(accuracy=correct / len(items) if items else 0.0, correct=correct, total=len(items),
                  seconds=time.time() - started, records=records)


def write_result(result: Result, path: Path, trace: Path | None) -> None:
    """The summary as JSON; the per-item records as JSONL when a trace path is given (train
    only: val and test traces would show the proposer the held-out items)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8")
    if trace is not None:
        with trace.open("w", encoding="utf-8") as handle:
            for record in result.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
