"""Build the built-in objective folders from the paper's data. Maintainers only.

    python scripts/import_paper_splits.py --reference-run /path/to/code/meta-harness/reference-run

Maps the Meta-Harness text-classification data and the paper's monitor splits onto the four
paper splits (see objectives/README.md) and writes, per objective,
{proposal,train,val,test}.jsonl and manifest.json. objective.toml and TASK.md are untouched.
"""
import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBJECTIVES = HERE.parent / "objectives"
LAWBENCH_MEMORY_SIZE = 150          # the paper's LawBench arms ran Meta-Harness with num_train 150
S2D_VAL_SIZE = 300                  # the monitor file has 301 rows; the paper reports 300, the last row is dropped
SHUFFLE_SEED = 42                   # the reference loader's own seed


@dataclass(frozen=True)
class Row:
    input: str
    target: str


@dataclass(frozen=True)
class SourceFiles:
    """Where one task's four splits come from, as Meta-Harness and the paper laid them out."""
    name: str
    memory: Path                    # Meta-Harness train.jsonl: the proposer's memory
    search: Path                    # Meta-Harness val.jsonl: the 50-item search set
    monitor: Path                   # the paper's held-out val
    test: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-run", required=True, type=Path,
                        help="the code/meta-harness/reference-run folder of the paper repository")
    arguments = parser.parse_args()
    for source in sources(arguments.reference_run):
        write_objective(source)


def sources(reference_run: Path) -> list[SourceFiles]:
    data = reference_run / "baseline/meta-harness-ref/reference_examples/text_classification/data"
    return [
        SourceFiles("s2d",
                    memory=data / "symptom_diagnosis/train.jsonl",
                    search=data / "symptom_diagnosis/val.jsonl",
                    monitor=reference_run / "splits/data/symptom_diagnosis/monitor.jsonl",
                    test=data / "symptom_diagnosis/test.jsonl"),
        SourceFiles("lawbench",
                    memory=data / "crime_prediction/train.jsonl",
                    search=data / "crime_prediction/val.jsonl",
                    monitor=reference_run / "splits/data_shuffle_holdout/crime_prediction/monitor.jsonl",
                    test=data / "crime_prediction/test.jsonl"),
    ]


@dataclass(frozen=True)
class Split:
    name: str
    rows: list[Row]


def write_objective(source: SourceFiles) -> None:
    proposal = read_rows(source.memory)
    if source.name == "lawbench":
        # The loader shuffles the 200 memory rows and keeps the first num_train; the other 50
        # are the monitor, so the proposal split is exactly what the proposer could read.
        random.Random(SHUFFLE_SEED).shuffle(proposal)
        proposal = proposal[:LAWBENCH_MEMORY_SIZE]
    val = read_rows(source.monitor)
    if source.name == "s2d":
        val = val[:S2D_VAL_SIZE]
    splits = [Split("proposal", proposal), Split("train", read_rows(source.search)),
              Split("val", val), Split("test", read_rows(source.test))]
    overlaps = count_overlaps(splits, source.name)
    folder = OBJECTIVES / source.name
    folder.mkdir(parents=True, exist_ok=True)
    sizes = {}
    digests = {}
    for split in splits:
        path = folder / f"{split.name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in split.rows:
                handle.write(json.dumps({"input": row.input, "target": row.target}, ensure_ascii=False) + "\n")
        sizes[split.name] = len(split.rows)
        digests[split.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"sizes": sizes, "sha256": digests, "overlapping_inputs": overlaps}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(source.name, sizes)


def read_rows(path: Path) -> list[Row]:
    """The source files hold question/answer; LawBench answers may carry a '罪名:' prefix,
    which the Meta-Harness loader strips, so it is stripped here once."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            target = str(raw["answer"])
            if "罪名:" in target:
                target = target.split("罪名:")[-1].strip()
            rows.append(Row(input=str(raw["question"]), target=target))
    return rows


def count_overlaps(splits: list[Split], objective: str) -> int:
    """Inputs that appear in two splits. The paper's data is kept exactly as its runs scored
    it, so an overlap is reported and recorded in the manifest rather than removed; one
    Symptom2Disease question sits in both the search set and the test set as shipped by
    Meta-Harness. The val split must stay disjoint from every other, or OVERFIT would be
    judged on data the proposer could have read: that case aborts."""
    owner = {}                       # normalised input -> the split it was first seen in
    overlaps = 0
    for split in splits:
        for row in split.rows:
            key = " ".join(row.input.split())
            if key in owner and owner[key] != split.name:
                if "val" in (owner[key], split.name):
                    raise SystemExit(f"{objective}: val shares an input with {owner[key]}: {key[:60]!r}")
                print(f"{objective}: WARNING an input is in both {owner[key]} and {split.name}: {key[:60]!r}")
                overlaps += 1
            owner.setdefault(key, split.name)
    return overlaps


if __name__ == "__main__":
    main()
