import csv
import json
from pathlib import Path

import pytest

from sigop.objective import SPLIT_NAMES, builtin_names, load_objective


def test_builtins_are_listed():
    assert builtin_names() == ["lawbench", "s2d"]


@pytest.mark.parametrize("name, sizes", [
    ("s2d", {"proposal": 200, "train": 50, "val": 300, "test": 212}),
    ("lawbench", {"proposal": 150, "train": 50, "val": 50, "test": 100}),
])
def test_builtin_split_sizes_match_the_paper(name, sizes):
    objective = load_objective(name)
    assert {split: len(objective.splits.of(split)) for split in SPLIT_NAMES} == sizes


@pytest.mark.parametrize("name", ["s2d", "lawbench"])
def test_builtin_val_is_disjoint_from_every_other_split(name):
    objective = load_objective(name)
    val = {item.input for item in objective.splits.val}
    for split in ("proposal", "train", "test"):
        assert not val & {item.input for item in objective.splits.of(split)}, f"val overlaps {split}"


@pytest.mark.parametrize("name", ["s2d", "lawbench"])
def test_builtin_overlaps_are_the_recorded_ones(name):
    """The data is kept as the paper ran it; S2D ships one question in both train and test."""
    objective = load_objective(name)
    assert objective.root is not None
    manifest = json.loads((objective.root / "manifest.json").read_text(encoding="utf-8"))
    seen = {}
    overlaps = 0
    for split in SPLIT_NAMES:
        for item in objective.splits.of(split):
            if item.input in seen and seen[item.input] != split:
                overlaps += 1
            seen.setdefault(item.input, split)
    assert overlaps == manifest["overlapping_inputs"]


@pytest.mark.parametrize("name", ["s2d", "lawbench"])
def test_builtin_targets_score_correct_against_themselves(name):
    objective = load_objective(name)
    tags = objective.spec.tags
    for split in ("train", "val", "test"):
        for item in objective.splits.of(split):
            assert objective.scorer(f"{tags.open}{item.target}{tags.close}", item.target).correct, item.target


def test_s2d_targets_are_within_the_label_set():
    objective = load_objective("s2d")
    labels = set(objective.spec.labels)
    for split in SPLIT_NAMES:
        for item in objective.splits.of(split):
            assert item.target in labels, item.target


def test_labels_reach_the_proposer_in_the_task():
    objective = load_objective("s2d")
    assert all(label in objective.task for label in objective.spec.labels)
    assert "objective.toml" not in objective.task
    assert "Labels:" not in load_objective("lawbench").task


def test_builtin_prompt_wraps_the_input():
    objective = load_objective("lawbench")
    assert "{input}" in objective.spec.prompt
    assert objective.spec.prompt.format(input="事实").endswith("事实")


def write_csv(path: Path, rows: int, start: int = 0) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "target"])
        for index in range(start, start + rows):
            writer.writerow([f"question {index}", "yes" if index % 2 else "no"])


def test_csv_is_split_into_four_equal_parts(tmp_path):
    path = tmp_path / "mine.csv"
    write_csv(path, rows=42)
    objective = load_objective(str(path))
    assert objective.spec.name == "mine"
    assert [len(objective.splits.of(split)) for split in SPLIT_NAMES] == [10, 11, 10, 11]
    assert objective.root is None


def test_csv_split_is_deterministic(tmp_path):
    path = tmp_path / "mine.csv"
    write_csv(path, rows=40)
    first = load_objective(str(path))
    second = load_objective(str(path))
    assert first.splits.train == second.splits.train


def test_folder_of_split_csvs_keeps_the_caller_split_and_grades_exact(tmp_path):
    folder = tmp_path / "symptoms"
    folder.mkdir()
    for index, (name, rows) in enumerate(zip(SPLIT_NAMES, (20, 5, 7, 9), strict=True)):
        write_csv(folder / f"{name}.csv", rows, start=100 * index)
    objective = load_objective(str(folder))
    assert objective.spec.name == "symptoms" and objective.spec.scorer == "exact"
    assert [len(objective.splits.of(split)) for split in SPLIT_NAMES] == [20, 5, 7, 9]
    assert objective.splits.val[0].input == "question 200"


def test_folder_missing_a_split_names_it(tmp_path):
    folder = tmp_path / "symptoms"
    folder.mkdir()
    for name in ("proposal", "train", "val"):
        write_csv(folder / f"{name}.csv", 4)
    with pytest.raises(FileNotFoundError, match="neither test.jsonl nor test.csv"):
        load_objective(str(folder))


def test_csv_too_small_for_four_splits_is_refused(tmp_path):
    path = tmp_path / "mine.csv"
    write_csv(path, rows=3)
    with pytest.raises(ValueError, match="cannot fill the four splits"):
        load_objective(str(path))


def test_csv_default_scorer_is_exact_on_answer_tags(tmp_path):
    path = tmp_path / "mine.csv"
    write_csv(path, rows=40)
    objective = load_objective(str(path))
    assert objective.scorer("I think...\n[ANSWER]Yes[/ANSWER]", "yes").correct
    assert objective.scorer("[ANSWER]no[/ANSWER]", "yes").correct is False


def test_unknown_objective_names_the_builtins(tmp_path):
    with pytest.raises(FileNotFoundError, match="lawbench, s2d"):
        load_objective("nope")


def test_folder_missing_a_split_is_refused(tmp_path):
    (tmp_path / "objective.toml").write_text('name = "x"\n', encoding="utf-8")
    (tmp_path / "proposal.jsonl").write_text('{"input": "a", "target": "b"}\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="neither train.jsonl nor train.csv"):
        load_objective(str(tmp_path))


def test_folder_of_four_csvs_without_toml_uses_the_defaults(tmp_path):
    folder = tmp_path / "symptoms"
    folder.mkdir()
    for name in SPLIT_NAMES:
        (folder / f"{name}.csv").write_text("input,target\nq1,yes\nq2,no\n", encoding="utf-8")
    objective = load_objective(str(folder))
    assert objective.spec.name == "symptoms" and objective.spec.scorer == "exact"
    assert [len(objective.splits.of(name)) for name in SPLIT_NAMES] == [2, 2, 2, 2]
    assert objective.scorer("[ANSWER]Yes[/ANSWER]", "yes").correct
