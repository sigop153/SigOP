from pathlib import Path

import pytest

from sigop.evaluate import learn_item, score, train, write_result
from sigop.harness import FewShotAll, LearnItem, NoMemory, extract_json_field, load_harness, write_seed
from sigop.objective import load_objective


def echo_target(prompt: str) -> str:
    """A stub executor that answers with the first 'A: <target>' it finds in the prompt,
    so FewShotAll on its own learnt items is always right."""
    for line in prompt.splitlines():
        if line.startswith("A: "):
            return '{"reasoning": "copied", "final_answer": "' + line[3:] + '"}'
    return '{"final_answer": "unknown"}'


def test_extract_json_field_handles_fences_prose_and_trailing_commas():
    assert extract_json_field('{"final_answer": "diabetes"}', "final_answer") == "diabetes"
    assert extract_json_field('```json\n{"final_answer": "malaria"}\n```', "final_answer") == "malaria"
    assert extract_json_field('Sure. {"reasoning": "x", "final_answer": "dengue",}', "final_answer") == "dengue"
    assert extract_json_field("no json here", "final_answer", default="?") == "?"


def test_no_memory_answers_through_the_executor():
    harness = NoMemory(lambda prompt: '{"final_answer": "typhoid"}')
    prediction = harness.predict("fever for a week")
    assert prediction.answer == "typhoid"
    assert harness.last_prompt().startswith("Answer the following question.")


def test_fewshot_all_learns_and_shows_examples():
    harness = FewShotAll(echo_target)
    harness.learn_from_batch([LearnItem("prompt", "I itch", "allergy", "allergy")])
    assert "Q: I itch\nA: allergy" in harness.examples_section()
    assert harness.predict("I itch again").answer == "allergy"
    restored = FewShotAll(echo_target)
    restored.set_state(harness.get_state())
    assert restored.examples == harness.examples


def test_seed_file_loads_standalone(tmp_path):
    path = write_seed("fewshot_all", tmp_path)
    harness = load_harness(path, echo_target)
    assert type(harness).__name__ == "FewShotAll"


def test_load_harness_requires_exactly_one_subclass(tmp_path):
    path = tmp_path / "two.py"
    path.write_text("from sigop.harness import Harness\n"
                    "class A(Harness):\n    pass\nclass B(Harness):\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_harness(path, echo_target)


def csv_objective(tmp_path: Path):
    rows = ["input,target"] + [f"question {i},{'yes' if i % 2 else 'no'}" for i in range(40)]
    path = tmp_path / "mine.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_objective(str(path))


def test_train_then_score_with_a_perfect_stub(tmp_path):
    objective = csv_objective(tmp_path)
    harness = FewShotAll(lambda prompt: '{"final_answer": "[ANSWER]' + ("yes" if "question 1" in prompt.split("**Problem:**")[-1] else "no") + '[/ANSWER]"}')
    train(harness, objective)
    assert len(harness.examples) == 10
    item = learn_item(objective.splits.proposal[0], objective)
    assert item.raw_question in item.input and item.prediction == item.ground_truth
    result = score(harness, objective, objective.splits.train, concurrency=4)
    assert result.total == 10 and 0.0 <= result.accuracy <= 1.0
    assert len(result.records) == 10 and result.records[0].prompt_length > 0


def test_write_result_writes_summary_and_optional_trace(tmp_path):
    objective = csv_objective(tmp_path)
    harness = NoMemory(lambda prompt: '{"final_answer": "[ANSWER]yes[/ANSWER]"}')
    result = score(harness, objective, objective.splits.val, concurrency=2)
    write_result(result, tmp_path / "s" / "val.json", None)
    write_result(result, tmp_path / "s" / "train.json", tmp_path / "s" / "train_trace.jsonl")
    assert (tmp_path / "s" / "val.json").exists()
    assert len((tmp_path / "s" / "train_trace.jsonl").read_text(encoding="utf-8").splitlines()) == 10
