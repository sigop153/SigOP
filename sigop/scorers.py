"""Grading a prediction against a target. The answer is cut out by its tags
("[DIAGNOSIS]diabetes[/DIAGNOSIS]"), then compared:

    exact   equal after normalisation (lower case, single spaces, no trailing punctuation)
    set     split on separators into label sets, which must be equal; set F1 alongside

These are the Meta-Harness text-classification evaluators, kept so scores match the paper.
"""
import re
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_SEPARATORS = (";", "；", ",", "，", "、")


@dataclass(frozen=True)
class Score:
    correct: bool            # what the search optimises
    f1: float                # 1 or 0 for exact match, the set F1 for set match


@dataclass(frozen=True)
class AnswerTags:
    """Markers around the final answer. An empty close runs the answer to the end of the text."""
    open: str
    close: str
    # Regex whose group 1 is the answer, tried when the tags are missing. Symptom2Disease's
    # evaluator accepted "diagnosis: X" this way.
    fallback_pattern: str = ""


Scorer = Callable[[str, str], Score]


def extract_answer(text: str, tags: AnswerTags) -> str:
    """Between the tags; else the fallback pattern's match; else the whole text."""
    text = text.strip()
    if tags.open and tags.open in text:
        after = text.split(tags.open, 1)[1]
        if tags.close and tags.close in after:
            after = after.split(tags.close, 1)[0]
        return after.strip()
    if tags.fallback_pattern:
        match = re.search(tags.fallback_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[.!?]+$", "", value)


def exact_scorer(tags: AnswerTags) -> Scorer:
    def score(prediction: str, target: str) -> Score:
        correct = normalize(extract_answer(prediction, tags)) == normalize(target)
        return Score(correct=correct, f1=1.0 if correct else 0.0)
    return score


def set_scorer(tags: AnswerTags, separators: tuple[str, ...] = DEFAULT_SEPARATORS) -> Scorer:
    def labels_of(text: str) -> set[str]:
        answer = extract_answer(text, tags)
        for separator in separators:
            if separator in answer:
                return {part.strip() for part in answer.split(separator) if part.strip()}
        return {answer} if answer else set()

    def score(prediction: str, target: str) -> Score:
        predicted = labels_of(prediction)
        expected = labels_of(target)
        true_positive = len(predicted & expected)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return Score(correct=predicted == expected, f1=f1)
    return score


SCORERS = {"exact": exact_scorer, "set": set_scorer}
