"""The harness interface, which every candidate the proposer writes must implement.

A harness is one Python file in the run's harnesses/ folder holding a Harness subclass:

    learn_from_batch(results)   called once per proposal item with the item and its target;
                                build whatever memory the harness keeps
    predict(input)              answer one item through the executor; returns the answer
                                and a dict of diagnostics that go into the trace
    get_state() / set_state()   the memory as a JSON string

The executor is reached only through self.call_llm(prompt). The interface is Meta-Harness's
MemorySystem; the two seed harnesses are its no_memory and fewshot_all baselines.
"""
import hashlib
import importlib.util
import inspect
import json
import random
import re
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

Executor = Callable[[str], str]


@dataclass(frozen=True)
class LearnItem:
    """One proposal item as learn_from_batch receives it. prediction equals ground_truth:
    the proposal split is learnt with its targets visible."""
    input: str                 # the prompt as the executor sees it
    raw_question: str          # the item alone, without the prompt template
    ground_truth: str
    prediction: str
    was_correct: bool = True


@dataclass
class Prediction:
    answer: str
    full_response: str
    extra: dict = field(default_factory=dict)


class Harness(ABC):
    def __init__(self, llm: Executor):
        self._llm = llm
        self._local = threading.local()

    def call_llm(self, prompt: str) -> str:
        self._local.last_prompt = prompt
        return self._llm(prompt)

    def last_prompt(self) -> str:
        return getattr(self._local, "last_prompt", "")

    @abstractmethod
    def predict(self, input: str) -> Prediction: ...

    @abstractmethod
    def learn_from_batch(self, results: list[LearnItem]) -> None: ...

    @abstractmethod
    def get_state(self) -> str: ...

    def set_state(self, state: str) -> None:
        pass


def extract_json_field(text: str, name: str, default: str = "") -> str:
    """A field of the JSON object the executor was asked to answer with. Tolerates code
    fences, surrounding prose and trailing commas; falls back to a regex on the key."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get(name, default))
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return str(data.get(name, default))
        except json.JSONDecodeError:
            pass
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth, position, in_string = 1, start + 1, False
        while position < len(text) and depth > 0:
            character = text[position]
            if character == '"' and text[position - 1] != "\\":
                in_string = not in_string
            elif not in_string:
                depth += 1 if character == "{" else (-1 if character == "}" else 0)
            position += 1
        if depth == 0:
            candidate = re.sub(r",\s*([\]}])", r"\1", text[start:position])
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return str(data.get(name, default))
            except json.JSONDecodeError:
                pass
    found = re.findall(rf'"{name}"\s*:\s*"([^"]*)"', text)
    return found[-1] if found else default


NO_MEMORY_PROMPT = """Answer the following question.

{input}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning process]",
  "final_answer": "[Your concise final answer here]"
}}
"""


class NoMemory(Harness):
    """Prompts the executor directly and learns nothing."""

    def predict(self, input: str) -> Prediction:
        response = self.call_llm(NO_MEMORY_PROMPT.format(input=input))
        return Prediction(answer=extract_json_field(response, "final_answer"), full_response=response)

    def learn_from_batch(self, results: list[LearnItem]) -> None:
        pass

    def get_state(self) -> str:
        return "{}"


FEWSHOT_PROMPT = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""

FEWSHOT_MAX_CHARS = 30000


class FewShotAll(Harness):
    """Every learnt item as a Q/A demonstration, shuffled per input, up to a character cap.
    The seed harness: the search starts from it."""

    def __init__(self, llm: Executor):
        super().__init__(llm)
        self.examples: list[LearnItem] = []

    def examples_section(self, seed: int | None = None) -> str:
        chosen = list(self.examples)
        if seed is not None:
            random.Random(seed).shuffle(chosen)
        parts = []
        total = 0
        for example in chosen:
            part = f"Q: {example.raw_question}\nA: {example.ground_truth}"
            if total + len(part) > FEWSHOT_MAX_CHARS:
                break
            parts.append(part)
            total += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> Prediction:
        seed = int.from_bytes(hashlib.sha256(input.encode()).digest()[:8], "big")
        prompt = FEWSHOT_PROMPT.format(examples_section=self.examples_section(seed), input=input)
        response = self.call_llm(prompt)
        return Prediction(answer=extract_json_field(response, "final_answer"), full_response=response,
                          extra={"num_examples": len(self.examples)})

    def learn_from_batch(self, results: list[LearnItem]) -> None:
        self.examples.extend(results)

    def get_state(self) -> str:
        return json.dumps([example.__dict__ for example in self.examples])

    def set_state(self, state: str) -> None:
        self.examples = [LearnItem(**row) for row in json.loads(state)]


SEEDS = {"no_memory": NoMemory, "fewshot_all": FewShotAll}

# The import lines every seed file starts with. The skill text shows the same lines, so a
# candidate copied from a seed keeps the prompt constants and modules the code uses.
SEED_HEADER = ("from sigop.harness import Executor, Harness, LearnItem, Prediction, extract_json_field, "
               "FEWSHOT_PROMPT, FEWSHOT_MAX_CHARS, NO_MEMORY_PROMPT\n"
               "import hashlib\nimport json\nimport random\n")


def load_harness(path: Path, llm: Executor) -> Harness:
    """Import a harness file and instantiate its one Harness subclass."""
    spec = importlib.util.spec_from_file_location(f"harness_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    classes = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
               if issubclass(obj, Harness) and obj is not Harness and obj.__module__ == module.__name__]
    if len(classes) != 1:
        raise ValueError(f"{path} must define exactly one Harness subclass, found {len(classes)}")
    return classes[0](llm)


def write_seed(name: str, directory: Path) -> Path:
    """Copy a seed harness's source into the run's harnesses/ folder as a standalone file,
    so the proposer can read and copy it like any candidate."""
    source = inspect.getsource(SEEDS[name])
    path = directory / f"{name}.py"
    path.write_text(SEED_HEADER + "\n\n" + source, encoding="utf-8")
    return path
