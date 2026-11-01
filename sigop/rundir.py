"""Where a run keeps its files.

    runs/<name>/                   the proposer's working folder; it reads and writes here
        TASK.md                    the objective's briefing
        harnesses/<name>.py        the seed and every candidate
        history.jsonl              one row per candidate: iteration, train score, outcome
        incumbent.json             the current h*: its name and train score
        scores/<name>/train.json   each candidate's train score
        scores/<name>/train_trace.jsonl   its per-item train predictions
        reports/                   the proposer's own notes, and the SIGNAL files
        pending.json               what the proposer hands back: its candidates
        sessions/                  proposer session logs, and iterNN.json with each session's
                                   wall-clock and outcome
    runs/<name>.heldout/           never shown to the proposer
        scores/<name>/{val,test}.json
        detector.jsonl             every decision the loop made
    runs/.cache/                   executor response cache, shared by every run

The proposer is a coding agent with shell access and is not sandboxed by this package, so
the heldout folder is kept out of its working folder but not out of its reach. Run it under
a sandbox (the paper used bubblewrap) when that matters.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def heldout(self) -> Path:
        return self.root.parent / f"{self.root.name}.heldout"

    @property
    def harnesses(self) -> Path:
        return self.root / "harnesses"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def history(self) -> Path:
        return self.root / "history.jsonl"

    @property
    def incumbent(self) -> Path:
        return self.root / "incumbent.json"

    @property
    def pending(self) -> Path:
        return self.root / "pending.json"

    @property
    def task(self) -> Path:
        return self.root / "TASK.md"

    @property
    def detector(self) -> Path:
        return self.heldout / "detector.jsonl"

    @property
    def cli_homes(self) -> Path:
        return self.heldout / "cli_homes"          # one Claude Code config dir per session

    def session_meta(self, iteration: int) -> Path:
        return self.sessions / f"iter{iteration:02d}.json"

    def event(self, row: dict) -> None:
        self.append(self.detector, {"t": time.strftime("%FT%T"), **row})

    def score_path(self, harness: str, split: str) -> Path:
        base = self.root if split == "train" else self.heldout
        return base / "scores" / harness / f"{split}.json"

    def trace_path(self, harness: str) -> Path:
        return self.root / "scores" / harness / "train_trace.jsonl"

    def create(self) -> None:
        for folder in (self.harnesses, self.reports, self.sessions, self.heldout / "scores"):
            folder.mkdir(parents=True, exist_ok=True)

    def append(self, path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_rows(self, path: Path) -> list[dict]:
        """Tolerant: the proposer can write history.jsonl too, and has left junk in it."""
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows
