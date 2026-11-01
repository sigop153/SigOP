"""Claude Code PostToolUse hook: tell the proposer how long its session has run.

The launcher writes `session_clock` beside this file: `<start epoch> <limit seconds>`. After
every tool call this prints a one-line note that the CLI adds to the model's context, so the
proposer always knows where it is in this session. A note read once at the start (the
TIME_BUDGET report) does not survive a hundred tool calls of diagnosis; this does.

No dependencies: it runs under the system python inside the sandbox. A missing clock file
means no note.
"""
import json
import os
import sys
import time


def main() -> None:
    clock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_clock")
    try:
        with open(clock_path, encoding="utf-8") as handle:
            start, limit_seconds = (float(x) for x in handle.read().split()[:2])
    except (OSError, ValueError):
        sys.exit(0)
    note = note_for((time.time() - start) / 60, limit_seconds / 60)
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": note}}))


def note_for(used_minutes: float, limit_minutes: float) -> str:
    used, limit = int(used_minutes), int(limit_minutes)
    note = (f"[time] {used} of {limit} minutes used; at {limit} the session is stopped and the harness "
            "files written so far are evaluated.")
    if used_minutes >= limit_minutes - 10:
        note += f" Write the file now; whatever is on disk at {limit} is evaluated."
    elif used_minutes >= limit_minutes - 20:
        note += " Start writing the harness file."
    return note


if __name__ == "__main__":
    main()
