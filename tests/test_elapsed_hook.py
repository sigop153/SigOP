import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sigop.elapsed_hook import note_for

HOOK = Path(__file__).resolve().parent.parent / "sigop" / "elapsed_hook.py"


def test_three_bands():
    assert note_for(14, 40) == ("[time] 14 of 40 minutes used; at 40 the session is stopped and the harness files "
                                "written so far are evaluated.")
    assert note_for(21, 40).endswith("evaluated. Start writing the harness file.")
    assert note_for(31, 40).endswith("evaluated. Write the file now; whatever is on disk at 40 is evaluated.")


def test_hook_reads_the_clock_beside_itself(tmp_path):
    shutil.copy(HOOK, tmp_path / "elapsed_hook.py")
    (tmp_path / "session_clock").write_text(f"{int(time.time()) - 25 * 60} 2400\n", encoding="utf-8")
    out = subprocess.run([sys.executable, str(tmp_path / "elapsed_hook.py")], check=True, capture_output=True, text=True)
    note = json.loads(out.stdout)["hookSpecificOutput"]["additionalContext"]
    assert note.startswith("[time] 25 of 40 minutes used") and "Start writing" in note


def test_hook_is_silent_without_a_clock(tmp_path):
    shutil.copy(HOOK, tmp_path / "elapsed_hook.py")
    out = subprocess.run([sys.executable, str(tmp_path / "elapsed_hook.py")], check=True, capture_output=True, text=True)
    assert out.stdout == ""
