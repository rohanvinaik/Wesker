"""The action's report survives a mutant-killed stdout (badge-run EBADF, 2026-08-05).

Self-profiling runs mutants OF the fd-capture machinery in-process; a mutant
that breaks restoration leaves fd 1 closed for the rest of the process, and
the whole-codebase badge run died at its first ::warning annotation with
EBADF — after completing the entire measurement. The reentry hand snapshots
the real descriptors before profiling and re-enters them before the report
speaks. Exercised in a subprocess because the damage under test IS
process-global fd state.
"""

from __future__ import annotations

import subprocess
import sys

_SCRIPT = """
import os, sys
from Wesker.action import _stream_reentry

reenter = _stream_reentry()

# The hostile-mutant damage, both layers: the Python object and the fd.
sys.stdout.close()
os.close(1)

reenter()
print("::warning file=x.py,line=1::report-survives")
print("banner-survives")
"""


def test_report_lines_survive_a_killed_stdout():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "::warning file=x.py,line=1::report-survives" in proc.stdout
    assert "banner-survives" in proc.stdout


def test_reentry_is_harmless_on_healthy_streams():
    script = """
import sys
from Wesker.action import _stream_reentry
reenter = _stream_reentry()
print("before")
reenter()
print("after")
sys.stderr.write("err-after\\n")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "before" in proc.stdout
    assert "after" in proc.stdout
    assert "err-after" in proc.stderr
