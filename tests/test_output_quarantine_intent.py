"""Code under measurement does not write into the run's own output (Wesker #52, the last item).

THE DEFECT, measured in the badge run 43102482321: one line of a nested Action's banner,
`[2/24] action.pyWesker — Whole codebase — 1 file(s).`, reached the job log, spliced onto the owner's own
progress line. A test run against a mutant is called as a plain callable, outside pytest's capture of
descriptors; `print` is caught by the `sys.stdout` redirect, but `os.write(1, ...)`, a child process that
inherits the descriptor, and a nested Action's re-entered streams go straight to fd 1/2, which is the
run's log. Reproduced on a fixture project whose test does exactly those three (`python -m
Wesker.action --targets mod.py`): LEAK-FD1 six times (once inside the owner's `[1/1] mod.py` line),
LEAK-FD2 seven times, LEAK-CHILD seven times; the plain CLI the same, and with no private progress
descriptor its own lines are the ones interleaved.

THE INVARIANT. For the length of one test run (a mutant's, and the baseline's), descriptors 1 and 2 of the
measuring process are a sink; they are the run's again the moment it ends. The owner prints progress
through a private duplicate taken before profiling, and the CLI prints between evaluations, so neither
loses a line. Every descriptor-level test runs in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from Wesker.isolation import quarantined_output  # noqa: F401  (the module under test)


def _run_py(script: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        **kw,
    )


def test_descriptor_writes_inside_the_block_go_nowhere_and_outside_it_come_back():
    done = _run_py(
        """
        import os, subprocess, sys
        from Wesker.isolation import quarantined_output
        os.write(1, b"before-out\\n")
        with quarantined_output() as took:
            assert took is True
            os.write(1, b"LEAK-FD1\\n")
            os.write(2, b"LEAK-FD2\\n")
            subprocess.run(["echo", "LEAK-CHILD"], check=False)
            sys.stdout.write("LEAK-REBOUND\\n"); sys.stdout.flush()
        os.write(1, b"after-out\\n")
        os.write(2, b"after-err\\n")
        """
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == "before-out\nafter-out\n"
    assert done.stderr == "after-err\n"


def test_an_exception_inside_the_block_still_restores_the_descriptors():
    done = _run_py(
        """
        import os
        from Wesker.isolation import quarantined_output
        try:
            with quarantined_output():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        os.write(1, b"alive\\n")
        """
    )
    assert done.stdout == "alive\n", done.stderr


def test_a_nested_entry_does_nothing_and_the_outer_one_restores_exactly_what_it_saved():
    done = _run_py(
        """
        import os
        from Wesker.isolation import quarantined_output
        with quarantined_output() as outer:
            with quarantined_output() as inner:
                assert (outer, inner) == (True, False)
                os.write(1, b"LEAK-INNER\\n")
            os.write(1, b"LEAK-STILL-QUARANTINED\\n")
        os.write(1, b"back\\n")
        """
    )
    assert done.stdout == "back\n", done.stderr


def test_a_closed_descriptor_degrades_to_no_quarantine_instead_of_failing():
    done = _run_py(
        """
        import os
        from Wesker.isolation import quarantined_output
        os.close(1)
        with quarantined_output() as took:
            print("ran", file=__import__("sys").stderr)
        """
    )
    assert done.returncode == 0, done.stderr
    assert "ran" in done.stderr


def test_an_owner_that_died_inside_the_block_is_taken_over_not_waited_on():
    done = _run_py(
        """
        import os, threading
        from Wesker.isolation import _QUARANTINE, quarantined_output
        class Dead(threading.Thread):
            def run(self): pass
        t = Dead(); t.start(); t.join()
        _QUARANTINE["owner"] = (object(), t)   # a stale owner whose thread is gone
        with quarantined_output() as took:
            assert took is True
        os.write(1, b"ok\\n")
        """
    )
    assert done.stdout == "ok\n", done.stderr


# ── end to end: the real commands on a project whose test writes at the descriptor ──────────

_MOD = "def add(a, b):\n    return a + b\n\n\ndef neg(a):\n    return -a\n"
_TEST = """\
import os, subprocess, sys
from mod import add, neg


def test_add():
    print("LEAK-PRINT")
    sys.stderr.write("LEAK-STDERR\\n")
    os.write(1, b"LEAK-FD1\\n")
    os.write(2, b"LEAK-FD2\\n")
    subprocess.run(["echo", "LEAK-CHILD"], check=False)
    assert add(1, 2) == 3


def test_neg():
    assert neg(2) == -2
"""


def _project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "mod.py").write_text(_MOD)
    (tmp_path / "tests" / "test_mod.py").write_text(_TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npythonpath = .\n"
    )


def _go(tmp_path, *argv):
    return subprocess.run(
        [sys.executable, *argv],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
        env={
            # PYTHONUNBUFFERED is dropped on purpose. A CI log is a pipe, which Python block-buffers,
            # and a machine that exports the variable cannot see an ordering defect between
            # ``sys.stdout`` and the owner's private progress descriptor: this file passed on such a
            # machine and failed on all six CI cells (the banner landed inside the first progress line).
            **{k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"},
            "HOME": str(tmp_path),
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
    )


def test_the_action_log_holds_none_of_the_measured_tests_output(tmp_path):
    _project(tmp_path)
    done = _go(tmp_path, "-m", "Wesker.action", "--targets", "mod.py")
    assert done.returncode == 0, done.stderr[-1500:]
    assert "LEAK" not in done.stdout + done.stderr
    assert "[1/1] mod.py 4/4" in done.stdout, "the owner's own progress line is whole"
    banner = done.stdout.index("Wesker — Whole codebase")
    assert banner < done.stdout.index("[1/1] mod.py"), (
        "the banner precedes the progress it announces"
    )


def test_the_plain_cli_keeps_its_own_lines_and_none_of_the_tests(tmp_path):
    _project(tmp_path)
    done = _go(tmp_path, "-m", "Wesker", "mod.py")
    assert done.returncode == 0, done.stderr[-1500:]
    assert "LEAK" not in done.stdout + done.stderr
    assert "[1/1] mod.py 4/4" in done.stdout
    assert "Kill rate: 100% (4/4)" in done.stdout
