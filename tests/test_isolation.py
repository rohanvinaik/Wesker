"""#19 — isolated worker execution: a mutant/test runs in a KILLABLE process group.

The in-process path cannot contain a test blocked in a subprocess/socket/C-extension: a Python
thread cannot be killed, only asked to stop, and `interrupt.abandon` reports honestly when the ask
fails. A separate process GROUP can be killed. These tests pin that guarantee — a runaway node, and
a CHILD it spawned, are both terminated with an uncatchable SIGKILL and the run reports contained —
which is the whole reason isolation is the gateable mode.

The pure `isolated_test_outcome` is characterized by converge; these add the intent: a timeout
outranks any code, an unknown exit is never a pass, and the end-to-end runs prove the containment.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from Wesker.isolation import isolated_test_outcome, run_pytest_node_isolated


# ── the pure decision ──────────────────────────────────────────────────────────


def test_a_timeout_outranks_any_exit_code():
    """The worker was killed mid-flight — its exit code is not trustworthy, so `timeout` wins."""
    assert isolated_test_outcome(0, timed_out=True) == "timeout"
    assert isolated_test_outcome(1, timed_out=True) == "timeout"


def test_exit_codes_map_to_typed_outcomes():
    assert isolated_test_outcome(0, False) == "passed"
    assert isolated_test_outcome(1, False) == "failed"
    assert isolated_test_outcome(5, False) == "no_tests"


def test_an_unknown_exit_is_error_never_passed():
    """A collection/internal/usage/signal exit must never read as green."""
    for rc in (2, 3, 4, -9, 137):
        assert isolated_test_outcome(rc, False) == "error"


# ── end-to-end: the isolated pytest lifecycle ───────────────────────────────────


def test_a_passing_node_runs_isolated_and_passes(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    r = run_pytest_node_isolated(str(tmp_path), ["test_ok.py::test_ok"], 30.0)
    assert r.contained and not r.timed_out
    assert r.outcome == "passed"


def test_a_failing_node_reports_failed(tmp_path):
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert 1 == 2\n")
    r = run_pytest_node_isolated(str(tmp_path), ["test_bad.py::test_bad"], 30.0)
    assert r.outcome == "failed"


# ── containment: the guarantee a thread abandon cannot give ─────────────────────


def test_a_runaway_test_is_killed_and_reported_contained(tmp_path):
    (tmp_path / "test_hang.py").write_text(
        "import time\n\n\ndef test_hang():\n    time.sleep(100)\n"
    )
    r = run_pytest_node_isolated(str(tmp_path), ["test_hang.py::test_hang"], 1.0)
    assert r.timed_out is True
    assert r.contained is True, (
        "a subprocess that can be SIGKILLed must report contained"
    )
    assert r.outcome == "timeout"


def test_a_child_the_test_spawned_is_killed_with_the_group(tmp_path):
    """The differentiator. killpg reaches the CHILD subprocess the test started — the exact thing a
    thread abandon leaves running while later measurement continues against a compromised box."""
    pidfile = tmp_path / "child.pid"
    (tmp_path / "test_child.py").write_text(
        "import subprocess, sys, time\n\n\n"
        "def test_child():\n"
        "    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"    open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "    time.sleep(120)\n"
    )
    r = run_pytest_node_isolated(str(tmp_path), ["test_child.py::test_child"], 3.0)
    assert r.timed_out is True and r.contained is True
    assert pidfile.exists(), "the test did not reach the point of spawning its child"
    child_pid = int(pidfile.read_text())

    # The child must die WITH the group. Poll: reparenting/reaping is not instant.
    gone = False
    for _ in range(40):
        try:
            os.kill(child_pid, 0)  # 0 = existence probe; raises if the process is gone
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.1)
    assert gone, (
        f"child {child_pid} survived the group kill — containment did not reach it"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_the_module_uses_a_real_process_group():
    """Guard the mechanism itself: the worker is started in a new session (hence a new process
    group), which is what makes killpg reach its children."""
    import inspect

    from Wesker import isolation

    src = inspect.getsource(isolation.run_pytest_node_isolated)
    assert "start_new_session=True" in src
    assert "killpg" in inspect.getsource(isolation._terminate_group)
