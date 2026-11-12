"""#17 — reach that happens only in a CHILD PROCESS is UNKNOWN, never complete-and-empty.

THE DEFECT (measured 2026-10-26). "A test whose only call of `f` runs in `subprocess.run` traces
`lines=()`, `reach=complete`, admissible", and `detective audit` printed `✗ uncovered 2 line(s)`
about a function a green test pins. Wesker's own self-measurement scores `_isolated_worker.py`
0/451 for the same reason.

THE CONTRACT. No tool in this interpreter can follow a child, so the engine does not pretend to: a
test that creates a process (CPython's audit events `subprocess.Popen`, `os.fork`, `os.exec*`,
`os.posix_spawn`, `os.system`, ...) has `reach=incomplete_subprocess` — its absences are unknown.
The lines it DID observe stay as admissible as the outcome says; nothing is added for the child;
the trace is never cached; a test that creates no process is untouched, and the next test is not
poisoned by the previous one's child.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading

from Wesker.line_coverage import trace_suite

_TARGET = "def triple(x):\n    y = x * 3\n    return y\n"


def _module(tmp_path, name):
    path = tmp_path / f"{name}.py"
    path.write_text(_TARGET)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, os.path.realpath(str(path)), str(tmp_path), name


def _child_calls_target(tmp_path_str, name):
    out = subprocess.run(
        [sys.executable, "-c", f"import {name}; print({name}.triple(2))"],
        cwd=tmp_path_str,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": tmp_path_str},
    )
    assert out.stdout.strip() == "6"


def test_a_test_whose_only_reach_is_a_child_process_is_unknown_not_negative(tmp_path):
    _mod, path, d, name = _module(tmp_path, "sub_only_unit")

    def t_child():
        _child_calls_target(d, name)

    incomplete: dict = {}
    cache: dict = {}
    traced = trace_suite([t_child], {path}, incomplete_reach=incomplete, cache=cache)
    (test_id,) = traced
    assert incomplete == {test_id: "incomplete_subprocess"}
    assert traced[test_id].get(path, set()) == set()
    assert cache == {}, (
        "an unknown reach was cached: a replay would read it as negative"
    )


def test_a_child_started_by_a_thread_the_test_started_is_also_named(tmp_path):
    _mod, path, d, name = _module(tmp_path, "sub_thread_unit")

    def t_thread_child():
        t = threading.Thread(target=_child_calls_target, args=(d, name))
        t.start()
        t.join()

    incomplete: dict = {}
    traced = trace_suite([t_thread_child], {path}, incomplete_reach=incomplete)
    (test_id,) = traced
    assert incomplete == {test_id: "incomplete_subprocess"}


def test_in_process_reach_beside_a_child_keeps_its_lines_and_names_the_unknown(
    tmp_path,
):
    mod, path, d, name = _module(tmp_path, "sub_mixed_unit")

    def t_mixed():
        assert mod.triple(1) == 3
        _child_calls_target(d, name)

    incomplete: dict = {}
    traced = trace_suite([t_mixed], {path}, incomplete_reach=incomplete)
    (test_id,) = traced
    assert traced[test_id][path] == {2, 3}
    assert incomplete == {test_id: "incomplete_subprocess"}


def test_a_test_that_creates_no_process_is_complete_and_the_next_is_not_poisoned(
    tmp_path,
):
    mod, path, d, name = _module(tmp_path, "sub_next_unit")

    def t_child():
        _child_calls_target(d, name)

    def t_plain():
        assert mod.triple(1) == 3

    incomplete: dict = {}
    traced = trace_suite([t_child, t_plain], {path}, incomplete_reach=incomplete)
    first, second = list(traced)
    assert incomplete == {first: "incomplete_subprocess"}
    assert traced[second][path] == {2, 3}


def test_a_process_created_outside_any_traced_window_is_not_recorded(tmp_path):
    from Wesker import line_coverage as LC

    _mod, _path, d, name = _module(tmp_path, "sub_outside_unit")
    _child_calls_target(d, name)  # no trace is active
    assert LC._spawn_watchers == {}


def test_the_watcher_is_removed_when_the_trace_ends(tmp_path):
    from Wesker import line_coverage as LC

    mod, path, _d, _name = _module(tmp_path, "sub_cleanup_unit")
    trace_suite([lambda: mod.triple(1)], {path})
    assert LC._spawn_watchers == {}


# ── end-to-end through a live pytest session ──────────────────────────────────────────

_CHILD_TESTS = """import os, subprocess, sys
from {mod} import double


def test_reaches_only_in_a_child_process():
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(
        [sys.executable, "-c", "import {mod}; print({mod}.double(3))"],
        cwd=here, capture_output=True, text=True,
        env={{**os.environ, "PYTHONPATH": here}},
    )
    assert out.stdout.strip() == "6"
"""


def test_a_child_only_test_row_is_unknown_in_the_ledger_and_the_payload(tmp_path):
    from test_child_thread_reach_intent import _profile

    r = _profile(tmp_path, "sub_e2e", _CHILD_TESTS)
    (row,) = r.trace_evidence
    assert row.reach == "incomplete_subprocess"
    assert row.lines == ()
    (payload,) = r.to_dict()["trace_evidence"]
    assert payload["reach"] == "incomplete_subprocess"
