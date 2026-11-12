"""One test the baseline cannot stop costs that test's evidence, not every function's (Wesker #29).

THE DEFECT, measured on the first whole-codebase run that printed its report (Actions run
43102482321, 2026-11-11): `WESKER REFUSED TO REPORT: 584 function(s) were only PARTIALLY evaluated`,
572 of them with ONE cause, `could not stop: the session's baseline trace`. No test was named, and
none could be: `_build_test_scope` turned the session's set of uncontained TEST IDS into the literal
marker `session_baseline` and added it to every function of the session, for the life of the process.
Reproduced locally at small scale (`python -m Wesker.action --targets Wesker/tce.py`, the per-test
trace budget lowered to 4 s to stand for a runner 2.8 times slower): the same line on every function.
The three tests the session's trace could not stop were `test_kill_relation_intent`,
`test_stack_dump_clock_intent` and `test_step_budget_wired_intent`, and each was parked in
`subprocess.run` -> `selector.poll`: a wait on a CHILD PROCESS.

THE FIX has three parts, each pinned below.

* A worker parked on a child process is stoppable: `release_child_waits` kills the children its own
  frames name (and everything beneath them), the wait ends, and the injection `abandon` left pending
  lands. Only a worker still alive after that is uncontained.
* A worker that is uncontained is recorded with the place it is parked and the test it was running,
  and the refusal names both.
* Whether it still matters is a fact about a LIVE thread: the baseline's set of uncontained tests is
  consulted when each function begins, and a test whose worker has since ended is no longer in it.

These are written from what the engine must do. A generated suite would pin today's answer.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
import time
import uuid

import pytest

from Wesker.ci import callable_test_id, describe_truncation
from Wesker.engine import (
    _SESSION_BASELINE,
    LazySessionBaseline,
    MutationCategory,
    _containment_lost,
    _run_test_with_timeout,
    build_session_baseline,
    run_function_profiling,
)
from Wesker.interrupt import (
    Abandoned,
    abandoned_site,
    abandoned_standing,
    abandoned_worker_standing,
    bounded_join,
    child_wait_disposition,
    claim_abandoned,
    process_tree_members,
)

# ── the pure decisions (#29, pinned) ─────────────────────────────────────────────────────────


def test_a_root_and_everything_beneath_it_is_stopped():
    table = {10: [11, 12], 11: [13], 99: [100]}
    assert process_tree_members([10], table, [1000]) == [10, 11, 12, 13]


def test_several_roots_are_each_walked_and_listed_once():
    table = {10: [11], 20: [11, 21]}
    assert process_tree_members([10, 20], table, [1000]) == [10, 11, 20, 21]


def test_the_measuring_process_and_init_are_never_named():
    table = {1000: [10], 10: [1000, 11]}
    assert process_tree_members([1000], table, [1000]) == [], "a protected root"
    assert process_tree_members([1, 0, -5], {1: [10]}, [1000]) == [], "init and bad ids"
    assert process_tree_members([10], table, [1000]) == [10, 11], (
        "a table that places the protected id beneath a root cannot widen the kill"
    )


def test_nothing_is_walked_beneath_a_protected_id():
    assert process_tree_members([10], {10: [1000], 1000: [11]}, [1000]) == [10]


def test_a_cyclic_table_terminates():
    assert process_tree_members([10], {10: [11], 11: [10]}, [1000]) == [10, 11]


def test_no_roots_name_nothing():
    assert process_tree_members([], {10: [11]}, [1000]) == []


@pytest.mark.parametrize(
    ("found", "gone", "expected"),
    [
        (0, True, "no_child_wait"),
        (0, False, "no_child_wait"),
        (-1, True, "no_child_wait"),
        (1, True, "released"),
        (3, True, "released"),
        (1, False, "survived"),
    ],
)
def test_stopping_children_names_what_it_achieved(found, gone, expected):
    assert child_wait_disposition(found, gone) == expected


@pytest.mark.parametrize(
    ("recorded", "alive", "expected"),
    [
        (0, 0, "unobserved"),
        (0, 1, "unobserved"),
        (1, 1, "live"),
        (2, 1, "live"),
        (2, 0, "ended"),
        (1, 0, "ended"),
    ],
)
def test_a_left_running_worker_is_a_hazard_until_it_ends(recorded, alive, expected):
    assert abandoned_worker_standing(recorded, alive) == expected


def test_an_unrecorded_label_is_never_read_as_ended():
    """Absence of a record is not a worker that ended."""
    assert abandoned_standing("never-" + uuid.uuid4().hex) == "unobserved"


# ── a worker parked on a child process is stopped ────────────────────────────────────────────

_SLEEPER = "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
_PARENT = (
    "import os, subprocess, sys, time\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "subprocess.Popen([sys.executable, '-c', sys.argv[3], sys.argv[2]])\n"
    "time.sleep(60)\n"
)


def _quiet(fn):
    """A thread target that ends quietly when the engine's stop lands (pytest reports an escaped one)."""

    def run():
        try:
            fn()
        except Abandoned:
            pass

    return run


def _wait_for(path, seconds=20.0) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            text = path.read_text()
        except OSError:
            text = ""
        if text.strip():
            return int(text)
        time.sleep(0.02)
    raise AssertionError(f"{path} was never written")


def _gone(pid: int, seconds=10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # a child that has been killed but not yet reaped is a zombie: still named, no longer alive
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                if fh.read().rsplit(b")", 1)[-1].split()[0] == b"Z":
                    return True
        except OSError:
            return True
        time.sleep(0.02)
    return False


def test_a_worker_waiting_on_a_child_process_is_stopped_and_its_child_killed(tmp_path):
    pidfile = tmp_path / "child.pid"
    worker = threading.Thread(
        target=_quiet(
            lambda: subprocess.run(
                [sys.executable, "-c", _SLEEPER, str(pidfile)],
                capture_output=True,
                check=False,
            )
        ),
        daemon=True,
    )
    worker.start()
    child = _wait_for(pidfile)
    started = time.monotonic()
    timed_out, contained = bounded_join(worker, 0.2)
    assert timed_out is True
    assert contained is True, "a wait on a child process is a stoppable wait"
    assert time.monotonic() - started < 15
    assert not worker.is_alive()
    assert _gone(child), "the child it was waiting on was killed"


def test_a_child_that_has_children_of_its_own_is_killed_with_them(tmp_path):
    """A grandchild that inherited the pipe keeps `communicate` waiting after its parent dies, so
    stopping the root alone would leave the worker parked."""
    parent_pid, grand_pid = tmp_path / "parent.pid", tmp_path / "grand.pid"
    worker = threading.Thread(
        target=_quiet(
            lambda: subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _PARENT,
                    str(parent_pid),
                    str(grand_pid),
                    _SLEEPER,
                ],
                capture_output=True,
                check=False,
            )
        ),
        daemon=True,
    )
    worker.start()
    parent, grand = _wait_for(parent_pid), _wait_for(grand_pid)
    timed_out, contained = bounded_join(worker, 0.2)
    assert (timed_out, contained) == (True, True)
    assert _gone(parent) and _gone(grand), "the whole tree went, not only its root"


def test_the_measuring_process_survives_stopping_a_child_wait(tmp_path):
    pidfile = tmp_path / "child.pid"
    worker = threading.Thread(
        target=_quiet(
            lambda: subprocess.run(
                [sys.executable, "-c", _SLEEPER, str(pidfile)], check=False
            )
        ),
        daemon=True,
    )
    worker.start()
    _wait_for(pidfile)
    bounded_join(worker, 0.2)
    assert os.getpid() > 1  # we are still here to assert it


def test_a_test_waiting_on_a_child_is_a_contained_timeout_not_an_uncontained_one(
    tmp_path,
):
    def test_waits():
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(60)"], check=False
        )

    outcome = _run_test_with_timeout(test_waits, None, True, 700)
    assert outcome == "undetermined", (
        "a contained timeout: a named non-kill, not uncontained"
    )
    assert abandoned_standing(callable_test_id(test_waits)) == "unobserved", (
        "nothing was left running, so nothing was recorded"
    )


# ── a worker that cannot be stopped is named, and its standing follows its thread ───────────


def _label() -> str:
    return "t.py::" + uuid.uuid4().hex


def test_a_worker_blocked_on_a_lock_is_uncontained_recorded_and_lapses_when_it_ends():
    release = threading.Event()
    worker = threading.Thread(target=_quiet(lambda: release.wait(30)), daemon=True)
    worker.start()
    label = _label()
    try:
        timed_out, contained = bounded_join(worker, 0.1)
        assert (timed_out, contained) == (True, False)
        claim_abandoned(label)
        assert abandoned_standing(label) == "live"
        assert "wait" in abandoned_site(label), abandoned_site(label)
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert abandoned_standing(label) == "ended", "the hazard lapsed with the thread"


def test_a_label_claims_only_the_workers_nobody_has_named_yet():
    first, second = _label(), _label()
    release = threading.Event()
    threads = [
        threading.Thread(target=_quiet(lambda: release.wait(30)), daemon=True)
        for _ in range(2)
    ]
    try:
        for thread in threads:
            thread.start()
        bounded_join(threads[0], 0.05)
        claim_abandoned(first)
        bounded_join(threads[1], 0.05)
        claim_abandoned(second)
        assert abandoned_standing(first) == "live"
        assert abandoned_standing(second) == "live"
        release.set()
        threads[0].join(10)
        assert abandoned_standing(first) == "ended"
        threads[1].join(10)
        assert abandoned_standing(second) == "ended"
    finally:
        release.set()


def test_the_mutation_path_names_the_test_it_could_not_stop():
    release = threading.Event()

    def test_blocks():
        release.wait(30)

    try:
        assert _run_test_with_timeout(test_blocks, None, True, 150) == "uncontained"
        name = callable_test_id(test_blocks)
        assert abandoned_standing(name) == "live"
        assert "wait" in abandoned_site(name)
    finally:
        release.set()


# ── the engine consumes the named fact ────────────────────────────────────────────────────────

_SRC = "def scoreit(a, b, flag):\n    if flag:\n        return a * 2 + b\n    return a - b\n"
_CATS = {MutationCategory.VALUE, MutationCategory.ARITHMETIC, MutationCategory.SWAP}


def _target():
    node = ast.parse(_SRC).body[0]
    ns: dict = {}
    # S102: test fixture source
    exec(compile(ast.parse(_SRC), "<uncontained-baseline>", "exec"), ns)  # noqa: S102
    return node, ns["scoreit"]


def _profile(tests, original, node, **budgets):
    target_files = {original.__code__.co_filename}

    def build(subset=None, fresh=False, **_kw):
        return build_session_baseline(
            tests if subset is None else list(subset), target_files, **budgets
        )

    holder = LazySessionBaseline(build)
    token = _SESSION_BASELINE.set(holder)
    try:

        def run():
            return run_function_profiling(
                node,
                f"{original.__code__.co_filename}::scoreit",
                _CATS,
                tests,
                original,
                max_per_category=0,
            )

        return holder, run
    finally:
        _SESSION_BASELINE.reset(token)


def _in_session(holder, run):
    token = _SESSION_BASELINE.set(holder)
    try:
        return run()
    finally:
        _SESSION_BASELINE.reset(token)


def test_one_stuck_test_is_named_in_every_function_it_can_touch_and_lapses_with_it():
    node, original = _target()
    release = threading.Event()

    def test_blocks():
        release.wait(30)

    def test_pins():
        assert scoreit(1, 2, True) == 4  # noqa: F821

    tests = [test_blocks, test_pins]
    for t in tests:
        t.__globals__["scoreit"] = original
    stuck = callable_test_id(test_blocks)
    holder, run = _profile(tests, original, node, timeout_ms=300, trace_budget_s=0.3)
    try:
        live = _in_session(holder, run)
        entries = [e for e in live.containment_lost if e["phase"] == "baseline"]
        assert [e["test"] for e in entries] == [stuck], live.containment_lost
        assert "wait" in entries[0]["where"], entries
        assert all(e["test"] != "session_baseline" for e in live.containment_lost)
        assert live.coverage_depth == "cut"
        assert live.is_gateable is False
        assert live.to_dict()["containment_lost"][0]["where"] == entries[0]["where"]
    finally:
        release.set()
    deadline = time.monotonic() + 15
    while abandoned_standing(stuck) != "ended" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert abandoned_standing(stuck) == "ended"

    after = _in_session(holder, run)
    assert after.containment_lost == (), (
        "a worker that has ended shares nothing with this run"
    )
    assert after.coverage_depth == "profiled"
    assert after.total_killed >= 1, "the other test's evidence was never lost"


def test_a_test_waiting_on_a_child_process_does_not_cut_the_session():
    """The measured case: the session's trace parked in `subprocess.run`. Its trace is truncated,
    which is a sampled count, and its child is dead; no function is made partial by it."""
    node, original = _target()

    def test_waits():
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(60)"], check=False
        )

    def test_pins():
        assert scoreit(1, 2, True) == 4  # noqa: F821

    tests = [test_waits, test_pins]
    for t in tests:
        t.__globals__["scoreit"] = original
    holder, run = _profile(tests, original, node, timeout_ms=600, trace_budget_s=0.6)
    result = _in_session(holder, run)
    session = holder.get()
    assert session.uncontained == set()
    assert callable_test_id(test_waits) in session.truncated, (
        "its trace was cut, and said so"
    )
    assert result.containment_lost == ()
    assert result.coverage_depth == "profiled"
    assert result.total_killed >= 1


def test_the_outcome_pass_alone_names_the_test(monkeypatch):
    """With the trace pass out of the way, the outcome pass is the only place left that meets the worker."""
    from Wesker import engine as E

    release = threading.Event()

    def test_blocks():
        release.wait(30)

    monkeypatch.setattr(E, "_trace_suite", lambda *a, **k: {})
    try:
        session = build_session_baseline(
            [test_blocks], {__file__}, timeout_ms=150, trace_budget_s=None
        )
        assert session.uncontained == {callable_test_id(test_blocks)}
    finally:
        release.set()


# ── the page ──────────────────────────────────────────────────────────────────────────────────


def test_the_refusal_names_the_test_and_where_it_is_blocked():
    cut = [
        {
            "function_key": "Wesker/tce.py::_code_key",
            "cause": "uncontained",
            "elapsed_ms": 1300.0,
            "tested": 8,
            "universe": 9,
            "containment_lost": [
                {
                    "phase": "baseline",
                    "test": "tests/test_stack_dump_clock_intent.py::test_dumps_survive",
                    "where": "selectors.py:398 in select, called from tests/t.py:90 in test_x",
                }
            ],
        }
    ]
    page = describe_truncation(cut)
    assert "tests/test_stack_dump_clock_intent.py::test_dumps_survive" in page
    assert "blocked at selectors.py:398 in select" in page
    assert "the session's baseline trace" not in page
    assert "STILL RUNNING" in page
    assert "working directory" in page and "patched module attributes" in page


def test_a_mutation_entry_carries_the_site_too():
    cut = [
        {
            "function_key": "f",
            "cause": "uncontained",
            "containment_lost": [
                {
                    "phase": "mutation",
                    "test": "tests/t.py::test_a",
                    "mutant": "m",
                    "mutated_line": 3,
                    "where": "threading.py:373 in wait",
                }
            ],
        }
    ]
    assert "(line 3) — blocked at threading.py:373 in wait" in describe_truncation(cut)


def test_an_entry_without_a_site_renders_as_before():
    lost = _containment_lost({"t.py::x"}, [])
    assert lost == ({"phase": "baseline", "test": "t.py::x"},)
