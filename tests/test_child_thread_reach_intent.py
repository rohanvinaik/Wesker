"""#17 — reach in a thread the test starts is OBSERVED, and what cannot be observed is said, never denied.

THE DEFECT (measured 2026-10-02 on current code). `sys.settrace` is per-thread and the tracer armed
it only in the worker running the test, so a test that hands the target to a thread it starts — a
worker it joins, a pool, a server — was traced in its own thread alone. Its row in the proof ledger
read `lines=()` and `reason=admissible`: an ADMISSIBLE NEGATIVE, "this green test reached nothing".
Scoping believed it, never ran the test against the mutants, and the function profiled 0/5 killed —
five false survivors on a result that reported itself gateable — while the test pins all five.

THE CONTRACT. Threads started while a test runs are traced into that test's reach for as long as it
runs. A thread that outlives the run may execute target code nobody observes, so that row's reach
is `incomplete_thread`: its missing lines are UNKNOWN, never "not reached", and such a trace is not
cached (a replayed cell has no field to carry that). `threading`'s own hook is handed back after
every window. NOT closed here: a thread that existed BEFORE the run and is handed work (a
module-level pool), and subprocesses — the per-run window cannot see either (#32's process-wide
monitoring revisits threads; a subprocess runs the ORIGINAL from disk, so it never meets a mutant).
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import threading

from Wesker.line_coverage import trace_suite

_TARGET = "def double(x):\n    y = x * 2\n    return y\n"


def _module(tmp_path, name):
    path = tmp_path / f"{name}.py"
    path.write_text(_TARGET)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, os.path.realpath(str(path))


def test_a_thread_the_test_starts_and_joins_is_traced_into_its_reach(tmp_path):
    mod, path = _module(tmp_path, "thr_join_unit")

    def t_join():
        box = []
        worker = threading.Thread(target=lambda: box.append(mod.double(3)))
        worker.start()
        worker.join()
        assert box == [6]

    incomplete: dict = {}
    traced = trace_suite([t_join], {path}, incomplete_reach=incomplete)
    reach = next(iter(traced.values()))
    assert reach.get(path) == {2, 3}, (
        "the child thread's lines were not observed — an admissible negative"
    )
    assert incomplete == {}, "a joined child is fully observed; nothing is unknown"


def test_a_thread_that_outlives_the_run_makes_the_reach_unknown_and_is_not_cached(
    tmp_path,
):
    mod, path = _module(tmp_path, "thr_late_unit")
    release = threading.Event()

    def t_fire_and_forget():
        def late():
            release.wait(5)
            mod.double(4)  # runs after the test returned: unobserved

        threading.Thread(target=late, daemon=True).start()

    incomplete: dict = {}
    cache: dict = {}
    hook_before = threading.gettrace()
    try:
        traced = trace_suite(
            [t_fire_and_forget], {path}, incomplete_reach=incomplete, cache=cache
        )
    finally:
        release.set()
    (test_id,) = traced
    assert incomplete == {test_id: "incomplete_thread"}
    assert cache == {}, "an incomplete observation was cached: a replay would deny it"
    assert threading.gettrace() is hook_before, (
        "threading's hook leaked past the window"
    )
    # The late call happened after the window closed, so it never reached the returned map.
    assert traced[test_id].get(path, set()) == set()


def test_an_unstoppable_worker_still_hands_threadings_hook_back(tmp_path, monkeypatch):
    """The adversarial case: the traced worker cannot be stopped, so its own `finally` does not run
    when the joiner gives up. The JOINER must close the window — else every thread the process
    starts from then on is traced into a measurement that is over — and the runaway's late
    `finally` must not reinstall anything."""
    from Wesker import interrupt as INTERRUPT
    from Wesker import line_coverage as LC

    mod, path = _module(tmp_path, "thr_unstoppable_unit")
    monkeypatch.setattr(INTERRUPT, "abandon", lambda _thread: False)
    done = threading.Event()

    def runaway():
        threading.Event().wait(
            0.5
        )  # outlives the 0.1 s budget; the patch injects nothing
        late = threading.Thread(target=lambda: mod.double(9))
        late.start()
        late.join()
        done.set()

    hits: set[int] = set()

    def local(frame, event, _arg):
        if event == "line":
            hits.add(frame.f_lineno)
        return local

    def dispatch(frame, event, _arg):
        if event == "call" and os.path.realpath(frame.f_code.co_filename) == path:
            return local
        return None

    hook_before = threading.gettrace()
    cut, contained = LC._traced_in_thread(runaway, dispatch, 0.1, [])
    assert cut is True and contained is False
    assert threading.gettrace() is hook_before, "the joiner left the window open"
    assert done.wait(5)
    assert threading.gettrace() is hook_before, (
        "the runaway's late finally clobbered it"
    )
    assert hits == set(), "a thread started after the window closed was traced into it"


# ── end-to-end through a live pytest session (Wesker only) ────────────────────────────

_THREAD_TESTS = """import threading
from {mod} import double


def test_reaches_only_in_a_child_thread():
    box = []
    worker = threading.Thread(target=lambda: box.append(double(3)))
    worker.start()
    worker.join()
    assert box == [6]
"""

_LATE_TESTS = """import threading
from {mod} import double

_RELEASE = threading.Event()


def test_leaves_a_thread_running():
    assert double(1) == 2

    def late():
        _RELEASE.wait(0.5)
        double(5)

    threading.Thread(target=late, daemon=True).start()
"""


def _profile(tmp_path, mod, tests_source):
    from Wesker.ci import (
        discover_test_callables,
        resolve_original_func,
        run_with_live_suite,
        walk_functions,
    )
    from Wesker.engine import run_function_profiling
    from Wesker.filter import filter_categories

    (tmp_path / f"{mod}.py").write_text(_TARGET)
    (tmp_path / f"test_{mod}.py").write_text(tests_source.format(mod=mod))
    for name in (mod, f"test_{mod}"):
        sys.modules.pop(name, None)
    root = str(tmp_path)

    def body():
        src = os.path.join(root, f"{mod}.py")
        with open(src) as fh:
            node = dict(walk_functions(ast.parse(fh.read())))["double"]
        tests = discover_test_callables(root, f"{mod}.py", ["double"])
        return run_function_profiling(
            node,
            f"{mod}.py::double",
            filter_categories(node),
            tests,
            resolve_original_func(src, "double"),
        )

    cwd = os.getcwd()
    try:
        os.chdir(root)
        return run_with_live_suite(root, body, target_files=[f"{mod}.py"])
    finally:
        os.chdir(cwd)


def test_a_child_thread_test_kills_the_mutants_it_pins(tmp_path):
    r = _profile(tmp_path, "thr_e2e", _THREAD_TESTS)
    (row,) = r.trace_evidence
    assert row.lines == (2, 3) and row.admissible and row.reach == "complete"
    assert r.total_mutants > 0
    assert r.total_killed == r.total_mutants, (
        f"false survivors from unobserved child-thread reach: {r.survivor_records}"
    )


def test_an_outliving_thread_is_reported_unknown_not_negative(tmp_path):
    r = _profile(tmp_path, "thr_late_e2e", _LATE_TESTS)
    (row,) = r.trace_evidence
    assert row.reach == "incomplete_thread"
    # What WAS observed stays proof: the main-thread call is green, contained and fresh.
    assert row.admissible and row.lines == (2, 3)
    (payload,) = r.to_dict()["trace_evidence"]
    assert payload["reach"] == "incomplete_thread"
