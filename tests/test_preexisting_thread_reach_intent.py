"""#17 — reach run by a thread that existed BEFORE the test is UNKNOWN, never complete-and-empty.

THE DEFECT (measured 2026-10-26). `_trace_one_multi` attributed line events only to threads that
were not alive when the test began. A test that calls the target through a module-level
`ThreadPoolExecutor` warmed at import got `outcome=passed reason=admissible reach=complete
lines=()`: "this green test reached nothing", consumed by Detective as five false survivors and an
`✗ uncovered` page, while the test pins every mutant.

WHY NOT SIMPLY OBSERVE THE POOL. A warmed pool serves whichever test submits to it, and a leaked
runaway from an earlier test is the same population; attributing every pre-existing thread's events
to the running test would manufacture reach (and admissible proof) the test never earned. So such
events are only WITNESSED: when a pre-existing thread ran target code during the window, the row
says `incomplete_preexisting_thread` (its absences are unknown), its lines stay what the test's own
threads observed, and the trace is never cached. Quiet pre-existing threads change nothing.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from Wesker.line_coverage import trace_suite

_TARGET = "def triple(x):\n    y = x * 3\n    return y\n"


def _module(tmp_path, name):
    path = tmp_path / f"{name}.py"
    path.write_text(_TARGET)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, os.path.realpath(str(path))


def _warm_pool():
    pool = ThreadPoolExecutor(max_workers=1)
    pool.submit(lambda: None).result()  # the worker thread now exists
    return pool


def test_a_pool_warmed_before_the_test_makes_its_reach_unknown(tmp_path):
    mod, path = _module(tmp_path, "pre_pool_unit")
    pool = _warm_pool()

    def t_pool():
        assert pool.submit(mod.triple, 2).result() == 6

    incomplete: dict = {}
    cache: dict = {}
    try:
        traced = trace_suite([t_pool], {path}, incomplete_reach=incomplete, cache=cache)
    finally:
        pool.shutdown()
    (test_id,) = traced
    assert incomplete == {test_id: "incomplete_preexisting_thread"}
    assert cache == {}, (
        "an unknown reach was cached: a replay would read it as negative"
    )
    assert traced[test_id].get(path, set()) == set(), (
        "unattributable lines were claimed"
    )


def test_a_quiet_preexisting_thread_changes_nothing(tmp_path):
    mod, path = _module(tmp_path, "pre_quiet_unit")
    pool = _warm_pool()  # alive, but never asked to run the target

    def t_direct():
        assert mod.triple(2) == 6

    incomplete: dict = {}
    try:
        traced = trace_suite([t_direct], {path}, incomplete_reach=incomplete)
    finally:
        pool.shutdown()
    assert incomplete == {}
    assert next(iter(traced.values()))[path] == {2, 3}


def test_a_thread_the_test_starts_is_still_attributed_not_unknown(tmp_path):
    mod, path = _module(tmp_path, "pre_own_unit")

    def t_own():
        box = []
        t = threading.Thread(target=lambda: box.append(mod.triple(2)))
        t.start()
        t.join()
        assert box == [6]

    incomplete: dict = {}
    traced = trace_suite([t_own], {path}, incomplete_reach=incomplete)
    assert incomplete == {}
    assert next(iter(traced.values()))[path] == {2, 3}


def test_the_next_test_is_not_poisoned_by_the_previous_ones_pool_call(tmp_path):
    mod, path = _module(tmp_path, "pre_next_unit")
    pool = _warm_pool()

    def t_pool():
        pool.submit(mod.triple, 1).result()

    def t_plain():
        mod.triple(1)

    incomplete: dict = {}
    try:
        traced = trace_suite([t_pool, t_plain], {path}, incomplete_reach=incomplete)
    finally:
        pool.shutdown()
    ids = list(traced)
    assert incomplete == {ids[0]: "incomplete_preexisting_thread"}
    assert traced[ids[1]][path] == {2, 3}


# ── end-to-end through a live pytest session ──────────────────────────────────────────

_POOL_TESTS = """from concurrent.futures import ThreadPoolExecutor
from {mod} import double

_POOL = ThreadPoolExecutor(max_workers=1)
_POOL.submit(lambda: None).result()


def test_reaches_only_through_a_warmed_pool():
    assert _POOL.submit(double, 3).result() == 6
"""


def test_a_warmed_pool_row_is_unknown_in_the_ledger_and_the_payload(tmp_path):
    from test_child_thread_reach_intent import _profile

    r = _profile(tmp_path, "pre_pool_e2e", _POOL_TESTS)
    (row,) = r.trace_evidence
    assert row.reach == "incomplete_preexisting_thread"
    assert row.lines == (), "the pool's lines were attributed to the test"
    (payload,) = r.to_dict()["trace_evidence"]
    assert payload["reach"] == "incomplete_preexisting_thread"


def test_an_outliving_thread_and_a_pool_call_are_named_together_never_one_of_them(
    tmp_path,
):
    mod, path = _module(tmp_path, "pre_both_unit")
    pool = _warm_pool()
    release = threading.Event()

    def t_both():
        pool.submit(mod.triple, 2).result()
        threading.Thread(target=release.wait, args=(5,), daemon=True).start()

    incomplete: dict = {}
    try:
        traced = trace_suite([t_both], {path}, incomplete_reach=incomplete)
    finally:
        release.set()
        pool.shutdown()
    (test_id,) = traced
    assert incomplete == {test_id: "incomplete_thread_and_preexisting_thread"}


def test_reach_qualification_names_each_state():
    from Wesker.trace_evidence import reach_qualification

    assert reach_qualification(False, False) == "complete"
    assert reach_qualification(True, False) == "incomplete_thread"
    assert reach_qualification(False, True) == "incomplete_preexisting_thread"
    assert reach_qualification(True, True) == "incomplete_thread_and_preexisting_thread"
