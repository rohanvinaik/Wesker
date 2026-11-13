"""#29 — a failed observation of ONE test's trace makes that test's reach unknown; it never ends the measurement.

THE DEFECT, from the first sharded badge run (GitHub run 43315433429 on 8fcadfe, shard 1/4, `engine.py`).
The run died six minutes in, with no result, on a traceback out of Wesker's own baseline trace:

    line_coverage._trace_one_multi -> h.check()
    monitoring.py: raise MonitoringObservationFailed(state.failure, self.sub_id)
    MonitoringObservationFailed: observation_inflight (subscription=4765)

`SubscriptionHandle.check` says `observation_inflight` when a callback delivery was still running on some
thread when the subscription's evidence was read. The tracer tolerated that only when the test's own worker
could not be stopped; for every other failure it raised out of `build_session_baseline`, through the
function being measured, and ended the run. The measured cause is a test whose thread outlives it and keeps
running the traced file (the suite of a mutation engine has tests that start runaways on purpose): its
deliveries are in flight at the instant the next trace is read, which happens at random, in CI, once in six
minutes.

THE RULING pinned here. The failure belongs to the test it happened in. That test's lines stay as recorded,
its reach is `incomplete_observation` (what is missing from them is UNKNOWN, never a negative), and its trace
is never cached. A caller that collects no reach has nowhere to say so and still gets the exception.
"""

from __future__ import annotations

import importlib.util
import os
import threading

import pytest

from Wesker import monitoring
from Wesker.line_coverage import (
    _trace_one_multi,
    observation_failure_standing,
    trace_suite,
)
from Wesker.trace_evidence import reach_qualification

_TARGET = "def triple(x):\n    y = x * 3\n    return y\n"


def _module(tmp_path, name):
    path = tmp_path / f"{name}.py"
    path.write_text(_TARGET)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, os.path.realpath(str(path))


@pytest.fixture
def failing_check(monkeypatch):
    """Make every subscription's `check` fail with `code` (the first `count` times, else always)."""

    def arm(code: str, count: int | None = None):
        calls = []

        def check(self):
            calls.append(code)
            if count is None or len(calls) <= count:
                raise monitoring.MonitoringObservationFailed(code, self.sub_id)

        monkeypatch.setattr(monitoring.SubscriptionHandle, "check", check)
        return calls

    return arm


# ── the pure decisions ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("code", "contained", "can_qualify", "expected"),
    [
        ("", True, True, "observed"),
        ("", False, False, "observed"),
        ("observation_inflight", False, True, "tolerated"),
        ("observation_inflight", False, False, "tolerated"),
        ("observation_inflight", True, True, "reach_unknown"),
        ("observation_inflight", True, False, "refuse"),
        ("observer_failed:ValueError", True, True, "reach_unknown"),
        ("observer_failed:ValueError", False, True, "reach_unknown"),
        ("observer_failed:ValueError", False, False, "refuse"),
        ("discovery_failed:ValueError", True, True, "reach_unknown"),
        ("target_frame_unavailable", True, True, "reach_unknown"),
        ("ownership_filter_failed:KeyError", True, False, "refuse"),
    ],
)
def test_where_a_failed_observation_lives(code, contained, can_qualify, expected):
    assert observation_failure_standing(code, contained, can_qualify) == expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((False, False, False, False), "complete"),
        ((False, False, False, True), "incomplete_observation"),
        ((True, False, False, True), "incomplete_thread+incomplete_observation"),
        (
            (True, True, True, True),
            (
                "incomplete_thread+incomplete_preexisting_thread"
                "+incomplete_subprocess+incomplete_observation"
            ),
        ),
        ((False, False, True, False), "incomplete_subprocess"),
    ],
)
def test_the_observation_cause_joins_the_other_causes_in_a_fixed_order(flags, expected):
    assert reach_qualification(*flags) == expected


# ── the tracer ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code",
    [
        "observation_inflight",
        "observer_failed:ValueError",
        "discovery_failed:ValueError",
        "target_frame_unavailable",
    ],
)
def test_a_failed_observation_is_that_tests_unknown_reach_not_the_runs_end(
    tmp_path, failing_check, code
):
    mod, path = _module(tmp_path, f"obs_{code[:6]}_unit")
    failing_check(code)

    def t_triple():
        assert mod.triple(2) == 6

    incomplete: dict = {}
    cache: dict = {}
    traced = trace_suite([t_triple], {path}, incomplete_reach=incomplete, cache=cache)
    (test_id,) = traced
    assert incomplete == {test_id: "incomplete_observation"}
    assert traced[test_id][path] == {2, 3}, "what was observed stays observed"
    assert cache == {}, (
        "an unknown reach was cached: a replay would read it as negative"
    )


def test_the_failure_is_one_tests_and_the_next_test_is_measured_whole(
    tmp_path, failing_check
):
    mod, path = _module(tmp_path, "obs_next_unit")
    # Four subscriptions per test (line, foreign line, return, unwind): only the first test's fail.
    failing_check("observer_failed:RuntimeError", count=4)

    def t_first():
        assert mod.triple(1) == 3

    def t_second():
        assert mod.triple(2) == 6

    incomplete: dict = {}
    traced = trace_suite([t_first, t_second], {path}, incomplete_reach=incomplete)
    assert set(incomplete.values()) == {"incomplete_observation"}
    assert len(incomplete) == 1, "the second test's trace was whole"
    assert all(traced[t][path] == {2, 3} for t in traced)


def test_an_unobserved_reach_joins_a_thread_that_outlived_the_test(
    tmp_path, failing_check
):
    mod, path = _module(tmp_path, "obs_thread_unit")
    failing_check("observation_inflight")
    stop = threading.Event()
    started = threading.Event()

    def spin():
        started.set()
        while not stop.is_set():
            mod.triple(1)

    def t_leaks():
        threading.Thread(target=spin, daemon=True).start()
        assert started.wait(5)

    incomplete: dict = {}
    try:
        traced = trace_suite([t_leaks], {path}, incomplete_reach=incomplete)
    finally:
        stop.set()
    (test_id,) = traced
    assert incomplete == {test_id: "incomplete_thread+incomplete_observation"}


def test_a_caller_that_collects_no_reach_still_gets_the_exception(
    tmp_path, failing_check
):
    mod, path = _module(tmp_path, "obs_noreach_unit")
    failing_check("observer_failed:ValueError")

    def t_triple():
        mod.triple(2)

    with pytest.raises(monitoring.MonitoringObservationFailed):
        _trace_one_multi(t_triple, {path}, 5.0, reach=None)


def test_a_leaked_busy_thread_does_not_end_the_trace_of_the_tests_after_it(tmp_path):
    """No forced failure: a thread that outlives its test and keeps running the traced file."""
    mod, path = _module(tmp_path, "obs_busy_unit")
    stop = threading.Event()
    started = threading.Event()

    def spin():
        started.set()
        while not stop.is_set():
            mod.triple(1)

    def t_leaks():
        threading.Thread(target=spin, daemon=True).start()
        assert started.wait(5)

    def t_after():
        assert mod.triple(2) == 6

    incomplete: dict = {}
    try:
        for _ in range(20):
            traced = trace_suite(
                [t_leaks, t_after], {path}, incomplete_reach=incomplete
            )
    finally:
        stop.set()
    assert len(traced) == 2
    causes = {c for code in incomplete.values() for c in code.split("+")}
    assert causes <= {
        "incomplete_thread",
        "incomplete_preexisting_thread",
        "incomplete_observation",
    }
