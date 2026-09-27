"""The step budget: a DETERMINISTIC bound on a function's loop iterations (EP-B3, item 1 of the plan in
Detective docs/ENGINEERING_PASS_2026-09-26.md §6).

Today a mutant that runs past a WALL-CLOCK allowance is scored as killed — so a verdict can depend on how
busy the machine was. The step budget bounds something that does not: how many loop iterations the
monitored code makes. The count is identical on every machine at every load, and the stop is raised
into the looping frame itself, so a runaway unwinds on the spot instead of being waited out.

Pinned here from intent: exact counts; a deterministic stop just past the cap, twice the same; a stop a
broad ``except Exception`` cannot swallow; forward jumps are not iterations; only the opening thread's
iterations count; budgets do not nest and leave no events behind; the tool id is chosen from the free
ones.
"""

from __future__ import annotations

import sys
import threading

import pytest

from Wesker import monitoring
from Wesker.monitoring import (
    EXCEEDED,
    UNBOUNDED,
    WITHIN,
    MonitoringUnavailable,
    StepBudgetExceeded,
    step_budget,
)


def _finite(n):
    total = 0
    for i in range(n):
        total += i
    return total


def _runaway(_n):
    i = 0
    while i != -1:
        i += 1
    return i


def _swallows_everything(_n):
    i = 0
    while True:
        try:
            i += 1
        # BLE001,S110: the fixture's whole point — a broad handler that must not absorb the stop
        except Exception:  # noqa: BLE001,S110
            pass


def _no_loop(x):
    if x > 0:
        y = 1
    else:
        y = 2
    return y


def _comprehension(n):
    return [i * 2 for i in range(n)]


def _run(fn, arg, cap):
    with step_budget([fn.__code__], cap) as count:
        try:
            fn(arg)
        except StepBudgetExceeded:
            pass
    return count


def test_a_finite_loop_is_counted_exactly():
    count = _run(_finite, 1000, None)
    assert count.iterations == 1000
    assert count.verdict == UNBOUNDED and not count.stopped


def test_a_runaway_stops_just_past_the_cap_and_identically_every_time():
    first = _run(_runaway, 0, 10_000)
    second = _run(_runaway, 0, 10_000)
    assert first.stopped and first.iterations == 10_001 and first.verdict == EXCEEDED
    assert (second.iterations, second.verdict) == (first.iterations, first.verdict)


def test_the_stop_outranks_a_broad_except_exception():
    count = _run(_swallows_everything, 0, 500)
    assert count.stopped and count.verdict == EXCEEDED


def test_a_function_within_its_cap_is_untouched():
    with step_budget([_finite.__code__], 1000) as count:
        assert _finite(1000) == 499500
    assert count.verdict == WITHIN and not count.stopped


def test_forward_jumps_are_control_flow_not_iterations():
    assert _run(_no_loop, 1, None).iterations == 0
    assert _run(_no_loop, -1, None).iterations == 0


def test_an_inlined_comprehension_is_counted_as_the_loop_it_is():
    assert _run(_comprehension, 50, None).iterations == 50


def test_only_the_opening_threads_iterations_count():
    """sys.monitoring is process-wide: the same code on another thread must not spend this budget."""
    done = threading.Event()

    def other_thread():
        _finite(5000)
        done.set()

    with step_budget([_finite.__code__], None) as count:
        worker = threading.Thread(target=other_thread)
        worker.start()
        done.wait(10)
        worker.join(10)
    assert count.iterations == 0


def test_budgets_do_not_nest_and_leave_no_events_behind():
    with (
        step_budget([_finite.__code__], None),
        pytest.raises(RuntimeError, match="do not nest"),
        step_budget([_finite.__code__], None),
    ):
        pass
    tool = monitoring.tool_id()
    assert sys.monitoring.get_local_events(tool, _finite.__code__) == 0
    assert (
        _run(_finite, 10, None).iterations == 10
    )  # a fresh budget opens cleanly afterwards


def test_the_tool_id_is_chosen_from_the_free_ones(monkeypatch):
    """coverage.py holds id 1 when a suite runs under it; this owner must never assume an id."""
    mon = sys.monitoring
    ours = monitoring.tool_id()
    mon.free_tool_id(ours)
    monkeypatch.setattr(monitoring, "_tool", None)
    taken = []
    try:
        mon.use_tool_id(3, "someone-else")
        taken.append(3)
        chosen = monitoring.tool_id()
        assert chosen != 3 and mon.get_tool(chosen) == monitoring.TOOL_NAME
        assert _run(_finite, 7, None).iterations == 7
    finally:
        for tid in taken:
            mon.free_tool_id(tid)
        mon.free_tool_id(monitoring.tool_id())
        monkeypatch.setattr(monitoring, "_tool", None)
    assert (
        monitoring.tool_id() is not None
    )  # and it re-claims cleanly for the rest of the suite


def test_every_id_taken_is_a_named_refusal_not_a_crash(monkeypatch):
    mon = sys.monitoring
    mon.free_tool_id(monitoring.tool_id())
    monkeypatch.setattr(monitoring, "_tool", None)
    taken = []
    try:
        for tid in range(6):
            if mon.get_tool(tid) is None:
                mon.use_tool_id(tid, "someone-else")
                taken.append(tid)
        with pytest.raises(MonitoringUnavailable, match="in use"):
            monitoring.tool_id()
    finally:
        for tid in taken:
            mon.free_tool_id(tid)
        monkeypatch.setattr(monitoring, "_tool", None)
    assert monitoring.tool_id() is not None
