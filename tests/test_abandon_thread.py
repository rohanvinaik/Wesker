"""Stopping a timed-out test's thread, instead of abandoning it.

WHY THIS EXISTS: `_run_test_with_timeout` bounded the WAIT, not the WORK — on timeout it returned
"timeout" and left the daemon thread running ("cleaned up on process exit", i.e. never, within a
run). Every timeout leaked a live thread burning a core, so later mutants timed out BECAUSE
earlier ones were still running: the failure compounded across a session. The abandoned thread's
writes also escaped to the real stdout once the redirect around start+join exited, corrupting the
engine's own report minutes later.

`_abandon` injects `_Abandoned` via CPython's async-exception API, unwinding the thread at its
next BYTECODE boundary — in-process and stdlib-only, so Wesker keeps both its architecture and its
zero-dependency promise. What it cannot reach is pinned here too: a thread blocked OUTSIDE the
interpreter executes no bytecode, so it survives, and `_abandon` says so rather than pretending.
"""

import sys
import threading
import time

import pytest

from Wesker.interrupt import Abandoned as _Abandoned
from Wesker.interrupt import abandon as _abandon
from Wesker.interrupt import injection_landed as _unwind_dead_end


def _runaway(flag: dict) -> None:
    try:
        for i in range(10**9):
            flag["n"] = i
    # BLE001: the injection unwinds through here
    except BaseException:  # noqa: BLE001
        flag["unwound"] = True


def _spawn(target, *args) -> threading.Thread:
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    time.sleep(0.05)  # let it actually enter the loop before injecting
    return t


def test_abandon_stops_a_pure_python_runaway():
    """The leak this exists to remove: without the injection this thread runs until the process
    exits, competing with every mutant that follows it."""
    flag: dict = {}
    t = _spawn(_runaway, flag)
    assert _abandon(t) is True
    assert t.is_alive() is False


def test_abandon_outranks_the_test_s_own_broad_except():
    """`_Abandoned` is a BaseException ON PURPOSE — a test that swallows `Exception` (a real and
    ordinary thing for a test to do) would otherwise eat the injection and keep running, leaving
    the leak in place precisely where the code looks most innocent."""
    assert issubclass(_Abandoned, BaseException)
    assert not issubclass(_Abandoned, Exception)

    def _swallower() -> None:
        try:
            try:
                for _ in range(10**9):
                    pass
            # BLE001: the point of the test: this must NOT stop it
            except Exception:  # noqa: BLE001
                pass
        # BLE001: mirrors _target's own outer catch
        except BaseException:  # noqa: BLE001
            pass

    t = _spawn(_swallower)
    assert _abandon(t) is True
    assert t.is_alive() is False


@pytest.mark.skipif(
    sys.gettrace() is not None,
    reason="KNOWN AND UNEXPLAINED: under a trace function the injection LANDS on a thread "
    "blocked in a C call, so `abandon` stops it and truthfully returns True — the opposite "
    "of the boundary asserted here. Measured, reproducible, mechanism not understood. See "
    "the test body.",
)
def test_abandon_reports_false_for_a_thread_blocked_outside_the_interpreter():
    """THE HONEST BOUNDARY, UNTRACED. A thread blocked in a C call runs no bytecode, so the
    injection cannot land until that call returns on its own. `abandon` returns False rather than
    claiming a stop it did not make — bounding this needs process isolation, a different engine.

    SKIPPED UNDER A TRACER, AND THAT SKIP IS A RECORDED UNKNOWN, NOT A CONVENIENCE.
    Measured with `time.sleep(3)` (a C block whose only Python is the call itself), injecting and
    waiting 0.2s — ample for it to land if it could:

        bare        marked=1   alive after injection = True    <- boundary holds
        --cov       marked=1   alive after injection = False   <- injection landed anyway

    So under `sys.settrace` the interpreter does something to a C-blocked thread that this module's
    model does not account for. `abandon` is not wrong there — it stops the thread and reports True,
    which is the truth. What is wrong is the claim that it *cannot*.

    THIS MATTERS BEYOND CI. Wesker installs `sys.settrace` itself, in `line_coverage._trace_one`,
    around every test of the traced baseline pass — and `trace_budget_s` calls `abandon` from inside
    it. So the untraced boundary asserted below is NOT the condition the engine's own hot path runs
    under. Whether that is a gift (runaways in C become stoppable exactly where they were not) or a
    hazard (a stop lands somewhere the model says it cannot) is unknown, and worth knowing.

    Do not delete this skip to make the suite green. Delete it by explaining the mechanism.
    """
    entered = threading.Event()

    def _blocked() -> None:
        try:
            entered.set()
            time.sleep(3)  # C-level block: the GIL is released and no bytecode executes
        except BaseException:  # noqa: BLE001
            pass

    t = threading.Thread(target=_blocked, daemon=True)
    t.start()
    assert entered.wait(2.0), "the thread never started"
    time.sleep(0.1)  # settle INTO the C call — two bytecodes away
    assert _abandon(t) is False
    assert t.is_alive() is True


def test_abandon_is_false_for_a_thread_that_never_started():
    """No ident = nothing to inject into. Degrades to "not stopped", never an error."""
    assert _abandon(threading.Thread(target=lambda: None)) is False


def test_unwind_dead_end_only_accepts_exactly_one_marked_thread():
    """0 = a benign race (the thread finished between is_alive and the injection). >1 is CPython's
    documented "you're in trouble" case, and the caller must undo it — we cannot know which other
    threads were poisoned, so it can never count as success."""
    assert _unwind_dead_end(1) is True
    assert _unwind_dead_end(0) is False
    assert _unwind_dead_end(2) is False
