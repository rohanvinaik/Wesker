"""`interrupt.bounded_join` — a bounded wait that leaves no runaway behind, on EVERY way out.

THE ORPHAN (measured 2026-09-06, `detective converge` on its own engine): a bounded join is often
NESTED. Detective's classifier joins a runaway mutant with a timeout from inside a test, and that
test was being run by the traced baseline in a worker of its own, under a budget. The budget
abandoned the worker while it was parked in the inner join. The injected exception lands at the
first bytecode after the join returns — BEFORE the `if thread.is_alive(): abandon(thread)` that
followed it — so the runaway it was bounding was never stopped: a live thread hogging the GIL for
the rest of the process (two of them in the faulthandler dump, thread numbers past 4,600), and
every later phase crawled. All four bounded-join sites in the pair were written in that shape.

Pinned from intent, with the same runaway the abandon tests use:

- the defect, as a control: the old shape (join, then abandon-if-alive) orphans the runaway when
  the joiner is abandoned mid-wait;
- the fix: `bounded_join` stops the runaway on that same exit, and reports the wait as timed out;
- the ordinary paths keep their meaning: a timeout abandons and reports (True, True); a thread that
  finishes in time is (False, True); ``None`` waits unbounded;
- the honest boundary survives: a thread blocked outside the interpreter is (True, False), and the
  caller's extra unwind allowance is granted only when the injection did not land.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from Wesker.interrupt import abandon, bounded_join


def _runaway(flag: dict) -> None:
    try:
        for i in range(10**9):
            flag["n"] = i
    except BaseException:  # noqa: BLE001 — the injection unwinds through here
        flag["unwound"] = True


def _spawn(target, *args) -> threading.Thread:
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    time.sleep(0.05)  # let it actually enter the loop
    return t


def _abandon_after(thread: threading.Thread, delay_s: float) -> None:
    time.sleep(delay_s)
    abandon(thread)


# --- the defect, then the fix --------------------------------------------------------------------


def test_the_old_shape_orphans_the_runaway_when_the_joiner_is_abandoned_mid_wait():
    """The control: what every bounded-join site did before 2026-09-06."""
    flag: dict = {}
    runaway = _spawn(_runaway, flag)

    def old_shape() -> None:
        try:
            # Parked here when the abandonment is injected. It CANNOT land in a C-level lock wait
            # (the boundary): it lands at the first bytecode after the join returns — the next line.
            runaway.join(0.5)
            if runaway.is_alive():  # …so this line is never reached
                abandon(runaway)
        except BaseException:  # noqa: BLE001 — the expected abandonment; keep it out of pytest's thread warning
            pass

    joiner = threading.Thread(target=old_shape, daemon=True)
    joiner.start()
    _abandon_after(joiner, 0.1)
    joiner.join(2.0)
    assert not joiner.is_alive(), (
        "the joiner itself was abandoned once its join returned"
    )
    try:
        assert runaway.is_alive(), (
            "the orphan: the runaway outlived the joiner that was bounding it"
        )
    finally:
        abandon(runaway)  # do not leak it into the rest of the suite


def test_bounded_join_stops_the_runaway_when_the_joiner_is_abandoned_mid_wait():
    """The fix: the same exit, and the runaway is stopped on the way out."""
    flag: dict = {}
    runaway = _spawn(_runaway, flag)
    seen: dict = {}

    def new_shape() -> None:
        try:
            seen["result"] = bounded_join(runaway, 0.5)
        except BaseException as exc:  # noqa: BLE001 — the abandonment propagates out of the join
            seen["raised"] = type(exc).__name__

    joiner = threading.Thread(target=new_shape, daemon=True)
    joiner.start()
    _abandon_after(joiner, 0.1)
    joiner.join(2.0)
    assert not joiner.is_alive()
    assert seen.get("raised") == "Abandoned", (
        "the joiner's own abandonment still propagates"
    )
    runaway.join(0.5)
    assert not runaway.is_alive(), (
        "no orphan: the runaway was stopped on the joiner's way out"
    )
    assert flag.get("unwound") is True


# --- the ordinary paths keep their meaning ---------------------------------------------------------


def test_a_timeout_abandons_the_runaway_and_reports_timed_out_and_contained():
    flag: dict = {}
    runaway = _spawn(_runaway, flag)
    assert bounded_join(runaway, 0.1) == (True, True)
    assert not runaway.is_alive()
    assert flag.get("unwound") is True


def test_a_thread_that_finishes_in_time_is_neither_timed_out_nor_abandoned():
    done: dict = {}

    def quick() -> None:
        done["ran"] = True

    t = threading.Thread(target=quick, daemon=True)
    t.start()
    assert bounded_join(t, 1.0) == (False, True)
    assert done == {"ran": True}


def test_none_waits_unbounded():
    t = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
    t.start()
    assert bounded_join(t, None) == (False, True)


@pytest.mark.skipif(
    sys.gettrace() is not None,
    reason="under a trace function the injection lands on a C-blocked thread — the recorded unknown "
    "in test_abandon_thread.py; the boundary asserted here is the UNTRACED one",
)
def test_a_thread_blocked_outside_the_interpreter_is_timed_out_but_not_contained():
    """The honest boundary, unchanged: no bytecode runs in a C block, so nothing lands."""
    entered = threading.Event()

    def blocked() -> None:
        try:
            entered.set()
            time.sleep(1.5)
        except BaseException:  # noqa: BLE001
            pass

    t = threading.Thread(target=blocked, daemon=True)
    t.start()
    assert entered.wait(2.0)
    time.sleep(0.1)  # settle INTO the C call
    timed_out, contained = bounded_join(t, 0.1, unwind_s=0.1)
    assert (timed_out, contained) == (True, False)
    t.join(
        2.0
    )  # let the sleep return so the pending injection lands and the thread ends
