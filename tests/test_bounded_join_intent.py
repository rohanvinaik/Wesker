"""`interrupt.bounded_join` — a bounded wait that leaves no runaway behind, on EVERY way out.

THE ORPHAN (measured 2026-09-06, `detective converge` on its own engine): a bounded join is often
NESTED. Detective's classifier joins a runaway mutant with a timeout from inside a test, and that
test was being run by the traced baseline in a worker of its own, under a budget. The budget
abandoned the worker while it was parked in the inner join. The injected exception lands at the
first bytecode after the join returns — BEFORE the `if thread.is_alive(): abandon(thread)` that
followed it — so the runaway it was bounding was never stopped: a live thread hogging the GIL for
the rest of the process (two of them in the faulthandler dump, thread numbers past 4,600), and
every later phase crawled. All four bounded-join sites in the pair were written in that shape.

WHAT IS PINNED, AND HOW. The mid-wait claim is about CONTROL FLOW — whether the wait's exit hands
the thread it was bounding to `abandon` — and it is pinned with a FAKE thread whose `join` raises
the abandonment itself and which always reports alive. Not with a real runaway: where an injection
lands, and what else it does, varies with the interpreter and with a tracer. Two earlier versions
of these tests used real runaways; under coverage's C tracer — every CI cell, 3.10 through 3.13,
and Detective's pinned 3.11 locally — the joiner's injection landed INSIDE the C-level join AND
the runaway ended on its own without unwinding through its handler
(the recorded unknown in test_abandon_thread.py, one step stranger), so the "old shape orphans
it" control could not hold there, and the first version leaked runaways into the rest of the
suite — the very defect, committed by its own test. A fake removes the interpreter from the
claim; the real-thread tests below cover the ordinary paths, where the model holds everywhere.

- the defect, as a control: the old shape (join, then abandon-if-alive) never reaches the abandon
  when the joiner is abandoned inside the join;
- the fix: `bounded_join` reaches it on that same exit, and the joiner's own abandonment still
  propagates;
- the ordinary paths keep their meaning: a timeout abandons a real runaway and reports
  (True, True); a thread that finishes in time is (False, True); ``None`` waits unbounded;
- the honest boundary survives: a thread blocked outside the interpreter is (True, False) —
  untraced, per the recorded unknown.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from Wesker import interrupt
from Wesker.interrupt import Abandoned, bounded_join


class _CutInsideTheJoin:
    """A thread whose `join` is where the joiner's own abandonment lands, and which is still
    running afterwards — the nested case, with no interpreter in the loop."""

    ident = None
    name = "fake (_run)"

    def __init__(self) -> None:
        self.joined_with: list = []

    def join(self, timeout=None) -> None:
        self.joined_with.append(timeout)
        raise Abandoned

    def is_alive(self) -> bool:
        return True


@pytest.fixture
def recorded_abandon(monkeypatch):
    """`abandon`, observed: which threads it was asked to stop, in order. The real primitive still
    runs (a fake has no ident, so it honestly reports False); the recorder is the witness."""
    seen: list = []
    real = interrupt.abandon

    def recording(thread):
        seen.append(thread)
        return real(thread)

    monkeypatch.setattr(interrupt, "abandon", recording)
    return seen


# --- the defect, then the fix --------------------------------------------------------------------


def test_the_old_shape_never_reaches_the_abandon_when_the_joiner_is_cut_inside_the_join(
    recorded_abandon,
):
    """The control: what every bounded-join site did before 2026-09-06."""
    runaway = _CutInsideTheJoin()

    def old_shape() -> None:
        runaway.join(0.5)  # the joiner's abandonment lands here…
        if runaway.is_alive():  # …and this line is never reached
            interrupt.abandon(runaway)

    with pytest.raises(Abandoned):
        old_shape()
    assert recorded_abandon == [], "the orphan: nothing ever tried to stop the runaway"


def test_bounded_join_reaches_the_abandon_when_the_joiner_is_cut_inside_the_join(
    recorded_abandon,
):
    """The fix: the same exit, and the runaway is handed to `abandon` on the way out."""
    runaway = _CutInsideTheJoin()
    with pytest.raises(Abandoned):
        bounded_join(runaway, 0.5)
    assert recorded_abandon == [runaway], (
        "no orphan: the runaway was handed to abandon on the way out"
    )
    assert runaway.joined_with == [0.5], "the bounded wait itself was the one join"


def test_the_extra_unwind_allowance_is_granted_only_when_the_injection_did_not_land(
    recorded_abandon,
):
    runaway = _CutInsideTheJoin()
    with pytest.raises(Abandoned):
        bounded_join(runaway, 0.5, unwind_s=0.2)
    # abandon() reports False for a thread with no ident, so the allowance is paid: one more join.
    assert runaway.joined_with == [0.5, 0.2]


# --- the ordinary paths keep their meaning, on real threads --------------------------------------


def _runaway(flag: dict, stop: threading.Event) -> None:
    """A pure-Python runaway that ALSO honours a stop flag, so no test can leak it."""
    try:
        i = 0
        while not stop.is_set():
            flag["n"] = i
            i += 1
    except BaseException:  # noqa: BLE001 — the injection unwinds through here
        flag["unwound"] = True


def test_a_timeout_abandons_a_real_runaway_and_reports_timed_out_and_contained():
    flag: dict = {}
    stop = threading.Event()
    runaway = threading.Thread(target=_runaway, args=(flag, stop), daemon=True)
    runaway.start()
    time.sleep(0.05)  # let it actually enter the loop
    try:
        assert bounded_join(runaway, 0.1) == (True, True)
        assert not runaway.is_alive()
    finally:
        stop.set()


def test_a_thread_that_finishes_in_time_is_neither_timed_out_nor_abandoned(
    recorded_abandon,
):
    done: dict = {}

    def quick() -> None:
        done["ran"] = True

    t = threading.Thread(target=quick, daemon=True)
    t.start()
    assert bounded_join(t, 1.0) == (False, True)
    assert done == {"ran": True}
    assert recorded_abandon == []


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
