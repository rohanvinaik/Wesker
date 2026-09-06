"""A traced worker that could not be STOPPED is not an ordinary cut (issue #19).

`interrupt.abandon` already reports honestly whether a runaway thread is gone — it returns
False for one blocked outside the interpreter, where the async-exception injection cannot
land. `_traced_in_thread` called it and threw that answer away, returning a bare `True` for
"cut". So a worker still executing in a subprocess, socket or C extension left the baseline
trace reporting an ordinary budget trim, and every later measurement in the session ran
against a process hosting a runaway — the exact condition `all_contained` exists to make
non-gateable, reached through the one path that could not say so.

#14 closed this for the mutation runner. This is the baseline-trace half.

`abandon` is patched rather than genuinely blocked outside the interpreter: the real
condition needs a C-level block that is neither portable nor reliably reproducible in a test
suite, while the DEFECT was never about detecting it — `abandon` detected it correctly all
along. The defect was discarding the answer, and that is exactly what these pin.
"""

from __future__ import annotations

import time

import pytest

from Wesker import interrupt as INTERRUPT
from Wesker import line_coverage as LC

_FILE = __file__


def _spin(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total


def _slow() -> None:
    assert _spin(3_000_000) >= 0


def _fast() -> None:
    assert _spin(10) >= 0


_LINES = {33, 34, 35, 36}


@pytest.fixture
def unstoppable(monkeypatch):
    """`abandon` reporting False — a worker confirmed still alive after the injection. Patched at
    the primitive (`interrupt.abandon`): since 2026-09-06 the trace worker is bounded through
    `interrupt.bounded_join`, which reaches `abandon` by name in its own module, so the seam is
    the same one every bounded join in the pair consumes."""
    monkeypatch.setattr(INTERRUPT, "abandon", lambda _thread: False)


def test_a_cut_worker_that_stops_is_reported_contained():
    """The control. An ordinary budget cut on a pure-Python test unwinds, so containment holds
    and nothing downstream should treat it as an invalidated measurement."""
    _covered, was_cut, contained = LC._trace_one(_slow, _FILE, _LINES, budget_s=0.25)
    assert was_cut is True
    assert contained is True


def test_a_cut_worker_that_cannot_be_stopped_is_reported_uncontained(unstoppable):
    """The defect. Same cut, but the stop did not land — and that has to reach the caller
    rather than being flattened into the same `True` a clean cut returns."""
    _covered, was_cut, contained = LC._trace_one(_slow, _FILE, _LINES, budget_s=0.25)
    assert was_cut is True
    assert contained is False


def test_trace_suite_names_the_test_whose_worker_survived(unstoppable):
    """Named, not counted — for the same reason `truncated` is. A caller that cannot say WHICH
    test left a thread running cannot report it, and an unreported runaway is indistinguishable
    from a clean session."""
    cut: set[str] = set()
    loose: set[str] = set()
    LC.trace_suite([_slow], {_FILE}, budget_s=0.25, truncated=cut, uncontained=loose)
    assert loose, "the uncontained worker was not named"
    assert loose <= cut, "an uncontained trace is always also a cut"


def test_a_contained_pass_names_nobody():
    """The other control: a healthy pass must leave the set empty. A propagation bug that
    reported every test as uncontained would satisfy the test above and make every run
    non-gateable forever."""
    cut: set[str] = set()
    loose: set[str] = set()
    LC.trace_suite([_fast], {_FILE}, truncated=cut, uncontained=loose)
    assert loose == set()
    assert cut == set()


def test_the_injection_is_not_left_pending():
    """Housekeeping the real `abandon` owns: repeated cuts must not accumulate live threads."""
    import threading

    before = threading.active_count()
    for _ in range(3):
        LC._trace_one(_slow, _FILE, _LINES, budget_s=0.2)
    time.sleep(0.3)
    assert threading.active_count() <= before + 1
