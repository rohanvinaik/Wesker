"""Intent: `_ExcCapture`'s makereport wrapper never re-raises the wrapped call's exception in its teardown.

The defect, seen in Detective's suite under `-W default`: `PluggyTeardownRaisedWarning: A plugin raised an
exception during an old-style hookwrapper teardown. Hook: pytest_runtest_makereport. Abandoned`.
`_ExcCapture.pytest_runtest_makereport` called `outcome.get_result()` unconditionally after `yield`. When
the wrapped makereport call itself raised — an abandoned test's `Abandoned` unwinding through the chain —
`get_result()` re-raised it inside the wrapper's teardown. The exception still propagated, so no verdict
changed; the warning was the only symptom, and under `filterwarnings = error` a warning is a failure.

The wrapper now returns when the outcome carries an exception, and pluggy propagates the original. These
tests drive the generator directly with a stand-in outcome: on the old code the raised-call cases fail
because `send()` re-raises instead of finishing.
"""

import pytest

from Wesker.interrupt import Abandoned
from Wesker.pytest_runner import _ExcCapture


class _Outcome:
    """Pluggy's old-style result shape: `excinfo` is None when the call returned."""

    def __init__(self, excinfo=None, result=None) -> None:
        self.excinfo = excinfo
        self._result = result
        self.get_result_called = False

    def get_result(self):
        self.get_result_called = True
        if self.excinfo is not None:
            raise self.excinfo[1]
        return self._result


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid


class _ExcInfo:
    def __init__(self, value: BaseException) -> None:
        self.value = value


class _Call:
    def __init__(self, excinfo=None) -> None:
        self.excinfo = excinfo


class _Report:
    def __init__(self, when: str, failed: bool) -> None:
        self.when = when
        self.failed = failed


def _finish(capture: _ExcCapture, item: _Item, call: _Call, outcome: _Outcome) -> None:
    gen = capture.pytest_runtest_makereport(item, call)
    next(gen)
    with pytest.raises(StopIteration):
        gen.send(outcome)


@pytest.mark.parametrize("exc", [RuntimeError("makereport raised"), Abandoned()])
def test_a_raised_makereport_call_finishes_the_wrapper_without_re_raising(exc) -> None:
    capture = _ExcCapture()
    outcome = _Outcome(excinfo=(type(exc), exc, None))
    _finish(capture, _Item("t.py::raised"), _Call(), outcome)
    assert not outcome.get_result_called
    assert capture.last == {}, "nothing to record when there is no report"


def test_a_passing_call_still_records_no_exception() -> None:
    capture = _ExcCapture()
    _finish(
        capture,
        _Item("t.py::passes"),
        _Call(),
        _Outcome(result=_Report("call", failed=False)),
    )
    assert capture.last == {"t.py::passes": None}


def test_a_failing_call_still_records_its_real_exception() -> None:
    capture = _ExcCapture()
    boom = AssertionError("the value pin")
    _finish(
        capture,
        _Item("t.py::fails"),
        _Call(_ExcInfo(boom)),
        _Outcome(result=_Report("call", failed=True)),
    )
    assert capture.last == {"t.py::fails": boom}
