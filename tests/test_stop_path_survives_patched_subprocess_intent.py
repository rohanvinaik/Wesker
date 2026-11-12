"""The stop path must not trust names the measured tests can replace (#29).

The defect, from the first whole-codebase run that carried ``release_child_waits`` (Specification (full)
run 43303307762, 2026-11-12): the measurement ended 24 minutes in, inside its first file, on

    File ".../Wesker/interrupt.py", line 384, in _popen_pids
        isinstance(value, subprocess.Popen)
    TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union

``bounded_join`` was stopping a worker whose test had replaced ``subprocess.Popen`` with a stand-in. The
stop path runs inside the measuring process, between the tests it measures, so it looked the name up on
the module and found the test's object. A containment helper that reads the measured process's state by
name can be broken by that state, and here it took the whole run down with it.

What these tests state:

* the real ``Popen`` class is bound at import, so a worker parked on a real child is still found and
  released while ``subprocess.Popen`` and ``subprocess.run`` are replaced by things that are not types;
* a frame local that is a mock, a proxy, or an object whose attributes raise names no child and raises
  nothing;
* when the search for children itself fails, the release reports that the worker was not released. It
  never raises: an unreleased worker is a named, contained outcome, a crash is the end of the run.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from typing import Self
from unittest import mock

import pytest

from Wesker import interrupt
from Wesker.interrupt import _is_unreaped_child, _popen_pids, release_child_waits

_SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


class _Parked:
    """A worker thread parked in ``Popen.wait`` on a real child, as a test in ``subprocess.run`` is."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._body, daemon=True)

    def _body(self) -> None:
        proc = subprocess.Popen(_SLEEPER)
        self.proc = proc
        self.started.set()
        proc.wait()

    def __enter__(self) -> Self:
        self.thread.start()
        assert self.started.wait(10), "the child never started"
        # let the worker reach the wait, so its frames hold the child
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not _popen_pids(self.thread):
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.thread.join(10)


def _not_a_type(*args: object, **kwargs: object) -> None:
    raise AssertionError("the stop path called a name a test replaced")


def test_a_parked_worker_is_found_while_popen_is_replaced_by_a_non_type(monkeypatch):
    with _Parked() as parked:
        assert parked.proc is not None
        monkeypatch.setattr(subprocess, "Popen", _not_a_type)
        assert _popen_pids(parked.thread) == [parked.proc.pid]


def test_a_parked_worker_is_released_while_popen_and_run_are_replaced(monkeypatch):
    with _Parked() as parked:
        assert parked.proc is not None
        monkeypatch.setattr(subprocess, "Popen", mock.MagicMock())
        monkeypatch.setattr(subprocess, "run", _not_a_type)
        assert release_child_waits(parked.thread, grace_s=10) is True
        assert not parked.thread.is_alive()
        assert parked.proc.poll() is not None, (
            "the child the worker waited on was stopped"
        )


def test_the_real_class_is_the_one_bound_at_import():
    assert interrupt._REAL_POPEN is subprocess.Popen
    assert interrupt._REAL_RUN is subprocess.run


class _Raises:
    @property
    def returncode(self) -> int:
        raise RuntimeError("an attribute a test's object raises from")

    @property
    def pid(self) -> int:
        raise RuntimeError("an attribute a test's object raises from")


class _PopenWhoseAttributesRaise(subprocess.Popen):
    """A real subclass, so the isinstance holds and the attribute read is what fails."""

    def __init__(self) -> None:  # no process is spawned
        pass

    def __del__(
        self,
    ) -> None:  # the base finalizer reads attributes this object never set
        pass

    @property
    def returncode(self) -> int:  # type: ignore[override]
        raise RuntimeError("returncode raises")

    @returncode.setter
    def returncode(self, value: object) -> None:
        pass


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        "Popen",
        object(),
        mock.MagicMock(),
        _Raises(),
        _PopenWhoseAttributesRaise(),
    ],
    ids=[
        "none",
        "int",
        "str",
        "object",
        "magicmock",
        "raises",
        "popen_subclass_that_raises",
    ],
)
def test_a_local_that_is_not_an_unreaped_child_names_none_and_raises_nothing(value):
    assert _is_unreaped_child(value) is False


def test_a_reaped_child_is_not_a_child_to_stop():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(30)
    assert _is_unreaped_child(proc) is False


def test_an_unreaped_child_is_one():
    proc = subprocess.Popen(_SLEEPER)
    try:
        assert _is_unreaped_child(proc) is True
    finally:
        proc.kill()
        proc.wait(30)


def test_a_search_that_fails_reports_not_released_and_does_not_raise(monkeypatch):
    def _boom(thread: object) -> list[int]:
        raise TypeError(
            "isinstance() arg 2 must be a type, a tuple of types, or a union"
        )

    monkeypatch.setattr(interrupt, "_popen_pids", _boom)
    with _Parked() as parked:
        assert release_child_waits(parked.thread, grace_s=0.2) is False
        assert parked.thread.is_alive(), "nothing was stopped, and the answer says so"


def test_a_process_table_that_cannot_be_read_reports_not_released(monkeypatch):
    def _boom() -> dict[int, list[int]]:
        raise OSError("the process table is not readable")

    monkeypatch.setattr(interrupt, "_children_map", _boom)
    with _Parked() as parked:
        assert release_child_waits(parked.thread, grace_s=0.2) is False


def test_a_thread_with_no_frames_names_no_child():
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    assert _popen_pids(done) == []
    assert release_child_waits(done, grace_s=0.1) is False
