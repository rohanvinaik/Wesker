"""EP-C1: a measurement session formats no traceback nobody reads — and loses no outcome.

(Detective docs/ENGINEERING_PASS_2026-09-26.md.) Every killed mutant is a failing test to pytest, and
pytest's ``repr_failure`` formatted a source-annotated traceback for each one — re-parsing source with
``ast`` per traceback entry — for a report this session reads only for ``failed`` and the raw
exception. Measured on a converge of an 18-line function: ~45% of the run's CPU and 99% of its AST
allocation. The session now installs a cheap repr on its own items.

What must NOT change, pinned here from intent: the runner re-raises the ORIGINAL exception — an
AssertionError stays an assertion (a value kill) and anything else stays a crash, because the engine's
assertion-vs-crash precedence is built on exactly that — a setup-phase failure still fails, and a green
test still passes. A SAME check through the real CLI (identical FINAL lines and byte-identical
generated suites and certificates on two fixtures) is recorded in the commit that added this.

ZERO-ARG tests and a fresh temp dir per case, for the reasons tests/test_reset_item.py gives.
"""

from __future__ import annotations

import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest
from _pytest.runner import runtestprotocol

from Wesker.pytest_runner import _cheap_failure_repr, run_in_session

_SUITE = """
    import pytest

    @pytest.fixture
    def broken():
        raise RuntimeError("setup blew up")

    def test_value():
        assert 1 + 1 == 3

    def test_crash():
        raise KeyError("missing")

    def test_setup(broken):
        pass

    def test_green():
        assert True
"""


def _run(name, body):
    """UNIQUE module name per call: a nested session imports test modules BY NAME into sys.modules, so
    two cases sharing a filename collide and the second session collects nothing (measured while
    writing this file — the second case read ``None``)."""
    root = Path(tempfile.mkdtemp(prefix="wesker_cheap_repr_"))
    try:
        (root / f"test_cheap_repr_{name}.py").write_text(textwrap.dedent(_SUITE))
        return run_in_session(str(root), body)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _classify(calls):
    out = {}
    for call in calls:
        try:
            call()
            out[call.__name__] = "passed"
        except AssertionError:
            out[call.__name__] = "assertion"
        # BLE001: the test's own exception, named by type; the breadth is what is being observed
        except Exception as exc:  # noqa: BLE001
            out[call.__name__] = type(exc).__name__
    return out


def test_the_engine_still_receives_the_original_exception():
    installed, outcomes = _run(
        "outcomes",
        lambda calls, session: (
            all(
                getattr(item, "repr_failure", None) is _cheap_failure_repr
                for item in session.items
            ),
            _classify(calls),
        ),
    )
    assert installed
    assert outcomes == {
        "test_value": "assertion",
        "test_crash": "KeyError",
        "test_setup": "RuntimeError",
        "test_green": "passed",
    }


def test_a_failing_report_carries_the_short_text_not_a_formatted_traceback():
    def body(calls, session):
        item = next(i for i in session.items if i.name == "test_crash")
        return [
            r.longrepr
            for r in runtestprotocol(item, nextitem=None, log=False)
            if r.failed
        ]

    assert _run("longrepr", body) == ["KeyError: 'missing'"]


def test_the_cheap_repr_is_the_type_and_the_message():
    try:
        raise ValueError("bad value")
    except ValueError as exc:
        info = pytest.ExceptionInfo.from_exception(exc)
    assert _cheap_failure_repr(info) == "ValueError: bad value"


def test_an_exception_that_cannot_be_printed_still_yields_its_type():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no string for you")

    try:
        raise Unprintable
    except (
        Unprintable
    ) as exc:  # raised, so it carries the __traceback__ from_exception requires
        info = pytest.ExceptionInfo.from_exception(exc)
    assert _cheap_failure_repr(info) == "Unprintable"
