"""The `exception` kill class, and `--version`.

WHY `exception` EXISTS. pytest signals a violated `pytest.raises(...)` contract by raising
`Failed`, whose MRO is `Failed -> OutcomeException -> BaseException` — NOT `AssertionError`.
So `except AssertionError` cannot see it and it landed in the BaseException fallback beside
genuine crashes. That is a category error with teeth: `value_survivor_records` re-lists every
non-value kill as unpinned, so a mutant killed by an error-path test came back as a survivor,
was re-classified killable off the same witness, and the caller was asked for an input that
would rebuild the same test and be discarded again — a loop with no exit, on a mutant that was
being killed the whole time. Any function that validates its inputs by raising hit it.

A crash means the mutant blew up and no test said anything about its VALUE. A declared failure
means a test STATED a contract and the mutant broke it. The second pins behaviour; the first
does not. Raising IS the return behaviour of an error path, and this is the only way to pin it.
"""

from __future__ import annotations

import pytest

from Wesker.cli import main
from Wesker.engine import CategoryResult, ProfilingResult, _is_declared_failure


# ── _is_declared_failure ─────────────────────────────────────────────
def test_violated_raises_contract_is_a_declared_failure():
    # B017,PT011: the exact type IS the test
    with pytest.raises(BaseException) as exc:  # noqa: B017,PT011
        with pytest.raises(ValueError):
            pass  # nothing raised -> pytest declares failure
    assert _is_declared_failure(exc.value)


def test_pytest_fail_is_a_declared_failure():
    try:
        pytest.fail("x")
    except BaseException as e:  # noqa: BLE001
        assert _is_declared_failure(e)


def test_skip_is_not_a_failure():
    """`Skipped` shares the `OutcomeException` base and would match a base-only check. A skip
    is not a failure and must never read as a kill."""
    try:
        pytest.skip("x")
    except BaseException as e:  # noqa: BLE001
        assert not _is_declared_failure(e)


def test_plain_exceptions_are_not_declared_failures():
    assert not _is_declared_failure(AssertionError("x"))
    assert not _is_declared_failure(ValueError("x"))
    assert not _is_declared_failure(TypeError("x"))


def test_identified_by_base_not_module():
    """`Failed.__module__` is the string "builtins" — pytest rewrites it so tracebacks read
    `Failed` rather than `_pytest.outcomes.Failed`. Trusting that attribute matches NOTHING,
    which is worse than not checking: every raises-kill keeps its crash verdict and the
    classifier looks like it simply does not work. `OutcomeException.__module__` is honest."""
    from _pytest.outcomes import Failed, OutcomeException

    assert Failed.__module__ == "builtins"  # the trap
    assert OutcomeException.__module__.startswith("_pytest")  # the signal actually used
    assert not issubclass(
        Failed, AssertionError
    )  # why `except AssertionError` misses it


def test_a_lookalike_class_is_not_matched():
    """Name alone is not enough — an unrelated user exception called `Failed` is not pytest's."""

    class Failed(Exception):
        pass

    assert not _is_declared_failure(Failed())


# ── an exception kill PINS the value ─────────────────────────────────
def _rec(mid: str, killed_by: str) -> dict:
    return {"mutant_id": mid, "killed_by": killed_by, "diff_summary": "- a\n+ b"}


def test_exception_kills_are_not_re_listed_as_value_survivors():
    """THE regression. A crash kill is re-listed because it proves the code RAN, not what it
    returned. An exception kill proves what it returned (that it raised), so re-listing it
    reported a pin as a gap."""
    r = ProfilingResult(
        function_key="m.py::f",
        killed_records=[
            _rec("A", "assertion"),
            _rec("E", "exception"),
            _rec("C", "crash"),
        ],
        survivor_records=[],
    )
    ids = {rec["mutant_id"] for rec in r.value_survivor_records}
    assert ids == {"C"}  # crash only — assertion and exception both pin the value


def test_value_killed_counts_assertion_and_exception():
    cr = CategoryResult(
        category="BOUNDARY",
        total=3,
        killed=3,
        killed_by_assertion=1,
        killed_by_exception=1,
    )
    assert cr.value_killed == 2


def test_value_survived_still_excludes_exception():
    """The other side of the same contract: an exception kill must not leak into the
    unspecified count either."""
    cr = CategoryResult(
        category="BOUNDARY",
        total=4,
        killed=3,
        killed_by_assertion=1,
        killed_by_exception=1,
        killed_by_crash=1,
        survived=1,
        timed_out=0,
    )
    assert cr.value_survived == 2  # survived + crash — never the exception kill


# ── --version ────────────────────────────────────────────────────────
def test_version_reports_the_single_owner(capsys):
    """There was no `--version` at all: it printed an argparse usage error, so the only way to
    ask an installed engine what it was was to import it. This engine DECIDES the verdict and
    its consumers key their caches on this number, so it has to be able to say what it is —
    and from `Wesker.__version__`, the one owner. 0.6.0 shipped to PyPI announcing itself as
    0.5.1 because that number lived in two places."""
    import Wesker

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"wesker {Wesker.__version__}"
