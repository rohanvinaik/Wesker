"""Tests for the pytest discovery backend — in particular that @parametrize
cases are bound into runnable in-process callables (not skipped), which is what
lets Wesker consume idiomatic parametrized suites.
"""

from __future__ import annotations

import textwrap

from Wesker.pytest_discovery import collect_pytest_callables


def _write(tmp_path, name: str, body: str) -> None:
    # Unique module name per test: nested pytest.main imports test modules by
    # name into sys.modules, so a shared filename would collide across cases.
    (tmp_path / f"test_{name}.py").write_text(textwrap.dedent(body))


def test_parametrized_test_binds_one_callable_per_case(tmp_path):
    _write(
        tmp_path,
        "param",
        """
        import pytest

        @pytest.mark.parametrize("x, expected", [(1, 1), (2, 2), (3, 99)])
        def test_identity(x, expected):
            assert x == expected
        """,
    )
    callables = collect_pytest_callables(str(tmp_path))
    assert callables is not None
    assert len(callables) == 3  # one bound callable per parametrize case

    # Each runs in-process; a value mismatch raises AssertionError (an
    # "assertion" kill for Wesker), NOT a TypeError (a false "crash" kill).
    outcomes = []
    for c in callables:
        try:
            c()
            outcomes.append("pass")
        except AssertionError:
            outcomes.append("assertion")
    assert outcomes == ["pass", "pass", "assertion"]


def test_fixture_requiring_test_is_skipped_plain_kept(tmp_path):
    _write(
        tmp_path,
        "fixture",
        """
        def test_needs_fixture(tmp_path):
            assert tmp_path.exists()

        def test_plain():
            assert True
        """,
    )
    callables = collect_pytest_callables(str(tmp_path))
    assert callables is not None
    names = [getattr(c, "__name__", "") for c in callables]
    assert any("test_plain" in n for n in names)
    assert not any("needs_fixture" in n for n in names)


def test_zero_arg_test_runs_directly(tmp_path):
    _write(
        tmp_path,
        "plain",
        """
        def test_truth():
            assert 1 + 1 == 2
        """,
    )
    callables = collect_pytest_callables(str(tmp_path))
    assert callables is not None
    assert len(callables) == 1
    callables[0]()  # runs without raising


def test_parametrized_cases_fingerprint_apart(tmp_path):
    """Each bound case must carry its own trace-cache identity — nodeid on
    ``__qualname__``, the user's test on ``__wrapped__``.

    With neither set (the 0.9.2 regression), `test_fingerprint` hashed the
    bound closure itself: two Wesker-internal constants, identical for every
    parametrized case in every project. The per-test trace cache then served
    whichever case was traced first as the coverage of all of them — a stale
    line-17-style gap no re-run could clear, because the poisoned entry's key
    matched forever.
    """
    from Wesker.trace_cache import test_fingerprint

    _write(
        tmp_path,
        "fpident",
        """
        import pytest

        @pytest.mark.parametrize("x, expected", [(5, 10), (-3, -4)])
        def test_double_or_dec(x, expected):
            assert (x * 2 if x > 0 else x - 1) == expected
        """,
    )
    callables = collect_pytest_callables(str(tmp_path))
    assert callables is not None
    cases = [c for c in callables if "test_double_or_dec" in getattr(c, "__name__", "")]
    assert len(cases) == 2

    # The identity contract itself, not just its consequence: __wrapped__ is
    # the USER's function (so a source edit still invalidates the entry) and
    # __qualname__ discriminates the cases (the nodeid).
    assert all(getattr(c, "__wrapped__", None) is not None for c in cases)
    quals = {getattr(c, "__qualname__", "") for c in cases}
    assert len(quals) == 2

    fps = {test_fingerprint(c) for c in cases}
    assert len(fps) == 2  # sibling cases fingerprint APART


def test_collected_callables_carry_origin_tag(tmp_path):
    """Every collected callable resolves to its defining test FILE via
    ci.callable_origin — including parametrized closures, whose raw
    ``co_filename`` is this discovery module, not the test's."""
    from Wesker.ci import callable_origin

    _write(
        tmp_path,
        "origin",
        """
        import pytest

        @pytest.mark.parametrize("x", [1, 2])
        def test_param(x):
            assert x > 0

        def test_plain():
            assert True
        """,
    )
    callables = collect_pytest_callables(str(tmp_path))
    assert callables
    expected = str(tmp_path / "test_origin.py")
    for c in callables:
        assert callable_origin(c) == expected
