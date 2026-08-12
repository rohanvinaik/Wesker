"""Discovery honors the repo's real pytest test config, not a hardcoded `test_*.py` glob.

Found by dogfooding python-slugify: its whole 82-test suite is a bare ``test.py`` (collected by
pytest only because ``testpaths`` names it), and Wesker saw ZERO tests — so every function read a
misleading "0 pinned". Two independent collectors shared the bug:

1. STATIC discovery (`_discover_all_test_files`) globbed only ``test_*.py`` — missing both ``test.py``
   and ``*_test.py`` (a pytest DEFAULT), and never consulting ``testpaths``.
2. The LIVE pytest session (`run_in_session`) passed ``["."]`` as an explicit path arg, which makes
   pytest IGNORE ``testpaths`` entirely, so a testpaths-only ``test.py`` came back empty.

INTENT tests: a characterization of the buggy output would pin "0 tests" as correct. These assert the
contract — the suite pytest would collect IS the suite Wesker measures against — and each fails
without the fix.
"""

from __future__ import annotations

from Wesker.ci import (
    _discover_all_test_files,
    _DEFAULT_TEST_PATTERNS,
    relevant_test_files,
)
from Wesker.pytest_runner import run_in_session


def _names(paths):
    return {p.rsplit("/", 1)[-1] for p in paths}


# ── static discovery: honor python_files defaults AND explicit testpaths ──


def test_star_test_py_is_a_default_pattern_not_just_test_prefix(tmp_path):
    """``*_test.py`` is one of pytest's two DEFAULT ``python_files`` patterns; the old glob matched
    only ``test_*.py`` and dropped it. Both defaults must be honored with no config at all."""
    (tmp_path / "foo_test.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "test_bar.py").write_text("def test_b():\n    assert True\n")
    found = _names(_discover_all_test_files(str(tmp_path)))
    assert found == {"foo_test.py", "test_bar.py"}


def test_a_bare_test_py_is_invisible_by_default_but_found_when_testpaths_names_it(
    tmp_path,
):
    """A bare ``test.py`` matches NEITHER default pattern — exactly as under pytest, which collects it
    only because ``testpaths`` names it. So it is absent by default and present once ``testpaths``
    points at it: the same rule pytest applies, not a wider net."""
    (tmp_path / "test.py").write_text("def test_ok():\n    assert True\n")
    assert _names(_discover_all_test_files(str(tmp_path))) == set()
    assert _names(_discover_all_test_files(str(tmp_path), testpaths=("test.py",))) == {
        "test.py"
    }


def test_relevant_test_files_threads_testpaths_so_a_bare_suite_reaches_a_target(
    tmp_path,
):
    """End of the static chain: a ``test.py`` that exercises the target is selected as relevant ONLY
    when its testpaths reach discovery. This is the exact slugify shape — the suite calls the target
    by name, but the file it lives in is a bare ``test.py``."""
    (tmp_path / "lib.py").write_text("def widen(x):\n    return x + 1\n")
    (tmp_path / "test.py").write_text(
        "from lib import widen\n\n\ndef test_widen():\n    assert widen(1) == 2\n"
    )
    without = relevant_test_files(str(tmp_path), str(tmp_path / "lib.py"), ["widen"])
    with_tp = relevant_test_files(
        str(tmp_path), str(tmp_path / "lib.py"), ["widen"], testpaths=("test.py",)
    )
    assert _names(without) == set()
    assert _names(with_tp) == {"test.py"}


def test_default_patterns_are_pytest_s_real_defaults():
    """Guard the constant itself: pytest's default ``python_files`` is BOTH, not one."""
    assert _DEFAULT_TEST_PATTERNS == ("test_*.py", "*_test.py")


# ── live session: an unscoped run must resolve collection the way the suite does ──


def test_live_session_honors_testpaths_when_unscoped(tmp_path):
    """`run_in_session(paths=None)` must let pytest honor ``testpaths``. Passing ``["."]`` made pytest
    ignore it, so a testpaths-only ``test.py`` collected nothing (returned None). With the fix the
    82-test-shaped case — a bare ``test.py`` named by testpaths — is collected."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = "test.py"\n'
    )
    (tmp_path / "test.py").write_text("def test_ok():\n    assert True\n")
    collected = run_in_session(
        str(tmp_path), lambda calls, _session: len(calls), paths=None
    )
    assert collected == 1
