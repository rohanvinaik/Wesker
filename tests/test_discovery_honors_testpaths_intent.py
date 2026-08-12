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
    callable_origin,
    discover_test_callables,
    relevant_test_files,
    run_with_live_suite,
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


# ── isolation: pin a function against its RELEVANT tests, never the whole global regime ──


def test_testpaths_scopes_discovery_excluding_non_testpaths_dirs(tmp_path):
    """pytest with ``testpaths`` collects ONLY under those paths. A ``test_*.py`` in ``bench/`` /
    ``examples/`` that pytest never collects (and that needs its own deps) must NOT enter the impact
    map — else profiling one function scopes onto a test that errors the collection. Found dogfooding
    structlog, whose ``bench/test_benchmarks.py`` (needs pytest-codspeed) was pulled in under
    ``testpaths = "tests"`` for a pure log-level function."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_lib.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "test_benchmarks.py").write_text(
        "def test_b():\n    assert True\n"
    )
    scoped = _names(_discover_all_test_files(str(tmp_path), testpaths=("tests",)))
    assert scoped == {"test_lib.py"}, (
        scoped
    )  # bench/ excluded — pytest wouldn't collect it
    # With NO testpaths, pytest recurses the whole tree, so both are candidates.
    unscoped = _names(_discover_all_test_files(str(tmp_path)))
    assert unscoped == {"test_lib.py", "test_benchmarks.py"}


def test_a_scoped_collection_error_does_not_widen_to_the_whole_suite(tmp_path):
    """The isolation contract. If the tests reaching THIS function cannot collect (a broken import in
    them), do NOT widen to the whole suite — widening measures the target against IRRELEVANT tests it
    never reaches and drags in every other module's failure. `run_with_live_suite` returns None with
    reason ``collection_errors`` (the caller refuses with the relevant error), never silently swapping
    in a different suite. `test_clean.py` here is what the old whole-suite widen WOULD have run."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_broken.py").write_text(
        "import totally_nonexistent_xyz  # noqa\n\n\ndef test_b():\n    assert True\n"
    )
    (tmp_path / "tests" / "test_clean.py").write_text(
        "def test_c():\n    assert True\n"
    )
    diag: dict = {}
    ran = run_with_live_suite(
        str(tmp_path),
        lambda: "BODY_RAN",
        paths=[str(tmp_path / "tests" / "test_broken.py")],
        diagnostic=diag,
    )
    assert ran is None  # did NOT widen to collect test_clean and run the body
    assert diag.get("reason") == "collection_errors"


def test_a_clean_function_profiles_despite_an_unrelated_broken_module(tmp_path):
    """The user-facing payoff (structlog): a broken UNRELATED test module must not block profiling a
    function whose OWN reachable tests are clean. Plain pytest aborts collection on the broken module;
    scoped discovery finds the target's real tests and ignores the rest."""
    import sys

    (tmp_path / "lib.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_lib.py").write_text(
        "from lib import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )
    (tmp_path / "tests" / "test_broken.py").write_text(
        "import totally_nonexistent_xyz  # noqa\n\n\ndef test_b():\n    assert True\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:

        def body():
            cs = discover_test_callables(
                str(tmp_path), "lib.py", ["f"], testpaths=("tests",)
            )
            return [callable_origin(c) or "" for c in cs]

        origins = run_with_live_suite(str(tmp_path), body, target_files=["lib.py"])
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("lib", None)
    assert origins is not None, "the clean function's tests could not be found"
    assert any(o.endswith("test_lib.py") for o in origins)
    assert not any(o.endswith("test_broken.py") for o in origins)
