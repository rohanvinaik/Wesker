"""A baseline-FAILING test's trace must not close the line ledger (issue #17).

`_build_test_scope` already bars a test that fails on the unmutated program from KILL
attribution — it cannot distinguish a mutant from correct code — and then judged LINE
completeness from a union that still contained that same test's coverage. One body of
evidence, two admissibility rules, and the weaker one decided completeness.

The counterexample is the issue's own: a green test covering the true branch, and a FAILING
test that is the only observation of the false branch. Every executable line appears covered.
The failing test is correctly known to be failing; the defect is that its trace still counted
as proof.

Two views, deliberately. OBSERVED reach stays what mutants are SCOPED with, because routing
must stay conservative — a failing test still executes the line, and dropping it there would
turn a real kill into a reported gap. ADMISSIBLE reach is what a completeness claim may rest
on. `line_coverage` remains the observed map so consumers do not silently change behaviour
when this engine updates; the switchover belongs to Detective #59.
"""

from __future__ import annotations

import ast
import importlib.util
import sys

import pytest

from Wesker.engine import run_function_profiling
from Wesker.filter import filter_categories

_SRC = "def choose(flag):\n    if flag:\n        return 1\n    return 0\n"


@pytest.fixture
def target(tmp_path):
    """A real module on disk imported under its own name, so `co_filename` is a real path the
    module-qualified patch can match — the same shape a consumer repo has."""
    path = tmp_path / "choose_mod.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("choose_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["choose_mod"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod, str(path)
    finally:
        sys.modules.pop("choose_mod", None)


def _profile(tests, mod, path):
    node = ast.parse(_SRC).body[0]
    return run_function_profiling(
        node,
        "choose_mod.py::choose",
        filter_categories(node, True),
        tests,
        mod.choose,
        max_per_category=0,
    )


def _union(cov):
    out: set[int] = set()
    for lines in cov.values():
        out |= set(lines)
    return out


def test_a_failing_tests_reach_is_observed_but_not_admissible(target):
    """The defect, end to end. The failing test is the ONLY observation of the false branch,
    so the two views must disagree — and the admissible one must be the smaller."""
    mod, path = target

    def test_green_branch():
        assert mod.choose(True) == 1

    def test_failing_only_cover():
        assert mod.choose(False) == 1  # fails on the UNMUTATED function

    res = _profile([test_green_branch, test_failing_only_cover], mod, path)

    observed = _union(res.line_coverage)
    admissible = _union(res.admissible_line_coverage)

    assert observed, (
        "no line coverage was measured — the fixture is not exercising the engine"
    )
    assert admissible < observed, (
        f"the failing test's reach survived into the proof view: "
        f"observed={sorted(observed)} admissible={sorted(admissible)}"
    )


def test_the_observed_view_is_not_narrowed(target):
    """The control, and the guard against 'fixing' this by filtering the wrong map. Routing
    scopes on the observed view; narrowing it would drop a test that genuinely reaches the
    mutated line and turn its kills into reported gaps."""
    mod, path = target

    def test_green_branch():
        assert mod.choose(True) == 1

    def test_failing_only_cover():
        assert mod.choose(False) == 1

    both = _profile([test_green_branch, test_failing_only_cover], mod, path)
    green_only = _profile([test_green_branch], mod, path)

    assert _union(both.line_coverage) > _union(green_only.line_coverage), (
        "the failing test's reach must remain visible in the OBSERVED view"
    )


def test_an_all_green_suite_leaves_the_two_views_identical(target):
    """The other control: admissibility must cost nothing when every owner is admissible.
    A filter that quietly narrowed a healthy suite would be a worse bug than the one fixed."""
    mod, path = target

    def test_true_branch():
        assert mod.choose(True) == 1

    def test_false_branch():
        assert mod.choose(False) == 0

    res = _profile([test_true_branch, test_false_branch], mod, path)
    assert _union(res.line_coverage) == _union(res.admissible_line_coverage)
