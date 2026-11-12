"""The session baseline's one-time cost is not charged to the function that happens to build it (Wesker #29).

THE DEFECT, measured in the first whole-codebase run that printed its report (run 43102482321):
`Wesker/_isolated_worker.py::_build_mutant — 1035.9 s, 0/62 mutants evaluated`, listed beside eleven
engine functions as "ran past its budget, which applies to each function separately". It was the first
function profiled in the first file. The session baseline (the whole suite traced once) is built lazily by
whichever function asks first, and the build ran inside that function's own wall clock: the function's
budget paid for the suite, and the function was reported as the one that overran. Reproduced
(`profile_function('generate_mutants')` as the first function: 999.6 s, 0/0 mutants, budget exhausted, with
no mutant evaluated at all).

The baseline has its own bounds (a per-test and a session trace budget). The per-function budget is for the
function's own mutants.
"""

from __future__ import annotations

import ast
import time

from Wesker.engine import (
    _SESSION_BASELINE,
    LazySessionBaseline,
    MutationCategory,
    baseline_build_seconds,
    build_session_baseline,
    run_function_converged,
    run_function_profiling,
)

_SRC = "def scoreit(a, b, flag):\n    if flag:\n        return a * 2 + b\n    return a - b\n"
_CATS = {MutationCategory.VALUE, MutationCategory.ARITHMETIC}
_BUILD_S = 0.8
_BUDGET_MS = 400.0


def _fixture():
    node = ast.parse(_SRC).body[0]
    ns: dict = {}
    # S102: test fixture source
    exec(compile(ast.parse(_SRC), "<baseline-cost>", "exec"), ns)  # noqa: S102
    original = ns["scoreit"]

    def test_pins():
        assert scoreit(1, 2, True) == 4  # noqa: F821

    test_pins.__globals__["scoreit"] = original
    return node, original, [test_pins]


def _holder(tests, original):
    def build(subset=None, fresh=False, **_kw):
        time.sleep(
            _BUILD_S
        )  # a slow suite: the trace takes longer than the function's whole budget
        return build_session_baseline(
            tests if subset is None else list(subset),
            {original.__code__.co_filename},
        )

    return LazySessionBaseline(build)


def _run(runner, **kw):
    node, original, tests = _fixture()
    holder = _holder(tests, original)
    token = _SESSION_BASELINE.set(holder)
    try:
        return holder, runner(
            node,
            f"{original.__code__.co_filename}::scoreit",
            _CATS,
            tests,
            original,
            budget_ms=_BUDGET_MS,
            max_per_category=0,
            **kw,
        )
    finally:
        _SESSION_BASELINE.reset(token)


def test_the_holder_records_what_its_first_build_cost():
    _node, original, tests = _fixture()
    holder = _holder(tests, original)
    assert holder.build_seconds == 0.0, "unbuilt: no cost, not a measured zero build"
    holder.get()
    assert holder.build_seconds >= _BUILD_S
    cost = holder.build_seconds
    holder.get()
    assert holder.build_seconds == cost, "memoised: a second read builds nothing"


def test_outside_a_session_there_is_no_baseline_cost():
    assert baseline_build_seconds() == 0.0


def test_a_profile_that_builds_the_baseline_is_not_charged_for_it():
    holder, result = _run(run_function_profiling)
    assert holder.build_seconds >= _BUILD_S
    assert result.budget_exhausted is False
    assert result.total_mutants > 0, "its own mutants were evaluated"
    assert result.elapsed_ms < _BUILD_S * 1000, result.elapsed_ms


def test_a_converged_profile_that_builds_the_baseline_is_not_charged_for_it():
    holder, result = _run(run_function_converged)
    assert holder.build_seconds >= _BUILD_S
    assert result.budget_exhausted is False
    assert result.total_mutants > 0
    assert result.elapsed_ms < _BUILD_S * 1000, result.elapsed_ms


def test_a_function_that_overruns_on_its_own_mutants_is_still_cut():
    """The credit is for the baseline only: a budget too small for the function's own work still cuts."""
    _, result = _run(run_function_profiling, per_mutant_timeout_ms=5000)
    node, original, tests = _fixture()
    holder = _holder(tests, original)
    holder.get()  # built beforehand: nothing to credit
    token = _SESSION_BASELINE.set(holder)
    try:
        tight = run_function_profiling(
            node,
            f"{original.__code__.co_filename}::scoreit",
            _CATS,
            tests,
            original,
            budget_ms=0.0001,
            max_per_category=0,
        )
    finally:
        _SESSION_BASELINE.reset(token)
    assert tight.budget_exhausted is True
    assert result.budget_exhausted is False
