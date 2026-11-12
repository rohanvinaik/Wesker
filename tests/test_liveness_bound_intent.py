"""Intent: Wesker #27 — the wall that bounds a mutant's liveness follows the machine.

The defect, verbatim from the issue: "The isolated wall is `max(per_mutant_timeout_ms / 1000,
_ISOLATED_MIN_TIMEOUT_S = 10.0)`, plus armed cost. Work the count cannot see still ends
`undetermined` at a fixed wall: helper loops, C builtins, other threads." So a 140M-iteration helper
read COMPLETE on a fast machine and UNGATEABLE on a slow one: the verdict depended on how fast the
machine happened to be.

The bound is now a compute budget (reference seconds) stretched by the measured slowdown of this
machine on a fixed reference workload; a bound that fires is retried once, and a second fire is the
named non-kill `undetermined`. These tests pin the pure decisions (the arithmetic and the codes), the
property the issue asks for (a slower machine gets a proportionally longer bound, so a function that
fits on one fits on the other), and the engine wiring end to end through a real isolated worker.
"""

from __future__ import annotations

import ast
import importlib
import sys
import textwrap

import pytest

from Wesker import monitoring
from Wesker.engine import _isolated_wall_s
from Wesker.monitoring import (
    COMPUTE_BUDGET_ENV,
    LIVENESS_BUDGET_S,
    compute_budget_s,
    harness_slowdown,
    liveness_action,
    liveness_bound_ms,
    parse_compute_budget,
)

# ── the pure decisions ──────────────────────────────────────────────────────────


def test_slowdown_names_why_it_is_the_number_it_is():
    assert harness_slowdown(200, 100, 64) == (2.0, "slower")
    assert harness_slowdown(100, 100, 64) == (1.0, "reference")
    # A faster machine does not shrink the bound: the budget is a floor.
    assert harness_slowdown(50, 100, 64) == (1.0, "reference")
    assert harness_slowdown(100 * 1000, 100, 64) == (64.0, "capped")


def test_no_measurement_is_named_not_read_as_the_reference_machine():
    for measured, reference, cap in (
        (0, 100, 64),
        (-5, 100, 64),
        (100, 0, 64),
        (100, 100, 0),
    ):
        assert harness_slowdown(measured, reference, cap) == (1.0, "unmeasured")


def test_the_bound_stretches_with_the_machine_and_names_the_deciding_term():
    assert liveness_bound_ms(500.0, 10_000.0, 0.0, 1.0) == (10_000.0, "budget")
    assert liveness_bound_ms(500.0, 10_000.0, 0.0, 4.0) == (40_000.0, "budget")
    assert liveness_bound_ms(30_000.0, 10_000.0, 0.0, 2.0) == (60_000.0, "requested")
    # Armed: reaching the caps is paid for on top of the (stretched) budget, and not stretched twice.
    assert liveness_bound_ms(500.0, 10_000.0, 3_000.0, 2.0) == (23_000.0, "armed")
    # A factor below 1 is read as 1 (a bound never shrinks).
    assert liveness_bound_ms(500.0, 10_000.0, 0.0, 0.25) == (10_000.0, "budget")


@pytest.mark.parametrize("work_ms", [1_000.0, 6_000.0, 9_000.0, 12_000.0, 40_000.0])
@pytest.mark.parametrize("slowdown", [1.0, 2.5, 4.0, 10.0])
def test_a_function_that_fits_on_a_fast_machine_fits_on_a_slow_one(work_ms, slowdown):
    # The acceptance's "SAME check": work that takes `work_ms` here takes `work_ms * slowdown` on a
    # machine `slowdown` times slower. Whether it beats the bound must not depend on the machine.
    fast, _ = liveness_bound_ms(500.0, 10_000.0, 0.0, 1.0)
    slow, _ = liveness_bound_ms(500.0, 10_000.0, 0.0, slowdown)
    assert (work_ms <= fast) == (work_ms * slowdown <= slow)


def test_a_fired_wall_is_retried_once_then_undetermined():
    assert liveness_action(False, True, 1) == "accept"
    assert liveness_action(False, False, 2) == "accept"
    assert liveness_action(True, True, 1) == "retry"
    assert liveness_action(True, True, 2) == "undetermined"
    # A runaway that could not be stopped is not run again beside itself.
    assert liveness_action(True, False, 1) == "undetermined"


def test_the_compute_budget_setting():
    assert parse_compute_budget(None) == (LIVENESS_BUDGET_S, "default")
    assert parse_compute_budget("  ") == (LIVENESS_BUDGET_S, "default")
    assert parse_compute_budget("25") == (25.0, "set")
    assert parse_compute_budget("0.5") == (0.5, "set")
    for bad in ("fast", "0", "-3", "inf", "nan", "1e999"):
        assert parse_compute_budget(bad) == (LIVENESS_BUDGET_S, "invalid"), bad


def test_a_blank_setting_is_the_default_and_not_an_invalid_one():
    # Alone on purpose: a mutant that crashes on None fails `test_the_compute_budget_setting` at its
    # first line, so the blank-string assertion there never runs under it and pins nothing.
    assert parse_compute_budget("  ") == (LIVENESS_BUDGET_S, "default")


def test_a_number_is_a_set_budget_and_not_the_default():
    assert parse_compute_budget("25") == (25.0, "set")


def test_an_invalid_budget_is_refused_not_ignored(monkeypatch):
    monkeypatch.setenv(COMPUTE_BUDGET_ENV, "plenty")
    with pytest.raises(ValueError, match=COMPUTE_BUDGET_ENV):
        compute_budget_s()
    monkeypatch.setenv(COMPUTE_BUDGET_ENV, "42")
    assert compute_budget_s() == 42.0
    monkeypatch.delenv(COMPUTE_BUDGET_ENV)
    assert compute_budget_s() == LIVENESS_BUDGET_S


# ── the impure edge reads the calibration ─────────────────────────────────────────


def test_the_isolated_wall_is_the_budget_times_the_calibrated_slowdown(monkeypatch):
    monkeypatch.delenv(COMPUTE_BUDGET_ENV, raising=False)
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (1.0, "reference")
    )
    assert _isolated_wall_s(500.0, 0.0) == 10.0
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (4.0, "slower")
    )
    assert _isolated_wall_s(500.0, 0.0) == 40.0
    monkeypatch.setenv(COMPUTE_BUDGET_ENV, "20")
    assert _isolated_wall_s(500.0, 0.0) == 80.0


def test_the_calibration_is_taken_once_and_a_refresh_keeps_the_slower_reading(
    monkeypatch,
):
    readings = iter([100_000_000, 400_000_000, 150_000_000])
    monkeypatch.setattr(monitoring, "_harness_ns", None)
    monkeypatch.setattr(
        monitoring, "measure_harness_ns", lambda repeats=5: next(readings)
    )
    first = monitoring.machine_slowdown()
    again = monitoring.machine_slowdown()
    assert first == again, "a second call must not measure again"
    slower = monitoring.machine_slowdown(refresh=True)
    assert slower[0] > first[0]
    # A later refresh that reads faster does not shrink the bound back.
    assert monitoring.machine_slowdown(refresh=True) == slower


# ── end to end through a real isolated worker ─────────────────────────────────────

_TARGET = """
import os
import sys
import time


def helper():
    if "_isolated_worker" in sys.argv[0]:
        if os.environ.get("LIVENESS_STALL"):
            time.sleep(float(os.environ.get("LIVENESS_STALL_S", "5")))


def add_one(x):
    helper()
    return x + 1
"""


def _project(tmp_path):
    (tmp_path / "m.py").write_text(textwrap.dedent(_TARGET))
    (tmp_path / "test_m.py").write_text(
        "from m import add_one\n\n\ndef test_adds():\n    assert add_one(1) == 2\n"
    )
    sys.path.insert(0, str(tmp_path))
    m = importlib.import_module("m")
    tm = importlib.import_module("test_m")
    tm.test_adds.__qualname__ = "test_m.py::test_adds"
    node = next(
        n
        for n in ast.parse((tmp_path / "m.py").read_text()).body
        if getattr(n, "name", "") == "add_one"
    )
    return node, m.add_one, [tm.test_adds]


def _drop(tmp_path):
    sys.path[:] = [p for p in sys.path if p != str(tmp_path)]
    sys.modules.pop("m", None)
    sys.modules.pop("test_m", None)


def _profile(tmp_path, monkeypatch, env, slowdown):
    from Wesker.ci import _PROJECT_ROOT
    from Wesker.engine import MutationCategory, run_function_profiling

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (slowdown, "slower")
    )
    node, func, tests = _project(tmp_path)
    try:
        token = _PROJECT_ROOT.set(str(tmp_path))
        try:
            return run_function_profiling(
                node,
                "m.py::add_one",
                {MutationCategory.ARITHMETIC},
                tests,
                func,
                isolated=True,
            )
        finally:
            _PROJECT_ROOT.reset(token)
    finally:
        _drop(tmp_path)


def _dispositions(result):
    return sorted({r["disposition"] for r in result.unscored_records})


def test_work_beyond_the_bound_is_undetermined_after_one_retry(tmp_path, monkeypatch):
    env = {
        COMPUTE_BUDGET_ENV: "3",
        "LIVENESS_STALL": "1",
        "LIVENESS_STALL_S": "6",
    }
    result = _profile(tmp_path, monkeypatch, env, 1.0)
    assert result.unscored_records, "every mutant ran the stalled helper"
    assert _dispositions(result) == ["undetermined"]


def test_the_same_work_on_a_machine_four_times_slower_is_not_undetermined(
    tmp_path, monkeypatch
):
    # The helper's 6 s is "4x slower machine" work against a budget of 3 reference seconds: the
    # bound, 12 s, is stretched in the same proportion, so the verdict is the fast machine's.
    env = {
        COMPUTE_BUDGET_ENV: "3",
        "LIVENESS_STALL": "1",
        "LIVENESS_STALL_S": "6",
    }
    result = _profile(tmp_path, monkeypatch, env, 4.0)
    assert "undetermined" not in _dispositions(result)


def test_a_transient_stall_is_cured_by_the_one_retry(tmp_path, monkeypatch):
    # The stall ends with the first fired wall: the wrapper removes the environment the NEXT worker
    # would inherit, as a machine that was busy for a moment and is not any more.
    import os

    from Wesker.isolation import IsolatedMutantWorker

    real = IsolatedMutantWorker.evaluate
    fired: list[bool] = []

    def evaluate(self, *args, **kwargs):
        run = real(self, *args, **kwargs)
        fired.append(run.timed_out)
        if run.timed_out:
            os.environ.pop("LIVENESS_STALL", None)
        return run

    monkeypatch.setattr(IsolatedMutantWorker, "evaluate", evaluate)
    env = {COMPUTE_BUDGET_ENV: "3", "LIVENESS_STALL": "1", "LIVENESS_STALL_S": "6"}
    try:
        result = _profile(tmp_path, monkeypatch, env, 1.0)
    finally:
        os.environ.pop("LIVENESS_STALL", None)
    assert fired.count(True) == 1, "exactly one wall fired, and it was retried"
    assert fired.index(True) + 1 < len(fired), "a run followed the fired wall"
    assert "undetermined" not in _dispositions(result), "the retry ran clean"
