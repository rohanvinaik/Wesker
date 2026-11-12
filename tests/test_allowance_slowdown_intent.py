"""Intent: Wesker #27, plan item 4 — the unarmed allowance follows the machine too.

The defect: "Unarmed observations still share the #13 `_adaptive_allowance`: `max(50 ms, 50 x the
original's live baseline)`, capped." The multiplier rests on a baseline measured live, so it is
already machine-relative. The 50 ms floor and the per-mutant cap are not: they are reference-machine
milliseconds, so a slow machine met them sooner in proportion to its work and filed a terminating
mutant `undetermined` where a fast one filed a verdict. The allowance is NOT re-derived from the
liveness bound itself: its job (#13) is to cut a runaway in milliseconds, and the bound is ten
reference seconds; making the one the other would charge every runaway the full bound. Its two
reference-time terms are instead stretched by the same calibrated slowdown the bound uses.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from Wesker import engine, monitoring
from Wesker.engine import (
    MutationCategory,
    _adaptive_allowance,
    _machine_allowance,
    _original_count_wall_ms,
    _unbounded_ceiling_ms,
)


def test_without_a_slowdown_the_allowance_is_what_it_always_was():
    assert _adaptive_allowance(None, 500.0, 10_000.0) == 500.0
    assert _adaptive_allowance(0.0, 500.0, 10_000.0) == 50.0
    assert _adaptive_allowance(4.0, 500.0, 10_000.0) == 200.0
    assert _adaptive_allowance(4.0, 500.0, 120.0) == 120.0


def test_the_floor_and_the_cap_stretch_and_the_live_baseline_does_not():
    # floor 50 ms -> 200 ms on a machine four times slower
    assert _adaptive_allowance(0.0, 500.0, 100_000.0, 4.0) == 200.0
    # cap 500 ms -> 2000 ms
    assert _adaptive_allowance(None, 500.0, 100_000.0, 4.0) == 2000.0
    # a baseline measured on THIS machine is multiplied, not stretched again
    assert _adaptive_allowance(10.0, 500.0, 100_000.0, 4.0) == 500.0
    assert _adaptive_allowance(100.0, 500.0, 100_000.0, 4.0) == 2000.0


def test_a_real_deadline_is_never_stretched():
    assert _adaptive_allowance(None, 500.0, 300.0, 4.0) == 300.0
    assert _adaptive_allowance(1000.0, 500.0, 0.0, 4.0) == 0.0


def test_a_factor_below_one_is_read_as_one():
    assert _adaptive_allowance(None, 500.0, 10_000.0, 0.1) == 500.0
    assert _adaptive_allowance(None, 500.0, 10_000.0, -3.0) == 500.0


@pytest.mark.parametrize("slowdown", [1.0, 2.5, 4.0, 10.0])
@pytest.mark.parametrize("work_ms", [20.0, 40.0, 150.0, 400.0, 3000.0])
def test_a_mutant_that_fits_on_a_fast_machine_fits_on_a_slow_one(work_ms, slowdown):
    # No baseline (the case the floor and the cap decide). The mutant needs `work_ms` on the
    # reference machine and `work_ms * slowdown` on one that much slower.
    fast = _adaptive_allowance(None, 500.0, 1e12, 1.0)
    slow = _adaptive_allowance(None, 500.0, 1e12, slowdown)
    assert (work_ms <= fast) == (work_ms * slowdown <= slow)


def test_the_engines_allowance_reads_the_calibration(monkeypatch):
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (4.0, "slower")
    )
    assert _machine_allowance(None, 500.0, 1e9) == 2000.0
    assert _unbounded_ceiling_ms(500.0) == 2000.0
    assert _original_count_wall_ms(500.0) == 5000.0 * 4.0
    assert _original_count_wall_ms(60_000.0) == 60_000.0 * 4.0
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (1.0, "reference")
    )
    assert _machine_allowance(None, 500.0, 1e9) == 500.0
    assert _unbounded_ceiling_ms(500.0) == 500.0
    assert _original_count_wall_ms(500.0) == 5000.0


_SRC = """
def add_one(x):
    return x + 1
"""


def _profile_allowance_calls(monkeypatch, slowdown):
    """Every (cap, remaining, slowdown) the allowance was asked for during a real in-process profile."""
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (slowdown, "slower")
    )
    node = ast.parse(textwrap.dedent(_SRC)).body[0]
    ns: dict = {}
    # S102: this test compiles its own fixture source
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<fx>", "exec"), ns)  # noqa: S102
    func = ns["add_one"]

    def test_adds(f=func):
        assert f(1) == 2

    calls: list[tuple[float, float, float]] = []
    real = engine._adaptive_allowance

    def spy(baseline_ms, cap_ms, remaining_ms, slowdown=1.0):
        calls.append((cap_ms, remaining_ms, slowdown))
        return real(baseline_ms, cap_ms, remaining_ms, slowdown)

    monkeypatch.setattr(engine, "_adaptive_allowance", spy)
    engine.run_function_profiling(
        node,
        "fx.py::add_one",
        {MutationCategory.ARITHMETIC},
        [test_adds],
        func,
        per_mutant_timeout_ms=100.0,
    )
    return calls


def test_a_profile_asks_for_the_allowance_under_the_calibrated_slowdown(monkeypatch):
    for slowdown in (1.0, 4.0):
        calls = _profile_allowance_calls(monkeypatch, slowdown)
        assert calls, "the profile evaluated mutants"
        for cap_ms, remaining_ms, factor in calls:
            assert factor == slowdown
            # no aggregate deadline was set, so the ceiling is the cap, stretched the same way
            assert remaining_ms == cap_ms * slowdown
