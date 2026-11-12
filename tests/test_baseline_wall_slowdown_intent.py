"""Intent: Wesker #27 — the baseline's per-test wall follows the machine, like every wall that can decide.

Reproduced on a machine slowed by a competing load: a test whose helper loop the step budget cannot
count (250M iterations, about 4 s here) passed on the idle machine and hit the fixed 5 s per-test
baseline wall on the loaded one, so converge ended `FAILING NOW — tests/test_steps250.py::test_adds_one
reaches this function and fails on current code; no test written; certificate withheld`. That is a
verdict about the suite that was really a verdict about the machine, from the one wall the liveness
bound did not yet reach.
"""

from __future__ import annotations

import ast
import textwrap

from Wesker import engine, monitoring
from Wesker.engine import _baseline_failures, _stretched_wall_ms

_SRC = "def f(x):\n    return x + 1\n"


def _walls(monkeypatch, slowdown):
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (slowdown, "slower")
    )
    node = ast.parse(textwrap.dedent(_SRC)).body[0]
    ns: dict = {}
    # S102: this test compiles its own fixture source
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<fx>", "exec"), ns)  # noqa: S102
    func = ns["f"]

    def test_f(g=func):
        assert g(1) == 2

    seen: list[float] = []
    real = engine._run_test_with_timeout

    def spy(test_fn, mutated, patched, timeout_ms, **kwargs):
        seen.append(timeout_ms)
        return real(test_fn, mutated, patched, timeout_ms, **kwargs)

    monkeypatch.setattr(engine, "_run_test_with_timeout", spy)
    _baseline_failures([test_f], func, "f", timeout_ms=5000)
    return seen


def test_the_stretch_is_the_calibrated_slowdown_and_never_a_shrink(monkeypatch):
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (4.0, "slower")
    )
    assert _stretched_wall_ms(5000.0) == 20_000.0
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (1.0, "reference")
    )
    assert _stretched_wall_ms(5000.0) == 5000.0
    monkeypatch.setattr(
        monitoring, "machine_slowdown", lambda refresh=False: (0.2, "reference")
    )
    assert _stretched_wall_ms(5000.0) == 5000.0


def test_the_baseline_runs_each_test_under_the_stretched_wall(monkeypatch):
    assert _walls(monkeypatch, 1.0) == [5000.0]
    assert _walls(monkeypatch, 4.0) == [20_000.0]
