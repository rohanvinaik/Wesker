"""Intent: Wesker #27 — a mutant that finishes inside an ARMED loop budget and passes is a survivor.

The acceptance asks for an intent test per outcome (diverged / undetermined / killed / survived).
The first three were covered; none asserted a scored `survived` with the budget armed. The danger it
guards is the budget's own: an armed observation runs under instrumentation that can stop a run, and
a stop is `diverged`. A mutant that is NOT stopped, because it terminates inside its cap, must keep
reading as what it is: a survivor the suite failed to catch, scored in the denominator, with the
budget having stopped nothing.
"""

from __future__ import annotations

import ast
import textwrap

from Wesker import engine
from Wesker.engine import (
    SCORED_DISPOSITIONS,
    Mutant,
    MutationCategory,
    StepPlan,
    evaluate_mutant,
    mutant_disposition,
)
from Wesker.monitoring import STEP_BUDGET_FACTOR, STEP_BUDGET_FLOOR

_ADVANCE = """
def advance(flags, start):
    i = max(start, 0)
    while i < len(flags) and flags[i]:
        i += 1
    return i
"""


def _compile(src: str):
    node = ast.parse(textwrap.dedent(src)).body[0]
    ns: dict = {}
    # S102: this test compiles its own fixture source
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<fx>", "exec"), ns)  # noqa: S102
    return node, ns[node.name]


def _hand_mutant(old: str, new: str) -> Mutant:
    onode = ast.parse(textwrap.dedent(_ADVANCE)).body[0]
    mnode = ast.parse(textwrap.dedent(_ADVANCE).replace(old, new)).body[0]
    return Mutant(MutationCategory.ARITHMETIC, onode, mnode, f"{old} -> {new}")


def _evaluate_armed(mutant, original, tests, monkeypatch):
    caps: list[int | None] = []
    real = engine._run_test_with_timeout

    def spy(*args, **kwargs):
        caps.append(kwargs.get("step_cap"))
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_run_test_with_timeout", spy)
    plan = StepPlan(STEP_BUDGET_FACTOR, STEP_BUDGET_FLOOR)
    res = evaluate_mutant(
        mutant,
        tests,
        original,
        timeout_ms=30_000,
        qualname="advance",
        step_plan=plan,
    )
    return res, plan, caps


def test_a_mutant_that_terminates_inside_its_cap_and_passes_is_a_scored_survivor(
    monkeypatch,
):
    # `max(start, 1)` starts the scan one step in; from 0 over [True, True, False] it still returns 2.
    _node, original = _compile(_ADVANCE)
    mutant = _hand_mutant("max(start, 0)", "max(start, 1)")

    def test_weak(f=original):
        assert f([True, True, False], 0) == 2

    res, plan, caps = _evaluate_armed(mutant, original, [test_weak], monkeypatch)

    state, count = next(iter(plan.originals.values()))
    assert state == "passed", "the original passed under the count: the budget is armed"
    armed = [c for c in caps if c is not None]
    assert armed, "the mutant's observation ran under a loop budget"
    assert armed[-1] == count * STEP_BUDGET_FACTOR + STEP_BUDGET_FLOOR

    assert (res.killed, res.killed_by) == (False, None)
    assert res.tests_run == 1 and res.entered is True
    disposition = mutant_disposition(
        res.constructed, res.installed, res.entered, res.contained, res.killed
    )
    assert disposition == "survived_after_entry"
    assert disposition in SCORED_DISPOSITIONS


def test_the_same_budget_still_does_not_hide_a_kill_or_a_runaway(monkeypatch):
    # The control: with the budget armed the same mutant spelling that DISAGREES is a kill, and one
    # that never ends is `diverged`: three outcomes, three different dispositions.
    _node, original = _compile(_ADVANCE)

    def test_pins_the_start(f=original):
        assert f([False], 0) == 0

    killed, _plan, _ = _evaluate_armed(
        _hand_mutant("max(start, 0)", "max(start, 1)"),
        original,
        [test_pins_the_start],
        monkeypatch,
    )
    assert (killed.killed, killed.killed_by) == (True, "assertion")

    def test_runs_the_loop(f=original):
        assert f([True, True, False], 0) == 2

    diverged, _plan, _ = _evaluate_armed(
        _hand_mutant("i += 1", "i *= 2"), original, [test_runs_the_loop], monkeypatch
    )
    assert (diverged.killed, diverged.killed_by) == (False, "diverged")
