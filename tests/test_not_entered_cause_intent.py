"""Intent pins: a mutant that was installed but never entered says WHY, by name (#18).

THE DEFECT. `not_entered` was reported with ONE cause whatever the real one was — Detective rendered
it as "the namespace holds the mutant while the caller holds the original (the decorator/registry
capture); route a test through the patched name, or pin the caller". Field evidence (Detective #77):
five instances in one project, two of them plain functions with no decorator or registry at all, and
another on 2026-09-27 on `Wesker/monitoring.py::step_budget_verdict` — an undecorated function whose
mutant ran against a pool of tests that never reached it before its allowance ran out. The message
named a mechanism that was absent, the prescribed repair left the verdict unchanged, and the operator
had no next step. Wesker's own certificates carry three more (`describe_truncation`, `gate_truncation`,
`_describe_containment_lost`), "fixed" by rewriting imports when the mechanism was the pool.

WHAT IS DECIDABLE, and how each is decided:

  not_reached          -- the tests that RAN against the mutant were each traced on the baseline and
                          none executed a line of the function: nothing called it by any route.
  decorator_wrapper    -- a callable bound in a module wraps the original (`__wrapped__`, or a closure
                          over it) and the install, which rebinds by NAME and by code file, cannot
                          rebind it: callers through the wrapper run the original.
  captured_at_import   -- a module-level constant or registry holds the original (a container element,
                          an object's field — `Lens(..., vote=fn)` — a default argument, a partial).
  pre_bound_reference  -- a module binds the original under ANOTHER name (`from m import fn as alias`).
  holder_not_found     -- the tests reached the function, yet no module-level holder was found: the
                          reference lives where the bounded scan does not look.
  no_reach_data        -- no holder, and no complete trace to say whether the tests reached it.

These are written from what the engine must report, one per cause; the generated suites beside them
pin the two pure decisions' current answers.
"""

from __future__ import annotations

import ast
import importlib
import sys
import threading

import pytest

from Wesker.engine import (
    _original_holders,
    baseline_reach,
    not_entered_cause,
    run_function_converged,
    run_function_profiling,
)
from Wesker.filter import filter_categories

_TARGET = """\
import dataclasses

from nedeco import wrap


def direct(x):
    return x * 2


def aliased(x):
    return x + 3


def captured(x):
    return x - 1


@dataclasses.dataclass(frozen=True)
class Lens:
    name: str
    vote: object


LENS = Lens("one", captured)
MEMBERS = {"m": Lens("two", captured)}


@wrap
def decorated(x):
    return x * 5


def deep(x):
    return x * 7


DEEP = {"a": {"b": {"c": deep}}}
"""

_DECO = """\
import functools


def wrap(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner
"""

_HELPER = """\
from nemod import aliased as compute
from nemod import direct


def run(x):
    return compute(x)


def run_direct(x):
    return direct(x)
"""

_MODULES = ("nemod", "nedeco", "nehelper")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Three real modules on disk, imported under their own names, as a consumer repo has them."""
    (tmp_path / "nemod.py").write_text(_TARGET)
    (tmp_path / "nedeco.py").write_text(_DECO)
    (tmp_path / "nehelper.py").write_text(_HELPER)
    for name in _MODULES:
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(tmp_path))
    mods = {name: importlib.import_module(name) for name in _MODULES}
    try:
        yield mods, str(tmp_path / "nemod.py")
    finally:
        for name in _MODULES:
            sys.modules.pop(name, None)


def _node(name: str) -> ast.FunctionDef:
    for node in ast.parse(_TARGET).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(name)


def _profile(name, original, tests, runner=run_function_profiling, **kw):
    node = _node(name)
    extra = {"max_per_category": 0} if runner is run_function_profiling else {}
    return runner(
        node,
        f"nemod.py::{name}",
        filter_categories(node, True),
        tests,
        original,
        **extra,
        **kw,
    )


def _not_entered(res) -> list[dict]:
    records = [r for r in res.unscored_records if r["disposition"] == "not_entered"]
    assert records, (
        f"no mutant was not_entered — the fixture did not exercise #18: {res.to_dict()}"
    )
    return records


def _causes(res) -> set[str]:
    return {r["cause"] for r in _not_entered(res)}


# ── one cause each, through the real profiling path ──────────────────────────────────────────


@pytest.mark.parametrize("runner", [run_function_profiling, run_function_converged])
def test_a_function_no_test_reaches_is_named_not_reached(project, runner):
    """The step_budget_verdict shape: the tests that ran never touch the function at all. Nothing
    holds a stale reference — there is nothing to repair in the imports; route a test to it."""
    mods, _path = project

    def test_elsewhere():
        assert mods["nemod"].captured(5) == 4

    kw = {"scope_tests": False}
    res = _profile("direct", mods["nemod"].direct, [test_elsewhere], runner, **kw)
    assert _causes(res) == {"not_reached"}
    assert all(r["reach"] == "unreached" for r in _not_entered(res))
    assert res.not_entered_causes == {"not_reached": len(_not_entered(res))}


def test_an_alias_in_a_module_the_test_calls_through_is_a_pre_bound_reference(project):
    """`from nemod import aliased as compute` in the module the test calls through: the install
    rebinds `aliased` everywhere it is bound UNDER THAT NAME, and never `compute`."""
    mods, _path = project

    def test_through_helper():
        assert mods["nehelper"].run(1) == 4

    res = _profile("aliased", mods["nemod"].aliased, [test_through_helper])
    assert _causes(res) == {"pre_bound_reference"}
    record = _not_entered(res)[0]
    assert record["reach"] == "reached"
    assert {"kind": "alias", "where": "nehelper.compute"} in record["holders"]


def test_a_module_level_constant_holding_the_original_is_captured_at_import(project):
    """The Detective #77 `Lens(..., vote=fn)` shape — one level down, and two (a dict of them)."""
    mods, _path = project

    def test_through_the_lens():
        assert mods["nemod"].LENS.vote(5) == 4

    res = _profile("captured", mods["nemod"].captured, [test_through_the_lens])
    assert _causes(res) == {"captured_at_import"}
    wheres = {h["where"] for h in _not_entered(res)[0]["holders"]}
    assert "nemod.LENS.vote" in wheres
    assert "nemod.MEMBERS['m'].vote" in wheres


def test_a_wrapper_from_another_module_is_a_decorator_wrapper(project):
    """`nemod.decorated` IS the wrapper (its code lives in nedeco.py), so the install's code-file
    match skips it; callers through the module attribute run the original inside the wrapper."""
    mods, _path = project
    wrapper = mods["nemod"].decorated
    original = wrapper.__wrapped__

    def test_through_the_wrapper():
        assert mods["nemod"].decorated(2) == 10

    res = _profile("decorated", original, [test_through_the_wrapper])
    assert _causes(res) == {"decorator_wrapper"}
    assert {"kind": "wrapper", "where": "nemod.decorated"} in _not_entered(res)[0][
        "holders"
    ]


def test_a_holder_beyond_the_scan_is_named_as_not_found_not_guessed(project):
    """Reached, yet no module-level holder within the scan's depth: say so, rather than borrow a
    mechanism the evidence does not show."""
    mods, _path = project

    def test_deep():
        assert mods["nemod"].DEEP["a"]["b"]["c"](1) == 7

    res = _profile("deep", mods["nemod"].deep, [test_deep])
    assert _causes(res) == {"holder_not_found"}
    assert _not_entered(res)[0]["holders"] == []


def test_without_reach_data_the_cause_is_not_invented(project):
    """No trace (the caller supplied empty line data) and no holder: `no_reach_data`, never
    `not_reached` — absence of a line nobody recorded is not evidence."""
    mods, _path = project

    def test_elsewhere():
        assert mods["nemod"].captured(5) == 4

    res = _profile(
        "direct",
        mods["nemod"].direct,
        [test_elsewhere],
        precomputed_line_data=({}, []),
    )
    assert _causes(res) == {"no_reach_data"}
    assert all(r["reach"] == "unknown" for r in _not_entered(res))


def test_the_cause_is_carried_in_the_payload_detective_reads(project):
    mods, _path = project

    def test_through_helper():
        assert mods["nehelper"].run(1) == 4

    payload = _profile(
        "aliased", mods["nemod"].aliased, [test_through_helper]
    ).to_dict()
    assert payload["not_entered_causes"] == {
        "pre_bound_reference": payload["unscored_by"]["not_entered"]
    }
    assert all(
        r["cause"] == "pre_bound_reference"
        for r in payload["unscored_records"]
        if r["disposition"] == "not_entered"
    )


def test_a_same_name_import_in_a_module_the_test_calls_through_is_entered(project):
    """The control, and the reason the #77 heuristic ("the test imported the function by name")
    names the wrong mechanism: `nehelper` does `from nemod import direct`, and that binding IS
    rebound by the module-qualified patch, so a test calling through it enters every mutant."""
    mods, _path = project

    def test_through_same_name_import():
        assert mods["nehelper"].run_direct(4) == 8

    res = _profile("direct", mods["nemod"].direct, [test_through_same_name_import])
    assert not [r for r in res.unscored_records if r["disposition"] == "not_entered"]
    assert res.total_killed >= 1


def test_the_cause_is_about_the_tests_that_ran_not_the_whole_scope(project):
    """The `step_budget_verdict` shape: the allowance ends the scan on an unrelated test before the
    test that reaches the function gets its turn. Judged over the whole scope the reacher would say
    "reached" and send the operator hunting for a holder that does not exist; over the tests that
    RAN, nothing reached it — and `ended_by` says why the scan stopped."""
    from Wesker.ci import callable_test_id
    from Wesker.engine import _not_entered_detail, evaluate_mutant, generate_mutants

    mods, path = project
    stop = threading.Event()

    def test_spins_elsewhere():
        while not stop.is_set():  # pure Python: the allowance's abandon lands here
            pass

    def test_reaches_it():
        assert mods["nemod"].direct(4) == 8

    node = _node("direct")
    mutant = generate_mutants(
        node, filter_categories(node, True), max_per_category=0, pass_index=0
    )[0]
    scoped = [test_spins_elsewhere, test_reaches_it]
    try:
        res = evaluate_mutant(
            mutant,
            scoped,
            mods["nemod"].direct,
            timeout_ms=200,
            qualname="direct",
            source_path=path,
        )
    finally:
        stop.set()
    assert res.entered is False and res.killed_by == "timeout"
    assert res.tests_run == 1, "only the spinning test started"
    coverage = {
        callable_test_id(test_spins_elsewhere): [],
        callable_test_id(test_reaches_it): [8],
    }
    detail = _not_entered_detail(res, scoped, coverage, set(), [], "complete")
    assert detail["cause"] == "not_reached"
    assert detail["reach"] == "unreached"
    assert detail["ended_by"] == "timeout"


# ── the holder scan: what the install DOES rebind is never a holder ──────────────────────────


def test_bindings_the_install_rebinds_are_not_holders(project):
    """`nemod.direct` (the definition) and `nehelper.direct` (a same-name import) are both rebound
    by the module-qualified patch, so naming either as a holder would send the operator after a
    reference that was in fact replaced."""
    mods, path = project
    assert _original_holders(mods["nemod"].direct, "direct", "direct", path) == (
        [],
        "complete",
    )


def test_the_scan_names_each_holder_where_it_lives(project):
    mods, path = project
    assert _original_holders(mods["nemod"].aliased, "aliased", "aliased", path) == (
        [{"kind": "alias", "where": "nehelper.compute"}],
        "complete",
    )


_SHAPES = """\
import functools


def target(x):
    return x + 1


def _deco_without_wraps(f):
    def inner(*args):
        return f(*args)

    return inner


closed_over = _deco_without_wraps(target)


def with_default(x, fn=target):
    return fn(x)


bound_partial = functools.partial(target, 1)


class Holder:
    handler = target
    static = staticmethod(target)


class Owner:
    def meth(self):
        return 1
"""


@pytest.fixture
def shapes(tmp_path, monkeypatch):
    (tmp_path / "neshapes.py").write_text(_SHAPES)
    sys.modules.pop("neshapes", None)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield importlib.import_module("neshapes"), str(tmp_path / "neshapes.py")
    finally:
        sys.modules.pop("neshapes", None)


def test_each_reference_shape_is_named_where_it_lives(shapes):
    """A decorator written without `functools.wraps` (a closure), a default bound at definition, a
    partial, and class attributes — each holds the original where the install, which rebinds module
    attributes by name, never looks."""
    mod, path = shapes
    holders, scan = _original_holders(mod.target, "target", "target", path)
    assert scan == "complete"
    assert sorted((h["where"], h["kind"]) for h in holders) == [
        ("neshapes.Holder.handler", "captured"),
        ("neshapes.Holder.static", "captured"),
        ("neshapes.bound_partial", "captured"),
        ("neshapes.closed_over", "wrapper"),
        ("neshapes.with_default", "captured"),
    ]


def test_a_method_on_its_own_class_is_rebound_not_held(shapes):
    """For a method target the owner class's attribute IS rebound by the install's owner patch, so
    it must not be named as a holder of the original."""
    mod, path = shapes
    assert _original_holders(mod.Owner.meth, "meth", "Owner.meth", path) == (
        [],
        "complete",
    )


def test_the_scan_unwraps_the_wrapper_a_caller_may_hand_it(project):
    """Detective loads a target by module attribute, i.e. the decorator's wrapper; the references
    that matter are to the function the mutant replaces, and the answer must not depend on which
    of the two the caller held."""
    mods, path = project
    wrapper = mods["nemod"].decorated
    expected = ([{"kind": "wrapper", "where": "nemod.decorated"}], "complete")
    assert _original_holders(wrapper, "decorated", "decorated", path) == expected
    assert (
        _original_holders(wrapper.__wrapped__, "decorated", "decorated", path)
        == expected
    )


def test_the_scan_reads_the_project_before_its_libraries(tmp_path):
    """Under a budget, order decides what is seen: a project module that imports a library inserts
    the library's modules AFTER itself, so newest-first alone would read them first."""
    import types

    from Wesker.engine import _holder_scan_order

    def module(name, path):
        mod = types.ModuleType(name)
        mod.__file__ = path
        return mod

    home = module("proj.target", str(tmp_path / "proj" / "target.py"))
    project = module("proj.caller", str(tmp_path / "proj" / "caller.py"))
    library = module("lib.sub", "/elsewhere/site-packages/lib/sub.py")
    builtin = types.ModuleType("builtinish")  # no __file__
    order = _holder_scan_order(
        home, [home, project, library, builtin, None], str(tmp_path / "proj")
    )
    assert order == [home, project, builtin, library]


def test_a_record_says_whether_the_holder_scan_was_complete(project):
    mods, _path = project

    def test_deep():
        assert mods["nemod"].DEEP["a"]["b"]["c"](1) == 7

    record = _not_entered(_profile("deep", mods["nemod"].deep, [test_deep]))[0]
    assert record["holder_scan"] == "complete", (
        "a holder_not_found must say whether the scan that found nothing was whole"
    )


# ── the pure decisions (#18, pinned) ─────────────────────────────────────────────────────────


def test_any_ran_test_reaching_the_function_is_reach():
    assert baseline_reach(["a", "b"], {"a": [], "b": [3]}, []) == "reached"
    assert baseline_reach(["b"], {"b": [3]}, ["b"]) == "reached", (
        "a cut trace's lines are real lines"
    )


def test_unreached_needs_every_ran_test_traced_in_full():
    assert baseline_reach(["a", "b"], {"a": [], "b": []}, []) == "unreached"
    assert baseline_reach(["a", "b"], {"a": []}, []) == "unknown", "b was never traced"
    assert baseline_reach(["a"], {"a": []}, ["a"]) == "unknown", (
        "a's trace was cut short"
    )


def test_no_ran_tests_is_no_evidence():
    assert baseline_reach([], {"a": [3]}, []) == "unknown"


def test_unreached_outranks_every_holder():
    """If no test reached the function, no holder explains anything: nothing went through it."""
    for holders in ([], ["wrapper"], ["captured"], ["alias"]):
        assert not_entered_cause("unreached", holders) == "not_reached"


def test_holder_precedence_is_wrapper_then_captured_then_alias():
    assert not_entered_cause("reached", ["alias", "captured", "wrapper"]) == (
        "decorator_wrapper"
    )
    assert not_entered_cause("reached", ["alias", "captured"]) == "captured_at_import"
    assert not_entered_cause("reached", ["alias"]) == "pre_bound_reference"


def test_a_holder_is_named_even_without_reach_data():
    """A holder is a fact about the process whatever the trace says."""
    assert not_entered_cause("unknown", ["captured"]) == "captured_at_import"


def test_no_holder_splits_on_whether_reach_was_observed():
    assert not_entered_cause("reached", []) == "holder_not_found"
    assert not_entered_cause("unknown", []) == "no_reach_data"


def test_every_cause_is_reachable_and_distinct():
    seen = {
        not_entered_cause("unreached", []),
        not_entered_cause("reached", ["wrapper"]),
        not_entered_cause("reached", ["captured"]),
        not_entered_cause("reached", ["alias"]),
        not_entered_cause("reached", []),
        not_entered_cause("unknown", []),
    }
    assert seen == {
        "not_reached",
        "decorator_wrapper",
        "captured_at_import",
        "pre_bound_reference",
        "holder_not_found",
        "no_reach_data",
    }
