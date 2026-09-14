"""Intent tests: a worker that could not be stopped is NAMED — the test, and the mutant it was running.

THE GAP, from Wesker's own Specification (full) run on 21a69de. The refusal named five cut functions,
all cut because "a timed-out test could not be stopped", and told the maintainer to "bound the
blocking call, or run that test in a killable process". It could not say which test. The engine knew:
`evaluate_mutant` had the test in hand when `_run_test_with_timeout` returned "uncontained", and the
baseline kept the names of tests it could not stop. Both were reduced to one bool, `all_contained`,
before the result was built, so the report could refuse but not direct.

What is pinned here, from intent:
- `evaluate_mutant` records the first test whose worker could not be stopped, in the kill vocabulary;
- the isolated path names it too, and only when the run was uncontained;
- both profiling paths carry `containment_lost`: baseline names first, then each uncontained mutant
  with its test, and `to_dict` emits it;
- a contained profile carries nothing.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap

import pytest

from Wesker import engine as E
from Wesker.ci import callable_test_id
from Wesker.engine import (
    MutantResult,
    MutationCategory,
    evaluate_mutant,
    generate_mutants,
    run_function_converged,
    run_function_profiling,
)
from Wesker.filter import filter_categories
from Wesker.isolation import IsolatedRun

_SRC = "def add(n):\n    return n + 1\n"


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


@pytest.fixture
def target(tmp_path):
    """A real on-disk module, so the module-qualified patch has a path to match."""
    path = tmp_path / "blocked_add_mod.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("blocked_add_mod", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["blocked_add_mod"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod, str(path)
    finally:
        sys.modules.pop("blocked_add_mod", None)


def test_evaluate_mutant_names_the_test_whose_worker_could_not_be_stopped(
    target, monkeypatch
):
    mod, path = target

    def test_blocks() -> None:
        assert mod.add(2) == 3

    node = _fn(_SRC)
    mutant = generate_mutants(node, filter_categories(node, True), max_per_category=0)[
        0
    ]
    monkeypatch.setattr(E, "_run_test_with_timeout", lambda *a, **k: "uncontained")

    result = evaluate_mutant(
        mutant, [test_blocks], mod.add, qualname="add", source_path=path
    )

    assert result.contained is False
    assert result.uncontained_test == callable_test_id(test_blocks)


def test_a_contained_evaluation_names_no_test(target, monkeypatch):
    mod, path = target

    def test_times_out() -> None:
        assert mod.add(2) == 3

    node = _fn(_SRC)
    mutant = generate_mutants(node, filter_categories(node, True), max_per_category=0)[
        0
    ]
    monkeypatch.setattr(E, "_run_test_with_timeout", lambda *a, **k: "timeout")

    result = evaluate_mutant(
        mutant, [test_times_out], mod.add, qualname="add", source_path=path
    )

    assert result.contained is True
    assert result.uncontained_test is None


def _a_mutant():
    node = _fn("def f(a, b):\n    return a < b\n")
    return generate_mutants(node, {MutationCategory.BOUNDARY}, max_per_category=0)[0]


def test_the_isolated_path_names_the_test_only_when_uncontained():
    mutant = _a_mutant()
    stuck = IsolatedRun(
        returncode=-9,
        timed_out=True,
        contained=False,
        stdout="",
        test_name="t.py::test_wedged",
    )
    reaped = IsolatedRun(
        returncode=-9,
        timed_out=True,
        contained=True,
        stdout="",
        test_name="t.py::test_slow",
    )

    assert (
        E._isolated_result(mutant, stuck, 1.0).uncontained_test == "t.py::test_wedged"
    )
    assert E._isolated_result(mutant, reaped, 1.0).uncontained_test is None


def test_the_record_lists_baseline_names_then_each_uncontained_mutant():
    mutant = _a_mutant()
    stopped = MutantResult(mutant=mutant, contained=True)
    stuck = MutantResult(
        mutant=mutant, contained=False, uncontained_test="t.py::test_wedged"
    )

    lost = E._containment_lost({"t.py::test_z", "session_baseline"}, [stopped, stuck])

    assert lost == (
        {"phase": "baseline", "test": "session_baseline"},
        {"phase": "baseline", "test": "t.py::test_z"},
        {
            "phase": "mutation",
            "test": "t.py::test_wedged",
            "mutant_id": mutant.mutant_id,
            "mutant": mutant.description,
            "mutated_line": mutant.mutated_line,
        },
    )


def test_nothing_lost_records_nothing():
    mutant = _a_mutant()

    assert (
        E._containment_lost(set(), [MutantResult(mutant=mutant, contained=True)]) == ()
    )


_SIX_BOUNDARIES = """
def f(a, b, c, d, e, g):
    return a < b, a < c, a < d, a < e, b > c, d == g
"""


def _uncontained_evaluate(test_name: str):
    calls: list = []

    def fake(mutant, *args, **kwargs):
        calls.append(mutant)
        return MutantResult(
            mutant=mutant, killed=False, contained=False, uncontained_test=test_name
        )

    return fake, calls


def test_the_converged_path_names_the_uncontained_mutant_and_its_test(monkeypatch):
    fake, calls = _uncontained_evaluate("t.py::test_wedged")
    monkeypatch.setattr(E, "evaluate_mutant", fake)
    monkeypatch.setattr(E, "check_equivalent", lambda *a, **k: False)

    res = run_function_converged(
        _fn(_SIX_BOUNDARIES),
        "m::f",
        {MutationCategory.BOUNDARY},
        test_functions=[],
        original_func=None,
        max_per_category=4,
    )

    assert res.coverage_depth == "cut"
    assert [e["phase"] for e in res.containment_lost] == ["mutation"]
    entry = res.containment_lost[0]
    assert entry["test"] == "t.py::test_wedged"
    assert entry["mutant_id"] == calls[0].mutant_id
    assert entry["mutated_line"] == calls[0].mutated_line
    assert res.to_dict()["containment_lost"] == [entry]


def test_the_profiling_path_names_the_uncontained_mutant_and_its_test(monkeypatch):
    fake, calls = _uncontained_evaluate("t.py::test_wedged")
    monkeypatch.setattr(E, "evaluate_mutant", fake)

    res = run_function_profiling(
        _fn(_SIX_BOUNDARIES),
        "m::f",
        {MutationCategory.BOUNDARY},
        test_functions=[],
        original_func=None,
    )

    assert res.coverage_depth == "cut"
    mutation = [e for e in res.containment_lost if e["phase"] == "mutation"]
    assert [(e["test"], e["mutant_id"]) for e in mutation] == [
        ("t.py::test_wedged", calls[0].mutant_id)
    ]


def test_the_converged_path_names_a_baseline_test_it_could_not_stop(monkeypatch):
    def scope_with_a_stuck_baseline(*args, **kwargs):
        uncontained = args[10] if len(args) > 10 else kwargs.get("uncontained")
        uncontained.add("t.py::test_hangs_on_baseline")
        return (lambda mutant: []), {}, [], []

    monkeypatch.setattr(E, "_build_test_scope", scope_with_a_stuck_baseline)
    monkeypatch.setattr(
        E,
        "evaluate_mutant",
        lambda mutant, *a, **k: MutantResult(mutant=mutant, contained=True),
    )
    monkeypatch.setattr(E, "check_equivalent", lambda *a, **k: False)

    res = run_function_converged(
        _fn(_SIX_BOUNDARIES),
        "m::f",
        {MutationCategory.BOUNDARY},
        test_functions=[],
        original_func=None,
        max_per_category=4,
    )

    assert res.coverage_depth == "cut"
    assert res.containment_lost == (
        {"phase": "baseline", "test": "t.py::test_hangs_on_baseline"},
    )


def test_a_contained_profile_carries_no_record(monkeypatch):
    monkeypatch.setattr(
        E,
        "evaluate_mutant",
        lambda mutant, *a, **k: MutantResult(mutant=mutant, contained=True),
    )
    monkeypatch.setattr(E, "check_equivalent", lambda *a, **k: False)

    res = run_function_converged(
        _fn(_SIX_BOUNDARIES),
        "m::f",
        {MutationCategory.BOUNDARY},
        test_functions=[],
        original_func=None,
        max_per_category=4,
    )

    assert res.containment_lost == ()
    assert "containment_lost" not in res.to_dict()
