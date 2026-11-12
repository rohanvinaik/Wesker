"""Intent: a per-mutant outcome is not a cut, and the badge still refuses it (W#27; Detective #80).

The defect, reproduced on 2026-10-25 against Detective #80's fixture (a `while` loop whose
`i += 1 -> pass` mutant never terminates on the test's input): the W#27 vocabulary set
`coverage_depth="cut"` whenever ANY mutant ended `diverged` or `undetermined`. Every mutant had been
evaluated, so every consumer was told something false — Detective printed "the profile was cut
before the universe was measured" beside "every operator-universe mutant tested" and refused to
classify ten unrelated survivors, and the codebase rollup filed the function under "a timed-out test
could not be stopped", the remedy for a different fault.

Intent: the depth is "cut" only for a RUN that stopped short (budget, containment, memory). A mutant
with no verdict is evaluated-but-unresolved: it stays out of the gateable verdict through its
unscored record, and the badge refuses the run under its own cause and remedy, exactly as strictly
as before — `--allow-truncation` still turns the refusal into a warning, as it did when these were
reported as cuts.
"""

from __future__ import annotations

import ast
import importlib.util
import sys

import pytest

from Wesker.action import _completeness_refusal, gate_unresolved
from Wesker.ci import describe_unresolved, truncation_cause, unresolved_causes
from Wesker.engine import (
    converged_coverage_depth,
    run_function_converged,
    run_function_profiling,
)
from Wesker.filter import filter_categories

# ── the decisions ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    (
        "budget_exhausted",
        "contained",
        "evaluated",
        "universe",
        "passes",
        "depth",
    ),
    [
        (True, True, 9, 9, 1, "cut"),
        (False, False, 9, 9, 1, "cut"),
        # W#45: a memory cap that stopped one mutant is not a row here. It cannot cut the RUN: that
        # mutant is evaluated, unscored and named (`memory_cut`), like `diverged`.
        (False, True, 9, 9, 1, "profiled"),
        (False, True, 10, 9, 1, "profiled"),
        (False, True, 5, 9, 2, "converged"),
        (False, True, 5, 9, 1, "sampled"),
        (False, True, 0, 0, 1, "sampled"),
    ],
)
def test_only_a_run_that_stopped_short_is_cut(
    budget_exhausted, contained, evaluated, universe, passes, depth
):
    assert (
        converged_coverage_depth(
            budget_exhausted, contained, evaluated, universe, passes
        )
        == depth
    )


@pytest.mark.parametrize(
    ("unscored_by", "causes"),
    [
        ({}, ()),
        ({"not_entered": 2, "nested_measurement": 1}, ()),
        ({"diverged": 1}, ("diverged",)),
        ({"undetermined": 1}, ("undetermined",)),
        # W#51: every cause the function holds, the one with no evidence first.
        ({"undetermined": 1, "diverged": 3}, ("undetermined", "diverged")),
        ({"undetermined": 0, "diverged": 0}, ()),
    ],
)
def test_the_unresolved_causes_name_the_outcome_with_no_evidence_first(
    unscored_by, causes
):
    assert unresolved_causes(unscored_by) == causes


def test_the_unresolved_functions_are_rendered_under_their_own_remedy():
    text = describe_unresolved(
        [
            {
                "function_key": "m.py::f",
                "cause": "undetermined",
                "mutants": ["STMT_a", "VALUE_b"],
            },
            {"function_key": "m.py::g", "cause": "diverged", "mutants": []},
            {"function_key": "m.py::h", "cause": "something_new", "mutants": ["X_1"]},
        ]
    )
    assert "wall-clock bound without a verdict" in text
    assert "m.py::f — STMT_a, VALUE_b" in text
    assert "deterministic step budget" in text
    assert "m.py::g — mutants not recorded" in text
    assert "No verdict, cause not recorded: something_new." in text
    assert describe_unresolved([]) == ""


# ── the gate: still refused, never as "PARTIALLY evaluated" ─────────────────────────────────────


def _report(truncated: int = 0, unresolved: int = 0) -> dict:
    return {
        "total_truncated": truncated,
        "truncated_functions": [
            {"function_key": "m.py::t", "cause": "budget", "tested": 1, "universe": 2}
        ][:truncated],
        "total_unresolved": unresolved,
        "unresolved_functions": [
            {"function_key": "m.py::f", "cause": "undetermined", "mutants": ["STMT_a"]}
        ][:unresolved],
    }


def test_a_run_with_no_verdict_mutants_is_refused_under_its_own_name():
    refusal = gate_unresolved(_report(unresolved=1))
    assert refusal is not None
    assert "evaluated in full, but some mutants reached no verdict" in refusal
    assert "PARTIALLY evaluated" not in refusal
    assert "m.py::f — STMT_a" in refusal
    assert gate_unresolved(_report()) is None
    assert _completeness_refusal(_report(unresolved=1), allow_sample=False) == refusal


def test_allow_truncation_still_warns_instead_of_refusing(capsys):
    assert _completeness_refusal(_report(unresolved=1), allow_sample=True) is None
    assert "reached no verdict" in capsys.readouterr().err


def test_a_truncated_run_is_refused_first():
    refusal = _completeness_refusal(
        _report(truncated=1, unresolved=1), allow_sample=False
    )
    assert refusal is not None and "PARTIALLY evaluated" in refusal


# ── through the real engine: a mutant that never terminates ─────────────────────────────────────

_SRC = (
    "def first_false(flags, start):\n"
    "    i = max(start, 0)\n"
    "    while i < len(flags) and flags[i]:\n"
    "        i += 1\n"
    "    return i\n"
)


@pytest.fixture
def loop_target(tmp_path):
    path = tmp_path / "loop_outcome_mod.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("loop_outcome_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["loop_outcome_mod"] = mod
    spec.loader.exec_module(mod)

    def test_first_false():
        assert mod.first_false([True, True, False], 0) == 2

    try:
        yield mod, ast.parse(_SRC).body[0], test_first_false
    finally:
        sys.modules.pop("loop_outcome_mod", None)


def _no_verdict(result) -> int:
    return sum(
        cr.unscored_by.get("undetermined", 0) + cr.unscored_by.get("diverged", 0)
        for cr in result.per_category
    )


def test_the_profiling_path_never_calls_a_no_verdict_mutant_a_cut(loop_target):
    mod, node, test = loop_target
    result = run_function_profiling(
        node,
        "loop_outcome_mod.py::first_false",
        filter_categories(node, True),
        [test],
        mod.first_false,
        200,
        max_per_category=0,
    )
    assert _no_verdict(result) >= 1, result.to_dict()
    assert result.coverage_depth == "profiled", result.to_dict()
    assert not result.is_gateable, (
        "a mutant with no verdict must still keep the verdict ungateable"
    )
    record = result.to_dict()
    assert truncation_cause(record["coverage_depth"], record["budget_exhausted"]) == (
        "not_truncated"
    )
    assert set(unresolved_causes(record["unscored_by"])) & {"undetermined", "diverged"}


def test_the_converged_path_counts_a_no_verdict_mutant_as_evaluated(loop_target):
    mod, node, test = loop_target
    result = run_function_converged(
        node,
        "loop_outcome_mod.py::first_false",
        filter_categories(node, True),
        [test],
        mod.first_false,
        budget_ms=60000,
    )
    assert _no_verdict(result) >= 1, result.to_dict()
    assert result.coverage_depth != "cut", result.to_dict()
    assert not result.is_gateable
