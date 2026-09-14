"""Intent tests: a truncated run names each cut function under the remedy its cause needs.

THE DEFECT, seen on Wesker's own Specification (full) workflow. Every run from 2026-08-14 to
2026-09-14 refused with "N function(s) were only PARTIALLY evaluated — cut by the per-file budget,
or by a timed-out worker that could not be stopped", followed by both remedies. It named no function,
and the JSON report that could have been read is written only after the gates pass. So nobody could
tell which functions to look at, or whether the workflow's own rule (raise the budget, never lower
the claim) applied at all: for a worker blocked outside the interpreter, more budget is the wrong
fix. The message also called the budget per-file; it is measured from each function's own start.

`truncation_cause` decides the cause from the fields `ProfilingResult.to_dict` already emits.
`describe_truncation` renders the causes for both reporting paths, the Action's refusal and the CLI's
warning. `profile_codebase` records each cut function with its cause in `truncated_functions`.
"""

from __future__ import annotations

import pytest

# describe_truncation and gate_truncation are called through their modules. Detective's converge
# refused both when these tests imported the names: "the namespace holds the mutant while the
# caller holds the original ... route a test through the patched name".
from Wesker import action, ci
from Wesker.ci import truncation_cause


@pytest.mark.parametrize(
    ("depth", "budget_exhausted", "memory_standing", "expected"),
    [
        ("profiled", False, "n/a", "not_truncated"),
        ("sampled", False, "n/a", "not_truncated"),
        ("converged", False, "enforced", "not_truncated"),
        ("cut", True, "n/a", "budget"),
        ("sampled", True, "n/a", "budget"),
        ("cut", False, "cut", "memory"),
        ("cut", False, "n/a", "uncontained"),
        ("cut", False, "enforced", "uncontained"),
        ("cut", True, "cut", "budget"),
    ],
)
def test_each_cut_is_named_for_its_cause(
    depth, budget_exhausted, memory_standing, expected
):
    """The engine marks a profile "cut" for three reasons: the budget ran out, a worker could not be
    stopped, or (on the isolated path) the memory cap was hit, which also leaves the run uncontained.
    So a cut that is neither the budget's nor the memory cap's is a worker that could not be stopped.
    When the budget ran out, containment is not recorded separately, and the code claims no more
    than "budget"."""
    assert truncation_cause(depth, budget_exhausted, memory_standing) == expected


def _cut(
    key: str,
    cause: str,
    elapsed_ms: float = 150000.0,
    tested: int = 40,
    universe: int = 212,
):
    return {
        "function_key": key,
        "cause": cause,
        "elapsed_ms": elapsed_ms,
        "tested": tested,
        "universe": universe,
    }


def test_the_description_names_every_function_under_its_remedy():
    text = ci.describe_truncation(
        [
            _cut("Wesker/ci.py::slow", "budget"),
            _cut("Wesker/engine.py::blocked", "uncontained", 8000.0, 3, 50),
        ]
    )

    assert "Wesker/ci.py::slow — 150.0 s, 40/212 mutants evaluated" in text
    assert "Wesker/engine.py::blocked — 8.0 s, 3/50 mutants evaluated" in text
    # The cause no budget can fix comes first, and says that a budget will not fix it.
    assert text.index("could not be stopped") < text.index("ran past its budget")
    assert "A larger budget will not help" in text


def test_the_budget_remedy_says_what_the_budget_bounds():
    """`budget` is measured from each function's start, not the file's. Someone raising it needs to
    know what it bounds, or the number they pick is a guess."""
    text = ci.describe_truncation([_cut("a.py::f", "budget")])

    assert "applies to each function separately" in text
    assert "Raise `budget`" in text


def test_a_long_group_is_capped_and_the_rest_are_counted():
    text = ci.describe_truncation(
        [_cut(f"a.py::f{i}", "budget") for i in range(25)], limit=20
    )

    assert "a.py::f19 " in text
    assert "a.py::f20 " not in text
    assert "… and 5 more" in text


def test_an_empty_record_renders_nothing():
    assert ci.describe_truncation([]) == ""


def test_an_unrecognised_cause_is_listed_not_dropped():
    """A cause the renderer does not know is shown under its own name. A refusal that silently left a
    cut function out would bring back the defect this file is about."""
    text = ci.describe_truncation([_cut("a.py::f", "mystery")])

    assert "a.py::f" in text
    assert "mystery" in text


def test_the_refusal_names_the_cut_functions():
    reason = action.gate_truncation(
        {
            "total_truncated": 2,
            "truncated_functions": [
                _cut("Wesker/ci.py::slow", "budget"),
                _cut("Wesker/engine.py::blocked", "uncontained"),
            ],
        }
    )

    assert reason is not None
    assert "2 function(s)" in reason
    assert "sample" in reason
    assert "Wesker/ci.py::slow" in reason
    assert "Wesker/engine.py::blocked" in reason


def test_a_report_without_the_record_still_refuses_with_both_remedies():
    """A report written before `truncated_functions` existed carries only the count. It must still be
    refused, and the message still names both possible remedies."""
    reason = action.gate_truncation({"total_truncated": 3})

    assert reason is not None
    assert "Raise `budget`" in reason
    assert "killable process" in reason


def test_the_rollup_records_each_cut_function_with_its_cause(tmp_path, monkeypatch):
    """Through `profile_codebase` itself: the per-function results it receives become a count AND a
    named list, and a clean result appears in neither."""
    results = [
        {
            "function_key": "m.py::clean",
            "coverage_depth": "sampled",
            "budget_exhausted": False,
            "memory_standing": "n/a",
            "elapsed_ms": 10.0,
            "total_mutants": 5,
            "universe_size": 5,
        },
        {
            "function_key": "m.py::slow",
            "coverage_depth": "cut",
            "budget_exhausted": True,
            "memory_standing": "n/a",
            "elapsed_ms": 130000.0,
            "total_mutants": 9,
            "universe_size": 30,
        },
        {
            "function_key": "m.py::blocked",
            "coverage_depth": "cut",
            "budget_exhausted": False,
            "memory_standing": "n/a",
            "elapsed_ms": 900.0,
            "total_mutants": 2,
            "universe_size": 12,
        },
    ]
    monkeypatch.setattr(ci, "profile_file", lambda *args, **kwargs: results)

    report = ci.profile_codebase(str(tmp_path), ["m.py"], verbose=False)

    assert report["total_truncated"] == 2
    assert [(t["function_key"], t["cause"]) for t in report["truncated_functions"]] == [
        ("m.py::slow", "budget"),
        ("m.py::blocked", "uncontained"),
    ]
    assert report["truncated_functions"][0]["elapsed_ms"] == 130000.0
