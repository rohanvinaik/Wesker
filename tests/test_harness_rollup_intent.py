"""Intent: the mutants the HARNESS keeps out of the kill rate are counted and named beside it (W#54).

THE DEFECT. `ci.UNSCORED_ROLLUP` assigns every unscored disposition an owner, and the rollup named two
owners and dropped the third. A function whose mutants were never built (`harness_error`), never
installed (`not_installed`), never entered (`not_entered`) or refused as a nested measurement
(`nested_measurement`) reported a kill rate over the rest, and no line of `python -m Wesker`'s summary
or of the `--json` rollup said that some had left the denominator, or how many. Measured on a module
with one `functools.lru_cache`-wrapped function: `Kill rate: 88% (7/8)`, `total_unresolved: 0`, and 4 of
its 12 mutants (`unscored_by == {"harness_error": 4}`) mentioned nowhere.

THE RULING (Detective tasks 14.3a, C1; COMMUNICATING_DETERMINISM section 4: absence and emptiness never
render alike): a harness line beside the kill rate, counted per disposition, the functions named, and
`not_entered` carrying its named cause. The badge's gating is unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from Wesker.ci import (
    UNSCORED_ROLLUP,
    describe_harness,
    harness_dispositions,
    harness_standing,
    profile_codebase,
)
from Wesker.engine import ORIGINAL_ALSO_FAILS
from Wesker.write_authority import WRITE_OUTSIDE_AUTHORITY

_HARNESS = [d for d, owner in UNSCORED_ROLLUP if owner == "harness"]


def test_harness_dispositions_are_exactly_the_tables_harness_owner():
    every = {d: 1 for d, _ in UNSCORED_ROLLUP}
    assert harness_dispositions(every) == tuple(_HARNESS)
    assert set(_HARNESS) == {
        "harness_error",
        "not_installed",
        "not_entered",
        "nested_measurement",
    }


@pytest.mark.parametrize(
    "unscored",
    [{}, {"harness_error": 0}, {"undetermined": 3, "cut": 2}, {"brand_new": 4}],
)
def test_a_function_with_no_harness_mutant_names_none(unscored):
    """A zero count is absence; a cut, an unresolved cause and an unknown disposition belong to
    other owners and are never read as harness."""
    assert harness_dispositions(unscored) == ()


def test_standing_never_renders_absence_like_emptiness():
    assert harness_standing(False, 0) == "not_recorded"
    assert harness_standing(False, 5) == "not_recorded"
    assert harness_standing(True, 0) == "none"
    assert harness_standing(True, -1) == "none"
    assert harness_standing(True, 1) == "left_ratio"


def test_old_report_empty_report_and_populated_report_render_three_different_lines():
    old = describe_harness({"kill_pct": 90})
    empty = describe_harness({"harness_functions": [], "total_harness_mutants": 0})
    full = describe_harness(
        {
            "harness_functions": [
                {"function_key": "m.py::f", "disposition": "not_installed", "count": 2}
            ],
            "total_harness_mutants": 2,
            "total_harness": 1,
            "harness_mutants": {"not_installed": 2},
        }
    )
    assert len({old, empty, full}) == 3
    assert "not recorded" in old
    assert "0 mutants" in empty
    assert "m.py::f" in full and "not_installed 2" in full


def _result(key: str, unscored_by: dict, **extra) -> dict:
    return {
        "function_key": key,
        "unscored_by": unscored_by,
        "unscored_records": [
            {"mutant_id": f"M{i}_{d}", "disposition": d}
            for d, n in unscored_by.items()
            for i in range(n)
        ],
        "total_mutants": 4,
        "total_killed": 4,
        **extra,
    }


def _rollup(monkeypatch, results):
    monkeypatch.setattr("Wesker.ci.profile_file", lambda *a, **k: results)
    monkeypatch.setattr("Wesker.ci._load_cached_state", lambda root: None)
    return profile_codebase(".", ["m.py"], verbose=False)


def test_rollup_counts_every_harness_disposition_per_function(monkeypatch):
    report = _rollup(
        monkeypatch,
        [
            _result("m.py::a", {"harness_error": 2, "not_installed": 1}),
            _result(
                "m.py::b",
                {"not_entered": 3, "nested_measurement": 1},
                not_entered_causes={"not_reached": 2, "decorator_wrapper": 1},
            ),
            _result("m.py::c", {}),
        ],
    )
    assert report["total_harness_mutants"] == 7
    assert report["total_harness"] == 2
    assert report["harness_mutants"] == {
        "harness_error": 2,
        "not_installed": 1,
        "not_entered": 3,
        "nested_measurement": 1,
    }
    entered = [
        e for e in report["harness_functions"] if e["disposition"] == "not_entered"
    ]
    assert entered[0]["causes"] == {"not_reached": 2, "decorator_wrapper": 1}
    text = describe_harness(report)
    for needle in (
        "m.py::a",
        "m.py::b",
        "not_reached 2",
        "decorator_wrapper 1",
        "7 mutant",
    ):
        assert needle in text
    assert "m.py::c" not in text
    # the badge's gating is untouched: harness mutants are not unresolved and not truncated
    assert report["total_unresolved"] == 0 and report["total_truncated"] == 0


def test_a_cut_function_still_names_its_harness_mutants(monkeypatch):
    report = _rollup(
        monkeypatch,
        [_result("m.py::a", {"harness_error": 1}, budget_exhausted=True)],
    )
    assert report["total_truncated"] == 1
    assert report["total_harness_mutants"] == 1


def test_a_run_with_none_records_an_empty_harness_not_a_missing_one(monkeypatch):
    report = _rollup(monkeypatch, [_result("m.py::a", {})])
    assert report["harness_functions"] == []
    assert report["total_harness_mutants"] == 0
    assert "not recorded" not in describe_harness(report)


def test_original_also_fails_is_named_wherever_write_outside_authority_is():
    """The unresolved owners are four, and each is handled by every rollup table beside the others."""
    from Wesker.ci import _UNRESOLVED_REMEDIES, unresolved_causes

    owned = {d for d, owner in UNSCORED_ROLLUP if owner == "unresolved"}
    assert {WRITE_OUTSIDE_AUTHORITY, ORIGINAL_ALSO_FAILS} <= owned
    assert {c for c, _ in _UNRESOLVED_REMEDIES} == owned
    both = {WRITE_OUTSIDE_AUTHORITY: 1, ORIGINAL_ALSO_FAILS: 1}
    assert set(unresolved_causes(both)) == set(both)
    assert harness_dispositions(both) == ()


_CALC = """\
import functools


def add(a, b):
    if a > b:
        return a - b
    return a + b


@functools.lru_cache(maxsize=None)
def five(x):
    return x * 5
"""
_TEST = """\
from calc import add


def test_add():
    assert add(1, 2) == 3
    assert add(5, 2) == 3
"""


def test_the_cli_summary_and_json_say_it_end_to_end(tmp_path):
    """The issue's own shape through the real command: a wrapped target whose mutants are never built."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "calc.py").write_text(_CALC)
    (tmp_path / "tests" / "test_calc.py").write_text(_TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npythonpath = .\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": os.pathsep.join(sys.path)}
    text = subprocess.run(
        [sys.executable, "-m", "Wesker", "calc.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert text.returncode == 0, text.stderr
    assert "Kill rate:" in text.stdout and "Harness:" in text.stdout
    assert text.stdout.index("Kill rate:") < text.stdout.index("Harness:")
    assert "calc.py::five" in text.stdout
    raw = subprocess.run(
        [sys.executable, "-m", "Wesker", "calc.py", "--json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    mutation = json.loads(raw.stdout)["mutation"]
    assert mutation["total_harness_mutants"] > 0
    assert mutation["harness_functions"][0]["function_key"] == "calc.py::five"
