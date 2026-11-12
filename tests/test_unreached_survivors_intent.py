"""A survivor no test ran against says so, instead of reading as a suite that ran and missed (Wesker #29, item 4).

THE DEFECT, from the badge run 43102482321: `_isolated_worker.py` 0/451, `pytest_discovery.py` 0/80,
`sarif.py` 0/63, `trace_cache.py` 0/9. Measured on this tree: `sarif.py` has NO test at all (no test file
names it), `_isolated_worker.py` is reached only through `python -m Wesker._isolated_worker` subprocesses
(`tests/test_isolation.py` and others), which a mutant installed in the measuring process never reaches.
Both are evaluated against an EMPTY test list, and every mutant was recorded `survived_after_entry`
exactly like one a suite really failed to detect. The scoring is right (an untested function must never
read as fully specified: see `THE entered=None RULE`); the record was silent about what kind of zero it
was.

THE FIX is the reporting. Each survivor carries `reach`: `no_test_ran` / `tests_ran` / `not_recorded`
(`engine.survivor_reach`), the rollup lists the functions whose survivors had no test run against them,
and the completeness note says what that does and does not mean.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from Wesker.ci import completeness_warnings, describe_unreached, unreached_entries
from Wesker.engine import survivor_reach


@pytest.mark.parametrize(
    ("tests_run", "expected"),
    [(0, "no_test_ran"), (1, "tests_ran"), (7, "tests_ran"), (-1, "not_recorded")],
)
def test_a_survivor_is_a_finding_about_the_suite_only_when_a_test_ran(
    tests_run, expected
):
    assert survivor_reach(tests_run) == expected


def test_only_a_positive_reading_makes_an_entry():
    records = [
        {"reach": "no_test_ran"},
        {"reach": "no_test_ran"},
        {"reach": "tests_ran"},
        {"reach": "not_recorded"},
        {},
    ]
    assert unreached_entries(
        {"function_key": "m.py::f", "survivor_records": records, "total_mutants": 9}
    ) == [{"function_key": "m.py::f", "count": 2, "of": 9}]
    assert unreached_entries({"survivor_records": records[2:]}) == []
    assert unreached_entries({}) == []


def test_the_note_names_the_function_and_what_the_zero_means():
    note = describe_unreached([{"function_key": "m.py::f", "count": 2, "of": 9}])
    assert "2 surviving mutant(s) in 1 function(s) had NO test run" in note
    assert "m.py::f — 2/9 mutants" in note
    assert "subprocess" in note and "not measured" in note
    assert describe_unreached([]) == ""
    assert "… and 1 more" in describe_unreached(
        [{"function_key": f"m.py::f{i}", "count": 1, "of": 1} for i in range(3)],
        limit=2,
    )


def test_the_completeness_text_carries_the_note_and_is_silent_without_it():
    with_note = completeness_warnings(
        {"unreached_functions": [{"function_key": "m.py::f", "count": 1, "of": 2}]}
    )
    assert "NO test run" in with_note and "WARNING" not in with_note
    assert completeness_warnings({}) == ""


_MOD = "def add(a, b):\n    return a + b\n\n\ndef lonely(a):\n    return a * 3\n"
_TEST = "from mod import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def test_end_to_end_the_untested_function_is_named_and_the_tested_one_is_not(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "mod.py").write_text(_MOD)
    (tmp_path / "tests" / "test_mod.py").write_text(_TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npythonpath = .\n"
    )
    done = subprocess.run(
        [sys.executable, "-m", "Wesker", "mod.py", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
    )
    assert done.returncode == 0, done.stderr[-1500:]
    mutation = json.loads(done.stdout)["mutation"]
    keys = {e["function_key"] for e in mutation["unreached_functions"]}
    assert keys == {"mod.py::lonely"}, mutation["unreached_functions"]
    lonely = [s for s in mutation["survivors"] if s["function_key"] == "mod.py::lonely"]
    assert lonely and all(s["reach"] == "no_test_ran" for s in lonely)
    assert all(
        s.get("reach") != "no_test_ran"
        for s in mutation["survivors"]
        if s["function_key"] == "mod.py::add"
    )
