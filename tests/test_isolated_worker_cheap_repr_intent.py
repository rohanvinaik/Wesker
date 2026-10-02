"""W#34 — the isolated worker formats no failure traceback nobody reads, and loses no verdict.

THE DEFECT. EP-C1 (Detective docs/ENGINEERING_PASS_2026-09-26.md): every killed mutant is a failing
test to pytest, and building its report formatted a full source-annotated traceback — re-parsing
source with `ast` per traceback entry — ~45% of a converge and 99% of its AST allocation, for text
nothing reads. `49f7e91` gave the in-process measurement session a cheap `repr_failure`; the ISOLATED
worker (`Wesker/_isolated_worker.py`, a separate process running its own `pytest.main` per mutant and
per determinism baseline) still formatted every one.

THE CONTRACT, pinned through the REAL worker process (the server protocol the engine drives, and the
`--baseline` mode the determinism check runs): every failing report the worker builds carries the
cheap "Type: message" text — no source lines, no traceback — for a call-phase failure AND a setup
error; and the verdict is untouched: the exit code, `killed_by`, the killing node. The SAME check
through `detective converge --isolated` (identical FINAL, byte-identical generated suite and
certificates) and the per-mutant kill-record A/B are recorded in the commit that added this.

A project `conftest.py` records each report's `longrepr` as pytest's own logreport hook sees it — the
observation point is the report itself, not anything this test infers.
"""

from __future__ import annotations

import json

from Wesker.isolation import (
    IsolatedMutantWorker,
    mutant_verdict,
    run_baseline_traced_isolated,
)

_RECORDER = """
import json
from pathlib import Path

_OUT = Path(__file__).parent / "reports.jsonl"


def pytest_runtest_logreport(report):
    if report.failed:
        with _OUT.open("a") as fh:
            fh.write(json.dumps({"when": report.when, "nodeid": report.nodeid,
                                 "longrepr": str(report.longrepr)}) + "\\n")
"""


def _project(tmp_path):
    (tmp_path / "conftest.py").write_text(_RECORDER)
    (tmp_path / "t34.py").write_text("def double(x):\n    y = x * 2\n    return y\n")
    (tmp_path / "test_t34.py").write_text(
        "import pytest\n"
        "from t34 import double\n\n\n"
        "def _helper(v):\n    return double(v)\n\n\n"
        "def test_value():\n    assert _helper(3) == 6\n\n\n"
        "@pytest.fixture\ndef broken():\n    raise RuntimeError('setup blew up')\n\n\n"
        "def test_setup_error(broken):\n    assert double(1) == 2\n"
    )
    return str(tmp_path)


def _reports(tmp_path):
    path = tmp_path / "reports.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _is_cheap(longrepr: str, type_name: str) -> bool:
    # The full repr carries the test's SOURCE ("def test_...", the `>` failure marker, `E` lines,
    # "file:line" locations); the cheap one is a single "Type: message" with none of them.
    return (
        longrepr.startswith(f"{type_name}: ")
        and "def test_" not in longrepr
        and "\n>" not in longrepr
        and ".py:" not in longrepr
    )


def test_a_killed_mutant_report_carries_the_cheap_text_and_the_verdict_survives(
    tmp_path,
):
    root = _project(tmp_path)
    worker = IsolatedMutantWorker(root, ["test_t34.py::test_value"], "t34.py", "double")
    try:
        run = worker.evaluate("def double(x):\n    y = x * 3\n    return y\n", 60.0)
    finally:
        worker.close()
    assert mutant_verdict(run.outcome) == "killed"
    assert run.killed_by == "assertion"
    assert run.test_name == "test_t34.py::test_value"
    (call,) = [r for r in _reports(tmp_path) if r["when"] == "call"]
    assert _is_cheap(call["longrepr"], "AssertionError"), call["longrepr"]


def test_a_setup_error_report_carries_the_cheap_text_too(tmp_path):
    """A setup failure is formatted through `_repr_failure_py`, the second entry point."""
    root = _project(tmp_path)
    worker = IsolatedMutantWorker(
        root, ["test_t34.py::test_setup_error"], "t34.py", "double"
    )
    try:
        run = worker.evaluate("def double(x):\n    y = x * 2\n    return y\n", 60.0)
    finally:
        worker.close()
    assert (
        mutant_verdict(run.outcome) == "killed"
    )  # pytest's exit code: the node errored
    (setup,) = [r for r in _reports(tmp_path) if r["when"] == "setup"]
    assert _is_cheap(setup["longrepr"], "RuntimeError"), setup["longrepr"]


def test_the_determinism_baseline_worker_formats_no_traceback_either(tmp_path):
    root = _project(tmp_path)
    (tmp_path / "test_t34_red.py").write_text(
        "from t34 import double\n\n\ndef test_red():\n    assert double(2) == 5\n"
    )
    lines, outcome, contained = run_baseline_traced_isolated(
        root, ["test_t34_red.py::test_red"], "t34.py", 60.0
    )
    assert contained and outcome == "failed"
    assert {2, 3} <= set(lines)  # the trace still sees the body
    (call,) = [r for r in _reports(tmp_path) if r["when"] == "call"]
    assert _is_cheap(call["longrepr"], "AssertionError"), call["longrepr"]
