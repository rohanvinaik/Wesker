"""#17 — a skipped, expected-failure or errored test is not a passing one, and its reach proves nothing.

THE DEFECT (measured 2026-10-02 on current code). The baseline outcome was `baseline_passed: bool`
plus `reason`, and the bool was "not inert": the live-session runner raises only when a pytest
report FAILED, so a skipped test and an `xfail` test both returned normally and read as PASSED.
Their reach was therefore ADMISSIBLE proof — an xfail test asserting the wrong value on the false
branch closed that branch's line, which is Detective #59's failing-only counterexample with a
marker on it. And a fixture error read `refuse_failed`, indistinguishable from a wrong expectation.

THE CONTRACT. Each ledger row carries the typed outcome `passed | failed | skipped | xfailed |
error` (pytest's own categories), `baseline_passed` is True only for `passed`, every non-green
outcome is refused for proof under its OWN name, both proof views (`trace_evidence` and
`admissible_line_coverage`) bar them, and a warm build never re-reads a persisted "not failing" as
"passed". Kill attribution is deliberately untouched: a skip that turns into a failure under a
mutant is the suite going red, which is what a kill is.

INTENT tests: the old output was self-consistent, so only a test written from the contract catches it.
"""

from __future__ import annotations

import ast
import json
import os
import sys

import pytest

from Wesker.trace_evidence import (
    BASELINE_OUTCOMES,
    baseline_outcome,
    build_trace_ledger,
    item_run_status,
    trace_admissibility,
)

# ── pytest's category for one item's run, from its phase reports ─────────────────────


@pytest.mark.parametrize(
    ("setup", "call", "teardown", "expected"),
    [
        ("passed", "passed", "passed", "passed"),
        ("passed", "failed", "passed", "failed"),
        ("failed", "", "passed", "error"),  # a fixture failed: the test never ran
        ("passed", "passed", "failed", "error"),  # only teardown failed
        (
            "passed",
            "failed",
            "failed",
            "failed",
        ),  # the call's verdict is the one stated
        ("skipped", "", "passed", "skipped"),  # a skip marker / skipif
        ("passed", "skipped", "passed", "skipped"),  # pytest.skip() in the body
        ("passed", "xfailed", "passed", "xfailed"),  # the expected failure happened
        ("xfailed", "", "passed", "xfailed"),  # xfail(run=False)
        (
            "passed",
            "xpassed",
            "passed",
            "passed",
        ),  # non-strict XPASS: every assert held
        ("", "", "", "unobserved"),  # no reports at all
    ],
)
def test_item_run_status_reads_the_phase_reports_the_way_pytest_does(
    setup, call, teardown, expected
):
    assert item_run_status(setup, call, teardown) == expected


# ── the typed baseline outcome: the engine's channel and pytest's, combined ──────────


@pytest.mark.parametrize(
    ("run_code", "status", "expected"),
    [
        (None, "passed", "passed"),
        (None, "unobserved", "passed"),  # a plain callable that returned
        (None, "skipped", "skipped"),  # returned normally — but did not pass
        (None, "xfailed", "xfailed"),
        ("assertion", "failed", "failed"),
        ("crash", "failed", "failed"),  # a call-phase KeyError is pytest's FAILED
        (
            "assertion",
            "error",
            "error",
        ),  # a fixture's AssertionError is not a wrong expectation
        ("assertion", "unobserved", "failed"),
        (
            "exception",
            "unobserved",
            "failed",
        ),  # a declared failure (pytest.fail / raises)
        ("crash", "unobserved", "error"),  # ambiguous under direct call: never accused
        ("timeout", "passed", "error"),  # a stopped run reached no verdict
        ("timeout", "failed", "error"),  # ...and its partial report is not one either
        ("uncontained", "failed", "error"),
        ("uncontained", "unobserved", "error"),
        ("some_new_kill_reason", "unobserved", "error"),  # unrecognised -> not green
    ],
)
def test_baseline_outcome_never_reads_a_non_pass_as_passed(run_code, status, expected):
    assert baseline_outcome(run_code, status) == expected
    assert expected in BASELINE_OUTCOMES


# ── admissibility: every non-green outcome refused, each under its own name ──────────


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("skipped", "refuse_skipped"),
        ("xfailed", "refuse_xfailed"),
        ("error", "refuse_error"),
        ("failed", "refuse_failed"),
    ],
)
def test_a_non_green_outcome_is_refused_under_its_own_name(outcome, reason):
    # `baseline_passed=True` on purpose: the old boolean said "not inert", which a skip satisfied.
    assert trace_admissibility(True, False, True, True, outcome) == reason


def test_only_an_affirmative_pass_is_admissible():
    assert trace_admissibility(True, False, True, True, "passed") == "admissible"
    # an outcome the vocabulary does not know is not green
    assert trace_admissibility(True, False, True, True, "bogus") == "refuse_failed"
    # a contradiction resolves to the refusal, never to proof
    assert trace_admissibility(False, False, True, True, "passed") == "refuse_failed"


def test_an_untyped_row_keeps_the_boolean_meaning():
    """A caller (or a cached row) with no typed outcome is `unrecorded`: the boolean decides, as before."""
    assert trace_admissibility(True, False, True) == "admissible"
    assert trace_admissibility(False, False, True) == "refuse_failed"


def test_measurement_refusals_still_outrank_the_outcome():
    assert trace_admissibility(True, False, False, True, "skipped") == (
        "refuse_uncontained"
    )
    assert trace_admissibility(True, True, True, True, "xfailed") == "refuse_truncated"
    # a replay of a green item is still only routing; a replay of a skipped one is refused as skipped
    assert trace_admissibility(True, False, True, False, "passed") == "refuse_replayed"
    assert trace_admissibility(True, False, True, False, "skipped") == "refuse_skipped"


def test_the_ledger_types_each_row_and_keeps_the_skipped_reach_as_observed_only():
    ledger = build_trace_ledger(
        line_coverage={"t_green": [2, 3], "t_skip": [2, 4], "t_untyped": [5]},
        failed_ids=set(),
        truncated_ids=set(),
        contained=True,
        outcomes={"t_green": "passed", "t_skip": "skipped"},
    )
    by_id = {ev.test_id: ev for ev in ledger}
    assert by_id["t_skip"].baseline_outcome == "skipped"
    assert by_id["t_skip"].baseline_passed is False
    assert by_id["t_skip"].reason == "refuse_skipped"
    assert by_id["t_skip"].lines == (2, 4)  # observed reach is kept
    assert by_id["t_green"].admissible is True
    assert by_id["t_untyped"].baseline_outcome == "unrecorded"
    assert (
        by_id["t_untyped"].admissible is True
    )  # no typed outcome: the old meaning stands


# ── end-to-end through a live pytest session (Wesker only) ────────────────────────────

_TARGET = "def pick(flag):\n    if flag:\n        return 1\n    return 0\n"

_TESTS = """import pytest
from {mod} import pick


def test_green():
    assert pick(True) == 1


def test_failing():
    assert pick(False) == 1


def test_skip_after_reaching():
    pick(False)
    pytest.skip("reached the target, then skipped")


@pytest.mark.xfail(reason="a known wrong expectation")
def test_xfail_reaching():
    assert pick(False) == 1


@pytest.fixture
def broken():
    raise RuntimeError("fixture setup error")


def test_setup_error(broken):
    assert pick(False) == 0
"""


def _project(tmp_path, mod):
    (tmp_path / f"{mod}.py").write_text(_TARGET)
    (tmp_path / f"test_{mod}.py").write_text(_TESTS.format(mod=mod))
    for name in (mod, f"test_{mod}"):
        sys.modules.pop(name, None)
    return str(tmp_path)


def _in_live_session(root, target, body):
    from Wesker.ci import run_with_live_suite

    cwd = os.getcwd()
    try:
        os.chdir(root)
        return run_with_live_suite(root, body, target_files=[target])
    finally:
        os.chdir(cwd)


def _profile(root, mod):
    from Wesker.ci import (
        discover_test_callables,
        resolve_original_func,
        walk_functions,
    )
    from Wesker.engine import run_function_profiling
    from Wesker.filter import filter_categories

    def body():
        src = os.path.join(root, f"{mod}.py")
        with open(src) as fh:
            node = dict(walk_functions(ast.parse(fh.read())))["pick"]
        tests = discover_test_callables(root, f"{mod}.py", ["pick"])
        return run_function_profiling(
            node,
            f"{mod}.py::pick",
            filter_categories(node),
            tests,
            resolve_original_func(src, "pick"),
        )

    return _in_live_session(root, f"{mod}.py", body)


def test_each_baseline_outcome_is_typed_and_only_the_pass_proves_anything(tmp_path):
    root = _project(tmp_path, "otype_e2e")
    r = _profile(root, "otype_e2e")
    by_name = {ev.test_id.rsplit("::", 1)[-1]: ev for ev in r.trace_evidence}
    assert {n: ev.baseline_outcome for n, ev in by_name.items()} == {
        "test_green": "passed",
        "test_failing": "failed",
        "test_skip_after_reaching": "skipped",
        "test_xfail_reaching": "xfailed",
        "test_setup_error": "error",
    }
    assert {n: ev.reason for n, ev in by_name.items()} == {
        "test_green": "admissible",
        "test_failing": "refuse_failed",
        "test_skip_after_reaching": "refuse_skipped",
        "test_xfail_reaching": "refuse_xfailed",
        "test_setup_error": "refuse_error",
    }
    assert [n for n, ev in by_name.items() if ev.baseline_passed] == ["test_green"]
    # Line 4 (`return 0`) is reached only by the failing, skipped and xfail tests: observed, never proved.
    assert 4 in r.observed_union
    assert 4 not in r.admissible_union, (
        "a skipped/xfail test's reach closed a line — #17"
    )
    # The derived view Detective's certificate reads agrees with the ledger.
    admissible_owners = {k.rsplit("::", 1)[-1] for k in r.admissible_line_coverage}
    assert admissible_owners == {"test_green"}
    rows = {
        row["test_id"].rsplit("::", 1)[-1]: row for row in r.to_dict()["trace_evidence"]
    }
    assert rows["test_xfail_reaching"]["baseline_outcome"] == "xfailed"


def test_a_warm_build_reuses_the_typed_outcome_and_never_rereads_a_skip_as_a_pass(
    tmp_path, monkeypatch
):
    """The outcome pass is cached per TestId. Before typing, the cache held only `failing`/`inert`,
    so a reused outcome could not say "skipped"; reuse is gated on the persisted type, and a cache
    written without one is re-measured rather than read as green."""
    from Wesker import engine
    from Wesker.ci import callable_test_id

    root = _project(tmp_path, "otype_warm")
    target = os.path.join(root, "otype_warm.py")

    def _skip_id(sb):
        return next(t for t in sb.outcomes if t.endswith("test_skip_after_reaching"))

    def _build_twice(tamper=None):
        runs: list[str] = []

        def body():
            from Wesker.ci import _LIVE_SUITE

            calls = _LIVE_SUITE.get()
            first = engine.build_session_baseline(calls, {target}, project_root=root)
            if tamper is not None:
                tamper()
            real = engine._run_test_with_timeout

            def counting(test_fn, *a, **k):
                runs.append(callable_test_id(test_fn))
                return real(test_fn, *a, **k)

            monkeypatch.setattr(engine, "_run_test_with_timeout", counting)
            try:
                second = engine.build_session_baseline(
                    calls, {target}, project_root=root
                )
            finally:
                monkeypatch.setattr(engine, "_run_test_with_timeout", real)
            return first, second

        return _in_live_session(root, "otype_warm.py", body), runs

    (first, second), runs = _build_twice()
    assert first.outcomes[_skip_id(first)] == "skipped"
    assert second.outcomes == first.outcomes
    assert runs == [], "the warm build re-ran outcomes it had persisted with their type"

    cache_file = os.path.join(root, ".wesker", "trace_cache.json")

    def _strip_types():
        with open(cache_file) as fh:
            blob = json.load(fh)
        blob.pop("outcome_status", None)
        with open(cache_file, "w") as fh:
            json.dump(blob, fh)

    (first, second), runs = _build_twice(tamper=_strip_types)
    assert runs, (
        "an outcome persisted WITHOUT its type was reused — it reads a skip as a pass"
    )
    assert second.outcomes[_skip_id(second)] == "skipped"
