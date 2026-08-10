"""#19 — isolated worker execution: a mutant/test runs in a KILLABLE process group.

The in-process path cannot contain a test blocked in a subprocess/socket/C-extension: a Python
thread cannot be killed, only asked to stop, and `interrupt.abandon` reports honestly when the ask
fails. A separate process GROUP can be killed. These tests pin that guarantee — a runaway node, and
a CHILD it spawned, are both terminated with an uncatchable SIGKILL and the run reports contained —
which is the whole reason isolation is the gateable mode.

The pure `isolated_test_outcome` is characterized by converge; these add the intent: a timeout
outranks any code, an unknown exit is never a pass, and the end-to-end runs prove the containment.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from Wesker.isolation import (
    IsolatedMutantWorker,
    isolated_test_outcome,
    mutant_verdict,
    run_mutant_isolated,
    run_pytest_node_isolated,
    should_recycle,
)


# ── the pure decision ──────────────────────────────────────────────────────────


def test_a_timeout_outranks_any_exit_code():
    """The worker was killed mid-flight — its exit code is not trustworthy, so `timeout` wins."""
    assert isolated_test_outcome(0, timed_out=True) == "timeout"
    assert isolated_test_outcome(1, timed_out=True) == "timeout"


def test_exit_codes_map_to_typed_outcomes():
    assert isolated_test_outcome(0, False) == "passed"
    assert isolated_test_outcome(1, False) == "failed"
    assert isolated_test_outcome(5, False) == "no_tests"


def test_an_unknown_exit_is_error_never_passed():
    """A collection/internal/usage/signal exit must never read as green."""
    for rc in (2, 3, 4, -9, 137):
        assert isolated_test_outcome(rc, False) == "error"


# ── end-to-end: the isolated pytest lifecycle ───────────────────────────────────


def test_a_passing_node_runs_isolated_and_passes(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    r = run_pytest_node_isolated(str(tmp_path), ["test_ok.py::test_ok"], 30.0)
    assert r.contained and not r.timed_out
    assert r.outcome == "passed"


def test_a_failing_node_reports_failed(tmp_path):
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert 1 == 2\n")
    r = run_pytest_node_isolated(str(tmp_path), ["test_bad.py::test_bad"], 30.0)
    assert r.outcome == "failed"


# ── containment: the guarantee a thread abandon cannot give ─────────────────────


def test_a_runaway_test_is_killed_and_reported_contained(tmp_path):
    (tmp_path / "test_hang.py").write_text(
        "import time\n\n\ndef test_hang():\n    time.sleep(100)\n"
    )
    r = run_pytest_node_isolated(str(tmp_path), ["test_hang.py::test_hang"], 1.0)
    assert r.timed_out is True
    assert r.contained is True, (
        "a subprocess that can be SIGKILLed must report contained"
    )
    assert r.outcome == "timeout"


def test_a_child_the_test_spawned_is_killed_with_the_group(tmp_path):
    """The differentiator. killpg reaches the CHILD subprocess the test started — the exact thing a
    thread abandon leaves running while later measurement continues against a compromised box."""
    pidfile = tmp_path / "child.pid"
    (tmp_path / "test_child.py").write_text(
        "import subprocess, sys, time\n\n\n"
        "def test_child():\n"
        "    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"    open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "    time.sleep(120)\n"
    )
    r = run_pytest_node_isolated(str(tmp_path), ["test_child.py::test_child"], 3.0)
    assert r.timed_out is True and r.contained is True
    assert pidfile.exists(), "the test did not reach the point of spawning its child"
    child_pid = int(pidfile.read_text())

    # The child must die WITH the group. Poll: reparenting/reaping is not instant.
    gone = False
    for _ in range(40):
        try:
            os.kill(child_pid, 0)  # 0 = existence probe; raises if the process is gone
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.1)
    assert gone, (
        f"child {child_pid} survived the group kill — containment did not reach it"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_the_module_uses_a_real_process_group():
    """Guard the mechanism itself: the worker is started in a new session (hence a new process
    group), which is what makes killpg reach its children."""
    import inspect

    from Wesker import isolation

    src = inspect.getsource(isolation.run_pytest_node_isolated)
    assert "start_new_session=True" in src
    assert "killpg" in inspect.getsource(isolation._terminate_group)


# ── the mutant verdict decision ─────────────────────────────────────────────────


def test_a_failed_or_timed_out_node_is_a_kill():
    assert mutant_verdict("failed") == "killed"
    assert mutant_verdict("timeout") == "killed"


def test_all_nodes_passing_is_a_survivor():
    assert mutant_verdict("passed") == "survived"


def test_a_collection_or_empty_run_is_harness_never_a_kill():
    """A harness/collection state measures the engine, not the suite — never a kill."""
    assert mutant_verdict("no_tests") == "harness"
    assert mutant_verdict("error") == "harness"


# ── end-to-end: isolated mutant evaluation is SOUND ─────────────────────────────


def _tiny_project(tmp_path) -> None:
    (tmp_path / "t.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "test_t.py").write_text(
        "from t import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )


def test_a_mutant_a_test_kills_reports_killed(tmp_path):
    """The soundness proof: a mutant that changes the pinned value is detected as killed, through
    the real pytest lifecycle in an isolated process."""
    _tiny_project(tmp_path)
    r = run_mutant_isolated(
        str(tmp_path),
        ["test_t.py::test_f"],
        "t.py",
        "f",
        "def f(x):\n    return x - 1\n",
        30.0,
    )
    assert r.contained and not r.timed_out
    assert mutant_verdict(r.outcome) == "killed"


def test_a_mutant_no_test_kills_reports_survived(tmp_path):
    """The other half: a mutant the suite does NOT distinguish survives — the isolated verdict must
    not manufacture a kill the tests did not make."""
    _tiny_project(tmp_path)
    r = run_mutant_isolated(
        str(tmp_path),
        ["test_t.py::test_f"],
        "t.py",
        "f",
        # differs only at x == 42, which the single test never exercises
        "def f(x):\n    return x + 1 if x != 42 else 0\n",
        30.0,
    )
    assert r.contained and mutant_verdict(r.outcome) == "survived"


def test_a_hanging_mutant_is_killed_by_timeout_and_contained(tmp_path):
    """A mutant that never returns hangs the node; the isolated worker terminates its process group
    and reports a contained timeout kill — the containment an in-process thread cannot guarantee."""
    _tiny_project(tmp_path)
    r = run_mutant_isolated(
        str(tmp_path),
        ["test_t.py::test_f"],
        "t.py",
        "f",
        "def f(x):\n    while True:\n        pass\n",
        2.0,
    )
    assert r.timed_out is True and r.contained is True
    assert mutant_verdict(r.outcome) == "killed"


# ── the persistent recycled worker (increment 3) ────────────────────────────────


def test_recycle_fires_at_the_cap_and_never_when_disabled():
    assert should_recycle(50, 50) is True
    assert should_recycle(51, 50) is True
    assert should_recycle(49, 50) is False
    assert should_recycle(9999, 0) is False  # 0 = never recycle on count


def test_one_worker_serves_many_mutants_with_correct_verdicts(tmp_path):
    """The efficiency AND the soundness together: a single interpreter evaluates several mutants —
    one import, N verdicts — and each verdict is right (kill vs survive), with the mutant torn down
    per test so it never leaks into the next."""
    _tiny_project(tmp_path)
    worker = IsolatedMutantWorker(str(tmp_path), ["test_t.py::test_f"], "t.py", "f")
    try:
        k = worker.evaluate("def f(x):\n    return x - 1\n", 30.0)
        assert mutant_verdict(k.outcome) == "killed"
        s = worker.evaluate("def f(x):\n    return x + 1 if x != 42 else 0\n", 30.0)
        assert mutant_verdict(s.outcome) == "survived"
        assert worker.evaluated == 2 and worker.alive  # one import, two mutants
    finally:
        worker.close()


def test_a_hanging_mutant_retires_the_worker_but_stays_contained(tmp_path):
    """A hang on the reused worker kills its whole group and marks it dead, so the caller recycles a
    fresh one — the runaway cannot linger and perturb the next mutant."""
    _tiny_project(tmp_path)
    worker = IsolatedMutantWorker(str(tmp_path), ["test_t.py::test_f"], "t.py", "f")
    try:
        h = worker.evaluate("def f(x):\n    while True:\n        pass\n", 2.0)
        assert h.timed_out is True and h.contained is True
        assert worker.alive is False, "a hung worker must be retired, not reused"
        assert mutant_verdict(h.outcome) == "killed"
    finally:
        worker.close()


# ── the typed kill vocabulary survives isolation (increment 4) ───────────────────
#
# pytest's exit code says only killed/survived. The engine's `value_killed` split needs to know
# WHY — a value pin (assertion/exception) versus a run-only kill (crash/timeout). These prove the
# reason crosses back out of the isolated process intact, matching the in-process classifier, so
# routing profiling through the worker does not silently collapse the split.


def test_an_assertion_kill_reports_killed_by_assertion(tmp_path):
    """A test whose assert catches the changed value pins it: killed_by='assertion' — a value kill."""
    _tiny_project(tmp_path)  # test_f asserts f(1) == 2
    worker = IsolatedMutantWorker(str(tmp_path), ["test_t.py::test_f"], "t.py", "f")
    try:
        r = worker.evaluate("def f(x):\n    return x - 1\n", 30.0)
        assert mutant_verdict(r.outcome) == "killed"
        assert r.killed_by == "assertion"
        assert r.constructed is True
        assert r.test_name == "test_t.py::test_f"
    finally:
        worker.close()


def test_a_violated_raises_contract_reports_killed_by_exception(tmp_path):
    """A `pytest.raises` contract the mutant breaks is a DECLARED failure (pytest's 'DID NOT RAISE'),
    the same strength as an assertion — killed_by='exception', which `value_killed` also counts.
    This is the case a bare pass/fail exit code cannot tell apart from a crash."""
    (tmp_path / "g.py").write_text(
        "def g(x):\n    if x < 0:\n        raise ValueError('neg')\n    return x\n"
    )
    (tmp_path / "test_g.py").write_text(
        "import pytest\n\nfrom g import g\n\n\n"
        "def test_g():\n    with pytest.raises(ValueError):\n        g(-1)\n"
    )
    worker = IsolatedMutantWorker(str(tmp_path), ["test_g.py::test_g"], "g.py", "g")
    try:
        # The mutant drops the guard, so g(-1) no longer raises → the contract is violated.
        r = worker.evaluate("def g(x):\n    return x\n", 30.0)
        assert mutant_verdict(r.outcome) == "killed"
        assert r.killed_by == "exception"
    finally:
        worker.close()


def test_an_unexpected_raise_reports_killed_by_crash(tmp_path):
    """A mutant that blows up where no test stated a contract only proves it RUNS differently —
    killed_by='crash', a run-only kill the value-spec view treats as a survivor."""
    _tiny_project(tmp_path)  # test_f only asserts f(1) == 2; no exception contract
    worker = IsolatedMutantWorker(str(tmp_path), ["test_t.py::test_f"], "t.py", "f")
    try:
        r = worker.evaluate("def f(x):\n    raise RuntimeError('boom')\n", 30.0)
        assert mutant_verdict(r.outcome) == "killed"
        assert r.killed_by == "crash"
    finally:
        worker.close()


def test_a_noncompiling_mutant_is_not_constructed_never_a_survivor(tmp_path):
    """A mutant that will not compile installed nothing — the node ran the ORIGINAL and passed. That
    is a fact about the harness, not the suite: constructed=False, so the engine scores it
    harness_error, never a false survivor that deflates the kill score."""
    _tiny_project(tmp_path)
    worker = IsolatedMutantWorker(str(tmp_path), ["test_t.py::test_f"], "t.py", "f")
    try:
        r = worker.evaluate("def f(x)\n    return x - 1\n", 30.0)  # missing colon
        assert r.constructed is False
    finally:
        worker.close()


def test_per_mutant_node_ids_override_the_session_default(tmp_path):
    """A mutant may carry its own node_ids (per-mutant test scoping). Here the session default names
    a passing test, but the mutant is evaluated only against a DIFFERENT test that catches it —
    proving the override, not the session set, selected the run."""
    (tmp_path / "t.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "test_t.py").write_text(
        "from t import f\n\n\n"
        "def test_never_reached():\n    assert True\n\n\n"
        "def test_catches():\n    assert f(1) == 2\n"
    )
    worker = IsolatedMutantWorker(
        str(tmp_path), ["test_t.py::test_never_reached"], "t.py", "f"
    )
    try:
        r = worker.evaluate(
            "def f(x):\n    return x - 1\n",
            30.0,
            node_ids=["test_t.py::test_catches"],
        )
        assert mutant_verdict(r.outcome) == "killed"
        assert r.killed_by == "assertion"
    finally:
        worker.close()


# ── the engine routes profiling through the isolated worker (increment 4b) ───────


def _real_project(tmp_path):
    """An on-disk project whose test callables carry REAL pytest nodeids, and the function node +
    its imported test callables (nodeid stamped on __qualname__ exactly as Wesker's collection does,
    so the isolated path can address them). Returns (func_node, func_obj, test_callables)."""
    import ast
    import importlib
    import sys

    (tmp_path / "m.py").write_text(
        "def in_range(x, lo, hi):\n    return lo <= x <= hi\n"
    )
    (tmp_path / "test_m.py").write_text(
        "from m import in_range\n\n\n"
        "def test_inside():\n    assert in_range(5, 0, 10) is True\n\n\n"
        "def test_below():\n    assert in_range(-1, 0, 10) is False\n"
    )
    node = ast.parse((tmp_path / "m.py").read_text()).body[0]
    sys.path.insert(0, str(tmp_path))
    m = importlib.import_module("m")
    tm = importlib.import_module("test_m")
    tm.test_inside.__qualname__ = "test_m.py::test_inside"
    tm.test_below.__qualname__ = "test_m.py::test_below"
    return node, m.in_range, [tm.test_inside, tm.test_below]


def _drop_project(tmp_path):
    import sys

    sys.path[:] = [p for p in sys.path if p != str(tmp_path)]
    sys.modules.pop("m", None)
    sys.modules.pop("test_m", None)


def test_isolated_profiling_agrees_with_in_process_and_marks_the_mode(tmp_path):
    """THE soundness proof for the wiring: routing `run_function_profiling` through the killable
    worker process yields the SAME bottom line as the in-process path — same kills, same survivors,
    same value-spec split — and stamps the mode so a consumer can tell which containment guarantee
    measured it. A verdict that disagreed would mean the isolated crossing lost or invented a kill."""
    from Wesker.ci import _PROJECT_ROOT
    from Wesker.engine import run_function_profiling
    from Wesker.filter import filter_categories

    node, func_obj, tests = _real_project(tmp_path)
    try:
        cats = filter_categories(node)
        token = _PROJECT_ROOT.set(str(tmp_path))
        try:
            inproc = run_function_profiling(
                node, "m.py::in_range", cats, tests, func_obj
            )
            iso = run_function_profiling(
                node, "m.py::in_range", cats, tests, func_obj, isolated=True
            )
        finally:
            _PROJECT_ROOT.reset(token)
    finally:
        _drop_project(tmp_path)

    assert inproc.execution_mode == "in_process"
    assert iso.execution_mode == "isolated"
    # A real suite kills something here, or the comparison is vacuous.
    assert inproc.total_killed > 0
    # The bottom line agrees — the crossing neither lost nor invented a verdict.
    assert (iso.total_mutants, iso.total_killed, iso.total_survived) == (
        inproc.total_mutants,
        inproc.total_killed,
        inproc.total_survived,
    )
    # And the value-specification split (assertion pins vs run-only kills) survives isolation.
    assert iso.value_killed == inproc.value_killed


def test_isolated_profiling_is_gateable_on_a_clean_run(tmp_path):
    """The whole reason the isolated mode exists: a contained, budgeted, exhaustive run through it is
    gateable — the containment is a real SIGKILL guarantee, not a best-effort thread abandon."""
    from Wesker.ci import _PROJECT_ROOT
    from Wesker.engine import run_function_profiling
    from Wesker.filter import filter_categories

    node, func_obj, tests = _real_project(tmp_path)
    try:
        token = _PROJECT_ROOT.set(str(tmp_path))
        try:
            iso = run_function_profiling(
                node,
                "m.py::in_range",
                filter_categories(node),
                tests,
                func_obj,
                isolated=True,
            )
        finally:
            _PROJECT_ROOT.reset(token)
    finally:
        _drop_project(tmp_path)

    assert iso.coverage_depth == "profiled"
    assert iso.is_gateable is True


# ── mode -> gateability standing (increment 4c) ──────────────────────────────────
#
# The tier a result earns from its execution mode, layered on top of the measurement validity.
# Informational — it does NOT change is_gateable, so no in-process certificate is downgraded before
# the increment-5 shape check gives "conditional" its teeth.


def test_execution_mode_standing_is_the_intended_tiering():
    from Wesker.isolation import execution_mode_standing

    # isolated + a valid measurement is fully gateable — real SIGKILL containment.
    assert execution_mode_standing("isolated", True) == "gateable"
    # in_process + valid is only CONDITIONAL — best-effort containment, pending the shape check.
    assert execution_mode_standing("in_process", True) == "conditional"
    # an invalid measurement is cut regardless of mode — the mode cannot rescue it.
    assert execution_mode_standing("isolated", False) == "cut"
    assert execution_mode_standing("in_process", False) == "cut"


def test_a_profiling_result_reports_its_standing_without_changing_gateability():
    """The property wires the pure decision to a real result, and — the load-bearing invariant — an
    in_process result stays is_gateable=True while its standing reads 'conditional': the tier is
    surfaced, acceptance is untouched.

    Built by DIRECT construction, not a profiling run. The standing is pure over two fields
    (execution_mode, is_gateable), so a full profile would only add a subprocess-spawning test to
    this pure decision's covering set — the codebase-scale category error that makes `converge` on
    it trace the whole suite and hang. The A/B soundness of the profiling path is proved separately
    above; here the unit is the property alone."""
    from Wesker.engine import ProfilingResult

    iso = ProfilingResult(execution_mode="isolated", is_gateable=True)
    assert iso.execution_standing == "gateable"
    assert iso.to_dict()["execution_standing"] == "gateable"
    # The invariant: in_process is surfaced as conditional but its gateability is NOT downgraded.
    inproc = ProfilingResult(execution_mode="in_process", is_gateable=True)
    assert inproc.execution_standing == "conditional"
    assert inproc.is_gateable is True
    # And an invalid measurement is cut regardless of mode.
    assert (
        ProfilingResult(execution_mode="isolated", is_gateable=False).execution_standing
        == "cut"
    )


# ── fast-mode shape refusal + baseline determinism (increment 5a) ────────────────
#
# The two proof-facing gates that give the in_process fast mode its limits. Pure decisions,
# pinned in isolation before any wiring — the shape detector and the repeated-baseline harness
# (5b/5c) will supply their inputs.


def test_a_hermetic_shape_clears_the_fast_mode():
    from Wesker.isolation import fast_mode_standing

    assert fast_mode_standing(False, False, False, False, False) == "hermetic"


def test_each_hazardous_shape_is_refused_by_name():
    """A shape in_process containment cannot honestly measure is refused to the isolated mode, and
    NAMED so a consumer can say which hazard refused it."""
    from Wesker.isolation import fast_mode_standing

    assert fast_mode_standing(True, False, False, False, False) == "refuse_subprocess"
    assert fast_mode_standing(False, True, False, False, False) == "refuse_thread"
    assert fast_mode_standing(False, False, True, False, False) == "refuse_collector"
    assert fast_mode_standing(False, False, False, True, False) == "refuse_signal"
    assert fast_mode_standing(False, False, False, False, True) == "refuse_fixture"


def test_the_loudest_hazard_wins_and_over_refusal_is_safe():
    """When several hazards coincide, one refusal is reported (precedence = the issue's listing);
    the point is that ANY hazard refuses, routing the test to the always-sound isolated mode."""
    from Wesker.isolation import fast_mode_standing

    assert fast_mode_standing(True, True, True, True, True) == "refuse_subprocess"
    assert fast_mode_standing(False, True, False, True, False) == "refuse_thread"


def test_matched_fresh_baselines_are_deterministic():
    from Wesker.isolation import baseline_determinism

    assert (
        baseline_determinism([1, 2, 3], "passed", [1, 2, 3], "passed")
        == "deterministic"
    )
    # Coverage is a SET: order and repeats from the tracer are not signal.
    assert (
        baseline_determinism([3, 1, 2, 2], "passed", [1, 2, 3], "passed")
        == "deterministic"
    )


def test_a_flipped_outcome_or_shifted_coverage_is_nondeterministic():
    """Either a changed pass/fail OR a changed covered-line set across two fresh baselines means the
    function cannot ground a gateable verdict."""
    from Wesker.isolation import baseline_determinism

    assert (
        baseline_determinism([1, 2], "passed", [1, 2], "failed") == "nondeterministic"
    )
    assert (
        baseline_determinism([1, 2], "passed", [1, 2, 3], "passed")
        == "nondeterministic"
    )
