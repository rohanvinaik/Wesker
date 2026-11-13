"""#27/#40/#42/#45: execution boundaries survive the real CLI and aggregation.

These engine/worker boundaries take live callables and cannot be literal-pinned;
the tests exercise the production adapters and adversarial execution outcomes.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types

import pytest

from Wesker import engine
from Wesker.isolation import IsolatedRun


def _case():
    node = ast.parse("def target(x):\n    return x + 1\n").body[0]
    mutant = engine.generate_mutants(node, {engine.MutationCategory.VALUE})[0]
    return node, mutant


@pytest.mark.parametrize("isolated", [False, True])
def test_real_cli_profiles_a_function(tmp_path, isolated):
    (tmp_path / "calc.py").write_text("def scale(x):\n    return x * 2\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import scale\ndef test_scale():\n    assert scale(2) == 4\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\ntestpaths = ["."]\n'
    )
    argv = [sys.executable, "-m", "Wesker", "calc.py", "--json", "--budget", "30000"]
    if isolated:
        argv += ["--isolated", "--mem-limit-mb", "2048"]
    proc = subprocess.run(
        argv,
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)["mutation"]
    assert result["total_functions"] == 1
    assert result["total_mutants"] > 0
    assert result["total_killed"] > 0


@pytest.mark.parametrize("contained", [False, True])
def test_isolated_timeout_is_not_detection(contained):
    _, mutant = _case()
    result = engine._isolated_result(
        mutant,
        IsolatedRun(1, True, contained, "", installed=True, ran=1, entered_probe=True),
        1.0,
    )
    assert not result.killed
    assert result.killed_by == "undetermined"
    assert (
        engine.mutant_disposition(
            result.constructed,
            result.installed,
            result.entered,
            result.contained,
            result.killed,
            result.refusal,
            killed_by=result.killed_by,
        )
        == "undetermined"
    )


@pytest.mark.parametrize("outcome", ["diverged", "undetermined", "timeout"])
@pytest.mark.parametrize(
    "runner", [engine.run_function_converged, engine.run_function_profiling]
)
def test_converged_unresolved_outcome_keeps_identity_and_denominators(
    monkeypatch, outcome, runner
):
    node, mutant = _case()
    monkeypatch.setattr(engine, "generate_mutants", lambda *a, **k: [mutant])
    monkeypatch.setattr(
        engine,
        "evaluate_mutant",
        lambda *a, **k: engine.MutantResult(
            mutant=mutant,
            killed=False,
            killed_by=outcome,
            entered=True,
            test_name="test_target",
        ),
    )

    def forbidden_probe(*args):
        raise AssertionError("unresolved execution is not an equivalence candidate")

    monkeypatch.setattr(engine, "check_equivalent", forbidden_probe)
    result = runner(
        node,
        "calc.py::target",
        {engine.MutationCategory.VALUE},
        [],
        None,
        scope_tests=False,
        budget_ms=10000,
    )
    assert result.total_mutants == result.total_killed == result.dof_covered == 0
    assert not result.is_gateable
    # A per-mutant outcome was evaluated; it is never a run-level cut (Detective #80). It stays out
    # of the gateable verdict through its unscored record, asserted above and below. The converged
    # path reads `sampled` here only because generation is patched to one mutant of a larger universe.
    assert result.coverage_depth != "cut"
    assert result.survivor_records == []
    assert result.unscored_records[0]["disposition"] == (
        "undetermined" if outcome == "timeout" else outcome
    )
    assert result.unscored_records[0]["killed_by"] == outcome


def test_finite_test_has_no_unmeasured_5000_iteration_ceiling():
    def finite():
        total = 0
        for i in range(12000):
            total += i
        assert total == 71994000

    assert engine._run_test_with_timeout(finite, finite, True, 2000) is None


@pytest.mark.parametrize("full_matrix", [False, True])
def test_real_crash_outranks_an_earlier_budget_exhaustion(monkeypatch, full_matrix):
    _, mutant = _case()
    outcomes = iter(["diverged", "crash"])
    monkeypatch.setattr(engine, "_run_test_with_timeout", lambda *a: next(outcomes))
    result = engine.evaluate_mutant(
        mutant,
        [lambda: None, lambda: None],
        None,
        record_all_killers=full_matrix,
    )
    assert result.killed
    assert result.killed_by == "crash"
    if full_matrix:
        assert len(result.killed_by_tests) == 1


def test_live_globals_are_read_after_same_size_rebinding():
    namespace = {"__name__": "live", "__package__": "", "limit": 3}
    # S102: a live function namespace is the object whose rebinding is under test.
    exec("def target():\n    return limit\n", namespace)  # noqa: S102
    before = engine._prepare_base_globals(namespace["target"])
    size = len(namespace)
    namespace["limit"] = 9
    after = engine._prepare_base_globals(namespace["target"])
    assert len(namespace) == size
    assert before["limit"] == 3
    assert after["limit"] == 9
    assert after["__package__"] == ""


def test_owner_discovery_observes_same_size_module_replacement(monkeypatch, tmp_path):
    path = tmp_path / "calc.py"
    source = "def target(x):\n    return x + 1\n"
    path.write_text(source)
    namespace = {}
    # S102: module binding discovery must use a real code object's source filename.
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
    first = types.ModuleType("phase2_owner")
    first.target = namespace["target"]
    monkeypatch.setitem(sys.modules, "phase2_owner", first)
    assert (first, False) in engine._precompute_patch_targets("target", str(path))
    size = len(sys.modules)
    second = types.ModuleType("phase2_owner")
    second.target = namespace["target"]
    monkeypatch.setitem(sys.modules, "phase2_owner", second)
    assert len(sys.modules) == size
    with engine._execution_guard():
        saved = engine._patch_module_qualified(
            engine._held_patch_proof(), "target", lambda x: 99, str(path)
        )
        try:
            assert second.target(1) == 99
            assert first.target(1) == 2
        finally:
            for owner, original in saved:
                owner.target = original


def test_converged_closes_worker_when_later_progress_interrupts(monkeypatch):
    node, mutant = _case()

    class Worker:
        closed = False

        def close(self):
            self.closed = True

    worker = Worker()
    monkeypatch.setattr(engine, "generate_mutants", lambda *a, **k: [mutant, mutant])
    calls = 0

    def evaluate(*args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted worker evaluation")
        return (
            engine.MutantResult(
                mutant=mutant, killed=True, killed_by="assertion", entered=True
            ),
            worker,
            None,
        )

    # Distinct ids ensure that both evaluations run.
    from dataclasses import replace

    other = replace(mutant, mutant_id=mutant.mutant_id + "_other")
    monkeypatch.setattr(engine, "generate_mutants", lambda *a, **k: [mutant, other])
    monkeypatch.setattr(engine, "_evaluate_isolated", evaluate)
    with pytest.raises(RuntimeError, match="interrupted worker"):
        engine.run_function_converged(
            node,
            "calc.py::target",
            {engine.MutationCategory.VALUE},
            [],
            None,
            scope_tests=False,
            isolated=True,
            budget_ms=10000,
        )
    assert worker.closed
