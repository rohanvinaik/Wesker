"""Issue #15 intent: cached reach may route only under the exact execution context.

The false-gap hazard is a cached ``not_reached`` surviving a test helper, fixture/conftest, target,
or pytest-regime edit. Every such edit must turn the item back into UNKNOWN; only an exact prior
observation may become ``impossible_observed``.
"""

from __future__ import annotations

import importlib.util
import time

import pytest

import Wesker.engine as engine
from Wesker.ci import callable_test_id
from Wesker.engine import build_session_baseline
from Wesker.trace_cache import (
    observed_function_reach,
    save,
    targets_fingerprint,
    test_fingerprint as _test_fingerprint,
)


def _call(origin, nodeid, fixture=()):
    def run():
        return None

    run.__qualname__ = nodeid
    run.__wesker_origin__ = str(origin)
    run.__wesker_fixture_origins__ = tuple(str(p) for p in fixture)
    return run


def _seed_cache(
    root, target, tests, lines_by_test, regime="regime-a", outcomes_observed=True
):
    budgets = (50.0, 300.0)
    entries = {}
    for test in tests:
        entries[_test_fingerprint(test)] = {
            str(target): {"lines": list(lines_by_test[test]), "arcs": []}
        }
    save(
        str(root),
        targets_fingerprint({str(target)}),
        budgets,
        entries,
        [],
        [],
        regime,
        [callable_test_id(test) for test in tests] if outcomes_observed else [],
        (
            {callable_test_id(test): _test_fingerprint(test) for test in tests}
            if outcomes_observed
            else {}
        ),
    )
    return budgets


def test_exact_cells_become_per_test_reached_or_not_reached(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("def target():\n    return 1\n")
    test_file = tmp_path / "test_mod.py"
    test_file.write_text("def test_one(): pass\ndef test_two(): pass\n")
    reached = _call(test_file, "test_mod.py::test_one")
    missed = _call(test_file, "test_mod.py::test_two")
    budgets = _seed_cache(
        tmp_path, target, [reached, missed], {reached: {2}, missed: set()}
    )

    got = observed_function_reach(
        str(tmp_path),
        {str(target)},
        budgets,
        "regime-a",
        [reached, missed],
        str(target),
        {2},
    )

    assert got == {
        callable_test_id(reached): "reached",
        callable_test_id(missed): "not_reached",
    }


def test_test_module_conftest_fixture_or_regime_edit_voids_nonreach(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("def target():\n    return 1\n")
    test_file = tmp_path / "test_mod.py"
    test_file.write_text("def helper(): return 1\ndef test_one(): helper()\n")
    conftest = tmp_path / "conftest.py"
    conftest.write_text("def helper_fixture(): return 1\n")
    fixture_file = tmp_path / "fixture_plugin.py"
    fixture_file.write_text("def fixture_value(): return 1\n")
    test = _call(test_file, "test_mod.py::test_one", [fixture_file])
    budgets = _seed_cache(tmp_path, target, [test], {test: set()})

    def observed(regime="regime-a"):
        return observed_function_reach(
            str(tmp_path),
            {str(target)},
            budgets,
            regime,
            [test],
            str(target),
            {2},
        )

    assert observed() == {callable_test_id(test): "not_reached"}
    test_file.write_text("def helper(): return target()\ndef test_one(): helper()\n")
    assert observed() == {}

    # Re-seed under the edited module, then independently invalidate both indirect contexts.
    budgets = _seed_cache(tmp_path, target, [test], {test: set()})
    conftest.write_text("def helper_fixture(): return target()\n")
    assert observed() == {}
    budgets = _seed_cache(tmp_path, target, [test], {test: set()})
    fixture_file.write_text("def fixture_value(): return target()\n")
    assert observed() == {}

    budgets = _seed_cache(tmp_path, target, [test], {test: set()})
    assert observed("regime-b") == {}


def test_fresh_partial_observations_merge_and_persist_for_sibling_routing(tmp_path):
    """A function's fresh seed/widen is reusable as routing—not proof—for the next function."""
    target = tmp_path / "mod.py"
    target.write_text("def target():\n    return 1\n")
    spec = importlib.util.spec_from_file_location("routing_cache_uut", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def reaches():
        assert module.target() == 1

    def misses():
        assert 1 + 1 == 2

    budgets = (50.0, 300.0)
    kwargs = {
        "target_files": {str(target)},
        "project_root": str(tmp_path),
        "fresh": True,
        "regime_digest": "regime-a",
        "trace_budget_s": budgets[0],
        "trace_session_budget_s": budgets[1],
    }
    build_session_baseline([reaches], **kwargs)
    build_session_baseline([misses], **kwargs)

    got = observed_function_reach(
        str(tmp_path),
        {str(target)},
        budgets,
        "regime-a",
        [reaches, misses],
        str(target),
        {2},
    )
    assert got == {
        callable_test_id(reaches): "reached",
        callable_test_id(misses): "not_reached",
    }


def test_one_cut_item_does_not_erase_other_complete_routing_cells(tmp_path):
    """A cut TestId stays unknown; its complete siblings remain reusable observations (#15)."""
    target = tmp_path / "mod.py"
    target.write_text("def target():\n    return 1\n")

    def fast_miss():
        assert 1 + 1 == 2

    def slow_cut():
        time.sleep(0.1)

    budgets = (0.02, 300.0)
    baseline = build_session_baseline(
        [fast_miss, slow_cut],
        {str(target)},
        project_root=str(tmp_path),
        fresh=True,
        regime_digest="regime-a",
        trace_budget_s=budgets[0],
        trace_session_budget_s=budgets[1],
    )
    assert callable_test_id(slow_cut) in baseline.truncated

    got = observed_function_reach(
        str(tmp_path),
        {str(target)},
        budgets,
        "regime-a",
        [fast_miss, slow_cut],
        str(target),
        {2},
    )
    assert got == {callable_test_id(fast_miss): "not_reached"}


def test_trace_checkpoint_survives_interruption_before_outcome_pass(
    tmp_path, monkeypatch
):
    """Paid reach survives Ctrl-C; its missing outcome is not fabricated as green (#15/#17)."""
    target = tmp_path / "mod.py"
    target.write_text("def target():\n    return 1\n")

    def miss():
        assert True

    test_context = tmp_path / "test_mod.py"
    test_context.write_text("def helper(): return 1\n")
    miss.__wesker_origin__ = str(test_context)

    budgets = (50.0, 300.0)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "_run_test_with_timeout", interrupt)
    with pytest.raises(KeyboardInterrupt):
        build_session_baseline(
            [miss],
            {str(target)},
            project_root=str(tmp_path),
            fresh=True,
            regime_digest="regime-a",
            trace_budget_s=budgets[0],
            trace_session_budget_s=budgets[1],
        )

    assert (
        observed_function_reach(
            str(tmp_path),
            {str(target)},
            budgets,
            "regime-a",
            [miss],
            str(target),
            {2},
        )
        == {}
    ), "a checkpointed trace without an outcome is not negative proof"

    calls = 0

    def observe_outcome(*args, **kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(engine, "_run_test_with_timeout", observe_outcome)
    build_session_baseline(
        [miss],
        {str(target)},
        project_root=str(tmp_path),
        regime_digest="regime-a",
        trace_budget_s=budgets[0],
        trace_session_budget_s=budgets[1],
    )
    assert calls == 1, "checkpointed reach without an outcome must re-run qualification"
    assert observed_function_reach(
        str(tmp_path),
        {str(target)},
        budgets,
        "regime-a",
        [miss],
        str(target),
        {2},
    ) == {callable_test_id(miss): "not_reached"}

    test_context.write_text("def helper(): return 2\n")
    calls = 0
    build_session_baseline(
        [miss],
        {str(target)},
        project_root=str(tmp_path),
        regime_digest="regime-a",
        trace_budget_s=budgets[0],
        trace_session_budget_s=budgets[1],
    )
    assert calls == 1, "an edited TestId must not reuse its prior outcome"
