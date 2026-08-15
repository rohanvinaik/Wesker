"""Issue #15 intent: a cached observation may ROUTE only as a positive, under the exact context.

X1/G1 (TEST_BASIS §15.1/§16) demoted the replayed NEGATIVE: `test_fingerprint` cannot certify that
the modules a test imports are unchanged, so a cached non-reach may be stale and must NEVER become
an exclusion. `observed_function_reach` is therefore POSITIVE-ONLY — a cached hit that intersects the
target routes as "reached" (a re-traced seed hint), and a cached MISS is left absent (UNKNOWN) so
routing re-traces it fresh. What still matters, and is pinned here: a stale POSITIVE must not survive
a helper/conftest/fixture/target/regime edit (the fingerprint/key voids it), and a cut item must not
erase a complete sibling's cell.
"""

from __future__ import annotations

import importlib.util
import time

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


def _seed_cache(root, target, tests, lines_by_test, regime="regime-a"):
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
        [callable_test_id(test) for test in tests],
        {callable_test_id(test): _test_fingerprint(test) for test in tests},
    )
    return budgets


def test_an_intersecting_cell_is_reached_and_a_miss_is_absent(tmp_path):
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

    # Positive-only: the reacher routes; the miss is UNKNOWN (absent), never "not_reached" (G1).
    assert got == {callable_test_id(reached): "reached"}


def test_a_cached_positive_is_voided_by_a_helper_conftest_fixture_or_regime_edit(
    tmp_path,
):
    target = tmp_path / "mod.py"
    target.write_text("def target():\n    return 1\n")
    test_file = tmp_path / "test_mod.py"
    test_file.write_text("def helper(): return 1\ndef test_one(): helper()\n")
    conftest = tmp_path / "conftest.py"
    conftest.write_text("def helper_fixture(): return 1\n")
    fixture_file = tmp_path / "fixture_plugin.py"
    fixture_file.write_text("def fixture_value(): return 1\n")
    test = _call(test_file, "test_mod.py::test_one", [fixture_file])
    budgets = _seed_cache(tmp_path, target, [test], {test: {2}})

    def observed(regime="regime-a"):
        return observed_function_reach(
            str(tmp_path), {str(target)}, budgets, regime, [test], str(target), {2}
        )

    assert observed() == {callable_test_id(test): "reached"}
    # Each edit changes the fingerprint (the item's own origin file, an ancestor conftest, or a
    # fixture-origin file — all hashed into the key) or the regime digest, so the cached POSITIVE is
    # voided back to UNKNOWN and re-traced rather than served stale.
    test_file.write_text("def helper(): return 2\ndef test_one(): helper()\n")
    assert observed() == {}
    budgets = _seed_cache(tmp_path, target, [test], {test: {2}})
    conftest.write_text("def helper_fixture(): return 2\n")
    assert observed() == {}
    budgets = _seed_cache(tmp_path, target, [test], {test: {2}})
    fixture_file.write_text("def fixture_value(): return 2\n")
    assert observed() == {}
    budgets = _seed_cache(tmp_path, target, [test], {test: {2}})
    assert observed("regime-b") == {}


def test_fresh_positive_observations_merge_and_persist_for_sibling_routing(tmp_path):
    """A function's fresh seed/widen positive is reusable as routing—not proof—for the next."""
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
    # The reacher persists as positive routing; the miss is absent (UNKNOWN), never an exclusion.
    assert got == {callable_test_id(reaches): "reached"}


def test_one_cut_item_does_not_erase_another_complete_positive_cell(tmp_path):
    """A cut TestId stays unknown; a complete sibling's positive cell remains reusable (#15)."""
    target = tmp_path / "mod.py"
    target.write_text("def target():\n    return 1\n")
    spec = importlib.util.spec_from_file_location("routing_cut_uut", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fast_reacher():
        assert module.target() == 1

    def slow_cut():
        time.sleep(0.1)

    budgets = (0.02, 300.0)
    baseline = build_session_baseline(
        [fast_reacher, slow_cut],
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
        [fast_reacher, slow_cut],
        str(target),
        {2},
    )
    # The cut sibling contributes nothing; the reacher's complete positive cell survives.
    assert got == {callable_test_id(fast_reacher): "reached"}
