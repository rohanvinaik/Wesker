"""load_outcomes validates the shared cache by CONSTRUCTION, exactly as load does (#D4 repair 5, §4.6).

Before: `load_outcomes` re-read the same cache file `load` reads, but with NO
version/engine/targets/budgets/regime check. So the reach view (`load`) could refuse a stale or
wrong-regime file while the outcome view served it — two views of ONE file disagreeing on whether it
is fresh. The invariant held only by caller discipline (each caller ran `load` first). Now both share
`_load_valid_blob`, so a caller that skips `load` cannot be handed a stale outcome row.
"""

from __future__ import annotations

from Wesker import trace_cache

_BUDGETS = (50.0, 1800.0)


def _saved(root) -> None:
    trace_cache.save(
        str(root),
        "tfp-A",
        _BUDGETS,
        entries={"fp1": {"file.py": {"lines": [1]}}},
        failing=["t_fail"],
        inert_names=["t_inert"],
        regime_digest="regime-A",
        outcomes_observed=["t_out"],
        outcome_fingerprints={"t_out": "fp"},
    )


def test_matching_keys_return_the_outcomes(tmp_path):
    _saved(tmp_path)
    failing, inert, outcomes, fps = trace_cache.load_outcomes(
        str(tmp_path), "tfp-A", _BUDGETS, "regime-A"
    )
    assert failing == ["t_fail"]
    assert inert == ["t_inert"]
    assert outcomes == ["t_out"]
    assert fps == {"t_out": "fp"}


def test_a_wrong_regime_returns_empties_not_a_stale_row(tmp_path):
    _saved(tmp_path)
    assert trace_cache.load_outcomes(str(tmp_path), "tfp-A", _BUDGETS, "regime-B") == (
        [],
        [],
        [],
        {},
    )


def test_wrong_targets_or_budgets_also_refuse(tmp_path):
    _saved(tmp_path)
    assert trace_cache.load_outcomes(
        str(tmp_path), "tfp-OTHER", _BUDGETS, "regime-A"
    ) == (
        [],
        [],
        [],
        {},
    )
    assert trace_cache.load_outcomes(
        str(tmp_path), "tfp-A", (1.0, 2.0), "regime-A"
    ) == (
        [],
        [],
        [],
        {},
    )


def test_load_and_load_outcomes_agree_on_freshness(tmp_path):
    """The whole point: the reach view and the outcome view of ONE file never disagree on fresh."""
    _saved(tmp_path)
    for regime, fresh in (("regime-A", True), ("regime-B", False)):
        reach = trace_cache.load(str(tmp_path), "tfp-A", _BUDGETS, regime)
        outcomes = trace_cache.load_outcomes(str(tmp_path), "tfp-A", _BUDGETS, regime)
        assert bool(reach) == fresh
        assert (outcomes != ([], [], [], {})) == fresh
