"""A replayed cache miss is never an exclusion — observed_function_reach is positive-only (X1/G1).

TEST_BASIS §15.1 / §16 X1 (Part V gap ledger, G1 — soundness, a false COMPLETE). `test_fingerprint`
keys the test's own source, fixture origins, and ancestor conftests, but NOT the modules the test
plainly imports. So a helper edit that opens a new path to the target leaves the fingerprint
byte-identical (reproduced), and a stale cached non-reach could be replayed to EXCLUDE a test that
now reaches the target — manufacturing a false COMPLETE. The ruling (demote at the producer):
`observed_function_reach` emits only positives; a non-intersecting hit is left absent (UNKNOWN) so
routing re-traces it fresh. This mirrors the already-pinned `basis_membership` rule (replayed
non-reach → "pending", never "disjoint").
"""

from __future__ import annotations

import os

from Wesker import trace_cache
from Wesker.ci import callable_test_id
from Wesker.trace_cache import observed_function_reach


def _t():  # a plain test callable; underscore-prefixed so pytest does not collect it
    assert True


def _run(tmp_path, monkeypatch, covered_lines):
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n")
    target_real = os.path.realpath(str(target))
    fp = trace_cache.test_fingerprint(_t)
    fake_cache = {fp: {target_real: {"lines": list(covered_lines)}}}
    monkeypatch.setattr(trace_cache, "load", lambda *a, **k: fake_cache)
    return observed_function_reach(
        str(tmp_path),
        {target_real},
        (50.0, 300.0),
        "a-regime-digest",
        [_t],
        str(target),
        {1, 2},  # the target's executable lines
    )


def test_a_cached_miss_is_not_emitted_as_not_reached(tmp_path, monkeypatch):
    # The cached hit covers only line 999 — outside the target's executable set {1, 2}: a non-reach.
    out = _run(tmp_path, monkeypatch, covered_lines=[999])
    assert out == {}, (
        "a possibly-stale cached miss must NOT be replayed as an exclusion (G1)"
    )


def test_a_cached_hit_that_intersects_is_still_reached(tmp_path, monkeypatch):
    # An intersecting hit is still positive routing evidence — the demote only drops negatives.
    out = _run(tmp_path, monkeypatch, covered_lines=[1])
    assert out == {callable_test_id(_t): "reached"}
