"""Tests for the INCREMENTAL session-baseline refresh (`SessionBaseline.replaced`,
`LazySessionBaseline.refresh`, and the splice `refresh_live_suite` drives).

Writing one test file used to invalidate the whole baseline, so the next read re-traced the
entire suite to learn what one file changed — `O(passes x suite)` for a consumer that writes
tests in a loop. These pin the two properties that make replacing that with a splice safe:

  * it produces the SAME baseline a full rebuild would (else the speedup buys a wrong verdict);
  * a name the written file SHARES with another module does not take that module's coverage
    with it — the failure that matters, because a test whose coverage is absent is one the
    mutation loop never runs, so every mutant it kills reads as a surviving behavioral gap.
"""

from __future__ import annotations

import os
import textwrap

import Wesker.ci as ci
from Wesker.engine import (
    _SESSION_BASELINE,
    LazySessionBaseline,
    SessionBaseline,
)


def _bl(traced, failing=(), inert=(), n_tests=0, truncated=()):
    return SessionBaseline(
        dict(traced), list(failing), set(inert), n_tests, set(truncated)
    )


# ── SessionBaseline.replaced: the splice itself ──────────────────


def test_replaced_keeps_unaffected_tests_and_takes_the_partial_for_affected():
    base = _bl({"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}}, n_tests=2)
    partial = _bl({"test_b": {"f.py": {9}}}, n_tests=1)

    out = base.replaced({"test_b"}, set(), partial, n_tests=2)

    assert out.traced["test_a"] == {"f.py": {1}}  # untouched test survives verbatim
    assert out.traced["test_b"] == {
        "f.py": {9}
    }  # affected test takes the NEW measurement
    assert out.n_tests == 2


def test_replaced_drops_an_affected_test_the_partial_no_longer_reports():
    # The file was rewritten and no longer defines test_b: its entry must GO, not linger.
    # A lingering entry claims coverage for a test that does not exist, which reads as
    # specified behaviour that nothing pins.
    base = _bl({"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}}, n_tests=2)
    out = base.replaced({"test_b"}, set(), _bl({}, n_tests=0), n_tests=1)
    assert "test_b" not in out.traced
    assert out.traced["test_a"] == {"f.py": {1}}


def test_replaced_purges_removed_ids_from_inert_but_keeps_live_ones():
    # `inert` is keyed by id(); an id whose object was freed can be REUSED by a later
    # allocation, and a stale entry would bar an unrelated test from kill attribution.
    base = _bl({}, inert={111, 222}, n_tests=2)
    out = base.replaced(set(), {111}, _bl({}, inert={333}), n_tests=2)
    assert out.inert == {222, 333}  # 111 removed, 222 (still live) kept, 333 added


def test_replaced_splices_failing_and_truncated_by_name():
    base = _bl(
        {}, failing=["test_a", "test_b"], truncated={"test_a", "test_b"}, n_tests=2
    )
    partial = _bl({}, failing=["test_b"], truncated={"test_b"})
    out = base.replaced({"test_b"}, set(), partial, n_tests=2)
    assert out.failing == ["test_a", "test_b"]  # a kept, b re-derived (not duplicated)
    assert out.truncated == {"test_a", "test_b"}


def test_replaced_clears_a_stale_failing_flag_the_rewrite_fixed():
    # The written file's earlier version assert-failed; the new one passes. If the name were
    # not dropped first, converge would keep reporting a wrong-expectation it already fixed.
    base = _bl({}, failing=["test_b"], truncated={"test_b"}, n_tests=1)
    out = base.replaced(
        {"test_b"}, set(), _bl({}, failing=[], truncated=set()), n_tests=1
    )
    assert out.failing == []
    assert out.truncated == set()


# ── LazySessionBaseline.refresh: laziness and the safe degrade ───


def test_refresh_does_not_force_a_build_that_never_happened():
    # An unbuilt baseline is not stale: the lazy build already reads the CURRENT suite.
    # Forcing a trace to service a write is the eager cost the laziness exists to defer.
    calls = []
    holder = LazySessionBaseline(lambda subset=None: calls.append(subset) or _bl({}))
    assert holder.refresh({"test_b"}, set(), [], 0) is False
    assert calls == []
    assert holder.built is False


def test_refresh_splices_into_a_built_baseline():
    def build(subset=None, fresh=False):
        return (
            _bl({"test_b": {"f.py": {9}}}, n_tests=1)
            if subset
            else _bl({"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}}, n_tests=2)
        )

    holder = LazySessionBaseline(build)
    holder.get()  # force the full build
    assert holder.refresh({"test_b"}, set(), [lambda: None], 2) is True
    assert holder.get().traced == {"test_a": {"f.py": {1}}, "test_b": {"f.py": {9}}}


def test_refresh_degrades_to_invalidate_when_the_partial_build_raises():
    # A partial build RUNS the consumer's test code and can fail in ways this module cannot
    # enumerate. A half-spliced baseline would under-report coverage -> false survivors, so
    # the fast path must be skippable, never wrong: drop the value and let the next read
    # re-trace, which is exactly what invalidation did unconditionally.
    state = {"full": 0}

    def build(subset=None, fresh=False):
        if subset is not None:
            raise RuntimeError("the consumer's test blew up mid-trace")
        state["full"] += 1
        return _bl({"test_a": {"f.py": {1}}}, n_tests=1)

    holder = LazySessionBaseline(build)
    holder.get()
    assert holder.refresh({"test_a"}, set(), [lambda: None], 1) is False
    assert holder.built is False  # value dropped, not left half-spliced
    holder.get()
    assert (
        state["full"] == 2
    )  # the next read paid a full rebuild — correct, just slower


# ── The property the whole change rests on: same answer as a rebuild ──


def test_refresh_equals_a_full_rebuild():
    suite = {
        "test_a": {"f.py": {1}},
        "test_b": {"f.py": {2}},
        "test_new": {"f.py": {3}},
    }

    def build(subset=None, fresh=False):
        names = subset if subset is not None else list(suite)
        return _bl({n: suite[n] for n in names if n in suite}, n_tests=len(names))

    holder = LazySessionBaseline(build)
    holder.get()
    holder.refresh({"test_new"}, set(), ["test_new"], len(suite))
    spliced = holder.get()

    fresh = LazySessionBaseline(build)
    rebuilt = fresh.get()

    assert spliced.traced == rebuilt.traced
    assert spliced.n_tests == rebuilt.n_tests


# ── The collision, end to end through refresh_live_suite ─────────


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_refreshing_one_file_keeps_a_same_named_test_in_another_module(
    tmp_path, monkeypatch
):
    """A same-named test in ANOTHER module must not lose its coverage when one file is written.

    That loss is what the `__name__` union existed to prevent: dropping a shared name took the
    other owner's coverage with it, and every mutant that test killed then read as a survivor.
    Since issue #16 `traced` is keyed per ITEM, so the two owners occupy different entries and
    the written file's splice cannot reach the other one at all.

    THE INVARIANT IS UNCHANGED and is still what this asserts. What changed is that it now holds
    STRUCTURALLY rather than being bought by re-tracing every current owner of the name — so the
    second assertion below is the inverse of the one it replaces: the other owner must NOT be
    re-traced, because there is no longer anything to compensate for.
    """
    target = _write(tmp_path, "test_written.py", "def test_shared():\n    pass\n")

    # A live callable from a DIFFERENT module that happens to share the name.
    def test_shared():  # noqa: D103 — stands in for the other module's test
        pass

    other = _write(tmp_path, "test_other.py", "def test_shared():\n    pass\n")
    test_shared.__wesker_origin__ = other
    other_id = ci.callable_test_id(test_shared)

    traced_with: list[list[str]] = []

    def build(subset=None, fresh=False):
        ids = [ci.callable_test_id(c) for c in (subset or [])]
        traced_with.append(ids)
        return _bl({i: {"f.py": {1}} for i in ids}, n_tests=len(ids))

    holder = LazySessionBaseline(build)
    holder._value = _bl({other_id: {"f.py": {1}}}, n_tests=2)
    holder._built = True

    suite_token = ci._LIVE_SUITE.set([test_shared])
    base_token = _SESSION_BASELINE.set(holder)
    try:
        ci.refresh_live_suite(str(tmp_path), target)
    finally:
        ci._LIVE_SUITE.reset(suite_token)
        _SESSION_BASELINE.reset(base_token)

    assert traced_with, "the splice never ran"
    # THE INVARIANT: the other module's entry survives the write untouched.
    assert other_id in holder.get().traced
    # AND it cost nothing to keep. Distinct ids mean `affected` never contained the other
    # owner, so the splice could not reach it — where the union had to re-trace every current
    # owner of the name to protect exactly this entry.
    assert other_id not in traced_with[-1], (
        f"the other owner must not need re-tracing, got {traced_with[-1]}"
    )


def test_refresh_live_suite_is_a_noop_with_no_live_session(tmp_path):
    # The non-live path re-collects on every call and has nothing to invalidate.
    assert ci.refresh_live_suite(str(tmp_path), str(tmp_path / "test_x.py")) == 0


def test_refresh_live_suite_replaces_only_the_written_files_callables(tmp_path):
    target = _write(tmp_path, "test_w.py", "def test_one():\n    pass\n")

    def kept_test():
        pass

    kept_test.__wesker_origin__ = os.path.join(str(tmp_path), "test_kept.py")

    suite_token = ci._LIVE_SUITE.set([kept_test])
    try:
        ci.refresh_live_suite(str(tmp_path), target)
        live = ci._LIVE_SUITE.get()
    finally:
        ci._LIVE_SUITE.reset(suite_token)

    assert kept_test in live  # the other file's callable is untouched


# ── LazySessionBaseline.seed / expand: target-first incremental baseline (Fix B) ──


def _suite_build(suite):
    """A build closure over a name->coverage dict, treating `subset` as a name list (mirrors the
    real closure, which traces exactly the callables it is handed)."""

    def build(subset=None, fresh=False):
        names = list(suite) if subset is None else list(subset)
        return _bl({n: suite[n] for n in names if n in suite}, n_tests=len(names))

    return build


def test_seed_measures_only_the_candidate_subset():
    suite = {"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}, "test_c": {"f.py": {3}}}
    seen: list[list[str]] = []

    def build(subset=None, fresh=False):
        names = list(suite) if subset is None else list(subset)
        seen.append(names)
        return _bl({n: suite[n] for n in names if n in suite}, n_tests=len(names))

    holder = LazySessionBaseline(build)
    holder.seed(["test_a"])
    # Only the candidate was measured — never the whole suite.
    assert seen == [["test_a"]]
    assert set(holder.get().traced) == {"test_a"}


def test_seed_is_a_noop_once_built():
    suite = {"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}}
    holder = LazySessionBaseline(_suite_build(suite))
    holder.seed(["test_a"])
    holder.seed(["test_b"])  # must not re-measure or widen — once-per-session
    assert set(holder.get().traced) == {"test_a"}


def test_expand_adds_the_widened_batch_without_dropping_the_seed():
    suite = {"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}, "test_c": {"f.py": {3}}}
    holder = LazySessionBaseline(_suite_build(suite))
    holder.seed(["test_a"])
    assert holder.expand(["test_b", "test_c"]) is True
    got = holder.get().traced
    assert set(got) == {"test_a", "test_b", "test_c"}
    assert got["test_a"] == {"f.py": {1}}  # the seed survives the widening verbatim


def test_seed_then_expand_equals_a_full_rebuild():
    """THE soundness property: incrementally seeding candidates then widening to the rest yields
    the byte-identical baseline a full trace would — so early-stop-with-widen can never diverge
    from the full-suite verdict."""
    suite = {
        "test_a": {"f.py": {1}},
        "test_b": {"f.py": {2, 3}},
        "test_c": {"g.py": {9}},
        "test_d": {"f.py": {1}},
    }
    incremental = LazySessionBaseline(_suite_build(suite))
    incremental.seed(["test_a", "test_b"])
    incremental.expand(["test_c", "test_d"])
    spliced = incremental.get()

    full = LazySessionBaseline(_suite_build(suite))
    rebuilt = full.get()

    assert spliced.traced == rebuilt.traced
    assert spliced.n_tests == rebuilt.n_tests


def test_expand_degrades_to_invalidate_when_the_partial_build_raises():
    suite = {"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}}
    state = {"full": 0}

    def build(subset=None, fresh=False):
        if subset is not None and "test_b" in subset:
            raise RuntimeError("the consumer's test blew up mid-trace")
        if subset is None:
            state["full"] += 1
        names = list(suite) if subset is None else list(subset)
        return _bl({n: suite[n] for n in names if n in suite}, n_tests=len(names))

    holder = LazySessionBaseline(build)
    holder.seed(["test_a"])
    assert holder.expand(["test_b"]) is False
    assert holder.built is False  # value dropped, not left half-spliced
    holder.get()
    assert state["full"] == 1  # the next read re-traced in full — correct, just slower


def test_expand_is_a_noop_on_an_unbuilt_or_empty_batch():
    holder = LazySessionBaseline(_suite_build({"test_a": {"f.py": {1}}}))
    assert holder.expand(["test_a"]) is False  # nothing seeded yet -> nothing to widen
    holder.seed(["test_a"])
    assert holder.expand([]) is False  # empty batch -> no-op


def test_seed_and_expand_trace_every_proof_facing_test_fresh():
    # #15/#20: a cache may route, never prove. A stale "covers no target line" in either the seed
    # or widen would drop a real killer from `_tests_for`, so both subsets are observed this session.
    suite = {"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}}
    fresh_by_call: list[bool] = []

    def build(subset=None, fresh=False):
        fresh_by_call.append(fresh)
        names = list(suite) if subset is None else list(subset)
        return _bl({n: suite[n] for n in names if n in suite}, n_tests=len(names))

    holder = LazySessionBaseline(build)
    holder.seed(["test_a"])
    holder.expand(["test_b"])
    assert fresh_by_call == [True, True]


def test_fork_is_an_independent_unbuilt_holder_no_sibling_corruption():
    # Fix B #1: seeding one function's holder must not contaminate a sibling's. A fork is a fresh,
    # unbuilt holder over the SAME build closure; seeding it leaves the original untouched — the
    # reproduced `baseline seeded for alpha killed 0/3 of beta` failure, closed structurally.
    suite = {"test_a": {"f.py": {1}}, "test_b": {"f.py": {2}}}
    orig = LazySessionBaseline(_suite_build(suite))
    orig.seed(["test_a"])  # `alpha` seeds its candidate
    fork = orig.fork()
    assert fork.built is False  # a fork is fresh, never carrying the original's seed
    fork.seed(["test_b"])  # `beta` seeds its own candidate on its own holder
    assert set(orig.get().traced) == {"test_a"}  # alpha's baseline is untouched
    assert set(fork.get().traced) == {"test_b"}  # beta measured only its own
