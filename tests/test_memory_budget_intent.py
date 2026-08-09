"""A memory budget must describe THIS RUN, not the process's whole history (issue #21).

`over_budget()` compared `ru_maxrss` -- a process LIFETIME peak that never falls -- against a
per-run budget. In a long-lived MCP/server process one earlier spike therefore left every later
low-budget run reading as exhausted before it allocated anything.

W#21'S OWN COUNTEREXAMPLE DOES NOT REPRODUCE, and that mattered. The issue predicts that after a
release the instantaneous figure drops while the peak stays high:

    release + collect:        current ~20 MB, guard ~116 MB

Measured on macOS / CPython 3.14, both stay high, because the allocator keeps the pages:

    start                    resident   13.3 MB    resident_max   13.3 MB
    holding 200MB            resident  213.3 MB    resident_max  213.3 MB
    released + gc.collect    resident  213.3 MB    resident_max  213.3 MB

So the fix the issue's framing implies -- swap peak for current -- would have changed nothing
here. What makes the number honest is measuring GROWTH WITHIN A RUN. The same probe showed why
that is not a trick: reallocating 200MB on a later cycle left resident flat at 213.3, because
the retained pages were reused. A run that demands nothing new from the OS has grown by zero,
and a budget is about demand.

Verified end to end against the real functions:

    run 1 (allocates 200MB): growth 200.1 MB -> exhausted   over_budget = True
    run 2 (allocates   4MB): growth   4.0 MB -> within      over_budget = False
                                                            over_budget(OLD, absolute) = True
"""

from __future__ import annotations

from Wesker.memory_guard import (
    current_rss_bytes,
    memory_budget_standing,
    over_budget,
    process_rss_bytes,
    rss_capability,
    run_baseline_bytes,
    run_growth_bytes,
)

MB = 1024 * 1024


# ── the decision ──


def test_a_run_is_judged_on_its_own_growth():
    assert memory_budget_standing(4 * MB, 64 * MB, "current") == "within"
    assert memory_budget_standing(200 * MB, 64 * MB, "current") == "exhausted"


def test_history_cannot_exhaust_a_run_that_grew_by_nothing():
    """THE regression, in one line. A run that allocated nothing is within budget no matter how
    large the process already was -- which is exactly what the absolute check could not say."""
    assert memory_budget_standing(0, 64 * MB, "current") == "within"


def test_an_unmeasurable_platform_is_not_the_same_as_a_passing_one():
    """'We looked and it is fine' and 'we cannot look' are different facts. Collapsing them is
    how an absent guard comes to read as a passed one -- so this is a third state, not False."""
    assert memory_budget_standing(0, 64 * MB, "unavailable") == "unmeasurable"
    assert memory_budget_standing(10**12, 64 * MB, "unavailable") == "unmeasurable"


def test_an_unmeasurable_verdict_is_never_exhausted_either():
    """It must not manufacture a refusal on Windows. The run continues; what it may not do is
    describe itself as memory-bounded."""
    assert memory_budget_standing(10**12, 1, "unavailable") != "exhausted"


def test_a_nonpositive_budget_means_unbounded_not_instantly_exhausted():
    """`resolve_budget` returns `sys.maxsize` for an opt-out, but a 0 or negative budget arriving
    from a caller must read as 'no bound', not as 'every run is over'."""
    assert memory_budget_standing(10**12, 0, "current") == "within"


def test_the_boundary_is_strict():
    """Growth EQUAL to the budget is within it -- the budget is what the run may use."""
    assert memory_budget_standing(64 * MB, 64 * MB, "current") == "within"
    assert memory_budget_standing(64 * MB + 1, 64 * MB, "current") == "exhausted"


# ── the measurement it consumes ──


def test_growth_is_floored_at_zero():
    """A shrink is not negative demand; it is the allocator returning pages, which a budget has
    no opinion about."""
    assert run_growth_bytes(10**15) == 0


def test_a_later_small_run_is_not_poisoned_by_an_earlier_large_one():
    """End to end through the real functions, in one process -- the actual harm."""
    base1 = run_baseline_bytes()
    blob = bytearray(96 * MB)
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    assert over_budget(32 * MB, base1) is True, "a genuinely hungry run must still trip"
    del blob

    base2 = run_baseline_bytes()
    assert over_budget(32 * MB, base2) is False, (
        "a later run that allocated nothing must not inherit the earlier run's usage"
    )


def test_the_capability_is_named_rather_than_assumed():
    """A guarantee must not be claimed on a platform that only observes."""
    assert rss_capability() in {"current", "peak_only", "unavailable"}


def test_peak_never_falls_which_is_why_it_cannot_be_the_signal():
    """Documents the property that forced this change, so a future edit back to the absolute
    comparison fails here with the reason attached."""
    before = process_rss_bytes()
    blob = bytearray(64 * MB)
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    del blob
    assert process_rss_bytes() >= before, (
        "ru_maxrss is a lifetime peak; it does not come down"
    )


def test_current_rss_is_a_real_reading_where_it_is_available():
    if rss_capability() != "current":
        return  # peak_only / unavailable platforms have nothing to assert here
    assert current_rss_bytes() > 0
