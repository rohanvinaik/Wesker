"""The TRACE budget — the bound on the baseline's traced pass.

WHY THIS EXISTS: the baseline runs BEFORE any mutant and traces with a callback per executed line,
so a test that is merely slow untraced becomes effectively unbounded traced. That pass had no
budget of ANY kind (the mutation loop had `per_mutant_timeout_ms`; the trace that runs first had
nothing), so a heavy consumer test presented as a HANG — no output, no diagnosis. Found against a
real consumer (Regenesis, 2138 tests) where `detective diagnose` sat at 99% CPU emitting zero
bytes.

WHY IT IS A THREAD, not a deadline ticked from the trace callback — this is the whole design, and
the first attempt got it wrong: `dispatch` installs the line-callback ONLY for the target file, so
a callback-ticked budget only bounds tests that EXECUTE the file under analysis. In a real suite
that is a tiny minority (2138 tests, almost none of which touch one file), so almost every test
would be unbudgeted — the bug wearing the fix's clothes. `test_budget_binds_work_outside_the_
target_file` is that exact regression, and it FAILS against the callback-ticked design.

The contract pinned here: budgeted is OPT-IN (None = the historical unbounded pass), a cut test
KEEPS the lines it reached (partial coverage is real coverage — the same concession `_trace_one`
already makes for a test that RAISES), a cut is always NAMED, and a cut thread is STOPPED rather
than leaked.
"""

import ast
import os
import sys
import time

# The target must live in a real file (the tracer keys on co_filename), and this import must
# resolve OUTSIDE pytest too: Detective/Wesker discover tests by importing the module directly, and
# a bare `from _trace_budget_target import ...` only works under pytest's rootdir sys.path
# insertion — it raises anywhere else, and the whole file is then silently dropped from discovery
# (tests that pass while pinning nothing).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Wesker.line_coverage import (  # noqa: E402
    _trace_one,
    _trace_one_multi,
    executable_lines,
    trace_line_coverage,
    trace_suite,
)

from Wesker.ci import callable_test_id
from _trace_budget_target import spin  # noqa: E402

_FILE = spin.__code__.co_filename
# Parse the WHOLE module, not inspect.getsource(spin): the tracer records real file line numbers,
# so the AST must carry them too. Parsing the extracted source re-bases it at line 1 and the
# `hits & exec_lines` intersection silently comes back EMPTY.
_NODE = next(
    n
    for n in ast.parse(open(_FILE).read()).body
    if isinstance(n, ast.FunctionDef) and n.name == "spin"
)
_LINES = executable_lines(_NODE)


def _fast() -> None:
    assert spin(10) == 45


def _slow() -> None:
    assert spin(4_000_000) > 0


def _slow_elsewhere() -> None:
    """Burns time WITHOUT touching the target file — so it is never traced, and a budget riding
    the trace callback can never see it. The common case in any real suite."""
    total = 0
    for i in range(60_000_000):
        total += i
    assert total > 0


def test_unbudgeted_is_the_historical_unbounded_pass():
    """None (the default) must not change what the tracer does — only what it is ALLOWED to cost."""
    covered, truncated = _trace_one(_fast, _FILE, _LINES)
    assert truncated is False
    assert covered == _LINES  # the whole function body was reached


def test_budget_cuts_a_heavy_test_and_keeps_its_partial_coverage():
    """The cut is not a failure — the lines reached before it are real and are kept."""
    covered, truncated = _trace_one(_slow, _FILE, _LINES, budget_s=0.25)
    assert truncated is True
    assert covered  # partial coverage is real coverage
    assert covered <= _LINES


def test_budget_binds_work_outside_the_target_file():
    """THE REGRESSION. A budget ticked from the trace callback fires only while the TARGET file is
    executing, so this test — which never touches it, like nearly every test in a real suite —
    sailed through untouched (measured 4.2s under a 1.0s budget). Bounding by wall-clock in
    another thread does not care which module is running."""
    t0 = time.monotonic()
    _covered, truncated = _trace_one(_slow_elsewhere, _FILE, _LINES, budget_s=0.5)
    elapsed = time.monotonic() - t0
    assert truncated is True
    assert elapsed < 3.0, (
        f"budget did not bind work outside the target file ({elapsed:.1f}s)"
    )


def test_budget_does_not_touch_a_test_that_fits_in_it():
    """The common case pays nothing: a fast test's coverage is identical budgeted or not."""
    unbudgeted, _ = _trace_one(_fast, _FILE, _LINES)
    budgeted, truncated = _trace_one(_fast, _FILE, _LINES, budget_s=5.0)
    assert budgeted == unbudgeted and truncated is False


def test_a_cut_test_is_stopped_not_leaked():
    """A cut must not leave a live thread burning a core — that is what makes LATER tests time out
    because earlier ones are still running, compounding across a session."""
    import threading

    before = threading.active_count()
    for _ in range(3):
        _trace_one(_slow_elsewhere, _FILE, _LINES, budget_s=0.3)
    time.sleep(0.3)  # let the injections unwind
    assert threading.active_count() <= before + 1


def test_trace_line_coverage_names_every_test_the_budget_cut():
    """No silent caps: the caller learns WHICH tests are under-counted, by test id.

    Asserted through `callable_test_id` rather than a literal: the cut set and the coverage
    map must speak ONE vocabulary (issue #16), and hardcoding either spelling would let them
    drift apart while the test still passed."""
    cut: set[str] = set()
    cov = trace_line_coverage(
        [_fast, _slow], spin, _LINES, budget_s=0.25, truncated=cut
    )
    assert cut == {callable_test_id(_slow)}
    # both still contribute coverage
    assert cov[callable_test_id(_fast)] and cov[callable_test_id(_slow)]


def test_trace_line_coverage_reports_nothing_cut_when_unbudgeted():
    cut: set[str] = set()
    trace_line_coverage([_fast], spin, _LINES, truncated=cut)
    assert cut == set()


def test_trace_suite_budgets_each_test_and_reports_the_cuts():
    """The suite-global path matters MORE than the per-function one: its baseline is computed once
    and reused by every function, so one heavy test stalls the whole session before any mutant."""
    cut: set[str] = set()
    traced = trace_suite([_fast, _slow], {_FILE}, budget_s=0.25, truncated=cut)
    assert cut == {callable_test_id(_slow)}
    assert set(traced) == {callable_test_id(_fast), callable_test_id(_slow)}
    # the fast test's lines survive intact
    assert traced[callable_test_id(_fast)][_FILE]


def test_trace_one_multi_reports_its_own_cut_rather_than_the_caller_timing_it():
    """The trace reports the cut it made; the caller never infers it from a clock (which would
    false-positive on a test that merely happens to take about the budget)."""
    per_file, truncated = _trace_one_multi(_slow, {_FILE}, budget_s=0.25)
    assert truncated is True and per_file[_FILE]
    per_file2, truncated2 = _trace_one_multi(_fast, {_FILE})
    assert truncated2 is False and per_file2[_FILE]


# --- the SESSION budget: the aggregate bound the per-test cap cannot give ---------------------
def _heavy(name: str):
    def t() -> None:
        assert spin(3_000_000) >= 0

    t.__name__ = name
    return t


def test_session_budget_bounds_the_whole_pass_not_just_each_test():
    """A per-test cap × N tests is still N× unbounded — on a 2000-test suite the 50s per-test cap
    alone permits a day. Only this makes the phase finite."""
    tests = [_heavy(f"heavy_{i}") for i in range(3)]
    cut: set[str] = set()
    t0 = time.monotonic()
    trace_line_coverage(
        tests, spin, _LINES, budget_s=50.0, truncated=cut, session_budget_s=0.01
    )
    elapsed = time.monotonic() - t0
    # 3 heavy tests under a 50s PER-TEST cap could run for 150s; the session budget stops after
    # the first. The bound proven here is aggregate, which no per-test cap can give.
    assert elapsed < 6.0, f"session budget did not bound the pass ({elapsed:.1f}s)"
    assert cut, "tests left untraced by the session budget must be named"


def test_session_budget_names_the_tests_it_never_reached():
    """An UNTRACED test's coverage is absent, not zero — and downstream those are identical. So the
    unreached tail must be reported, or a budget cut reads as a real completeness gap.

    The budget is checked BEFORE each test, so the first always runs and the whole tail is named:
    that holds at any machine speed. (Pinning WHICH tests fit inside a 1s budget does not — it
    encodes how fast the box is, and a faster one silently fits more.)"""
    tests = [_heavy(f"heavy_{i}") for i in range(3)]
    cut: set[str] = set()
    traced = trace_suite(
        tests, {_FILE}, budget_s=50.0, truncated=cut, session_budget_s=0.01
    )
    ids = [callable_test_id(t) for t in tests]
    # These three are closures minted by ONE factory, so they share `__qualname__` and differ
    # only in `__name__`. That is precisely the shape that collapsed onto a single id while
    # #16 was being written — a 2-test cut reporting as 1 — so the distinctness is asserted
    # here rather than assumed by the two set comparisons below.
    assert len(set(ids)) == 3, f"factory closures must not share an id: {ids}"
    assert set(cut) == {ids[1], ids[2]}  # the entire untraced tail, by id
    assert set(traced) == {ids[0]}  # and only the one that ran is reported as traced


# --- progress: the half that makes the phase legible rather than merely finite ------------------
def test_progress_reports_each_test_of_the_pass():
    seen: list[tuple[int, int]] = []
    trace_line_coverage(
        [_fast, _fast, _fast],
        spin,
        _LINES,
        progress=lambda d, t, _ms: seen.append((d, t)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_progress_is_not_swallowed_by_the_consumer_output_redirect():
    """REGRESSION: the redirect that isolates a TEST's stdout/stderr was wrapped around the whole
    LOOP, so it also captured `progress` (which reports on stderr) — the callback fired into a
    StringIO and the phase stayed exactly as silent as if nothing reported at all."""
    import io

    written = io.StringIO()

    def report(done: int, total: int, _ms: float) -> None:
        sys.stderr.write(f"{done}/{total} ")

    real = sys.stderr
    sys.stderr = written
    try:
        trace_line_coverage([_fast, _fast], spin, _LINES, progress=report)
    finally:
        sys.stderr = real
    assert written.getvalue().strip() == "1/2 2/2"
