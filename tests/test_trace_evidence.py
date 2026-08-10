"""#17 — outcome-qualified per-TestId baseline evidence, without loss through early unioning.

"A trace observed this line" and "a baseline-green, contained test proves this line" are different
facts. Wesker used to return the observed line-coverage union, letting a failing test's reach close
a line ledger (Detective #59's counterexample). The typed ledger keeps each item's outcome so the
observed and admissible views can both be derived honestly.

INTENT tests: the defect is a false admissible-reach, so a characterization of current output
cannot catch it. The pure decision asserts the qualification contract; `build_trace_ledger` and an
end-to-end `profile` assert the two views separate the failing owner from the green one.
"""

from __future__ import annotations

import os
import sys

from Wesker.trace_evidence import build_trace_ledger, trace_admissibility


# ── the pure decision: only green, contained, whole is admissible ─────────────────


def test_green_contained_untruncated_is_admissible():
    assert (
        trace_admissibility(baseline_passed=True, truncated=False, contained=True)
        == "admissible"
    )


def test_a_failing_baseline_observation_proves_nothing():
    assert (
        trace_admissibility(baseline_passed=False, truncated=False, contained=True)
        == "refuse_failed"
    )


def test_a_truncated_trace_cannot_close_an_obligation():
    """An under-counted trace cannot be read as 'did not reach' — truncation bars admissibility even
    for a passing test."""
    assert (
        trace_admissibility(baseline_passed=True, truncated=True, contained=True)
        == "refuse_truncated"
    )


def test_containment_is_absorbing_and_checked_first():
    """An uncontained measurement may have perturbed every observation, so nothing it saw is proof —
    it outranks even a green, untruncated item."""
    assert (
        trace_admissibility(baseline_passed=True, truncated=False, contained=False)
        == "refuse_uncontained"
    )
    assert (
        trace_admissibility(baseline_passed=False, truncated=True, contained=False)
        == "refuse_uncontained"
    )


# ── the ledger: observed keeps everything, admissible keeps only proof ─────────────


def test_ledger_keeps_a_failing_owner_marked_inadmissible():
    """The failing owner is RETAINED (observed reach is a real fact) but marked inadmissible with
    its reason — not silently dropped, which would lose that the line WAS reached."""
    ledger = build_trace_ledger(
        line_coverage={"t_green": [2, 3], "t_fail": [2, 4]},
        failed_ids={"t_fail"},
        truncated_ids=set(),
        contained=True,
    )
    by_id = {ev.test_id: ev for ev in ledger}
    assert by_id["t_fail"].admissible is False
    assert by_id["t_fail"].reason == "refuse_failed"
    assert by_id["t_fail"].lines == (2, 4)  # the reach is kept
    assert by_id["t_green"].admissible is True


# ── end-to-end: the failing-only counterexample stays out of the admissible union ──


def test_the_failing_only_counterexample_is_not_admissibly_complete(tmp_path):
    """The #59/#17 counterexample. Line 4 (`return 0`) is reached ONLY by the failing test; the
    admissible union must leave it OUT while the observed union keeps it for routing."""
    (tmp_path / "choo.py").write_text(
        "def choose(flag):\n    if flag:\n        return 1\n    return 0\n"
    )
    (tmp_path / "test_choo.py").write_text(
        "from choo import choose\n\n\n"
        "def test_green_branch():\n    assert choose(True) == 1\n\n\n"
        "def test_failing_only_cover():\n    assert choose(False) == 1\n"
    )
    for name in ("choo", "test_choo"):
        sys.modules.pop(name, None)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        from Detective.engine import profile

        r = profile("choo.py", "choose", str(tmp_path))
    finally:
        os.chdir(cwd)
    assert 4 in r.observed_union, (
        "the observed view must keep the reached line (routing)"
    )
    assert 4 not in r.admissible_union, (
        "a failing test's reach closed the admissible ledger — #17"
    )
    ids = {ev.test_id.rsplit("::", 1)[-1]: ev for ev in r.trace_evidence}
    assert ids["test_failing_only_cover"].admissible is False
    assert ids["test_green_branch"].admissible is True


# ── arcs: branch edges, carrying the same admissibility as lines ───────────────────


def test_arcs_carry_the_same_admissibility_as_lines():
    """An arc from a failing owner is RETAINED but inadmissible — the same qualification lines get,
    since an arc is the same observation seen finer."""
    ledger = build_trace_ledger(
        line_coverage={"t_ok": [2, 3], "t_fail": [2, 4]},
        failed_ids={"t_fail"},
        truncated_ids=set(),
        contained=True,
        arc_coverage={"t_ok": [(2, 3)], "t_fail": [(2, 4)]},
    )
    by_id = {ev.test_id: ev for ev in ledger}
    assert by_id["t_ok"].arcs == ((2, 3),) and by_id["t_ok"].admissible is True
    assert by_id["t_fail"].arcs == ((2, 4),) and by_id["t_fail"].admissible is False


def test_arc_capture_distinguishes_both_branches_of_a_conditional(tmp_path):
    """The #17 branch obligation. Line coverage cannot tell the True edge of a conditional from the
    False edge; arc capture records `(2, 3)` for one and `(2, 4)` for the other."""
    import importlib.util
    import os

    from Wesker.line_coverage import trace_suite

    path = tmp_path / "cond.py"
    path.write_text("def pick(flag):\n    if flag:\n        return 1\n    return 0\n")
    spec = importlib.util.spec_from_file_location("condmod_arc_it", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def t_true():
        assert module.pick(True) == 1

    def t_false():
        assert module.pick(False) == 0

    arcs: dict = {}
    trace_suite([t_true, t_false], {os.path.realpath(str(path))}, arcs_out=arcs)
    observed = {a for files in arcs.values() for aset in files.values() for a in aset}
    assert (2, 3) in observed, "the True edge was not recorded"
    assert (2, 4) in observed, "the False edge was not recorded"


def test_arc_capture_is_off_by_default():
    """The hot path is untouched unless a caller opts in: no `arcs_out`, no arc work."""
    from Wesker.line_coverage import _trace_one_multi

    def _noop():
        pass

    _hits, _cut, _contained, arcs = _trace_one_multi(_noop, set(), capture_arcs=False)
    assert arcs == {}


def test_arcs_auto_populate_a_profiling_result_through_the_session_baseline(tmp_path):
    """The session-baseline auto-wiring (#17): a profile run under a live session carries arcs in
    every trace_evidence entry, and `admissible_arc_union` leaves the failing test's branch edge
    OUT — the whole chain (session baseline → _build_test_scope → ledger), not just the tracer."""
    (tmp_path / "arcapp.py").write_text(
        "def choose(flag):\n    if flag:\n        return 1\n    return 0\n"
    )
    (tmp_path / "test_arcapp.py").write_text(
        "from arcapp import choose\n\n\n"
        "def test_true():\n    assert choose(True) == 1\n\n\n"
        "def test_false_fails():\n    assert choose(False) == 1\n"
    )
    for name in ("arcapp", "test_arcapp"):
        sys.modules.pop(name, None)

    from Wesker.ci import run_with_live_suite

    seen: dict = {}

    def _body():
        from Detective.engine import profile

        r = profile("arcapp.py", "choose", str(tmp_path))
        seen["by_id"] = {ev.test_id.rsplit("::", 1)[-1]: ev for ev in r.trace_evidence}
        seen["adm_arcs"] = set(r.admissible_arc_union)

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        run_with_live_suite(str(tmp_path), _body, target_files=["arcapp.py"])
    finally:
        os.chdir(cwd)

    by_id = seen["by_id"]
    assert by_id["test_true"].arcs == ((2, 3),), (
        "the green owner's True-edge arc is missing"
    )
    assert by_id["test_false_fails"].arcs == ((2, 4),)  # observed, but inadmissible
    assert (2, 3) in seen["adm_arcs"], "the admissible True edge is absent"
    assert (2, 4) not in seen["adm_arcs"], (
        "a failing test's arc closed a branch obligation — #17"
    )
