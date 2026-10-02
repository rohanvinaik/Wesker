"""#31 — a test the baseline never screened is never credited with a kill, and is named where it was held back.

THE DEFECT (measured 2026-10-02 on current code). `_build_test_scope`'s resolver hands a mutant it
cannot scope — scoping off, no line data, no mutated line, or a line outside the traced body (a
default or annotation on the `def` line) — the WHOLE usable pool, and "usable" meant "not KNOWN to
fail on the original". Since Detective `071fc73` the pool is the CONSULTED set: the seeded candidates
plus the widen-admitted unknowns, which sit in the pool before the widen traces them. So an unknown
reached the fallback unscreened: a test asserting `scale(3) == 7` against a function returning 6 was
credited with killing both mutants of the `factor=2` default, by assertion, on a result reporting
itself gateable, with `failing_tests == []` — in-process and isolated alike.

THE CONTRACT (the issue's Acceptance). With a def-line mutant, a consulted-but-unscreened test that
fails on the original, and a screened candidate: the failing test never counts as a kill, and where
it was held back unscreened the report names it — a survivor's record carries `scope` (why it fell
back) and `unscreened_tests` (what its verdict was never measured against). Screening is done where
it can be: the in-process widen screens the consulted unknowns as it goes, and the isolated mode,
which never widens, screens them up front — without that, holding them back emptied `detective
audit` of a helper tested only through its caller (8/13 value-pinned -> 0/13, measured). A GREEN
unknown still kills once screened, so no real kill is lost.

The resolver is wired here exactly as Detective's target-first driver wires it: fork the session's
baseline holder, SEED it with the candidates, and profile the pool `candidates + unknowns`.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

from Wesker.engine import attribution_standing, mutant_scope_route

# ── the pure decisions ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("inert", "screened", "expected"),
    [
        (False, True, "usable"),
        (False, False, "unscreened"),  # the defect: this used to read "usable"
        (True, True, "barred"),
        (True, False, "barred"),  # a known failure is the most definite fact
    ],
)
def test_attribution_standing_keeps_unscreened_apart_from_usable(
    inert, screened, expected
):
    assert attribution_standing(inert, screened) == expected


@pytest.mark.parametrize(
    ("scope_tests", "has_line_data", "mutated_line", "line_in_body", "expected"),
    [
        (True, True, 7, True, "covering"),
        (True, True, 1, False, "off_body_line"),  # a default/annotation on the def line
        (True, True, None, False, "no_mutated_line"),
        # None means "the mutator could not say"; ANY int is a line — never a falsiness check.
        (True, True, 0, True, "covering"),
        (True, False, 7, True, "no_line_data"),
        (False, True, 7, True, "unscoped"),  # scoping off outranks everything
        (False, False, None, False, "unscoped"),
        (True, False, None, False, "no_line_data"),
    ],
)
def test_mutant_scope_route_names_why_a_mutant_falls_back(
    scope_tests, has_line_data, mutated_line, line_in_body, expected
):
    assert (
        mutant_scope_route(scope_tests, has_line_data, mutated_line, line_in_body)
        == expected
    )


# ── end-to-end: the issue's Acceptance, through a live pytest session ─────────────────

_TARGET = "def scale(x, factor=2):\n    y = x * factor\n    return y\n"

_TESTS = """from {mod} import scale


def test_candidate_explicit_factor():
    assert scale(3, factor=2) == 6


def test_unknown_fails_on_original():
    assert scale(3) == 7


def test_unknown_green_uses_default():
    assert scale(3) == 6
"""


def _profile(
    tmp_path,
    mod,
    unknowns,
    widen,
    *,
    seed=("test_candidate_explicit_factor",),
    isolated=False,
    converged=False,
):
    """Seed `seed`, pool = seed + `unknowns`, widen over `widen` (test names, without the path)."""
    from Wesker import engine
    from Wesker.ci import (
        callable_test_id,
        discover_test_callables,
        resolve_original_func,
        run_with_live_suite,
        walk_functions,
    )
    from Wesker.filter import filter_categories

    (tmp_path / f"{mod}.py").write_text(_TARGET)
    (tmp_path / f"test_{mod}.py").write_text(_TESTS.format(mod=mod))
    for name in (mod, f"test_{mod}"):
        sys.modules.pop(name, None)
    root = str(tmp_path)

    def body():
        src = os.path.join(root, f"{mod}.py")
        with open(src) as fh:
            node = dict(walk_functions(ast.parse(fh.read())))["scale"]
        tests = discover_test_callables(root, f"{mod}.py", ["scale"])
        by = {callable_test_id(t).rsplit("::", 1)[-1]: t for t in tests}
        seeded_tests = [by[n] for n in seed]
        pool = seeded_tests + [by[n] for n in unknowns]
        holder = engine._SESSION_BASELINE.get()
        seeded = holder.fork()
        seeded.seed(seeded_tests)
        token = engine._SESSION_BASELINE.set(seeded)
        try:
            if converged:
                return engine.run_function_converged(
                    node,
                    f"{mod}.py::scale",
                    filter_categories(node),
                    pool,
                    resolve_original_func(src, "scale"),
                    budget_ms=60000,
                    widen_tests=[by[n] for n in widen] or None,
                )
            return engine.run_function_profiling(
                node,
                f"{mod}.py::scale",
                filter_categories(node),
                pool,
                resolve_original_func(src, "scale"),
                widen_tests=[by[n] for n in widen] or None,
                isolated=isolated,
            )
        finally:
            engine._SESSION_BASELINE.reset(token)

    cwd = os.getcwd()
    try:
        os.chdir(root)
        return run_with_live_suite(root, body, target_files=[f"{mod}.py"])
    finally:
        os.chdir(cwd)


def _def_line(records):
    return [r for r in records if r["mutated_line"] == 1]


def _credited(result, short_name):
    return [
        r["mutant_id"]
        for r in result.killed_records
        if (r.get("test") or "").endswith(short_name)
    ]


@pytest.mark.parametrize("isolated", [False, True])
def test_a_consulted_unknown_that_fails_on_the_original_never_kills(tmp_path, isolated):
    """The Acceptance case. Both modes screen the widen-admitted unknown — the in-process widen as it
    goes, the isolated mode up front — so it is reported for what it is, a failing test, and the
    def-line mutants it used to "kill" survive the screened candidate honestly."""
    r = _profile(
        tmp_path,
        f"s31_fail_{int(isolated)}",
        ["test_unknown_fails_on_original"],
        ["test_unknown_fails_on_original"],
        isolated=isolated,
    )
    assert _credited(r, "test_unknown_fails_on_original") == [], (
        "a test that fails on the original was credited with a kill — #31"
    )
    survivors = _def_line(r.survivor_records)
    assert survivors, "the def-line default mutants must survive the screened candidate"
    assert all("unscreened_tests" not in rec for rec in survivors)
    assert any(t.endswith("test_unknown_fails_on_original") for t in r.failing_tests)


@pytest.mark.parametrize("isolated", [False, True])
def test_an_unknown_no_one_screens_is_held_back_and_named(tmp_path, isolated):
    """A pool member outside the widen is screened by no one, so it never runs, and every survivor
    that needed it names it: `scope` says why the mutant fell back, `unscreened_tests` what it was
    never measured against."""
    r = _profile(
        tmp_path,
        f"s31_named_{int(isolated)}",
        ["test_unknown_fails_on_original"],
        [],
        isolated=isolated,
    )
    assert _credited(r, "test_unknown_fails_on_original") == []
    survivors = _def_line(r.survivor_records)
    assert survivors
    for rec in survivors:
        assert rec["scope"] == "off_body_line"
        assert [t.rsplit("::", 1)[-1] for t in rec["unscreened_tests"]] == [
            "test_unknown_fails_on_original"
        ]
    # ...and the serialized report carries it too.
    assert any("unscreened_tests" in rec for rec in r.to_dict()["survivor_records"])


def test_a_green_unknown_still_kills_once_the_widen_screens_it(tmp_path):
    """The fix holds unscreened tests BACK; it does not drop them. Screened green, the default's
    mutants die to the unknown that uses the default."""
    r = _profile(
        tmp_path,
        "s31_green",
        ["test_unknown_green_uses_default"],
        ["test_unknown_green_uses_default"],
    )
    assert _def_line(r.survivor_records) == []
    assert _credited(r, "test_unknown_green_uses_default"), r.killed_records


def test_isolated_caller_only_reach_still_kills_after_screening_up_front(tmp_path):
    """The audit regression guard. An EMPTY seed (nothing names the function; only its callers'
    tests reach it) leaves every mutant on the no-line-data fallback with every consulted test
    unscreened. Held back without screening, the isolated mode — `detective audit`'s only
    measurement — would read every mutant as surviving a suite that pins them."""
    r = _profile(
        tmp_path,
        "s31_caller_only",
        ["test_unknown_green_uses_default"],
        ["test_unknown_green_uses_default"],
        seed=(),
        isolated=True,
    )
    assert r.total_killed > 0, r.survivor_records
    assert _credited(r, "test_unknown_green_uses_default")
    assert all("unscreened_tests" not in rec for rec in r.survivor_records)


def test_the_converged_path_holds_the_same_line(tmp_path):
    """`run_function_converged` shares the resolver, so it shares the rule."""
    r = _profile(
        tmp_path,
        "s31_conv",
        ["test_unknown_fails_on_original"],
        [],
        converged=True,
    )
    assert _credited(r, "test_unknown_fails_on_original") == []
    named = [rec for rec in _def_line(r.survivor_records) if "unscreened_tests" in rec]
    assert named, r.survivor_records
