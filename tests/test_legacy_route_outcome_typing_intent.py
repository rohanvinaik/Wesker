"""#17 — the legacy direct-call route says it CANNOT type a non-pass; it never types it wrongly.

THE DEFECT (measured 2026-10-26). An in-process run outside a live pytest session takes the legacy
direct-call route (test ids prefixed `legacy:`), which has no pytest phase. Library callers take it
(the CLI does not). On a suite holding a pass, a wrong expectation, a `pytest.skip()`, a
`pytest.xfail()`, a `@pytest.mark.xfail` test and a fixture that raises, the route filed the
`xfail`-marked wrong expectation as `failed` (so it was NAMED in `failing_tests`), and typed the
skip, the `xfail()` call and the setup error all `error` — a skip printed as a setup error.

THE CONTRACT. The route types what its channel can prove: a clean return is `passed`, an
AssertionError is `failed`. A callable carrying skip/skipif/xfail marks (which the route does not
evaluate) and any other raise are `untyped` — a named state meaning "this route cannot say which
kind of non-pass" — refused as `refuse_untyped`, never green, never accused in `failing_tests`.
Live-session items are untouched (their typing comes from pytest's own reports).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest

import Wesker
from Wesker.ci import callable_item_status
from Wesker.trace_evidence import (
    BASELINE_OUTCOMES,
    baseline_outcome,
    trace_admissibility,
)

_TARGET = "def f(x):\n    return x * 2\n"
_TESTS = """import pytest
from {mod} import f


@pytest.fixture
def boom():
    raise RuntimeError("setup")


def test_pass():
    assert f(2) == 4


def test_wrong():
    assert f(2) == 5


def test_skip():
    pytest.skip("no")
    assert f(2) == 4


def test_xfail_call():
    pytest.xfail("known")
    assert f(2) == 5


@pytest.mark.xfail
def test_marked_xfail():
    assert f(2) == 5


@pytest.mark.skip(reason="r")
def test_marked_skip():
    assert f(2) == 4


def test_setup_error(boom):
    assert f(2) == 4
"""


_SCRIPT = """
import ast, json, os, sys
from Wesker.ci import discover_test_callables, resolve_original_func, walk_functions
from Wesker.engine import run_function_profiling
from Wesker.filter import filter_categories

root, mod = sys.argv[1], sys.argv[2]
src = os.path.join(root, mod + ".py")
os.chdir(root)
sys.path.insert(0, root)
node = dict(walk_functions(ast.parse(open(src).read())))["f"]
tests = discover_test_callables(root, mod + ".py", ["f"], backend="legacy")
r = run_function_profiling(
    node, mod + ".py::f", filter_categories(node), tests, resolve_original_func(src, "f")
)
print(json.dumps({
    "failing": list(r.failing_tests),
    "rows": [[e.test_id, e.baseline_outcome, e.reason, e.admissible] for e in r.trace_evidence],
}))
"""


class _Result:
    def __init__(self, payload):
        self.failing_tests = payload["failing"]
        self.trace_evidence = [
            types.SimpleNamespace(test_id=t, baseline_outcome=o, reason=r, admissible=a)
            for t, o, r, a in payload["rows"]
        ]


def _profile_legacy(tmp_path, mod):
    """The legacy route in a FRESH interpreter: it is the library-caller route (no live session),
    and a fresh process keeps it independent of whatever session is measuring this test."""
    (tmp_path / f"{mod}.py").write_text(_TARGET)
    (tmp_path / f"test_{mod}.py").write_text(_TESTS.format(mod=mod))
    wesker_root = os.path.dirname(os.path.dirname(os.path.abspath(Wesker.__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([wesker_root, *sys.path])}
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path), mod],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return _Result(json.loads(out.stdout.strip().splitlines()[-1]))


def test_the_legacy_route_types_only_what_it_can_prove(tmp_path):
    r = _profile_legacy(tmp_path, "leg_route_unit")
    got = {row.test_id.split("::")[-1]: row for row in r.trace_evidence}
    assert got["test_pass"].baseline_outcome == "passed" and got["test_pass"].admissible
    assert got["test_wrong"].baseline_outcome == "failed"
    for name in (
        "test_skip",
        "test_xfail_call",
        "test_setup_error",
        "test_marked_xfail",
        "test_marked_skip",
    ):
        assert got[name].baseline_outcome == "untyped", name
        assert got[name].reason == "refuse_untyped", name
        assert not got[name].admissible, name


def test_a_marked_xfail_is_not_named_as_a_wrong_expectation(tmp_path):
    r = _profile_legacy(tmp_path, "leg_route_failing_unit")
    named = {t.split("::")[-1] for t in r.failing_tests}
    assert named == {"test_wrong"}, (
        f"a test the route cannot type was accused (or the real one dropped): {named}"
    )


def test_a_skip_is_never_called_an_error(tmp_path):
    r = _profile_legacy(tmp_path, "leg_route_skip_unit")
    typed = {
        row.test_id.split("::")[-1]: row.baseline_outcome for row in r.trace_evidence
    }
    assert typed["test_skip"] != "error"
    assert typed["test_skip"] != "failed"


@pytest.mark.parametrize(
    ("run_code", "status", "expected"),
    [
        (
            None,
            "marked",
            "untyped",
        ),  # a clean return of a callable whose marks were not evaluated
        ("assertion", "marked", "untyped"),
        ("timeout", "marked", "error"),  # a stopped run is still checked first
        ("uncontained", "marked", "error"),
        (None, "unobserved", "passed"),
        ("assertion", "unobserved", "failed"),
        ("crash", "unobserved", "untyped"),
        (None, "skipped", "skipped"),
    ],
)
def test_baseline_outcome_names_the_marked_and_untypable_states(
    run_code, status, expected
):
    assert baseline_outcome(run_code, status) == expected


def test_untyped_is_a_named_outcome_with_its_own_refusal():
    assert "untyped" in BASELINE_OUTCOMES
    assert trace_admissibility(False, False, True, True, "untyped") == "refuse_untyped"
    # contained/truncated still outrank it, as for every typed refusal
    assert trace_admissibility(False, True, True, True, "untyped") == "refuse_truncated"
    assert (
        trace_admissibility(False, False, False, True, "untyped")
        == "refuse_uncontained"
    )


def test_callable_item_status_reads_marks_only_without_a_live_status():
    import pytest as _pt

    @_pt.mark.xfail
    def marked():  # pragma: no cover
        pass

    def plain():  # pragma: no cover
        pass

    assert callable_item_status(marked) == "marked"
    assert callable_item_status(plain) == "unobserved"

    class _Status:
        value = "passed"

    marked.__wesker_item_status__ = _Status()  # type: ignore[attr-defined]
    assert callable_item_status(marked) == "passed", "a live status outranks marks"
