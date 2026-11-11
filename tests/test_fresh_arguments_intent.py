"""Intent: every evaluation gets the test's arguments as authored, and a failure the ORIGINAL also
produces is never a kill (Wesker #55; Detective #127).

THE DEFECT, measured 2026-10-26 on Wesker 2fbd029. The target writes into its argument only when its
`copy.deepcopy` is removed (mutant `new = written`), and carries a site that cannot change behaviour
(`tag = max(2, 2)`). The suite is ONE parametrized row and one duplicate-path test. One isolated worker
runs every mutant in one process; the test module stays in `sys.modules`, so the module-level
`parametrize` row is the same object for every mutant. After the writer ran, the row's `written` held
`{"objects": ["A", "B", "C"]}` and even the ORIGINAL returned `version:duplicate:B` on it, so the test
failed for every later mutant, and each failure counted as a kill:

    isolated:   the mutant IDENTICAL to the original, `tag = 2` and `min(2, 2)`: killed by assertion
    in-process: the two `TypeError` mutants (real crash kills on the authored row): survived

`detective converge --isolated` printed `✓ COMPLETE ... 27/28 killed` over it.

THE RULING (KNOWABILITY Def. 2.3): a suite kills `p∘f` iff `p(f(x)) != f(x)` for some `x` in the suite.
A kill is a DIFFERENCE FROM THE ORIGINAL ON THE SAME, AUTHORED INPUT. So:
 (A) every evaluation, in both modes, gets fresh test arguments; and where a value cannot be copied that
     is a named fact (`shared_arguments`), never a silent skip;
 (B) a node that fails under a mutant is re-run against the ORIGINAL in the same process state, and a
     failure the original reproduces is the unscored outcome `original_also_fails`, never a kill and
     never a survivor. (B) is what still guards shared state a copy cannot reach.

These are written from that ruling: the issue's own fixture in both modes, and the edge cases around it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from Wesker.engine import original_control_standing
from Wesker.pytest_runner import (
    FreshArguments,
    argument_copy_standing,
    identity_standing,
)

# ── the two pure decisions ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raised", "compare_raised", "equal", "standing"),
    [
        (False, False, True, "copy"),
        (False, False, False, "identity_bearing"),
        (False, True, False, "uncomparable"),
        (True, False, False, "uncopyable"),
        # A copy that never existed cannot be compared: copying is asked first.
        (True, True, True, "uncopyable"),
        (True, False, True, "uncopyable"),
    ],
)
def test_argument_copy_standing_names_why_an_argument_stays_shared(
    raised, compare_raised, equal, standing
):
    assert argument_copy_standing(raised, compare_raised, equal) == standing


@pytest.mark.parametrize(
    ("mutant", "control", "standing"),
    [
        ("assertion", None, "control_passed"),
        ("crash", None, "control_passed"),
        ("exception", None, "control_passed"),
        ("assertion", "assertion", "control_same"),
        ("crash", "crash", "control_same"),
        ("exception", "exception", "control_same"),
        # The original fails too, but differently: the runs differ, so the kill stands.
        ("assertion", "crash", "control_differs"),
        ("crash", "assertion", "control_differs"),
        ("assertion", "undetermined", "control_differs"),
        # Not a failure outcome of the mutant: there is nothing to control.
        (None, None, "not_a_failure"),
        (None, "assertion", "not_a_failure"),
        ("diverged", "diverged", "not_a_failure"),
        ("undetermined", None, "not_a_failure"),
        ("nested_measurement", "nested_measurement", "not_a_failure"),
    ],
)
def test_original_control_standing_is_a_difference_or_it_is_not_a_kill(
    mutant, control, standing
):
    assert original_control_standing(mutant, control) == standing


# ── FreshArguments on fake items: the snapshot, the refresh, the named exceptions ────────────────


def _item(nodeid, **params):
    return SimpleNamespace(nodeid=nodeid, callspec=SimpleNamespace(params=params))


def test_a_write_into_an_argument_does_not_reach_the_next_evaluation():
    row = {"objects": ["A"]}
    item = _item("t.py::t[0]", written=row, expected="x")
    fresh = FreshArguments()
    fresh.capture(item)

    fresh.refresh(item)
    item.callspec.params["written"]["objects"].append("B")  # the writer mutant
    fresh.refresh(item)

    assert item.callspec.params["written"] == {"objects": ["A"]}
    assert item.callspec.params["written"] is not row
    assert fresh.shared() == []


def test_the_snapshot_is_taken_once_before_any_evaluation_can_alter_the_row():
    """A second sight of the item (the next evaluation's collection) finds the row already altered; it
    must not become the new authored input."""
    row = {"objects": ["A"]}
    item = _item("t.py::t[0]", written=row)
    fresh = FreshArguments()
    fresh.capture(item)
    row["objects"].append("B")  # an earlier evaluation's writer

    fresh.capture(item)
    fresh.refresh(item)

    assert item.callspec.params["written"] == {"objects": ["A"]}


def test_two_parameters_that_name_one_object_still_name_one_object():
    shared = ["x"]
    item = _item("t.py::t[0]", left=shared, right=shared)
    fresh = FreshArguments()
    fresh.capture(item)
    fresh.refresh(item)

    params = item.callspec.params
    assert params["left"] is params["right"]
    assert params["left"] is not shared


def test_a_test_with_no_parameters_has_nothing_to_snapshot_or_refresh():
    for item in (
        SimpleNamespace(nodeid="t.py::plain"),
        SimpleNamespace(nodeid="t.py::plain", callspec=None),
        _item("t.py::empty"),
    ):
        fresh = FreshArguments()
        fresh.capture(item)
        fresh.refresh(item)
        assert fresh.shared() == []


def test_an_uncopyable_value_stays_shared_and_is_named_and_its_neighbours_are_still_copied():
    lock = threading.Lock()
    data = [1]
    item = _item("t.py::t[0]", guard=lock, data=data)
    fresh = FreshArguments()
    fresh.capture(item)
    fresh.refresh(item)

    params = item.callspec.params
    assert params["guard"] is lock, (
        "an uncopyable value is the same object, not a stand-in"
    )
    assert params["data"] == [1] and params["data"] is not data
    assert fresh.shared() == ["t.py::t[0] :: guard (uncopyable)"]


def test_a_value_whose_copy_is_not_equal_to_it_is_left_shared_and_named():
    """A sentinel compared by identity would FAIL on the original itself if it were copied: a broken
    suite is worse than a named gap."""
    sentinel = object()
    item = _item("t.py::t[0]", marker=sentinel)
    fresh = FreshArguments()
    fresh.capture(item)
    fresh.refresh(item)

    assert item.callspec.params["marker"] is sentinel
    assert fresh.shared() == ["t.py::t[0] :: marker (identity_bearing)"]


def test_a_value_whose_equality_has_no_truth_value_is_left_shared_and_named():
    class Array:
        def __eq__(self, other):
            raise ValueError("the truth value of an array is ambiguous")

        __hash__ = object.__hash__

    value = Array()
    item = _item("t.py::t[0]", array=value)
    fresh = FreshArguments()
    fresh.capture(item)
    fresh.refresh(item)

    assert item.callspec.params["array"] is value
    assert fresh.shared() == ["t.py::t[0] :: array (uncomparable)"]


# ── end to end: the issue's own fixture, both modes ─────────────────────────────────────────────

_REVISION = textwrap.dedent(
    """
    import copy


    def revision(current: dict, written: dict) -> str:
        new = copy.deepcopy(written)
        for name in current.get("objects", []):
            if name in new["objects"]:
                return "version:duplicate:" + name
            new["objects"].append(name)
        tag = max(2, 2)
        return "revised:" + ",".join(new["objects"]) + ":" + str(tag)
    """
)

# The same function with the deepcopy site AFTER another site, so the writer is generated second.
_REVISION_WRITER_LAST = textwrap.dedent(
    """
    import copy


    def revision(current: dict, written: dict) -> str:
        tag = max(2, 2)
        new = copy.deepcopy(written)
        for name in current.get("objects", []):
            if name in new["objects"]:
                return "version:duplicate:" + name
            new["objects"].append(name)
        return "revised:" + ",".join(new["objects"]) + ":" + str(tag)
    """
)

_REVISION_TESTS = textwrap.dedent(
    """
    import pytest
    from m import revision


    @pytest.mark.parametrize(
        "args,expected",
        [(({"objects": ["B", "C"]}, {"objects": ["A"]}), "revised:A,B,C:2")],
    )
    def test_revision_golden(args, expected):
        assert revision(*args) == expected


    def test_duplicate():
        assert revision({"objects": ["A"]}, {"objects": ["A"]}) == "version:duplicate:A"
    """
)

_RUNNER = textwrap.dedent(
    """
    import json, sys
    from Wesker.ci import profile_function

    r = profile_function(".", "m.py", sys.argv[1], isolated=(sys.argv[2] == "isolated"))
    rows = {}
    for kind in ("killed_records", "survivor_records", "unscored_records"):
        for rec in r.get(kind, []):
            rows[rec["mutant_id"]] = {
                "status": kind.split("_")[0],
                "by": rec.get("killed_by"),
                "disposition": rec.get("disposition"),
                "change": rec.get("change") or "",
                "test": rec.get("test") or rec.get("refused_test") or "",
            }
    print(json.dumps({
        "rows": rows,
        "shared": r.get("shared_arguments", []),
        "unscored_by": r.get("unscored_by", {}),
        "mode": r.get("execution_mode"),
    }))
    """
)


def _profile(tmp_path, source, tests, function, mode):
    """Profile `function` of m.py in a disposable project, in a fresh interpreter, in one mode."""
    project = tmp_path / f"proj_{mode}"
    (project / "tests").mkdir(parents=True)
    (project / "m.py").write_text(source)
    (project / "tests" / "test_m.py").write_text(tests)
    (project / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\n'
    )
    (project / "run.py").write_text(_RUNNER)
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "run.py", function, mode],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["mode"] == ("isolated" if mode == "isolated" else "in_process")
    return out


def _verdicts(profile):
    """`{change: status[:by]}`; the identical mutant's change is the empty string."""
    rows = profile["rows"].values()
    table = {}
    for row in rows:
        verdict = row["status"] + (f":{row['by']}" if row["by"] else "")
        table.setdefault(row["change"], []).append(verdict)
    return {change: sorted(v) for change, v in table.items()}


@pytest.fixture(scope="module", params=["in_process", "isolated"])
def revision_profile(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rev_" + request.param)
    return _profile(tmp, _REVISION, _REVISION_TESTS, "revision", request.param)


def test_the_mutant_identical_to_the_original_is_never_killed(revision_profile):
    identical = [r for r in revision_profile["rows"].values() if r["change"] == ""]
    assert identical, "the fixture must still generate the identical SWAP mutant (#53)"
    assert all(r["status"] != "killed" for r in identical), identical


def test_the_sites_no_test_can_kill_survive(revision_profile):
    table = _verdicts(revision_profile)
    assert table["tag = max(2, 2) → tag = 2"] == ["survivor"]
    assert table["tag = max(2, 2) → tag = min(2, 2)"] == ["survivor"]


def test_the_two_type_error_mutants_are_crash_kills_on_the_authored_row(
    revision_profile,
):
    table = _verdicts(revision_profile)
    join_dropped = [
        v
        for change, v in table.items()
        if change.endswith("':' + str(tag)")
        and "return 'revised:' + new['objects']" in change
    ]
    assert join_dropped == [["killed:crash"]], table
    str_dropped = [v for change, v in table.items() if change.endswith("+ ':' + tag")]
    assert str_dropped == [["killed:crash"]], table


def test_the_writer_mutant_survives_because_no_authored_input_distinguishes_it(
    revision_profile,
):
    table = _verdicts(revision_profile)
    assert table["new = copy.deepcopy(written) → new = written"] == ["survivor"]


def test_nothing_is_unscored_and_every_argument_was_copied(revision_profile):
    assert revision_profile["unscored_by"] == {}
    assert revision_profile["shared"] == []


def test_both_modes_print_the_same_per_mutant_verdict_table(tmp_path):
    in_process = _profile(
        tmp_path, _REVISION, _REVISION_TESTS, "revision", "in_process"
    )
    isolated = _profile(tmp_path, _REVISION, _REVISION_TESTS, "revision", "isolated")
    assert _verdicts(in_process) == _verdicts(isolated)
    assert {k: (v["status"], v["by"]) for k, v in in_process["rows"].items()} == {
        k: (v["status"], v["by"]) for k, v in isolated["rows"].items()
    }


@pytest.mark.parametrize("mode", ["in_process", "isolated"])
def test_the_order_the_writer_runs_in_does_not_change_any_verdict(tmp_path, mode):
    """The writer first and the writer second must give the same verdicts for the same changes."""
    first = _profile(tmp_path / "a", _REVISION, _REVISION_TESTS, "revision", mode)
    second = _profile(
        tmp_path / "b", _REVISION_WRITER_LAST, _REVISION_TESTS, "revision", mode
    )

    def by_site(profile):
        # The two layouts move lines, not meaning: compare the verdicts of the changes both contain.
        return {
            change: verdicts
            for change, verdicts in _verdicts(profile).items()
            if change in _verdicts(first) and change in _verdicts(second)
        }

    common = by_site(first)
    assert common, "the layouts share no mutant text"
    assert common == {c: _verdicts(second)[c] for c in common}
    writer = "new = copy.deepcopy(written) → new = written"
    assert _verdicts(second)[writer] == ["survivor"]
    identical = [r for r in second["rows"].values() if r["change"] == ""]
    assert all(r["status"] != "killed" for r in identical)


# ── test arguments with no parameters at all ────────────────────────────────────────────────────

_PLAIN = textwrap.dedent(
    """
    def total(items):
        out = 0
        for item in items:
            out = out + item
        return out
    """
)
_PLAIN_TESTS = textwrap.dedent(
    """
    from m import total


    def test_total():
        assert total([1, 2, 3]) == 6
    """
)


@pytest.mark.parametrize("mode", ["in_process", "isolated"])
def test_a_suite_with_no_parameters_is_measured_exactly_as_before(tmp_path, mode):
    profile = _profile(tmp_path, _PLAIN, _PLAIN_TESTS, "total", mode)
    kills = [r for r in profile["rows"].values() if r["status"] == "killed"]
    assert kills, "a suite that pins the sum must kill something"
    assert profile["shared"] == []
    assert profile["unscored_by"] == {}


# ── shared state a copy cannot reach: the original control guards it ────────────────────────────

# `table = dict(REGISTRY)` -> `table = REGISTRY` writes the module-level REGISTRY for the rest of the
# process. No copy of a test argument reaches it; every test that reads REGISTRY afterwards fails on the
# ORIGINAL too. `weight = max(1, 1)` is a site no test can kill.
_GLOBAL = textwrap.dedent(
    """
    REGISTRY = {"base": 10}


    def price(qty):
        table = dict(REGISTRY)
        table["base"] = table["base"] + qty
        weight = max(1, 1)
        return table["base"] * weight
    """
)
_GLOBAL_TESTS = textwrap.dedent(
    """
    import pytest
    from m import price


    def test_one():
        assert price(1) == 11


    def test_two():
        assert price(2) == 12


    @pytest.mark.parametrize("qty,expected", [(3, 13)])
    def test_three(qty, expected):
        assert price(qty) == expected
    """
)


@pytest.mark.parametrize("mode", ["in_process", "isolated"])
def test_a_mutant_that_corrupts_a_module_global_never_manufactures_a_kill(
    tmp_path, mode
):
    profile = _profile(tmp_path, _GLOBAL, _GLOBAL_TESTS, "price", mode)
    table = _verdicts(profile)

    # The sites no test can kill are never killed, whatever ran before them.
    for change, verdicts in table.items():
        if "max(1, 1)" in change and (
            "min(1, 1)" in change or change.endswith("weight = 1")
        ):
            assert verdicts == ["survivor"] or all(
                v.startswith("unscored") for v in verdicts
            ), (change, verdicts)
    identical = [r for r in profile["rows"].values() if r["change"] == ""]
    assert all(r["status"] != "killed" for r in identical), identical

    # The writer's own run is named, not scored: its later tests fail identically on the original.
    writer = [
        r
        for r in profile["rows"].values()
        if r["change"] == "table = dict(REGISTRY) → table = REGISTRY"
    ]
    assert len(writer) == 1
    assert writer[0]["status"] != "killed", writer
    if writer[0]["status"] == "unscored":
        assert writer[0]["disposition"] == "original_also_fails", writer


# ── an argument a copy cannot be made of: named, and still guarded ──────────────────────────────

_BOX = textwrap.dedent(
    """
    def extend(box, n):
        out = list(box.items)
        out.append(n)
        weight = max(1, 1)
        return out[:len(out)] if weight == 1 else out
    """
)
_BOX_TESTS = textwrap.dedent(
    """
    import threading
    import pytest
    from m import extend


    class Box:
        def __init__(self, items):
            self.items = items
            self.lock = threading.Lock()


    @pytest.mark.parametrize("box,n,expected", [(Box([1]), 2, [1, 2])])
    def test_extend(box, n, expected):
        assert extend(box, n) == expected


    def test_extend_twice():
        box = Box([5])
        assert extend(box, 6) == [5, 6]
        assert extend(box, 7) == [5, 7]
    """
)


@pytest.mark.parametrize("mode", ["in_process", "isolated"])
def test_an_uncopyable_parametrize_value_is_named_and_a_writer_still_cannot_forge_a_kill(
    tmp_path, mode
):
    profile = _profile(tmp_path, _BOX, _BOX_TESTS, "extend", mode)

    assert any(
        "test_extend[" in entry and ":: box (uncopyable)" in entry
        for entry in profile["shared"]
    ), profile["shared"]
    identical = [r for r in profile["rows"].values() if r["change"] == ""]
    assert all(r["status"] != "killed" for r in identical), identical
    # `out = box.items` writes into the shared Box; the sites no test can kill stay unkilled.
    for row in profile["rows"].values():
        if "max(1, 1)" in row["change"] and row["status"] == "killed":
            pytest.fail(f"a site no test can kill was killed: {row}")


# ── a healthy test must never read as failing because it was handed a copy ──────────────────────
#
# `argument_copy_standing` clears an argument whose copy compares EQUAL, and equal is not identical.
# A test can assert `lookup(k) is x` against a module-level object, or write through `x` and read the
# module-level name. Handed a copy, the ORIGINAL fails it: a healthy test would read as failing, go
# inert, and lose its kills (measured: 0 of 1, 0 of 1, 0 of 2 killed where 1, 1, 2 were before #55).
# Copying is therefore a MEASURED precondition (`identity_standing`): an item's arguments are copies
# only if the original passes on a copy; if it passes on the authored objects and fails on the copy,
# they stay shared, named `identity_sensitive`, and the original control guards them.


@pytest.mark.parametrize(
    ("authored", "copy", "standing"),
    [
        (True, True, "copy_ok"),
        (False, True, "copy_ok"),
        (True, False, "identity_sensitive"),
        (False, False, "fails_on_original"),
    ],
)
def test_identity_standing_is_decided_by_the_original_on_both_kinds_of_argument(
    authored, copy, standing
):
    assert identity_standing(authored, copy) == standing


def test_an_identity_sensitive_item_is_handed_its_authored_objects_and_named():
    shared = {"a": [1]}
    item = _item("t.py::t[0]", x=shared)
    fresh = FreshArguments()
    fresh.capture(item)
    assert fresh.unmeasured(item)
    fresh.refresh(item)
    assert item.callspec.params["x"] is not shared

    fresh.use_authored(item)
    assert item.callspec.params["x"] is shared

    assert fresh.mark_by_identity(item, True) is True
    fresh.measured(item)
    item.callspec.params["x"] = object()
    fresh.refresh(item)  # no pristine any more: nothing is replaced
    assert not fresh.unmeasured(item)
    assert fresh.shared() == ["t.py::t[0] :: x (identity_sensitive)"]
    # A later sight of the item (the next evaluation's collection) never re-snapshots it.
    fresh.capture(item)
    assert fresh.shared() == ["t.py::t[0] :: x (identity_sensitive)"]


def test_an_item_that_fails_on_the_authored_objects_too_keeps_its_copies():
    item = _item("t.py::t[0]", x=[1])
    fresh = FreshArguments()
    fresh.capture(item)
    assert fresh.mark_by_identity(item, False) is False
    fresh.measured(item)
    fresh.refresh(item)
    assert fresh.shared() == []


_IDENTITY = (
    "SHARED = {'a': [1]}\n\n\ndef fetch():\n    return SHARED\n",
    (
        "import pytest\nfrom m import fetch, SHARED\n\n\n"
        "@pytest.mark.parametrize('x', [SHARED])\n"
        "def test_fetch(x):\n    assert fetch() is x\n"
    ),
    "fetch",
)
_CACHE = (
    "CACHE = {'k': ['v']}\n\n\ndef lookup(k):\n    return CACHE[k]\n",
    (
        "import pytest\nfrom m import lookup, CACHE\n\n\n"
        "@pytest.mark.parametrize('x', [CACHE['k']])\n"
        "def test_lookup(x):\n    assert lookup('k') is x\n"
    ),
    "lookup",
)
_MUTATES = (
    "REG = {'n': 0}\n\n\ndef count():\n    return REG['n']\n",
    (
        "import pytest\nfrom m import count, REG\n\n\n"
        "@pytest.mark.parametrize('reg', [REG])\n"
        "def test_count(reg):\n    reg['n'] = 5\n    assert count() == 5\n"
    ),
    "count",
)


@pytest.mark.parametrize("mode", ["in_process", "isolated"])
@pytest.mark.parametrize(
    ("fixture", "argument"),
    [(_IDENTITY, "x"), (_CACHE, "x"), (_MUTATES, "reg")],
    ids=[
        "identity-against-a-module-object",
        "module-level-cache",
        "writes-through-the-parameter",
    ],
)
def test_a_test_that_depends_on_object_identity_keeps_its_kills_and_is_named(
    tmp_path, mode, fixture, argument
):
    source, tests, function = fixture
    profile = _profile(tmp_path, source, tests, function, mode)

    rows = profile["rows"].values()
    assert rows
    assert all(r["status"] == "killed" for r in rows), [
        (r["change"], r["status"]) for r in rows
    ]
    assert profile["unscored_by"] == {}
    assert [
        e for e in profile["shared"] if f":: {argument} (identity_sensitive)" in e
    ], profile["shared"]


@pytest.mark.parametrize("mode", ["in_process", "isolated"])
def test_an_ordinary_parameter_is_still_copied_and_the_55_results_are_unchanged(
    tmp_path, mode
):
    profile = _profile(tmp_path, _REVISION, _REVISION_TESTS, "revision", mode)
    assert profile["shared"] == []
    table = _verdicts(profile)
    assert table["tag = max(2, 2) → tag = 2"] == ["survivor"]
    assert table["new = copy.deepcopy(written) → new = written"] == ["survivor"]
    identical = [r for r in profile["rows"].values() if r["change"] == ""]
    assert identical and all(r["status"] != "killed" for r in identical)
