"""Intent: one profile run can carry the per-test kill relation, and asking for it changes no verdict
(Detective #145).

THE DEFECT, measured 2026-10-26 on Detective d2750f7 / Wesker 2fbd029. Detective's audit read its redundant
set and its minimal cover off Wesker's `kill_matrix`, which credits each mutant to the FIRST covering test
that kills it. Which test is first follows execution order, and execution order follows the trace cache's
warmth: the same tree and suite read a cover of 2 or of 3. The relation "test t kills mutant m"
(KNOWABILITY Def. 2.3) is per test, but `run_function_profiling` had no way to ask for it: `full_matrix`
reached only `run_function_converged`, and the isolated worker stopped at the first value kill and
returned one `test_name`. Detective paid for the relation by profiling the function once per test
(audit 0.9 s -> 3.6 s on the fixture, 27 s -> 109 s on an 18-test function).

THE RULING. The relation is a product of the run that produced the verdict, over the tests that run
reached, in either execution mode: `run_function_profiling(full_matrix=True)` runs every test that reaches
a mutant and records, per killed mutant, EVERY killer with that test's own reason. And it is only a field:
each mutant's killed / survived / unscored disposition and its `killed_by` are what they are with the mode
off. A test the original also fails on is determinately not a killer (#55); a test stopped by its loop
budget, a wall, or a refused measurement is an UNRESOLVED cell, never a killer and never a pass; a mutant
with no verdict has no row.

These are written from that ruling: the pure decisions, the in-process evaluator on hand-built mutants,
the isolated worker, and the whole profile in both modes.
"""

from __future__ import annotations

import ast
import sys

import pytest

from Wesker import engine
from Wesker.engine import (
    ORIGINAL_ALSO_FAILS,
    kill_relation_standing,
    killer_scan_standing,
    note_scan_cell,
    stronger_kill_reason,
)
from Wesker.isolation import IsolatedMutantWorker, decided_kill_reason

# ── the pure decisions ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("held", "new", "stronger"),
    [
        (None, "crash", "crash"),
        (None, "assertion", "assertion"),
        ("crash", "assertion", "assertion"),
        ("assertion", "crash", "assertion"),
        ("exception", "assertion", "assertion"),
        ("assertion", "exception", "assertion"),
        ("crash", "exception", "exception"),
        ("crash", "crash", "crash"),
        # a reason outside the kill vocabulary never displaces one inside it
        ("crash", "diverged", "crash"),
        (None, "diverged", "diverged"),
        ("bogus", "crash", "crash"),
    ],
)
def test_the_stronger_kill_reason_is_value_specification_precedence(
    held, new, stronger
):
    assert stronger_kill_reason(held, new) == stronger


@pytest.mark.parametrize(
    ("total", "scanned", "unresolved", "standing"),
    [
        (3, 3, 0, "complete"),
        (0, 0, 0, "complete"),  # no test reaches it: nothing was left unscanned
        (3, 2, 0, "cut"),
        (
            3,
            1,
            2,
            "cut",
        ),  # a cut outranks unresolved cells: the unrun cells are unknown too
        (3, 3, 1, "unresolved"),
    ],
)
def test_a_mutants_scan_is_complete_cut_or_unresolved(
    total, scanned, unresolved, standing
):
    assert killer_scan_standing(total, scanned, unresolved) == standing


@pytest.mark.parametrize(
    ("args", "standing"),
    [
        ((False, False, 3, 3, 0), "not_requested"),
        ((False, True, 0, 0, 5), "not_requested"),  # not asking outranks everything
        ((True, True, 3, 3, 0), "run_cut"),
        ((True, False, 3, 3, 1), "scan_incomplete"),
        ((True, False, 3, 2, 0), "row_missing"),
        ((True, False, 3, 3, 0), "complete"),
        (
            (True, False, 0, 0, 0),
            "complete",
        ),  # nothing killed: the empty relation is a claim
    ],
)
def test_the_profiles_relation_standing_names_why_it_is_or_is_not_whole(args, standing):
    assert kill_relation_standing(*args) == standing


@pytest.mark.parametrize(
    ("reasons", "decided"),
    [
        ([], ""),
        (["crash"], "crash"),
        (["crash", "crash"], "crash"),
        (["assertion"], "assertion"),
        (["crash", "assertion"], "assertion"),
        # two value pins: the FIRST in run order decides, as the unscanned run (which stops there) names it
        (["exception", "assertion"], "exception"),
        (["assertion", "exception"], "assertion"),
        (["crash", "exception", "assertion"], "exception"),
    ],
)
def test_the_first_value_pin_in_run_order_decides_the_kill(reasons, decided):
    assert decided_kill_reason(reasons) == decided


def test_note_scan_cell_keeps_kills_unresolved_cells_and_ignores_passes_and_voided_failures():
    reasons: dict[str, str] = {}
    unresolved: list[str] = []
    for test_id, outcome, refused in [
        ("a", "assertion", False),
        ("b", "crash", False),
        ("c", None, False),  # passed, or its failure was voided by the original control
        ("d", "diverged", False),
        ("e", "undetermined", False),
        ("f", "timeout", False),
        (
            "g",
            None,
            True,
        ),  # a refused measurement: the outcome is gone but the cell is unresolved
        ("a", "crash", False),  # the same test again: the stronger reason stands
    ]:
        note_scan_cell(reasons, unresolved, test_id, outcome, refused)
    assert reasons == {"a": "assertion", "b": "crash"}
    assert unresolved == ["d", "e", "f", "g"]


# ── the in-process evaluator, on hand-built mutants ─────────────────────────────────────────────

_ORIGINAL = "def subject(x=None):\n    return 7\n"
_MUTANT = (
    "def subject(x=None):\n"
    "    if x == 'boom':\n"
    "        raise TypeError('mutant')\n"
    "    return 8\n"
)


def _mutant_and_original():
    namespace: dict = {}
    # S102: an original outside live module bindings, so injection of the mutant is real.
    exec(compile(_ORIGINAL, "<relation-original>", "exec"), namespace)  # noqa: S102
    mutant = engine.Mutant(
        category=engine.MutationCategory.VALUE,
        original_node=ast.parse(_ORIGINAL).body[0],
        mutated_node=ast.parse(_MUTANT).body[0],
        description="relation mutant",
    )
    return mutant, namespace["subject"]


def by_assertion(fn=None):
    assert fn() == 7


def by_crash(fn=None):
    fn("boom")


def by_assertion_again(fn=None):
    assert fn(0) == 7


def never_kills(fn=None):
    fn(1)


def fails_on_the_original_too(fn=None):
    raise AssertionError("broken whatever is under test")


def _evaluate(tests, full=True, **kwargs):
    mutant, original = _mutant_and_original()
    return engine.evaluate_mutant(
        mutant,
        tests,
        original,
        qualname="subject",
        record_all_killers=full,
        **kwargs,
    )


def _ids(result):
    return {t.rsplit(".", 1)[-1].rsplit("::", 1)[-1] for t in result.killer_reasons}


def test_a_mutant_killed_by_assertion_in_one_test_and_crash_in_another_has_both_rows():
    result = _evaluate([by_crash, by_assertion])
    assert result.killed and result.killed_by == "assertion"
    assert _ids(result) == {"by_crash", "by_assertion"}
    assert sorted(result.killer_reasons.values()) == ["assertion", "crash"]
    assert result.killer_scan == "complete" and result.unresolved_tests == []


@pytest.mark.parametrize(
    "order",
    [
        [by_assertion, by_crash, by_assertion_again, never_kills],
        [by_crash, by_assertion, never_kills, by_assertion_again],
        [never_kills, by_assertion_again, by_crash, by_assertion],
    ],
)
def test_the_relation_does_not_follow_the_order_the_tests_ran_in(order):
    result = _evaluate(order)
    assert _ids(result) == {"by_assertion", "by_crash", "by_assertion_again"}
    assert result.killed_by == "assertion"


@pytest.mark.parametrize(
    "tests",
    [
        [by_assertion],
        [by_crash],
        [by_crash, by_assertion],
        [by_assertion, by_crash, by_assertion_again],
        [never_kills],
        [never_kills, by_crash],
        [fails_on_the_original_too],
        [fails_on_the_original_too, by_assertion],
        [],
    ],
)
def test_the_scan_changes_no_verdict(tests):
    scanned, plain = _evaluate(tests), _evaluate(tests, full=False)
    for field in (
        "killed",
        "killed_by",
        "test_name",
        "entered",
        "refusal",
        "refused_test",
        "contained",
    ):
        assert getattr(scanned, field) == getattr(plain, field), field
    # and the unscanned run claims no relation at all
    assert (
        plain.killer_scan == ""
        and plain.killer_reasons == {}
        and plain.killed_by_tests == []
    )


def test_a_single_test_is_the_whole_relation():
    result = _evaluate([by_assertion])
    assert (
        list(result.killer_reasons.values()) == ["assertion"]
        and result.killer_scan == "complete"
    )


def test_two_tests_with_identical_kill_sets_both_appear():
    result = _evaluate([by_assertion, by_assertion_again])
    assert _ids(result) == {"by_assertion", "by_assertion_again"}


def test_a_mutant_no_test_kills_has_an_empty_complete_row_and_is_a_survivor():
    result = _evaluate([never_kills])
    assert not result.killed and result.killer_reasons == {}
    assert result.killer_scan == "complete"


def test_a_test_the_original_also_fails_is_not_a_killer_and_not_unresolved():
    result = _evaluate([fails_on_the_original_too, by_assertion])
    assert _ids(result) == {"by_assertion"}
    assert result.unresolved_tests == [] and result.killer_scan == "complete"
    alone = _evaluate([fails_on_the_original_too])
    assert not alone.killed and alone.refusal == ORIGINAL_ALSO_FAILS
    assert alone.killer_reasons == {}


def test_not_asking_for_the_scan_claims_no_relation_and_stops_at_the_first_value_kill():
    ran: list[str] = []

    def first(fn=None):
        ran.append("first")
        assert fn() == 7

    def second(fn=None):
        ran.append("second")
        assert fn() == 7

    plain = _evaluate([first, second], full=False)
    # the original control re-runs the failing test, so compare which tests ran, not how often
    assert plain.killed and set(ran) == {"first"}
    assert plain.killer_scan == "" and plain.killer_reasons == {}
    ran.clear()
    scanned = _evaluate([first, second])
    assert set(ran) == {"first", "second"} and len(scanned.killer_reasons) == 2


def test_a_test_stopped_by_its_loop_budget_is_an_unresolved_cell_not_a_killer(
    monkeypatch,
):
    real = engine._run_test_with_timeout
    seen: list[str] = []

    def stopped_for_one(test_fn, *args, **kwargs):
        seen.append(test_fn.__name__)
        if test_fn.__name__ == "never_kills":
            return "diverged"
        return real(test_fn, *args, **kwargs)

    monkeypatch.setattr(engine, "_run_test_with_timeout", stopped_for_one)
    result = _evaluate([by_assertion, never_kills])
    assert result.killed and result.killed_by == "assertion"
    assert _ids(result) == {"by_assertion"}
    assert [
        t.rsplit("::", 1)[-1].rsplit(".", 1)[-1] for t in result.unresolved_tests
    ] == ["never_kills"]
    assert result.killer_scan == "unresolved"
    # the verdict is the unscanned run's
    assert (
        result.killed_by == _evaluate([by_assertion, never_kills], full=False).killed_by
    )


def test_an_allowance_spent_before_every_test_ran_is_a_cut_scan_with_the_verdict_intact(
    monkeypatch,
):
    real = engine._run_test_with_timeout

    def slow(test_fn, *args, **kwargs):
        import time

        time.sleep(0.05)
        return real(test_fn, *args, **kwargs)

    monkeypatch.setattr(engine, "_run_test_with_timeout", slow)
    result = _evaluate([by_assertion, by_crash], timeout_ms=20)
    assert result.killed and result.killed_by == "assertion"
    assert result.killer_scan == "cut"
    assert len(result.killer_reasons) == 1


# ── the isolated worker ─────────────────────────────────────────────────────────────────────────


def _project(tmp_path, tests):
    (tmp_path / "t.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "test_t.py").write_text("from t import f\n\n\n" + tests)
    return [
        "test_t.py::" + name
        for name in ("test_a", "test_b", "test_c", "test_broken")
        if name in tests
    ]


_TESTS = (
    "def test_a():\n    assert f(1) == 2\n\n\n"
    "def test_b():\n    assert f(5) == 6\n\n\n"
    "def test_c():\n    assert f(0) == 1\n"
)


def _isolated(tmp_path, tests, nodes, full):
    worker = IsolatedMutantWorker(str(tmp_path), nodes, "t.py", "f")
    try:
        return worker.evaluate(
            "def f(x):\n    return x - 1\n", 60.0, node_ids=nodes, full_matrix=full
        )
    finally:
        worker.close()


def test_the_isolated_worker_runs_every_node_and_reports_each_killers_own_reason(
    tmp_path,
):
    nodes = _project(tmp_path, _TESTS)
    run = _isolated(tmp_path, _TESTS, nodes, full=True)
    assert run.killed_by == "assertion"
    assert run.killers == tuple((node, "assertion") for node in nodes)
    assert run.unresolved_nodes == ()


def test_the_isolated_worker_without_the_scan_stops_at_the_first_value_kill_and_claims_no_row(
    tmp_path,
):
    nodes = _project(tmp_path, _TESTS)
    run = _isolated(tmp_path, _TESTS, nodes, full=False)
    assert run.killed_by == "assertion" and run.test_name == nodes[0]
    assert (
        run.killers is None
    )  # a list here would be a prefix of the relation, never the relation


def test_a_node_the_original_control_voids_is_not_a_killer_in_the_isolated_scan(
    tmp_path,
):
    tests = _TESTS + "\n\ndef test_broken():\n    assert False\n"
    nodes = _project(tmp_path, tests)
    run = _isolated(tmp_path, tests, nodes, full=True)
    assert "test_t.py::test_broken" not in {n for n, _ in run.killers}
    assert {n for n, _ in run.killers} == {
        "test_t.py::test_a",
        "test_t.py::test_b",
        "test_t.py::test_c",
    }
    assert run.killed_by == "assertion"


def test_a_crash_and_an_assertion_in_the_isolated_scan_keep_the_assertion_as_the_mutants_reason(
    tmp_path,
):
    tests = "def test_a():\n    assert f('x' if False else 1) == 2\n\n\ndef test_b():\n    f(None)\n"
    (tmp_path / "t.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "test_t.py").write_text("from t import f\n\n\n" + tests)
    nodes = ["test_t.py::test_b", "test_t.py::test_a"]
    worker = IsolatedMutantWorker(str(tmp_path), nodes, "t.py", "f")
    try:
        # the mutant raises on None only: test_b crashes under it AND under the original, so it is voided
        run = worker.evaluate(
            "def f(x):\n    return x - 1\n", 60.0, node_ids=nodes, full_matrix=True
        )
    finally:
        worker.close()
    reasons = dict(run.killers)
    assert reasons.get("test_t.py::test_a") == "assertion"
    assert (
        "test_t.py::test_b" not in reasons
    )  # f(None) raises under the original as well: no difference


# ── the whole profile, both modes ───────────────────────────────────────────────────────────────

_REVISION = (
    "import copy\n\n\n"
    "def revision(current: dict, written: dict) -> str:\n"
    "    new = copy.deepcopy(written)\n"
    "    for name in current.get('objects', []):\n"
    "        if name in new['objects']:\n"
    "            return 'version:duplicate:' + name\n"
    "        new['objects'].append(name)\n"
    "    tag = max(2, 2)\n"
    "    return 'revised:' + ','.join(new['objects']) + ':' + str(tag)\n"
)
_REVISION_TESTS = (
    "import pytest\nfrom m import revision\n\n\n"
    "@pytest.mark.parametrize(\n    'args,expected',\n"
    "    [(({'objects': ['B', 'C']}, {'objects': ['A']}), 'revised:A,B,C:2')],\n)\n"
    "def test_revision_golden(args, expected):\n    assert revision(*args) == expected\n\n\n"
    "def test_duplicate():\n"
    "    assert revision({'objects': ['A']}, {'objects': ['A']}) == 'version:duplicate:A'\n"
)


_RUNNER = """
import ast, importlib, json, os, sys
from Wesker.ci import _PROJECT_ROOT, discover_test_callables, run_with_live_suite
from Wesker.engine import run_function_profiling
from Wesker.filter import filter_categories

root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)
node = next(n for n in ast.parse(open("m.py").read()).body if getattr(n, "name", "") == "revision")
isolated, full = sys.argv[1] == "isolated", sys.argv[2] == "full"


def body():
    tests = discover_test_callables(root, "m.py", ["revision"])
    original = importlib.import_module("m").revision
    token = _PROJECT_ROOT.set(root)
    try:
        return run_function_profiling(
            node, "m.py::revision", filter_categories(node), tests, original,
            isolated=isolated, full_matrix=full,
        )
    finally:
        _PROJECT_ROOT.reset(token)


r = run_with_live_suite(root, body, target_files=["m.py"])
rows = {}
for kind in ("killed_records", "survivor_records", "unscored_records"):
    for rec in getattr(r, kind):
        rows[rec["mutant_id"]] = [kind, rec.get("killed_by"), rec.get("test"), rec.get("disposition")]
print(json.dumps({
    "rows": rows,
    "kill_matrix": r.kill_matrix,
    "relation": r.kill_relation,
    "standing": r.kill_relation_standing,
    "gateable": r.is_gateable,
    "totals": [r.total_mutants, r.total_killed, r.total_survived],
}, sort_keys=True))
"""


def _profile(tmp_path, isolated, full):
    """Profile #55's fixture (`m.py::revision`: one parametrized row, one duplicate-path test) in a
    disposable project, in a fresh interpreter, in one mode, with the scan on or off."""
    import os
    import subprocess

    project = tmp_path / f"proj_{isolated}_{full}"
    (project / "tests").mkdir(parents=True)
    (project / "m.py").write_text(_REVISION)
    (project / "tests" / "test_m.py").write_text(_REVISION_TESTS)
    (project / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\n'
    )
    (project / "run.py").write_text(_RUNNER)
    env = dict(os.environ, HOME=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, "run.py", isolated, full],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    import json

    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def profiles(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("relation")
    return {
        (mode, scan): _profile(tmp, mode, scan)
        for mode in ("in_process", "isolated")
        for scan in ("full", "plain")
    }


def test_the_scan_changes_no_verdict_in_either_mode(profiles):
    for mode in ("in_process", "isolated"):
        assert profiles[(mode, "full")]["rows"] == profiles[(mode, "plain")]["rows"], (
            mode
        )
        assert (
            profiles[(mode, "full")]["kill_matrix"]
            == profiles[(mode, "plain")]["kill_matrix"]
        ), mode
        assert (
            profiles[(mode, "full")]["totals"] == profiles[(mode, "plain")]["totals"]
        ), mode
        assert (
            profiles[(mode, "full")]["gateable"]
            == profiles[(mode, "plain")]["gateable"]
        ), mode


def test_without_the_scan_there_is_no_relation_and_it_says_so(profiles):
    for mode in ("in_process", "isolated"):
        plain = profiles[(mode, "plain")]
        assert plain["standing"] == "not_requested" and plain["relation"] == {}


def test_the_relation_is_the_same_in_process_and_isolated(profiles):
    inproc, iso = profiles[("in_process", "full")], profiles[("isolated", "full")]
    assert inproc["standing"] == iso["standing"] == "complete"
    assert inproc["relation"] == iso["relation"]
    assert inproc["relation"], (
        "the fixture kills mutants: an empty relation would make this vacuous"
    )


def test_every_killed_mutant_has_a_row_naming_a_test_and_a_reason_and_the_credit_is_inside_it(
    profiles,
):
    for mode in ("in_process", "isolated"):
        run = profiles[(mode, "full")]
        for mutant, killers in run["kill_matrix"].items():
            row = run["relation"][mutant]
            assert row and set(row.values()) <= {"assertion", "exception", "crash"}, (
                mode,
                mutant,
            )
            assert set(killers) <= set(row), (mode, mutant)


def test_a_mutant_two_tests_kill_is_credited_to_one_but_both_are_in_the_relation(
    profiles,
):
    run = profiles[("in_process", "full")]
    both = [m for m, row in run["relation"].items() if len(row) > 1]
    assert both, "the fixture has a mutant both tests kill"
    assert all(len(run["kill_matrix"][m]) == 1 for m in both)


def test_a_mutant_with_no_verdict_has_no_row(profiles):
    for mode in ("in_process", "isolated"):
        run = profiles[(mode, "full")]
        unscored = {m for m, row in run["rows"].items() if row[0] == "unscored_records"}
        survivors = {
            m for m, row in run["rows"].items() if row[0] == "survivor_records"
        }
        assert not (unscored | survivors) & {m.split(":")[0] for m in run["relation"]}
