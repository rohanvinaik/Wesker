"""#32 — the SAME check, as a property that can be re-run: per-TestId lines and arcs do not depend on
HOW the interpreter is instrumented or WHICH interpreter runs.

THE DEFECT. The migration of line coverage and arcs onto `sys.monitoring` was acceptance-gated on
"byte-identical line coverage, arcs, captures and FINAL verdicts, before vs after". The pre-migration
code cannot be re-run cheaply, so that comparison was never recorded. What can be pinned now is the
property the comparison stood for: what a test is credited with must not change with

  (i)   in-process vs the isolated worker (lines; the worker records no arcs),
  (ii)  a FOREIGN `sys.monitoring` tool being active beside the run — a coverage.py-like LINE tool
        on another tool id, and a tool that disables PY_START at every location,
  (iii) the Python version (3.12 vs 3.14): the golden file is written once and asserted under every
        interpreter this suite runs on, so equality across versions follows from both passing.

THE CORPUS. Two fixture projects written below (branches, loops with break/continue/else, try/except/
finally, comprehensions, generators, nested functions, recursion, `with`, `while`, short-circuits, a
raising path) with ten tests, and FIVE real functions of this package traced through a renamed
package copy (`WeskerUUT`; module identity is by `co_filename`, so the live tree cannot trace itself):
`item_run_status`, `baseline_outcome`, `trace_admissibility`, `reach_qualification` and
`build_trace_ledger`. A difference in any cell fails the test with the cell named; it is never
loosened to pass.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

import Wesker
from Wesker.isolation import run_baseline_traced_isolated
from Wesker.line_coverage import trace_suite

_GOLDEN = Path(__file__).parent / "data" / "trace_equality_golden.json"

_PROJECTS = {
    "corp_a": (
        """def classify(n):
    if n < 0:
        return "neg"
    elif n == 0:
        return "zero"
    total = 0
    for i in range(n):
        if i % 2:
            continue
        if i > 6:
            break
        total += i
    else:
        total += 100
    return "small" if total < 10 else "big"


def guarded(a, b):
    try:
        value = a // b
    except ZeroDivisionError:
        value = -1
    finally:
        a = 0
    return value or "falsy"


def squares(xs):
    return [x * x for x in xs if x % 2 == 0] or ["none"]


def count_up(limit):
    def gen():
        k = 0
        while k < limit:
            yield k
            k += 1

    return sum(gen())


def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)


def either(a, b):
    if a or b:
        return 1
    return 0
""",
        """from corp_a import classify, guarded, squares, count_up, fact, either


def test_neg():
    assert classify(-1) == "neg"


def test_zero_and_small():
    assert classify(0) == "zero"
    assert classify(3) == "big"


def test_big_breaks():
    assert classify(20) == "big"


def test_guarded():
    assert guarded(6, 3) == 2
    assert guarded(1, 0) == -1
    assert guarded(0, 5) == "falsy"


def test_squares():
    assert squares([1, 2, 3, 4]) == [4, 16]
    assert squares([1]) == ["none"]


def test_gen_and_recursion():
    assert count_up(4) == 6
    assert fact(5) == 120


def test_either():
    assert either(True, False) == 1
    assert either(False, False) == 0
""",
    ),
    "corp_b": (
        """import contextlib


def parse(text):
    parts = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(int(token))
        except ValueError:
            parts.append(token)
    return parts


def summarize(items):
    with contextlib.suppress(TypeError):
        return sum(items)
    return None


def pick(d, key, default=None):
    if key in d and d[key] is not None:
        return d[key]
    return default
""",
        """from corp_b import parse, summarize, pick


def test_parse():
    assert parse("1, 2,,x") == [1, 2, "x"]


def test_summarize():
    assert summarize([1, 2]) == 3
    assert summarize(None) is None


def test_pick():
    assert pick({"a": 1}, "a") == 1
    assert pick({"a": None}, "a", 0) == 0
    assert pick({}, "z") is None
""",
    ),
}


def _real_tests(uut):
    """Tests over five real functions of the package, from the renamed copy ``uut``."""
    te = uut

    def test_item_run_status():
        assert te.item_run_status("failed", "", "") == "error"
        assert te.item_run_status("passed", "failed", "passed") == "failed"
        assert te.item_run_status("passed", "passed", "failed") == "error"
        assert te.item_run_status("passed", "xfailed", "passed") == "xfailed"
        assert te.item_run_status("skipped", "", "") == "skipped"
        assert te.item_run_status("passed", "xpassed", "passed") == "passed"
        assert te.item_run_status("", "", "") == "unobserved"

    def test_baseline_outcome():
        assert te.baseline_outcome("timeout", "passed") == "error"
        assert te.baseline_outcome(None, "marked") == "untyped"
        assert te.baseline_outcome(None, "failed") == "failed"
        assert te.baseline_outcome("assertion", "unobserved") == "failed"
        assert te.baseline_outcome("crash", "unobserved") == "untyped"
        assert te.baseline_outcome(None, "skipped") == "skipped"
        assert te.baseline_outcome(None, "unobserved") == "passed"

    def test_trace_admissibility():
        assert te.trace_admissibility(True, False, True) == "admissible"
        assert te.trace_admissibility(True, False, False) == "refuse_uncontained"
        assert te.trace_admissibility(True, True, True) == "refuse_truncated"
        assert (
            te.trace_admissibility(False, False, True, True, "error") == "refuse_error"
        )
        assert (
            te.trace_admissibility(False, False, True, True, "untyped")
            == "refuse_untyped"
        )
        assert (
            te.trace_admissibility(False, False, True, True, "skipped")
            == "refuse_skipped"
        )
        assert (
            te.trace_admissibility(False, False, True, True, "xfailed")
            == "refuse_xfailed"
        )
        assert te.trace_admissibility(False, False, True) == "refuse_failed"
        assert te.trace_admissibility(True, False, True, False) == "refuse_replayed"

    def test_reach_qualification():
        assert te.reach_qualification(False, False) == "complete"
        assert te.reach_qualification(True, False) == "incomplete_thread"
        assert te.reach_qualification(True, True, True).count("+") == 2

    def test_build_trace_ledger():
        rows = te.build_trace_ledger(
            {"a": [1, 2], "b": [3]},
            ["b"],
            [],
            True,
            arc_coverage={"a": [(1, 2)]},
            outcomes={"a": "passed", "b": "failed"},
            incomplete_reach={"a": "incomplete_subprocess"},
        )
        assert [r.reason for r in rows] == ["admissible", "refuse_failed"]
        assert rows[0].reach == "incomplete_subprocess"

    return [
        test_item_run_status,
        test_baseline_outcome,
        test_trace_admissibility,
        test_reach_qualification,
        test_build_trace_ledger,
    ]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tests_of(module):
    return [
        getattr(module, n)
        for n in sorted(vars(module))
        if n.startswith("test_") and callable(getattr(module, n))
    ]


def _renamed_copy(tmp_path: Path) -> Path:
    """``trace_evidence.py`` of this package as ``WeskerUUT/trace_evidence.py`` (a path of its own)."""
    pkg = tmp_path / "uut" / "WeskerUUT"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    source = (Path(Wesker.__file__).parent / "trace_evidence.py").read_text()
    (pkg / "trace_evidence.py").write_text(re.sub(r"\bWesker\b", "WeskerUUT", source))
    return pkg / "trace_evidence.py"


def _cells(traced, arcs, target: str, tests) -> dict[str, dict[str, list]]:
    ids = {t.__name__: _id(t) for t in tests}
    out = {}
    for name, tid in ids.items():
        files = traced.get(tid, {})
        out[name] = {
            "lines": sorted(_at(files, target)),
            "arcs": sorted([a, b] for a, b in _at(arcs.get(tid, {}), target)),
        }
    return out


def _id(test) -> str:
    from Wesker.ci import callable_test_id

    return callable_test_id(test)


def _at(mapping, target):
    got = mapping.get(target)
    if got is not None:
        return got
    real = os.path.realpath(target)
    for key, value in mapping.items():
        if os.path.realpath(key) == real:
            return value
    return set()


def _trace(tests, target: str) -> dict[str, dict[str, list]]:
    arcs: dict = {}
    traced = trace_suite(tests, {target}, arcs_out=arcs)
    return _cells(traced, arcs, target, tests)


@contextmanager
def _coverage_like_tool():
    """A coverage.py-style tool on ANOTHER tool id: LINE everywhere, DISABLE after each first hit."""
    mon = sys.monitoring
    slot = next((i for i in range(6) if mon.get_tool(i) is None), None)
    if slot is None:
        pytest.skip("no free monitoring tool id for the foreign tool")
    seen = []
    mon.use_tool_id(slot, "foreign-coverage-like")
    mon.register_callback(
        slot, mon.events.LINE, lambda code, line: (seen.append(line), mon.DISABLE)[1]
    )
    mon.set_events(slot, mon.events.LINE)
    try:
        yield seen
    finally:
        mon.set_events(slot, 0)
        mon.register_callback(slot, mon.events.LINE, None)
        mon.free_tool_id(slot)


@contextmanager
def _start_disabler_tool():
    mon = sys.monitoring
    slot = next((i for i in range(6) if mon.get_tool(i) is None), None)
    if slot is None:
        pytest.skip("no free monitoring tool id for the foreign tool")
    mon.use_tool_id(slot, "foreign-start-disabler")
    mon.register_callback(slot, mon.events.PY_START, lambda *a: mon.DISABLE)
    mon.set_events(slot, mon.events.PY_START)
    try:
        yield
    finally:
        mon.set_events(slot, 0)
        mon.register_callback(slot, mon.events.PY_START, None)
        mon.free_tool_id(slot)


def _build(tmp_path: Path):
    """Write the corpus; return ``[(label, target_path, tests, project_dir_or_None)]``."""
    cases = []
    for mod, (target_src, tests_src) in _PROJECTS.items():
        proj = tmp_path / mod
        proj.mkdir()
        target = proj / f"{mod}.py"
        target.write_text(target_src)
        (proj / f"test_{mod}.py").write_text(tests_src)
        sys.path.insert(0, str(proj))
        sys.modules.pop(mod, None)
        _load(target, mod)
        tests = _tests_of(_load(proj / f"test_{mod}.py", f"test_{mod}"))
        cases.append((mod, os.path.realpath(target), tests, proj))
    copy = _renamed_copy(tmp_path)
    sys.path.insert(0, str(copy.parent.parent))
    for name in ("WeskerUUT", "WeskerUUT.trace_evidence"):
        sys.modules.pop(name, None)
    uut = _load(copy, "WeskerUUT.trace_evidence")
    cases.append(
        ("real_trace_evidence", os.path.realpath(copy), _real_tests(uut), None)
    )
    return cases


@pytest.fixture
def corpus(tmp_path):
    saved_path = list(sys.path)
    saved_mods = set(sys.modules)
    try:
        yield _build(tmp_path)
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_mods:
            sys.modules.pop(name, None)
        shutil.rmtree(tmp_path, ignore_errors=True)


def _baseline(corpus):
    return {label: _trace(tests, target) for label, target, tests, _ in corpus}


def _normalized(cells: dict) -> dict:
    """The golden is portable: keyed by corpus label and test name, lines and arcs only."""
    return json.loads(json.dumps(cells, sort_keys=True))


def test_every_corpus_test_passes_so_the_cells_are_green_observations(corpus):
    for _label, _target, tests, _proj in corpus:
        for test in tests:
            test()  # a failing corpus test would be traced all the same: say so here


def test_every_corpus_cell_observes_something(corpus):
    cells = _baseline(corpus)
    seen = sum(1 for label in cells.values() for c in label.values() if c["lines"])
    total = sum(len(label) for label in cells.values())
    assert total == 10 + 5
    assert seen == total, (
        "a corpus test traced nothing: the comparison below would be vacuous"
    )
    assert any(c["arcs"] for label in cells.values() for c in label.values())


def test_a_foreign_coverage_like_tool_changes_nothing(corpus):
    base = _baseline(corpus)
    with _coverage_like_tool() as seen:
        again = _baseline(corpus)
    assert seen, "the foreign tool never fired: the comparison would be vacuous"
    assert again == base


def test_a_foreign_py_start_disabler_changes_nothing(corpus):
    base = _baseline(corpus)
    with _start_disabler_tool():
        again = _baseline(corpus)
    assert again == base


def test_both_foreign_tools_together_change_nothing(corpus):
    base = _baseline(corpus)
    with _coverage_like_tool(), _start_disabler_tool():
        again = _baseline(corpus)
    assert again == base


def test_a_foreign_setprofile_hook_and_cprofile_change_nothing(corpus):
    import cProfile

    base = _baseline(corpus)
    calls = []

    def foreign(frame, event, arg):
        calls.append(event)

    sys.setprofile(foreign)
    try:
        with_hook = _baseline(corpus)
    finally:
        sys.setprofile(None)
    assert calls, (
        "the foreign profile hook never fired: the comparison would be vacuous"
    )
    assert with_hook == base
    prof = cProfile.Profile()
    prof.enable()
    try:
        with_cprofile = _baseline(corpus)
    finally:
        prof.disable()
    assert with_cprofile == base


def test_a_running_coverage_py_changes_nothing(corpus, tmp_path):
    import coverage

    base = _baseline(corpus)
    cov = coverage.Coverage(
        data_file=str(tmp_path / ".coverage-foreign"), source=[str(tmp_path)]
    )
    cov.start()
    try:
        under = _baseline(corpus)
    finally:
        cov.stop()
    assert under == base
    assert cov.get_data().measured_files(), (
        "coverage.py measured nothing: vacuous comparison"
    )


def test_in_process_and_isolated_credit_each_test_with_the_same_lines(corpus):
    """Lines only: the isolated worker records no arcs. Fixture projects (real pytest node ids)."""
    import ast

    from Wesker.line_coverage import executable_lines

    base = _baseline(corpus)
    compared = 0
    for label, target, tests, proj in corpus:
        if proj is None:
            continue
        # The fresh worker IMPORTS the target while it is traced, so it also records module-level
        # lines (`def`, `import`) that an already-imported in-process target never re-executes. Those
        # are not any function's reach; the engine intersects with a function's executable lines
        # (`exec_lines`) before use, so that is the comparison made here.
        body_lines = set()
        for node in ast.walk(ast.parse(Path(target).read_text())):
            if isinstance(node, ast.FunctionDef):
                body_lines |= executable_lines(node)
        for test in tests:
            node = f"test_{label}.py::{test.__name__}"
            lines, outcome, contained = run_baseline_traced_isolated(
                str(proj), [node], target, 60.0
            )
            assert contained and outcome == "passed", (node, outcome)
            assert (
                sorted(set(lines) & body_lines) == base[label][test.__name__]["lines"]
            ), f"{node}: isolated and in-process disagree"
            compared += 1
    assert compared == 10


def test_the_fixture_cells_equal_the_golden_recorded_under_every_interpreter(corpus):
    """(iii) 3.12 vs 3.14: one golden, asserted on whichever interpreters run this suite."""
    cells = _normalized(_baseline(corpus))
    golden = json.loads(_GOLDEN.read_text())
    fixtures = {k: v for k, v in cells.items() if k in _PROJECTS}
    assert fixtures == golden


def test_the_real_functions_are_credited_with_every_line_their_tests_reach(corpus):
    """The real-function cells move with the source, so they have no golden; what must hold under
    every interpreter and every instrumentation is that the lines credited to the tests cover the
    whole executable body of each function the tests exercise every branch of."""
    import ast

    from Wesker.line_coverage import executable_lines

    _label, target, tests, _ = next(c for c in corpus if c[0] == "real_trace_evidence")
    cells = _trace(tests, target)
    credited = {line for cell in cells.values() for line in cell["lines"]}
    tree = ast.parse(Path(target).read_text())
    for fn in (
        "item_run_status",
        "baseline_outcome",
        "trace_admissibility",
        "reach_qualification",
        "build_trace_ledger",
    ):
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn
        )
        missing = executable_lines(node) - credited
        assert not missing, f"{fn}: lines never credited: {sorted(missing)}"
    assert any(cell["arcs"] for cell in cells.values())


if (
    __name__ == "__main__"
):  # pragma: no cover  (regenerates the golden: python this_file.py)
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    _cells_all = _normalized(_baseline(_build(tmp)))
    print(
        json.dumps(
            {k: v for k, v in _cells_all.items() if k in _PROJECTS}, sort_keys=True
        )
    )
