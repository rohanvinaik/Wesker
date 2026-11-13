"""Wesker#18: a target a decorator wrapped is measured, named and scoped like any other.

THE DEFECT (found running the issue's own acceptance case end to end: `@functools.lru_cache def fee`,
reached by a test through a module constant, `FEE = fee`). Three faults, one cause each:

1. LINES. The module attribute is the decorator's WRAPPER. `_build_test_scope` and `evaluate_mutant` took
   the target's file from ``original_func.__code__`` and a cache wrapper has none, so the line ledger
   intersected the trace with the empty string: every body line read uncovered while the tests ran it,
   and hand-written tests were proposed for removal.
2. BUILD. `_prepare_base_globals` took ``original_func.__globals__``, which a cache wrapper does not
   have either, so the mutant's definition (its own `@functools.lru_cache`) raised NameError, every
   mutant read `harness_error` ("could not be BUILT, a defect to report"), and the real in-process
   standing, `not_entered (decorator_wrapper)`, never reached the page.
3. CAUSE. The widen re-recorded an unscored mutant through the bare `_unscored_record`, dropping the
   cause the first pass had named: "cause unclassified".

A memoized function also executes its body only on a cache MISS, so a traced test's reach depended on
what ran before it in the process: `_reset_target_memos` empties the memos over traced files before each
traced test.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types

import pytest

from Wesker import line_coverage as lc
from Wesker.engine import (
    _definition_file,
    _named_unscored_record,
    _prepare_base_globals,
    run_function_profiling,
)
from Wesker.filter import filter_categories
from Wesker.line_coverage import memo_reset_standing

_LRU = """\
import functools


@functools.lru_cache(maxsize=None)
def fee(n):
    if n > 10:
        return n * 2
    return n + 1


FEE = fee
"""

_WRAPS = """\
import functools


def deco(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
"""

_WRAPPED = """\
from dcdeco import deco


@deco
def fee(n):
    if n > 10:
        return n * 2
    return n + 1
"""

_PLAIN = """\
def fee(n):
    if n > 10:
        return n * 2
    return n + 1
"""

_NAMES = ("dcmod", "dcdeco", "dcwrapped", "dcplain", "dctests")


@pytest.fixture
def modules(tmp_path, monkeypatch):
    for name, src in (
        ("dcmod", _LRU),
        ("dcdeco", _WRAPS),
        ("dcwrapped", _WRAPPED),
        ("dcplain", _PLAIN),
    ):
        (tmp_path / f"{name}.py").write_text(src)
    for name in _NAMES:
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded = {n: importlib.import_module(n) for n in _NAMES[:4]}
    try:
        yield loaded, tmp_path
    finally:
        for name in _NAMES:
            sys.modules.pop(name, None)


def _node(path):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "fee"
    )


# ── 1. the target's file is the PLAIN function's, found through any wrapper ───────────────────


@pytest.mark.parametrize("name", ["dcmod", "dcwrapped", "dcplain"])
def test_the_definition_file_is_the_target_modules_whatever_wraps_it(modules, name):
    mods, tmp = modules
    assert _definition_file(mods[name].fee) == str(tmp / f"{name}.py")


def test_the_wrapper_of_another_module_does_not_name_the_decorators_file(modules):
    mods, tmp = modules
    assert mods["dcwrapped"].fee.__code__.co_filename == str(tmp / "dcdeco.py")
    assert _definition_file(mods["dcwrapped"].fee) == str(tmp / "dcwrapped.py")


@pytest.mark.parametrize("obj", [None, object(), len])
def test_an_object_with_no_reachable_code_has_no_file_not_an_invented_one(obj):
    assert _definition_file(obj) is None


def test_the_base_namespace_of_a_cache_wrapper_is_the_modules(modules):
    mods, _ = modules
    base = _prepare_base_globals(mods["dcmod"].fee)
    assert base["functools"] is sys.modules["functools"]
    assert _prepare_base_globals(mods["dcplain"].fee) == dict(vars(mods["dcplain"]))
    assert _prepare_base_globals(None) == {}


# ── 2. a cache-wrapped target builds its mutants; the standing is the wrapper's ───────────────


def _profile(mods, tmp, name, test_src):
    (tmp / "dctests.py").write_text(test_src)
    sys.modules.pop("dctests", None)
    tests_mod = importlib.import_module("dctests")
    tests = []
    for attr in sorted(a for a in vars(tests_mod) if a.startswith("test_")):
        fn = getattr(tests_mod, attr)
        fn.__qualname__ = f"dctests.py::{attr}"
        tests.append(fn)
    node = _node(tmp / f"{name}.py")
    return run_function_profiling(
        node,
        f"{name}.py::fee",
        filter_categories(node),
        tests,
        mods[name].fee,
        scope_tests=False,
        max_per_category=0,
    )


def test_a_cache_wrapped_target_reached_by_a_constant_is_not_a_build_failure(modules):
    mods, tmp = modules
    res = _profile(
        mods,
        tmp,
        "dcmod",
        "from dcmod import FEE as charge\n\n\ndef test_small():\n    assert charge(3) == 4\n\n\n"
        "def test_large():\n    assert charge(20) == 40\n",
    )
    unscored = {r["disposition"] for r in res.unscored_records}
    assert unscored == {"not_entered"}, res.to_dict()
    assert {r["cause"] for r in res.unscored_records} == {"decorator_wrapper"}
    assert res.not_entered_causes == {"decorator_wrapper": len(res.unscored_records)}


def test_a_wraps_wrapper_from_another_module_builds_and_names_its_cause(modules):
    mods, tmp = modules
    res = _profile(
        mods,
        tmp,
        "dcwrapped",
        "from dcwrapped import fee as charge\n\n\ndef test_small():\n    assert charge(3) == 4\n",
    )
    assert {r["disposition"] for r in res.unscored_records} <= {"not_entered"}
    assert all(
        r["cause"] == "decorator_wrapper"
        for r in res.unscored_records
        if r["disposition"] == "not_entered"
    )


def test_a_plain_function_is_unchanged_and_scores_every_mutant(modules):
    mods, tmp = modules
    res = _profile(
        mods,
        tmp,
        "dcplain",
        "import dcplain\n\n\ndef test_small():\n    assert dcplain.fee(3) == 4\n\n\n"
        "def test_large():\n    assert dcplain.fee(20) == 40\n\n\n"
        "def test_edge():\n    assert dcplain.fee(10) == 11\n    assert dcplain.fee(11) == 22\n",
    )
    assert res.unscored_records == []
    assert res.total_killed == res.total_mutants > 0


# ── 3. the record the widen re-writes keeps its cause ─────────────────────────────────────────


def test_a_not_entered_record_carries_its_cause_and_scans_holders_once(modules):
    mods, tmp = modules
    res = _profile(
        mods,
        tmp,
        "dcmod",
        "from dcmod import FEE as charge\n\n\ndef test_small():\n    assert charge(3) == 4\n",
    )
    mutants = [r for r in res.unscored_records if r["disposition"] == "not_entered"]
    assert mutants and all(r["cause"] == "decorator_wrapper" for r in mutants)
    # The builder itself, as the widen calls it: a not_entered result gets the detail, any other
    # disposition does not, and the scan runs at most once across calls.
    scans: list[int] = []

    def scan():
        scans.append(1)
        return [{"kind": "wrapper", "where": "dcmod.fee"}], "complete"

    result = types.SimpleNamespace(
        mutant=types.SimpleNamespace(
            mutant_id="X_1",
            description="d",
            category=types.SimpleNamespace(value="ARITHMETIC"),
            mutated_line=8,
            dimension="d",
            mutated_node=ast.parse("def fee(n):\n    return n\n").body[0],
            original_node=ast.parse("def fee(n):\n    return n + 1\n").body[0],
        ),
        killed_by=None,
        contained=True,
        test_name=None,
        uncontained_test=None,
        refused_test=None,
        blocked_writes=(),
        tests_run=-1,
    )
    record, held = _named_unscored_record(
        result, "not_entered", [], {}, set(), None, scan
    )
    assert record["cause"] == "decorator_wrapper"
    assert held is not None and len(scans) == 1
    again, held2 = _named_unscored_record(
        result, "not_entered", [], {}, set(), held, scan
    )
    assert again["cause"] == "decorator_wrapper" and len(scans) == 1 and held2 is held
    other, held3 = _named_unscored_record(
        result, "harness_error", [], {}, set(), None, scan
    )
    assert "cause" not in other and held3 is None and len(scans) == 1


# ── 4. a memo does not change what a traced test is credited with ─────────────────────────────


@pytest.mark.parametrize(
    ("clear", "info", "in_target", "expected"),
    [
        (True, True, True, "reset"),
        (True, True, False, "foreign"),
        (True, False, True, "not_a_cache"),
        (False, True, True, "not_a_cache"),
        (False, False, False, "not_a_cache"),
    ],
)
def test_memo_reset_standing_table(clear, info, in_target, expected):
    assert memo_reset_standing(clear, info, in_target) == expected


def _trace_two(tmp_path, monkeypatch, target_src, call):
    (tmp_path / "mmod.py").write_text(target_src)
    sys.modules.pop("mmod", None)
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("mmod")
    path = str(tmp_path / "mmod.py")

    def first():
        getattr(mod, call)(3)

    def second():
        getattr(mod, call)(3)

    first.__qualname__, second.__qualname__ = "mt.py::first", "mt.py::second"
    try:
        traced = lc.trace_suite([first, second], {path})
    finally:
        sys.modules.pop("mmod", None)
    return {tid: sorted(files.get(path, ())) for tid, files in traced.items()}


def test_a_second_test_is_credited_with_the_body_a_cache_would_have_served(
    tmp_path, monkeypatch
):
    got = _trace_two(tmp_path, monkeypatch, _LRU, "fee")
    assert got["mt.py::first"] == got["mt.py::second"] != []


def test_a_functools_cache_and_a_method_cache_are_reset_too(tmp_path, monkeypatch):
    src = (
        "import functools\n\n\n@functools.cache\ndef fee(n):\n    return n + 1\n\n\n"
        "class K:\n    @functools.lru_cache\n    def m(self, n):\n        return n - 1\n\n\n"
        "def both(n):\n    return fee(n), K().m(n)\n"
    )
    got = _trace_two(tmp_path, monkeypatch, src, "both")
    assert got["mt.py::first"] == got["mt.py::second"]
    assert {6, 12} <= set(got["mt.py::second"])


def test_a_wraps_wrapper_and_a_plain_function_trace_as_before(tmp_path, monkeypatch):
    plain = _trace_two(tmp_path, monkeypatch, _PLAIN, "fee")
    assert plain["mt.py::first"] == plain["mt.py::second"] != []


def test_a_cache_over_code_outside_the_traced_files_is_left_warm(tmp_path, monkeypatch):
    import functools

    calls: list[int] = []

    @functools.cache
    def elsewhere(n):
        calls.append(n)
        return n

    elsewhere(1)
    holder = types.ModuleType("holder_mod")
    holder.__file__ = str(tmp_path / "holder_mod.py")
    holder.elsewhere = elsewhere
    (tmp_path / "holder_mod.py").write_text("")
    sys.modules["holder_mod"] = holder
    try:
        lc._reset_target_memos({str(tmp_path / "holder_mod.py")})
    finally:
        sys.modules.pop("holder_mod", None)
    assert elsewhere.cache_info().currsize == 1
