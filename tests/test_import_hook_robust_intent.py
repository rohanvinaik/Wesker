"""Wesker#18, robust mode: the isolated worker gives the mutant to the target's IMPORT.

The defect: a decorator that captured the original function object when the target imported
(`functools.wraps`, `lru_cache`, a registry) keeps calling the original, because the plain install only
rebinds names afterwards. Under `--isolated` such a mutant read `not_installed` (no site could be
rebound; in-process the same shape reads `not_entered (decorator_wrapper)`), so a suite that DOES pin the
function never got credit for it, and a suite that does not never got blamed.

The repair is opt-in by condition. A mutant whose plain install was `not_entered` or `not_installed` with
cause `decorator_wrapper` is evaluated once more in a FRESH worker whose `sys.meta_path` finder compiles the
target module with the function's definition replaced by the mutant's (plus one innermost entry-probe
decorator), so every capture at import captures the mutant, and the probe is the acknowledgement of entry.
The second verdict replaces the first only when a test positively entered the mutant. The in-process path is
unchanged: it still names the shape `not_entered`.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import uuid

import pytest

from Wesker._isolated_worker import robust_hook_standing
from Wesker.ci import _PROJECT_ROOT
from Wesker.engine import (
    robust_adoption_standing,
    robust_retry_standing,
    run_function_profiling,
)
from Wesker.filter import filter_categories

_DECO = (
    "import functools\n\nREGISTRY = {}\n\n\n"
    "def register(fn):\n"
    "    @functools.wraps(fn)\n"
    "    def wrapper(*args, **kwargs):\n"
    "        return fn(*args, **kwargs)\n\n"
    "    REGISTRY[fn.__name__] = wrapper\n"
    "    return wrapper\n"
)


# Plain functions held by a copy taken AT IMPORT: (module tail, how a test calls it).
_CAPTURES = {
    "alias": ("DOUBLE = double\n", "DOUBLE"),
    "handlers": ("HANDLERS = {'d': double}\n", "HANDLERS['d']"),
    "default": ("def apply(value, fn=double):\n    return fn(value)\n", "apply"),
}


def _project(tmp_path, kind, strong=True):
    """Write a project of unique module names; returns (module, function, tests-module, names)."""
    tag = uuid.uuid4().hex[:8]
    mod, deco, tests = f"tgt_{tag}", f"deco_{tag}", f"test_{tag}"
    if kind == "lru":
        (tmp_path / f"{mod}.py").write_text(
            "import functools\n\n\n@functools.lru_cache\n"
            "def plus_one(value):\n    return value + 1\n"
        )
        body = (
            f"import {mod}\n\n\ndef test_pin():\n    assert {mod}.plus_one(3) == 4\n"
            if strong
            else f"import {mod}\n\n\ndef test_pin():\n    assert {mod}.plus_one(3) is not None\n"
        )
        func, names = "plus_one", ["test_pin"]
    elif kind in _CAPTURES:
        tail, call = _CAPTURES[kind]
        (tmp_path / f"{mod}.py").write_text(
            "def double(value):\n    return value * 2\n\n\n" + tail
        )
        (tmp_path / f"{tests}.py").write_text(
            f"import {mod}\n\n\ndef test_cap():\n    assert {mod}.{call}(4) == 8\n"
            if kind == "default"
            else f"from {mod} import {call.split('[')[0]}\n\n\ndef test_cap():\n    assert {call}(4) == 8\n"
        )
        return mod, "double", tests, ["test_cap"]
    else:
        (tmp_path / f"{deco}.py").write_text(_DECO)
        (tmp_path / f"{mod}.py").write_text(
            f"from {deco} import register\n\n\n@register\n"
            "def double(value):\n    return value * 2\n"
        )
        body = (
            f"import {deco}\nimport {mod}\n\n\n"
            f"def test_attr():\n    assert {mod}.double(4) == 8\n\n\n"
            f"def test_registry():\n    assert {deco}.REGISTRY['double'](5) == 10\n"
        )
        func, names = "double", ["test_attr", "test_registry"]
    (tmp_path / f"{tests}.py").write_text(body)
    return mod, func, tests, names


def _profile(tmp_path, kind, isolated, strong=True, only=None):
    mod, func, tests_mod, names = _project(tmp_path, kind, strong)
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(mod)
        original = getattr(module, func).__wrapped__
        tree = ast.parse((tmp_path / f"{mod}.py").read_text())
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func
        )
        tests_module = importlib.import_module(tests_mod)
        tests = []
        for name in only or names:
            fn = getattr(tests_module, name)
            fn.__qualname__ = f"{tests_mod}.py::{name}"
            tests.append(fn)
        token = _PROJECT_ROOT.set(str(tmp_path))
        try:
            return run_function_profiling(
                node,
                f"{mod}.py::{func}",
                filter_categories(node),
                tests,
                original,
                isolated=isolated,
            )
        finally:
            _PROJECT_ROOT.reset(token)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(tmp_path)]
        for name in list(sys.modules):
            if name.endswith(mod[-8:]) or name == tests_mod:
                sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("kind", "only"),
    [("lru", None), ("registry", ["test_attr"]), ("registry", ["test_registry"])],
)
def test_a_decorator_wrapper_mutant_is_scored_under_isolation(tmp_path, kind, only):
    result = _profile(tmp_path, kind, True, only=only)
    assert result.unscored_records == []
    assert result.total_mutants > 0
    assert result.total_killed == result.total_mutants


def test_a_pinning_suite_is_blamed_only_for_what_it_leaves_unpinned(tmp_path):
    """Entered mutants are scored both ways: a weak assertion lets some survive, none is unscored."""
    result = _profile(tmp_path, "lru", True, strong=False)
    assert result.unscored_records == []
    assert result.total_mutants > 0
    assert 0 < result.total_survived < result.total_mutants


@pytest.mark.parametrize(("kind", "only"), [("lru", None), ("registry", ["test_attr"])])
def test_the_in_process_path_still_names_the_shape_not_entered(tmp_path, kind, only):
    result = _profile(tmp_path, kind, False, only=only)
    assert result.total_mutants == 0
    assert {r["disposition"] for r in result.unscored_records} == {"not_entered"}
    assert {r["cause"] for r in result.unscored_records} == {"decorator_wrapper"}


_DRIVE = (
    "import json,sys\n"
    "from Wesker._isolated_worker import _evaluate_full\n"
    "pre = sys.argv[4] == 'pre'\n"
    "if pre:\n"
    "    __import__(sys.argv[5])\n"
    "out = _evaluate_full(sys.argv[1], sys.argv[2], [sys.argv[3]], sys.argv[6], robust=True)\n"
    "out['hook_left'] = any(type(f).__name__ == '_MutantImportHook' for f in sys.meta_path)\n"
    "print(json.dumps(out))\n"
)


def _drive(tmp_path, target, qualname, node, mutant, pre_import=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            _DRIVE,
            str(target),
            qualname,
            node,
            "pre" if pre_import else "fresh",
            pre_import or "-",
            mutant,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=env,
    )
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_the_hook_is_gone_after_the_run_and_a_method_definition_is_replaced(tmp_path):
    (tmp_path / "klass.py").write_text(
        "import functools\n\n\nclass K:\n    @functools.lru_cache\n"
        "    def m(self, v):\n        return v + 1\n"
    )
    (tmp_path / "test_klass.py").write_text(
        "from klass import K\n\n\ndef test_m():\n    assert K().m(3) == 4\n"
    )
    out = _drive(
        tmp_path,
        tmp_path / "klass.py",
        "K.m",
        "test_klass.py::test_m",
        "@functools.lru_cache\ndef m(self, v):\n    return v - 1\n",
    )
    assert out["constructed"] is True
    assert out["entered"] is True
    assert out["killed_by"] == "assertion"
    assert out["hook_left"] is False


def test_an_already_imported_target_cannot_be_hooked_and_is_not_built(tmp_path):
    (tmp_path / "early.py").write_text(
        "import functools\n\n\n@functools.lru_cache\ndef f(v):\n    return v + 1\n"
    )
    (tmp_path / "test_early.py").write_text(
        "import early\n\n\ndef test_f():\n    assert early.f(3) == 4\n"
    )
    out = _drive(
        tmp_path,
        tmp_path / "early.py",
        "f",
        "test_early.py::test_f",
        "def f(v):\n    return v - 1\n",
        pre_import="early",
    )
    assert out["constructed"] is False
    assert out["killed_by"] is None


def test_a_definition_the_hook_cannot_find_is_not_built_never_a_kill(tmp_path):
    (tmp_path / "gone.py").write_text("def g(v):\n    return v\n")
    (tmp_path / "test_gone.py").write_text(
        "import gone\n\n\ndef test_g():\n    assert gone.g(1) == 1\n"
    )
    out = _drive(
        tmp_path,
        tmp_path / "gone.py",
        "f",
        "test_gone.py::test_g",
        "def f(v):\n    return v - 1\n",
    )
    assert out["constructed"] is False
    assert out["killed_by"] is None


@pytest.mark.parametrize(("loaded", "code"), [(False, "install"), (True, "too_late")])
def test_robust_hook_standing(loaded, code):
    assert robust_hook_standing(loaded) == code


@pytest.mark.parametrize(
    ("disposition", "cause", "robust", "code"),
    [
        ("not_entered", "decorator_wrapper", False, "reevaluate_with_hook"),
        ("not_installed", "decorator_wrapper", False, "reevaluate_with_hook"),
        ("not_entered", "decorator_wrapper", True, "keep"),
        ("not_entered", "captured_at_import", False, "reevaluate_with_hook"),
        ("not_entered", "pre_bound_reference", False, "reevaluate_with_hook"),
        ("not_installed", "pre_bound_reference", False, "reevaluate_with_hook"),
        ("not_entered", "pre_bound_reference", True, "keep"),
        ("not_entered", "captured_at_import", True, "keep"),
        ("not_entered", "holder_not_found", False, "keep"),
        ("not_entered", "not_reached", False, "keep"),
        ("not_entered", "no_reach_data", False, "keep"),
        ("not_installed", "no_reach_data", False, "keep"),
        ("killed_after_entry", "decorator_wrapper", False, "keep"),
        ("survived_after_entry", "decorator_wrapper", False, "keep"),
        ("harness_error", "decorator_wrapper", False, "keep"),
        ("not_entered", "", False, "keep"),
    ],
)
def test_robust_retry_standing(disposition, cause, robust, code):
    assert robust_retry_standing(disposition, cause, robust) == code


@pytest.mark.parametrize(
    ("constructed", "entered", "timed_out", "memory_cut", "code"),
    [
        (True, True, False, False, "adopt"),
        (True, False, False, False, "keep_first"),
        (True, None, False, False, "keep_first"),
        (False, True, False, False, "keep_first"),
        (True, True, True, False, "keep_first"),
        (True, True, False, True, "keep_first"),
    ],
)
def test_robust_adoption_standing(constructed, entered, timed_out, memory_cut, code):
    assert robust_adoption_standing(constructed, entered, timed_out, memory_cut) == code


def _profile_plain(tmp_path, kind, isolated):
    """`_profile` for a plain (undecorated) target: the original is the function itself."""
    mod, func, tests_mod, _names = _project(tmp_path, kind)
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(mod)
        tree = ast.parse((tmp_path / f"{mod}.py").read_text())
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func
        )
        tests_module = importlib.import_module(tests_mod)
        fn = tests_module.test_cap
        fn.__qualname__ = f"{tests_mod}.py::test_cap"
        token = _PROJECT_ROOT.set(str(tmp_path))
        try:
            return run_function_profiling(
                node,
                f"{mod}.py::{func}",
                filter_categories(node),
                [fn],
                module.double,
                isolated=isolated,
            )
        finally:
            _PROJECT_ROOT.reset(token)
    finally:
        sys.path[:] = [p for p in sys.path if p != str(tmp_path)]
        for name in list(sys.modules):
            if name.endswith(mod[-8:]) or name == tests_mod:
                sys.modules.pop(name, None)


@pytest.mark.parametrize("kind", sorted(_CAPTURES))
def test_a_plain_function_captured_at_import_is_scored_under_isolation(tmp_path, kind):
    """An alias constant, a handler dict and a default argument each hold the original taken at import; the
    hook compiles the module with the definition replaced, so each holds the mutant and the test enters it."""
    result = _profile_plain(tmp_path, kind, True)
    assert result.unscored_records == []
    assert result.total_mutants > 0
    assert result.total_killed == result.total_mutants


@pytest.mark.parametrize("kind", sorted(_CAPTURES))
def test_in_process_the_same_capture_is_still_named_not_entered(tmp_path, kind):
    result = _profile_plain(tmp_path, kind, False)
    assert result.total_mutants == 0
    assert {r["disposition"] for r in result.unscored_records} == {"not_entered"}
    assert {r["cause"] for r in result.unscored_records} <= {
        "pre_bound_reference",
        "captured_at_import",
    }
