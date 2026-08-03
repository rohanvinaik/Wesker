"""Wesker CI runner — the next era of mutation testing.

In-process AST mutation engine with:
- 3-layer test discovery (convention → static impact → full fallback)
- Real equivalent mutant detection via boundary input evaluation
- Categorical profiling (VALUE, BOUNDARY, SWAP, STATE, TYPE, ARITHMETIC, LOGICAL)
- Clean, progressive terminal output

Zero external dependencies beyond the test framework.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextvars import ContextVar
import unittest
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any

from Wesker.engine import (
    run_function_converged,
)
from Wesker.filter import filter_categories, prioritize_categories


# ── ANSI colors for terminal output ──────────────────────────────

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Disable colors when not a terminal (CI logs, piped output)
if not sys.stderr.isatty() and not os.environ.get("WESKER_COLOR"):
    _GREEN = _RED = _YELLOW = _DIM = _RESET = ""


def _pct_color(pct: int) -> str:
    if pct == 100:
        return _GREEN
    if pct >= 80:
        return _YELLOW
    return _RED


# ── Layer 1: Convention-based test discovery ─────────────────────


def _name_matches_convention(
    base,
    base_stripped,
    generated_name,
    name,
    parent_dir,
    parent_qualified,
    partial_stems,
):
    match = (
        # Exact generated name (highest confidence)
        name == generated_name
        # Parent-qualified (wiki/config.py -> test_wiki_config.py)
        or (
            parent_qualified
            and (
                name == f"test_{parent_qualified}.py"
                or name.startswith(f"test_{parent_qualified}_")
            )
        )
        # Exact stem
        or name == f"test_{base}.py"
        or name == f"test_{base_stripped}.py"
        # Prefix match
        or name.startswith(f"test_{base}_")
        or name.startswith(f"test_{base_stripped}_")
        # Parent dir (extraction/det.py -> test_extraction.py)
        or (parent_qualified and name == f"test_{parent_dir}.py")
        # Contains-stem (test_prescriptive_deterministic.py)
        or f"_{base_stripped}." in name
        or f"_{base_stripped}_" in name
        # Partial stems (query_navigate -> test_navigate.py)
        or any(name == f"test_{s}.py" for s in partial_stems)
        or any(name.startswith(f"test_{s}_") for s in partial_stems)
    )
    return match


def _discover_by_convention(project_root: str, source_file: str) -> list[str]:
    """Find test files by naming convention (fast, high precision)."""
    base = Path(source_file).stem
    base_stripped = base.lstrip("_")
    tests_dir = Path(project_root) / "tests"
    generated_dir = tests_dir / "generated"

    # Path-safe generated test name
    try:
        rel = os.path.relpath(source_file, project_root)
    except ValueError:
        rel = base
    safe = rel.replace(os.sep, "_").replace("/", "_").replace(".", "_")
    if safe.endswith("_py"):
        safe = safe[:-3]
    generated_name = f"test_{safe}.py"

    # Parent-aware matching
    parent_dir = Path(source_file).parent.name
    # Skip qualification for top-level package dirs and src/
    _skip_dirs = {"src"}
    # Auto-detect: if parent is the package root (immediate child of src/), skip
    parent_path = Path(source_file).parent
    if parent_path.parent.name == "src" or parent_dir == "src":
        _skip_dirs.add(parent_dir)
    parent_qualified = f"{parent_dir}_{base}" if parent_dir not in _skip_dirs else None

    # Partial stems for compound names (query_navigate -> query, navigate)
    partial_stems = {p for p in base_stripped.split("_") if len(p) >= 4}

    # Ambiguous stems that exist at multiple paths
    ambiguous_stems = {"config", "base", "__main__", "utils", "helpers"}

    found: list[str] = []
    for search_dir in [tests_dir, generated_dir]:
        if not search_dir.is_dir():
            continue
        for entry in sorted(search_dir.iterdir()):
            if not entry.name.endswith(".py"):
                continue
            name = entry.name
            path_str = str(entry)

            match = _name_matches_convention(
                base,
                base_stripped,
                generated_name,
                name,
                parent_dir,
                parent_qualified,
                partial_stems,
            )

            # Suppress ambiguous bare-stem matches for common names in subdirs
            if match and parent_qualified and base_stripped in ambiguous_stems:
                # Only keep if it also matches parent dir or generated name
                if not (parent_dir in name or name == generated_name):
                    continue

            if match and path_str not in found:
                found.append(path_str)

    return found


# ── Layer 2: Static AST impact analysis ──────────────────────────


def _build_static_impact_map(test_files: list[str]) -> dict[str, list[str]]:
    """Build a map of function_name -> [test_file] by scanning test ASTs.

    Looks for function names referenced in test bodies via ast.Name nodes.
    This catches imports and direct references without executing anything.
    """
    impact: dict[str, set[str]] = {}
    for tf in test_files:
        try:
            with open(tf) as f:
                tree = ast.parse(f.read(), filename=tf)
        except (OSError, SyntaxError):
            continue
        # Collect all Name references in the file
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                impact.setdefault(node.id, set()).add(tf)
            elif isinstance(node, ast.Attribute):
                impact.setdefault(node.attr, set()).add(tf)
    return {k: sorted(v) for k, v in impact.items()}


# ── Layer 3: Full fallback ───────────────────────────────────────


def _discover_all_test_files(project_root: str) -> list[str]:
    """Find all ``test_*.py`` files ANYWHERE in the project — matching pytest's default
    collection, not just ``tests/`` — so a fresh install without the pytest extra still
    discovers root-level and package-level tests through the legacy loader (otherwise a
    user whose tests live at the repo root gets a misleading 0% kill rate)."""
    skip = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
        "node_modules",
        ".serena",
        ".lintgate",
    }
    root = Path(project_root)
    found: list[str] = []
    for py in sorted(root.rglob("test_*.py")):
        rel = py.relative_to(root)
        if any(part in skip or part.startswith(".") for part in rel.parts[:-1]):
            continue
        found.append(str(py))
    return found


# ── 3-Layer discovery orchestrator ───────────────────────────────


def discover_tests(
    project_root: str, source_file: str, func_names: list[str]
) -> list[str]:
    """3-layer test discovery: convention -> static impact -> full fallback.

    Layer 1: Convention matching (fast, filename-based)
    Layer 2: Static impact (AST scan for function name references)
    Layer 3: Full fallback (all test files)

    Each layer adds files not already found by previous layers.
    """
    # Layer 1: Convention
    found = _discover_by_convention(project_root, source_file)

    # Layer 2: Static impact — find additional test files that reference
    # any of the function names in this source file
    all_test_files = _discover_all_test_files(project_root)
    impact_map = _build_static_impact_map(all_test_files)
    found_set = set(found)
    for func_name in func_names:
        for tf in impact_map.get(func_name, []):
            if tf not in found_set:
                found.append(tf)
                found_set.add(tf)

    # Layer 3: Full fallback — add remaining test files
    for tf in all_test_files:
        if tf not in found_set:
            found.append(tf)
            found_set.add(tf)

    return found


# ── Test callable loading ────────────────────────────────────────


def _parametrize_cases(func: Any) -> "list[Any] | None":
    """Expand a ``@pytest.mark.parametrize``-decorated test into one bound, runnable callable per
    case — the legacy loader's parity with pytest for the parametrize forms it can resolve without a
    live session.

    The decorator only ATTACHES marks; it leaves the function's ORIGINAL signature intact, so appending
    the bare object yields an uncallable ``test(args, expected)`` that raises ``TypeError`` the moment
    the profiler calls it — and that raise reads as a crash, silently dropping the case's value-kills and
    line coverage (a parametrized golden then profiles as one case, so a re-profile disagrees with the
    live-pytest pass). Stacked marks take the cartesian product (pytest's own rule); ``pytest.param(...)``
    values are unwrapped. Returns None when there is no parametrize mark, or when a form cannot be
    resolved here (unrecognized shape, fixtures, indirect) — the caller then keeps its prior behavior for
    that callable, so this only ever ADDS coverage, never removes a case that used to load.
    """
    marks = [
        m
        for m in getattr(func, "pytestmark", ())
        if getattr(m, "name", "") == "parametrize"
    ]
    if not marks:
        return None

    def _is_paramset(v: Any) -> bool:
        # A pytest.param(...) ParameterSet — namedtuple(values, marks, id). Checking all three fields
        # (not just `.values`) avoids mis-reading a dict/namedtuple VALUE as a wrapper to unwrap.
        return hasattr(v, "values") and hasattr(v, "marks") and hasattr(v, "id")

    try:
        combined: list[dict] = [{}]
        for m in marks:
            argnames, argvalues = m.args[0], m.args[1]
            names = (
                [s.strip() for s in argnames.split(",")]
                if isinstance(argnames, str)
                else list(argnames)
            )
            frags: list[dict] = []
            for v in argvalues:
                if _is_paramset(v):
                    vals: tuple = tuple(v.values)
                elif len(names) == 1:
                    vals = (
                        v,
                    )  # a single argname takes each value whole (even a tuple is ONE value)
                else:
                    vals = tuple(v)
                frags.append(dict(zip(names, vals)))
            combined = [{**c, **f} for c in combined for f in frags]
    except Exception:
        return None

    cases: list[Any] = []
    for i, kwargs in enumerate(combined):

        def _case(_kwargs=kwargs, _i=i):
            return func(**_kwargs)

        # Mirror the live-item convention: siblings SHARE __name__ (the union key in trace_suite) and
        # differ on __qualname__ (the per-case discriminator test_fingerprint folds in), so the legacy
        # and pytest paths attribute a parametrized case's coverage identically.
        _case.__name__ = func.__name__
        _case.__qualname__ = f"{getattr(func, '__qualname__', func.__name__)}[{i}]"
        _case.__doc__ = func.__doc__
        cases.append(_case)
    return cases


def load_test_callables(
    test_files: list[str], project_root: str | None = None
) -> list[Any]:
    """Load all test_* callables from test files, including class methods.

    Intra-project imports in a test (``from calc import add``) resolve only if the code's
    directory is importable, so the project root and each test file's own directory are
    put on ``sys.path`` before import — the rootdir insertion pytest does for you, which
    the legacy loader must do itself (else a fresh no-pytest user gets import errors and a
    misleading 0% kill rate)."""
    callables: list[Any] = []
    for path in filter(
        None, [project_root, *(str(Path(tf).parent) for tf in test_files)]
    ):
        ap = os.path.abspath(path)
        if ap not in sys.path:
            sys.path.insert(0, ap)
    for tf in test_files:
        # Key the module cache on file CONTENT, not just its stem. A long-lived
        # process (or a converge loop) rewrites generated test files in place; a
        # stem-only cache would serve the stale prior version via sys.modules,
        # hiding freshly written killing tests as false survivors.
        try:
            with open(tf, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()[:16]
        except OSError:
            continue
        stem_prefix = f"_wesker_test_{Path(tf).stem}_"
        mod_name = f"{stem_prefix}{digest}"
        if mod_name in sys.modules:
            # Same content already loaded — reuse.
            mod = sys.modules[mod_name]
        else:
            # Evict any prior-content module for this file so sys.modules can't grow
            # without bound across rewrites.
            for stale in [m for m in list(sys.modules) if m.startswith(stem_prefix)]:
                sys.modules.pop(stale, None)
            try:
                spec = importlib.util.spec_from_file_location(mod_name, tf)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
            except Exception:
                continue

        for name in dir(mod):
            obj = getattr(mod, name)
            if name.startswith("test_") and callable(obj):
                # A @parametrize'd function is uncallable bare (its params are still required); expand
                # it into one bound callable per case so the fallback matches the pytest backend.
                cases = _parametrize_cases(obj)
                callables.extend(cases if cases is not None else [obj])
            elif isinstance(obj, type) and (
                (issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase)
                or name.startswith("Test")
            ):
                # unittest.TestCase subclasses use any naming (e.g. the common
                # `<Name>Tests` suffix); detect by type, not name prefix. Bind
                # the method name so TestCase instantiation succeeds.
                for mname in dir(obj):
                    if mname.startswith("test_"):
                        try:
                            callables.append(getattr(obj(mname), mname))
                        except Exception:
                            try:
                                callables.append(getattr(obj(), mname))
                            except Exception:
                                pass
    return callables


# The suite of a LIVE pytest session, when one is active (see `profile_codebase_live`).
# Threaded as ambient context rather than a parameter because the live-session runner
# INVERTS CONTROL: pytest calls us from inside `pytest_runtestloop`, so the callables
# cannot be passed down through profile_codebase -> profile_file -> discover_*
# without changing every signature Detective imports. Unset by default, so every
# existing caller — Detective included — takes the ordinary discovery path unchanged.
# `None` is already a MEANINGFUL budget — "unbounded, the historical pass" — so it cannot double as
# "caller said nothing". A sentinel keeps the three states distinct: unset (use the engine's
# default), None (explicitly unbounded), a number (that budget). Without it, a consumer that simply
# does not mention budgets would be indistinguishable from one asking for an unbounded trace, and
# the engine's defaults — the only thing making the baseline phase finite — would silently vanish
# for every caller of this seam.
_UNSET: Any = object()

_LIVE_SUITE: ContextVar[list[Any] | None] = ContextVar(
    "wesker_live_suite", default=None
)


DISCOVERED_CALLABLE_CONTRACT = """What every discovered test callable guarantees — THE single reference (issue #6).

Three backends hand out callables, in three shapes:

* live pytest items — ``pytest_runner._make_item_callable`` wrappers,
* re-collected pytest — ``pytest_discovery`` closures (parametrized bindings, TestCase runners),
* the legacy loader — plain function objects.

Which shape a consumer receives depends on in-process session state, so NO consumer may
branch on the shape. The contract every shape satisfies:

``__name__``
    The test's display/matrix name. A bracketed suffix (``test_x[case0]``) is a
    parametrized ROW of one live test, never a function of its own. NOT unique across
    files — never an identity on its own.
``__qualname__``
    The pytest nodeid when one exists (path::name[case]), else the name. The most
    discriminating identity string a callable carries.
``__wesker_origin__`` (optional, stamped at build time)
    Absolute path of the test FILE the callable stands for. The only truth for closures
    whose code object lives in Wesker's own modules. Read it through
    :func:`callable_origin`, never directly.
``__wrapped__`` (optional)
    The USER's underlying test function. ``inspect.getsource``/``unwrap`` follow it, and
    content-hashing consumers (trace cache, Detective's verdict cache) depend on it. Read
    it through :func:`callable_source`, never directly.
Invocation
    Zero-argument call, raising on test failure. Parametrized bindings are already bound.

Consumers resolve identity ONLY through the accessors below — a raw ``__code__`` /
``__wrapped__`` / ``__name__`` read is correct under one backend and silently wrong under
another (Detective's ``_locate`` bug: right under the legacy loader, a no-op under the
pytest backend, invisible to 312 green tests). Old call sites migrate as they are touched;
new ones start here.
"""


def callable_origin(call: Any) -> str | None:
    """Absolute path of the test FILE a discovered callable came from, or None.

    Accessor of :data:`DISCOVERED_CALLABLE_CONTRACT` — the ORIGIN axis. A
    ``__wesker_origin__`` tag outranks everything (stamped at build time, and
    the only truth for closures whose code object lives in Wesker's own modules);
    ``__wrapped__`` is the live-item wrapper's pointer to the real test function;
    the raw ``__code__.co_filename`` is the plain-function fallback. Consumers
    that need "which file defines this test" (Detective's ``suite_edit._locate``)
    MUST resolve through this — reading ``__code__`` directly attributes every
    wrapper to ``pytest_runner.py``/``pytest_discovery.py``.
    """
    tagged = getattr(call, "__wesker_origin__", None)
    if tagged:
        return str(tagged)
    real = getattr(call, "__wrapped__", call)
    code = getattr(real, "__code__", None)
    f = getattr(code, "co_filename", None)
    return os.path.abspath(f) if f else None


def callable_source(call: Any) -> Any:
    """The USER's underlying test function for a discovered callable — itself when it is
    already the plain function.

    Accessor of :data:`DISCOVERED_CALLABLE_CONTRACT` — the SOURCE axis. Wrapper shapes
    carry the real test as ``__wrapped__`` so introspection (``inspect.getsource``,
    content fingerprints) reads the user's code instead of Wesker's wrapper body, which is
    identical for every test in the suite and collapses any source-keyed cache to one
    entry. New consumers reach the source function through this, never through a raw
    ``__wrapped__`` read.
    """
    return getattr(call, "__wrapped__", call)


def callable_base_name(call: Any) -> str:
    """The test's FUNCTION name with any parametrize row id stripped —
    ``test_x[case0]`` → ``test_x``.

    Accessor of :data:`DISCOVERED_CALLABLE_CONTRACT` — the NAME axis. A bracketed
    ``__name__`` is a ROW of one live test, never a function of its own; consumers
    grouping rows to their test (row pruning, ownership matching) resolve through
    this instead of re-implementing the split.
    """
    name = str(getattr(call, "__name__", "") or "")
    return name.split("[", 1)[0]


def callable_case_id(call: Any) -> str:
    """The parametrize row id, or ``""`` for a non-parametrized test —
    ``test_x[case0]`` → ``case0``.

    Accessor of :data:`DISCOVERED_CALLABLE_CONTRACT` — the CASE axis, the complement
    of :func:`callable_base_name`. The principal backend shape puts the discriminator
    in the NODEID, not the display name: live and legacy wrappers deliberately share a
    base ``__name__`` and carry ``test_golden[args0]`` on ``__qualname__``. A
    bracketed ``__name__`` wins when present (the recollected-closure shape); else the
    nodeid's FINAL bracket suffix is the row id — final, because a nodeid can carry
    brackets in its path components too.
    """
    name = str(getattr(call, "__name__", "") or "")
    if "[" in name and name.endswith("]"):
        return name.split("[", 1)[1][:-1]
    node_id = callable_node_id(call)
    if node_id.endswith("]") and "[" in node_id:
        return node_id.rsplit("[", 1)[1][:-1]
    return ""


def callable_node_id(call: Any) -> str:
    """The most discriminating identity string a discovered callable carries: the
    pytest nodeid when one exists (stamped on ``__qualname__`` at build time), else
    the display name.

    Accessor of :data:`DISCOVERED_CALLABLE_CONTRACT` — the IDENTITY axis. This is
    what per-case caches key on (``trace_cache.test_fingerprint``): sibling
    parametrized cases share their function's SOURCE, and only the nodeid tells them
    apart.
    """
    return str(getattr(call, "__qualname__", "") or getattr(call, "__name__", "") or "")


def discover_test_callables(
    project_root: str,
    source_file: str,
    func_names: list[str],
    backend: str = "auto",
    extra_dirs: list[str] | None = None,
) -> list[Any]:
    """Discover runnable test callables — a dial over two backends.

      * ``"pytest"`` — pytest's own collection (robust across every convention:
        TestCase suffixes, mixins, parametrization, conftest);
      * ``"legacy"`` — Wesker's original hand-rolled loader
        (``discover_tests`` + ``load_test_callables``);
      * ``"auto"``   — try pytest, fall back to legacy (the default).

    pytest is the preferred/main path; the legacy loader stays intact as the
    fallback, so projects without pytest — or that pytest cannot collect —
    behave exactly as before.

    ``extra_dirs`` are additional roots to collect from, beyond ``project_root``
    — used when a caller wrote tests OUTSIDE the project tree (e.g. converge's
    ``--write-dir`` pointing at a scratch dir) and the kill count must still
    reflect them. Without this, tests written out-of-tree are invisible to
    discovery and the run reports a misleading 0% — the opposite of honest.
    """
    # Only EXISTING extra roots: a caller (converge) may pass its write-dir before
    # it has written anything there (the first profiling pass runs BEFORE tests are
    # written). Passing a nonexistent path to pytest's collector aborts collection
    # entirely → silent fallback to the legacy loader → a DIFFERENT test set and
    # inconsistent survivor counts. Filtering by existence is correct by lifecycle:
    # skip the empty/not-yet-created dir early, include it once tests land there.
    # A live pytest session outranks every backend: its items carry real fixtures,
    # conftest and lifecycle, which no re-collection here can reproduce. The session
    # already collected the whole suite, so the same list serves every file — for as long
    # as the suite is what it was when the session opened. A consumer that WRITES tests
    # must say so via `refresh_live_suite`; see there for what went wrong when it could not.
    live = _LIVE_SUITE.get()
    if live is not None:
        return live

    extra = [os.path.abspath(d) for d in (extra_dirs or []) if os.path.isdir(d)]
    if backend in ("auto", "pytest"):
        try:
            from Wesker.pytest_discovery import collect_pytest_callables

            # "." resolves to project_root inside the collector's chdir; the extra
            # roots are absolute so they collect regardless of cwd. No overlap with
            # "." when out-of-tree, so no double-collection.
            collected = collect_pytest_callables(
                project_root, paths=["."] + extra if extra else None
            )
        except Exception:
            collected = None
        if collected:
            return collected
        if backend == "pytest":
            return []
    # Legacy fallback: hand-rolled discovery + loader. Union the project-tree test
    # files with any found under the extra roots so out-of-tree tests still load.
    full_path = (
        os.path.join(project_root, source_file)
        if not os.path.isabs(source_file)
        else source_file
    )
    files = discover_tests(project_root, full_path, func_names)
    seen = set(files)
    for d in extra:
        for tf in _discover_all_test_files(d):
            if tf not in seen:
                files.append(tf)
                seen.add(tf)
    return load_test_callables(files, project_root)


# ── AST utilities ────────────────────────────────────────────────


def resolve_original_func(full_path: str, qualname: str) -> Any:
    """The LIVE function object for ``(source file, qualname)``, or None.

    Test-impact scoping needs the original callable: its ``__code__.co_filename`` is
    the authoritative identity the tracer attributes coverage to. The discovered tests
    have already imported the module under test (that is how they call it), so the
    live object is reachable from ``sys.modules`` — matched by FILE rather than by a
    guessed dotted name, which stays correct under src-layouts, namespace packages and
    same-named siblings. Walks the qualname so ``Class.method`` resolves too.

    Returns None when the module was never imported or the attribute path does not
    resolve; the caller then simply does not scope (full test set — always sound).
    """
    target = os.path.abspath(full_path)
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            if os.path.abspath(f) != target:
                continue
        except (OSError, ValueError):  # pragma: no cover — exotic __file__
            continue
        obj: Any = mod
        for part in qualname.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and callable(obj):
            return obj
    return None


def walk_functions(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Walk AST yielding (qualname, node) for each function."""
    results: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def _walk(scope: ast.AST, prefix: str) -> None:
        for node in getattr(scope, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}" if prefix else node.name
                results.append((name, node))
            elif isinstance(node, ast.ClassDef):
                cp = f"{prefix}{node.name}." if prefix else f"{node.name}."
                _walk(node, cp)

    _walk(tree, "")
    return results


# ── Cached state for Layer 2 predictive priors ─────────────────


def _load_cached_state(project_root: str) -> dict | None:
    """Load cached mutation report from a previous Wesker run.

    Reads ``.wesker/mutation_report.json`` which contains per-category
    aggregate survival data. Returns the full report dict (with a
    ``per_category`` list), or None if no cache exists.

    This enables Layer 2 (§6.2): historical survival rates inform which
    categories are most likely to contain specification gaps, so budget
    is spent where information gain is highest.
    """
    report_path = Path(project_root) / ".wesker" / "mutation_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text())
    except Exception:
        return None


# ── File profiling ───────────────────────────────────────────────


def profile_file(
    project_root: str,
    source_file: str,
    budget_ms: float = 10000,
    max_per_category: int | None = None,
    passes: int = 1,
    cached_state: dict | None = None,
    full_matrix: bool = False,
    test_discovery: str = "auto",
) -> list[dict]:
    """Profile all functions in a file with multi-pass convergence.

    Each function is profiled with ``passes`` rounds of sampling, each
    using a different seed. Equivalence detection is integrated into the
    evaluation loop — no post-hoc re-evaluation needed.

    When ``cached_state`` is provided (from a previous run's report),
    Layer 2 predictive priors order categories by historical survival
    rate — highest-survival first — so budget-limited runs test the
    most informative categories before less informative ones.
    """
    full_path = (
        os.path.join(project_root, source_file)
        if not os.path.isabs(source_file)
        else source_file
    )

    # Ensure src-layout packages are importable by tests
    abs_root = os.path.abspath(project_root)
    src_dir = os.path.join(abs_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        with open(full_path) as f:
            tree = ast.parse(f.read(), filename=full_path)
    except (OSError, SyntaxError):
        return []

    functions = walk_functions(tree)
    func_names = [name for name, _ in functions]

    # 3-layer test discovery
    tests = discover_test_callables(
        project_root, source_file, func_names, backend=test_discovery
    )

    results: list[dict] = []
    for qualname, func_node in functions:
        cats = filter_categories(func_node)
        if not cats:
            continue

        # Layer 2: order categories by historical survival prior
        priors = prioritize_categories(cats, cached_state)
        cat_order = [p.category for p in priors]

        rel = os.path.relpath(full_path, project_root)
        func_key = f"{rel}::{qualname}"

        sr = run_function_converged(
            func_node,  # type: ignore[arg-type]  # AsyncFunctionDef has same shape
            func_key,
            cats,
            tests,
            # The live callable, so test-impact scoping can trace a coverage baseline
            # against it. None when it cannot be resolved — scoping then simply does
            # not engage (full test set, sound but slower).
            resolve_original_func(full_path, qualname),
            budget_ms=budget_ms,
            max_per_category=max_per_category,
            passes=passes,
            category_order=cat_order,
            full_matrix=full_matrix,
            source_path=full_path,
        )
        results.append(sr.to_dict())

    return results


# ── Single-function profiling ──────────────────────────────────


def profile_function(
    project_root: str,
    source_file: str,
    function_name: str,
    budget_ms: float = 10000,
    max_per_category: int | None = None,
    passes: int = 1,
    cached_state: dict | None = None,
    full_matrix: bool = False,
    test_discovery: str = "auto",
) -> dict | None:
    """Profile a single function by name — the interactive/library entry point.

    Parses the file, finds the named function (supports ``Class.method``
    dotted names), discovers tests, and runs multi-pass convergence.
    Returns a full ProfilingResult dict (with kill_matrix, survivor/killed
    records, gateability) or None if the function was not found.

    This is the API that downstream consumers (LintGate, editors, MCP
    tools) should call when targeting a specific function rather than
    profiling an entire file.
    """
    full_path = (
        os.path.join(project_root, source_file)
        if not os.path.isabs(source_file)
        else source_file
    )

    abs_root = os.path.abspath(project_root)
    src_dir = os.path.join(abs_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        with open(full_path) as f:
            tree = ast.parse(f.read(), filename=full_path)
    except (OSError, SyntaxError):
        return None

    functions = walk_functions(tree)
    func_names = [name for name, _ in functions]

    # Find the target function
    func_node = None
    qualname = None
    for qn, node in functions:
        if qn == function_name or qn.split(".")[-1] == function_name:
            func_node = node
            qualname = qn
            break

    if func_node is None or qualname is None:
        return None

    cats = filter_categories(func_node)
    if not cats:
        return None

    priors = prioritize_categories(cats, cached_state)
    cat_order = [p.category for p in priors]

    tests = discover_test_callables(
        project_root, source_file, func_names, backend=test_discovery
    )

    rel = os.path.relpath(full_path, project_root)
    func_key = f"{rel}::{qualname}"

    result = run_function_converged(
        func_node,  # type: ignore[arg-type]
        func_key,
        cats,
        tests,
        # See profile_file: the live callable enables test-impact scoping; None just
        # means the full test set is used.
        resolve_original_func(full_path, qualname),
        budget_ms=budget_ms,
        max_per_category=max_per_category,
        passes=passes,
        category_order=cat_order,
        full_matrix=full_matrix,
        source_path=full_path,
    )
    return result.to_dict()


# ── Per-function result cache ──────────────────────────────────


# ── Codebase profiling with formatted output ─────────────────────


def live_suite_active() -> bool:
    """True when a LIVE pytest session is currently supplying the test suite.

    Consumers need this to decide whether work can leave the process. The live suite is
    a set of closures over LIVE pytest items — bound to this interpreter's session, its
    fixtures and its conftest — so it cannot cross a ``spawn`` boundary. A worker
    started from inside a live session re-discovers with the collect-only backend, which
    silently drops every fixture-taking test; the shard then reports those mutants as
    survivors and the parent merges the lie into an otherwise-correct result.

    ``Detective.engine.profile`` already refuses to fan out when the caller passed
    explicit callables, for exactly this reason ("workers re-discover; callables can't
    cross spawn"). This predicate extends that same rule to the live suite.
    """
    return _LIVE_SUITE.get() is not None


def refresh_live_suite(project_root: str, path: str) -> int:
    """Re-collect ONE test file into the live suite after writing it. Returns its test count.

    The live suite is a SNAPSHOT of the collection taken when the session opened, and
    `discover_test_callables` serves it to every later caller. That is exactly right for a
    consumer that only READS a suite. It is silently wrong for one whose product is WRITING
    tests: it writes a file, re-profiles, and is handed a list that predates its own work — so
    it scores the suite it had BEFORE it did anything. Measured on a 25-mutant function: the
    written tests were on disk, passing, and killing 18, while the run that wrote them reported
    2 and asked the user to supply inputs for the 14 it had already killed. Both features are
    correct alone; only their composition is not.

    ONLY the named file is re-collected, and only ITS prior callables are replaced. A blanket
    re-collect would be worse than the bug: the collect-only backend cannot bind a
    fixture-taking test, so refreshing the whole suite would silently DROP every one of them —
    reintroducing the false-survivor bug the live session exists to prevent. Restricting the
    blast radius to the written file is safe because the writer's own output is plain
    functions: whatever it generates, IT generates, and it does not generate fixtures.

    Identity has to survive BOTH shapes a callable can arrive in, and a tag is the only thing
    that does. A live item wraps its test as ``__wrapped__`` (see
    ``pytest_runner._make_item_callable``), so its file is recoverable — but a re-collected
    parametrized case is a closure built in ``pytest_discovery``, whose ``co_filename`` is that
    module, not the test's. Reading the code object alone therefore fails to recognise what a
    PREVIOUS refresh added, and each pass appends another copy of the same tests instead of
    replacing them. Tagging what we add makes the second refresh see the first's work.

    Re-measuring the session baseline is not an extra: the baseline measures which test covers
    which line, so a suite with a new test in it has no measurement for that test, and
    `_build_test_scope` finds no covering tests to run it against. Refreshing the list without
    the baseline changes what is DISCOVERED and nothing about what is RUN — the count stays
    exactly as wrong.

    It is a SPLICE, not an invalidation, and the difference is the cost of the consumer that
    needs this at all. Invalidating answered "one file changed" by re-measuring every file, so
    a consumer that writes tests in a loop paid the whole suite per pass — `O(passes x suite)`
    for a per-file edit. Measured on Detective's own 312-test suite: converge re-traced
    1,253 tests across 4 full passes (8.4s, 60% of wall clock) to write 3; the splice re-traces
    317 (2.2s, 27%) for the same verdict and the same generated file, byte for byte. The other
    tests' coverage is a CONSTANT across a write that did not touch them, and re-measuring a
    constant is the work `trace_suite` already refuses per function.

    The unit of replacement is a test NAME, not a file — see `SessionBaseline.replaced`, which
    is where that reasoning and the id()-reuse hazard live.

    Returns 0 and does nothing when no session is live: the non-live path re-collects on every
    call already and has nothing to invalidate.
    """
    live = _LIVE_SUITE.get()
    if live is None:
        return 0
    target = os.path.abspath(path)

    def _name(c: Any) -> str:
        # The key `SessionBaseline.traced` / `.failing` / `.truncated` use — `trace_suite` reads
        # the same attribute with the same fallback, so a splice keyed here lines up with what
        # the full build wrote. Not `_origin`: a parametrized case's code object names its
        # DEFINING module, which is why the file cannot be recovered from the name.
        return getattr(c, "__name__", "unknown")

    # Origin resolution is the module-level contract — one resolver for every
    # consumer, so a callable shape added later cannot be recognised here and
    # missed elsewhere (or vice versa).
    _origin = callable_origin

    # `gone` is held, not just counted: it carries the ids `SessionBaseline.inert` is keyed by,
    # and holding the objects until the splice is done is what stops a freed id being reused by
    # a later allocation and barring an unrelated test from kill attribution.
    gone = [c for c in live if _origin(c) == target]
    kept = [c for c in live if _origin(c) != target]
    fresh: list[Any] = []
    try:
        from Wesker.pytest_discovery import collect_pytest_callables

        fresh = list(collect_pytest_callables(project_root, paths=[target]) or [])
    except Exception:  # noqa: BLE001 — a failed refresh must not fail the caller's run
        fresh = []
    for c in fresh:
        with contextlib.suppress(Exception):  # builtins/C callables reject attributes
            c.__wesker_origin__ = target
    _LIVE_SUITE.set(kept + fresh)

    from Wesker.engine import (
        _SESSION_BASELINE,
    )  # local: engine imports ci at module scope

    holder = _SESSION_BASELINE.get()
    if holder is not None:
        # Re-measure ONLY what this write changed. `traced` is keyed by test ``__name__`` and
        # UNIONS duplicates across files, so the unit of replacement is a NAME, not a file: a
        # name this file merely shares with another module cannot be dropped on its own, or the
        # other owner's coverage goes with it and every mutant it kills reads as a survivor. So
        # re-trace every CURRENT owner of an affected name — this file's new tests, plus any
        # same-named test elsewhere whose entry the drop takes with it. Normally that is exactly
        # `fresh` (a writer's output is uniquely named), and the pass is milliseconds.
        affected = {_name(c) for c in gone} | {_name(c) for c in fresh}
        holder.refresh(
            affected,
            {id(c) for c in gone},
            [c for c in kept if _name(c) in affected] + fresh,
            len(kept) + len(fresh),
        )
    return len(fresh)


def run_with_live_suite(
    project_root: str,
    fn: Callable[[], Any],
    target_files: Iterable[str] | None = None,
    paths: list[str] | None = None,
    trace_progress: Callable[[int, int, float], None] | None = None,
    trace_budget_s: float | None = _UNSET,
    trace_session_budget_s: float | None = _UNSET,
    diagnostic: dict[str, Any] | None = None,
) -> Any:
    """Run ``fn()`` inside a LIVE pytest session — the public seam for any consumer.

    THE POINT: a caller wraps its entry point ONCE and every Wesker API it calls
    underneath transparently upgrades. ``discover_test_callables`` returns the live
    suite (fixtures, conftest, setup/teardown, parametrization — nothing skipped), and
    the suite-global baseline is computed once instead of per function. No signatures
    change; no caller passes callables around; nobody outside this module needs to know
    a ContextVar exists.

    This exists because the session CANNOT be handed out and left open — pytest owns
    the loop, so the work must happen INSIDE it. That inversion of control is the one
    thing a consumer cannot paper over itself, and re-deriving it per consumer is how
    the same bug lands in three places. ``Detective``'s profiler, for one, calls
    ``discover_test_callables`` directly; without this it silently drops every
    fixture-taking test in the target suite.

    ``target_files`` are the source files about to be mutated. Given, the session
    baseline is traced once for all of them (see :class:`~Wesker.engine.SessionBaseline`).
    Omitted, only the live suite is provided and the per-function baseline stands — still
    correct, just slower.

    ``trace_budget_s`` (per test) and ``trace_session_budget_s`` (the whole pass) bound the
    baseline trace this seam runs, and default to the engine's own. They are here because the
    baseline is traced HERE: a consumer exposing budget flags of its own had nowhere to send
    them, so the values reached only the per-function path while the pass that actually traces
    the suite kept the engine defaults — a documented opt-out that could not reach the thing it
    opts out of. ``None`` is a real value meaning unbounded, so "not passed" is a distinct state
    (see ``_UNSET``) and omitting them leaves the defaults exactly as they were.

    ``trace_progress(done, total, elapsed_ms)`` reports that baseline trace. It matters MOST
    here: this is the earliest thing that happens, it traces the WHOLE suite, and it runs before
    the consumer's own reporting can print anything at all — so without it a large suite spends
    minutes at 100% CPU emitting nothing, which reads as a wedged tool rather than a working one.
    A consumer that reports its own progress must pass this too, or its first phase is invisible.

    Returns ``fn()``'s value, or ``None`` when no live session could be started (pytest
    missing, collection failed, nothing collected). ``None`` is a DISTINCT outcome and
    callers must treat it as one: falling back silently to the collect-only path is the
    exact failure this seam exists to end.
    """
    from Wesker.engine import (
        _SESSION_BASELINE,
        DEFAULT_TRACE_BUDGET_S,
        DEFAULT_TRACE_SESSION_BUDGET_S,
        LazySessionBaseline,
        build_session_baseline,
    )
    from Wesker.pytest_runner import run_in_session

    resolved = {
        os.path.abspath(t if os.path.isabs(t) else os.path.join(project_root, t))
        for t in (target_files or ())
    }

    # Only forward a budget the caller actually named, so the engine's own defaults stay the
    # default. Passing `None` through unconditionally would read as "unbounded" and quietly
    # remove the only bound on the baseline phase.
    budgets: dict[str, Any] = {}
    if trace_budget_s is not _UNSET:
        budgets["trace_budget_s"] = trace_budget_s
    if trace_session_budget_s is not _UNSET:
        budgets["trace_session_budget_s"] = trace_session_budget_s

    # The budgets the baseline will ACTUALLY be built under: what the caller named, else the
    # engine's own defaults — resolved HERE because `budgets` deliberately records only what was
    # named, and "not named" is not a value a consumer can key a verdict on. Published on the
    # holder (below) so a cache can ask what produced a verdict without forcing the trace that
    # would produce it. `_UNSET` never escapes this function.
    effective: tuple[float | None, float | None] = (
        budgets.get("trace_budget_s", DEFAULT_TRACE_BUDGET_S),
        budgets.get("trace_session_budget_s", DEFAULT_TRACE_SESSION_BUDGET_S),
    )

    def _body(callables: list[Any], _session: Any) -> Any:
        suite_token = _LIVE_SUITE.set(callables)

        def _build(subset: list[Any] | None = None) -> Any:
            # The guard lives INSIDE the closure because the closure decides when it runs. The
            # baseline RUNS the consumer's whole suite — arbitrary third-party code — and any of
            # it can leave `sys.stdout` replaced: by assigning it, or by being cut mid-
            # `redirect_stdout` so its `__exit__` reinstalls a stale buffer on the way out. The
            # engine guards each test where it runs one; this is where such a leak stops being
            # the engine's problem and becomes the CONSUMER's, because `fn()` is the caller's
            # whole program and a dead `sys.stdout` means its report goes nowhere while it exits
            # 0. Re-entering the streams as they are on entry captures whatever the pass does and
            # hands them back intact. Wrapping the STORE below instead would guard nothing: this
            # now fires lazily, from deep inside `fn()`, long after any such block had exited.
            with (
                contextlib.redirect_stdout(sys.stdout),
                contextlib.redirect_stderr(sys.stderr),
            ):
                # The CURRENT live suite, not the list this closure captured. `refresh_live_suite`
                # replaces that list when a consumer writes tests, and this may run after it —
                # rebuilding from the captured snapshot would re-measure the suite we already
                # know is out of date, which is the whole bug.
                #
                # `subset` is the partial re-measure `LazySessionBaseline.refresh` splices in
                # after a write. It goes through THIS closure, not a second one, so a partial is
                # measured under the same target files and the same budgets as the whole — the
                # one property that makes the two safe to merge. It reports no progress: the
                # phase is a handful of tests, and a "baseline traced · 3 tests" line under a
                # label that meant 312 describes the splice as if it were the suite.
                return build_session_baseline(
                    subset if subset is not None else (_LIVE_SUITE.get() or callables),
                    resolved,
                    trace_progress=trace_progress if subset is None else None,
                    # The persistent cache lives under the CONSUMER's `.wesker/`, so the root has
                    # to reach it — this closure is the only place that has both.
                    project_root=project_root,
                    **budgets,
                )

        # Stored, not built. Whether the suite is traced at all is now the consumer's demand:
        # a run whose own cache answers the question never triggers it, and a run that needs it
        # gets it once. See `LazySessionBaseline` for why that is where the cost belongs.
        base_token = (
            _SESSION_BASELINE.set(LazySessionBaseline(_build, budgets=effective))
            if resolved
            else None
        )
        try:
            return fn()
        finally:
            if base_token is not None:
                _SESSION_BASELINE.reset(base_token)
            _LIVE_SUITE.reset(suite_token)

    result = run_in_session(project_root, _body, paths=paths, diagnostic=diagnostic)
    if result is None and paths:
        # The SCOPED collection bound nothing. That is a statement about the scope, not about
        # the suite — and the expensive pass is right there. `paths` is an optimisation: it
        # narrows collection to the files that could execute the target. When it narrows to a
        # set that will not collect (one stale import among them is enough — pytest's collection
        # is per-invocation, not per-file), the honest move is to pay for the whole suite rather
        # than report a suite that does not exist. Measured on TailChasingFixer: the 6 scoped
        # files could not collect, while the full 61 collect 815 tests.
        #
        # SAFE TO RETRY, and only because of the guard in `_Driver.pytest_runtestloop`: `body`
        # is the consumer's whole program — it writes test files — so a retry that could
        # double-run it would be far worse than a wrong number. `run_in_session` returns None
        # ONLY when body never ran, so the first attempt has no side effects to repeat. Never
        # retry on an exception: that means body DID run and raised.
        #
        # Coverage, not perfection: slow and correct beats fast and false. A user who wants the
        # cheap answer already got it when the scope collected.
        result = run_in_session(project_root, _body, paths=None, diagnostic=diagnostic)
    return result


def profile_codebase_live(
    project_root: str,
    targets: list[str],
    paths: list[str] | None = None,
    **kwargs: Any,
) -> dict | None:
    """:func:`profile_codebase`, executed inside a LIVE pytest session.

    This is Wesker-as-a-mutation-tester: the whole profile runs within
    ``pytest_runtestloop``, so every mutant is judged by pytest itself — real
    fixtures, conftest, parametrization, setup/teardown, markers, and pytest's own
    pass/fail verdict. Still one collection and no subprocess per mutant, so the
    in-process cost model (and every speed claim resting on it) is unchanged.

    WHY IT MATTERS: the ordinary path collects with ``--collect-only``, which tears
    the session down immediately, leaving items whose fixtures can never be
    supplied — so ``pytest_discovery`` SKIPS every fixture-taking test. A mutant that
    only such tests could kill is then scored a survivor, and on a suite where
    collection fails outright the silent fall back to the legacy loader can invert
    the error and manufacture kills instead. Neither number is about the suite.

    A thin wrapper over :func:`run_with_live_suite`, which is the reusable seam —
    other consumers (Detective) wrap their own entry points with it rather than
    re-deriving the inversion of control.

    The report gains a ``suite`` block describing the baseline the numbers were measured
    against — see :func:`suite_health`. Without it, a run whose entire suite was already
    broken (unmet dependency, import error) is indistinguishable from a run against a
    codebase nobody had specified: both report every mutant surviving. The first is a
    broken environment and the second is a real finding, and a consumer must be able to
    tell them apart before publishing either.

    Returns ``None`` when no live session could be started. Callers MUST treat that as
    a distinct outcome and say so. ``profile_codebase`` remains available and unchanged.
    """

    def _profile_and_describe_suite() -> dict:
        report = profile_codebase(project_root, targets, **kwargs)
        health = suite_health()
        if health is not None:
            report["suite"] = health
        return report

    return run_with_live_suite(
        project_root,
        _profile_and_describe_suite,
        target_files=targets,
        paths=paths,
    )


def suite_health() -> dict | None:
    """What the session baseline learned about the suite, for the report.

    Only meaningful INSIDE a live session with a session baseline (i.e. called from within
    :func:`run_with_live_suite` with target files); returns ``None`` otherwise rather than
    inventing a reading.

    ``inert`` is the load-bearing number. A test that fails with the ORIGINAL code in place
    is barred from kill attribution — correctly, since it cannot testify about a mutant. But
    if that is true of the whole suite, every mutant survives and the run reports a confident
    0% "specified", which is a statement about a broken environment wearing the costume of a
    statement about the code. Surfacing the count is what lets a caller refuse to publish it.
    """
    from Wesker.engine import session_baseline

    # Resolving BUILDS the baseline if nothing has yet (see `LazySessionBaseline`). Correct here
    # rather than merely convenient: these numbers ARE the baseline, so a caller asking for them
    # is the demand, and returning "no data" because nobody happened to profile first would make
    # the answer depend on call order. Outside a live session it is still None — the honest
    # "there is no suite-global baseline to report", which is a different claim from zero.
    baseline = session_baseline()
    if baseline is None:
        return None
    return {
        "tests": baseline.n_tests,
        # Tests that fail with the unmutated original in place, and so cannot testify
        # about any mutant.
        "inert": len(baseline.inert),
        # The narrower, nameable subset: an assertion that fails on correct code is a
        # WRONG EXPECTATION, and a human can act on the name.
        "failing_on_baseline": list(baseline.failing),
        # Tests whose coverage trace was cut short by the trace budget — their line
        # coverage is under-counted by construction.
        "trace_truncated": len(baseline.truncated),
    }


def profile_codebase(
    project_root: str,
    targets: list[str],
    budget_ms_per_file: float = 10000,
    max_per_category: int | None = None,
    passes: int = 1,
    *,
    verbose: bool = True,
) -> dict:
    """Profile all functions across multiple files with multi-pass convergence.

    Automatically loads cached state from ``.wesker/mutation_report.json``
    (written by previous runs) to enable Layer 2 predictive priors. On
    first run, all category priors are uniform; subsequent runs prioritize
    categories with historically higher survival rates.

    Args:
        max_per_category: Per-category mutant budget. ``None`` (the default) is
            DOF mode: the budget is derived per function from its own degrees of
            freedom (``engine.dimension_budget``), so one pass covers every
            behavioral dimension exactly once — no constant to tune, and no
            budget spent re-covering a dimension already pinned. ``0`` tests
            every mutant (exhaustive); a positive int pins an explicit budget.
        passes: Convergence passes per function. In DOF mode one pass already
            reaches full DOF coverage, so extra passes deepen WITHIN covered
            dimensions (a second mutant per dimension, a third, …) rather than
            reaching new ones — they buy kill evidence, not coverage.
    """
    # Layer 2: load historical priors from previous run
    cached_state = _load_cached_state(project_root)
    if verbose and cached_state and cached_state.get("per_category"):
        n_cats = len(cached_state["per_category"])
        print(f"  {_DIM}(loaded {n_cats}-category priors from previous run){_RESET}")

    total_killed = 0
    total_mutants = 0
    total_equivalent = 0
    total_truncated = 0
    total_universe = 0
    total_dof = 0
    total_dof_covered = 0
    total_dof_pinned = 0
    total_functions = 0
    per_file: dict[str, dict] = {}
    global_cats: dict[str, dict] = {}
    # The actionable half of the report. Per-function results carry a record for every
    # surviving mutant — the source line and the behavioral dimension no test pins — and the
    # file-level aggregation below reduces them to counts. Collected here so the report can
    # say WHERE a specification is incomplete, not merely how much of it is: a count is a
    # score, a located dimension is a task. Without this the survivors exist only inside this
    # loop and are discarded when it ends.
    survivors: list[dict] = []
    start = time.monotonic()

    for i, target in enumerate(targets, 1):
        if verbose:
            short = target.rsplit("/", 1)[-1]
            print(f"  {_DIM}[{i}/{len(targets)}]{_RESET} {short}", end="", flush=True)

        file_start = time.monotonic()
        results = profile_file(
            project_root,
            target,
            budget_ms=budget_ms_per_file,
            max_per_category=max_per_category,
            passes=passes,
            cached_state=cached_state,
        )
        file_ms = (time.monotonic() - file_start) * 1000

        file_killed = sum(r.get("total_killed", 0) for r in results)
        file_total = sum(r.get("total_mutants", 0) for r in results)
        file_equiv = sum(r.get("total_equivalent", 0) for r in results)
        file_universe = sum(r.get("universe_size", 0) for r in results)
        file_dof = sum(r.get("dof_total", 0) for r in results)
        file_dof_covered = sum(r.get("dof_covered", 0) for r in results)
        file_dof_pinned = sum(r.get("dof_pinned", 0) for r in results)
        total_killed += file_killed
        total_mutants += file_total
        total_equivalent += file_equiv
        total_universe += file_universe
        total_dof += file_dof
        total_dof_covered += file_dof_covered
        total_dof_pinned += file_dof_pinned
        total_functions += len(results)
        # A function whose budget ran out was only PARTIALLY evaluated: its unevaluated
        # mutants are absent from both numerator and denominator, so the ratio is a
        # sample of the cheap-to-reach mutants, not a mutation score. Aggregated here
        # because the per-function flag never reached the report — a truncated run and a
        # complete one published byte-identical badges.
        total_truncated += sum(1 for r in results if r.get("budget_exhausted"))

        # Carry each survivor up with the function it came from. ``function_key`` is
        # "path::qualname", so the record is self-locating: file, line, and the dimension
        # left unspecified — everything an annotation or a SARIF result needs, and nothing
        # a consumer would have to re-derive.
        # value_survivor_records, NOT survivor_records: this report's headline is `spec_pct`,
        # which counts assertion kills alone, so the gap it names has to be the VALUE-
        # unspecified set — true survivors plus the crash/timeout kills that ran the code
        # without ever checking what it returned. Listing only true survivors would report a
        # gap and then name none of it. Falls back to the raw records for a result produced by
        # a path that does not distinguish them.
        for r in results:
            key = r.get("function_key", "")
            records = r.get("value_survivor_records") or r.get("survivor_records", [])
            for rec in records:
                survivors.append({**rec, "function_key": key})

        # Aggregate per-category stats for the report (feeds next run's priors)
        for r in results:
            for cat_data in r.get("per_category", []):
                cat_name = cat_data.get("category", "")
                if not cat_name:
                    continue
                agg = global_cats.setdefault(
                    cat_name,
                    {
                        "category": cat_name,
                        "total": 0,
                        "killed": 0,
                        "survived": 0,
                        "equivalent": 0,
                    },
                )
                agg["total"] += cat_data.get("total", 0)
                agg["killed"] += cat_data.get("killed", 0)
                agg["survived"] += cat_data.get("survived", 0)
                agg["equivalent"] += cat_data.get("equivalent", 0)

        if file_total > 0:
            effective_total = file_total - file_equiv
            kill_pct = (
                round(100 * file_killed / effective_total)
                if effective_total > 0
                else 100
            )
            per_file[target] = {
                "functions": len(results),
                "killed": file_killed,
                "total": file_total,
                "equivalent": file_equiv,
                "universe": file_universe,
                "dof": file_dof,
                "dof_covered": file_dof_covered,
                "dof_pinned": file_dof_pinned,
                "spec_pct": round(100 * file_dof_pinned / max(file_dof, 1)),
                "kill_pct": kill_pct,
                "elapsed_ms": round(file_ms),
            }
            if verbose:
                c = _pct_color(kill_pct)
                equiv_note = (
                    f" {_DIM}({file_equiv} equiv){_RESET}" if file_equiv else ""
                )
                coverage = (
                    f" {_DIM}[{file_total}/{file_universe}]{_RESET}"
                    if file_universe > file_total
                    else ""
                )
                print(
                    f" {c}{file_killed}/{file_total}{_RESET}{equiv_note}{coverage}"
                    f" {_DIM}{file_ms:.0f}ms{_RESET}"
                )
        else:
            if verbose:
                print(f" {_DIM}(no mutants){_RESET}")

    elapsed = (time.monotonic() - start) * 1000
    effective_total = total_mutants - total_equivalent
    kill_pct = round(100 * total_killed / max(effective_total, 1))

    return {
        "total_killed": total_killed,
        "total_mutants": total_mutants,
        "total_equivalent": total_equivalent,
        "total_universe": total_universe,
        "total_dof": total_dof,
        "total_dof_covered": total_dof_covered,
        # Did the SELECTION reach every behavioral dimension? Under the DOF budget this is
        # the greedy bound being met — a statement about the engine, not about the suite.
        "dof_pct": round(100 * total_dof_covered / max(total_dof, 1)),
        "total_dof_pinned": total_dof_pinned,
        # SPECIFICATION COMPLETENESS — the headline. What fraction of this codebase's
        # behavioral dimensions do its tests actually pin? The denominator comes from the
        # AST, so unlike a kill rate it means the same thing in every repo at every budget.
        "spec_pct": round(100 * total_dof_pinned / max(total_dof, 1)),
        "kill_pct": kill_pct,
        "total_functions": total_functions,
        # Functions whose per-file budget ran out before every selected mutant was
        # evaluated. Non-zero means ``kill_pct`` is a PARTIAL result: raise
        # ``budget_ms_per_file`` before quoting it as a mutation score.
        "total_truncated": total_truncated,
        "passes": passes,
        "elapsed_ms": round(elapsed),
        "per_file": per_file,
        "per_category": list(global_cats.values()),
        # Every surviving mutant, located. The report's only per-mutant detail: what a
        # consumer needs to ANNOTATE the gap rather than just score it.
        "survivors": survivors,
    }
