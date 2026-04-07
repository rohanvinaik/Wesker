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
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
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

            match = (
                # Exact generated name (highest confidence)
                name == generated_name
                # Parent-qualified (wiki/config.py -> test_wiki_config.py)
                or (parent_qualified and (
                    name == f"test_{parent_qualified}.py"
                    or name.startswith(f"test_{parent_qualified}_")
                ))
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
    """Find all test_*.py files under tests/."""
    found: list[str] = []
    tests_dir = Path(project_root) / "tests"
    if not tests_dir.is_dir():
        return found
    for py in sorted(tests_dir.rglob("*.py")):
        if py.name.startswith("test_") and "__pycache__" not in str(py):
            found.append(str(py))
    return found


# ── 3-Layer discovery orchestrator ───────────────────────────────


def discover_tests(project_root: str, source_file: str, func_names: list[str]) -> list[str]:
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


def load_test_callables(test_files: list[str]) -> list[Any]:
    """Load all test_* callables from test files, including class methods."""
    callables: list[Any] = []
    for tf in test_files:
        mod_name = f"_wesker_test_{Path(tf).stem}"
        if mod_name in sys.modules:
            # Already loaded — reuse
            mod = sys.modules[mod_name]
        else:
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
                callables.append(obj)
            elif isinstance(obj, type) and name.startswith("Test"):
                for mname in dir(obj):
                    if mname.startswith("test_"):
                        try:
                            callables.append(getattr(obj(), mname))
                        except Exception:
                            pass
    return callables


# ── AST utilities ────────────────────────────────────────────────


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
    max_per_category: int = 5,
    passes: int = 3,
    cached_state: dict | None = None,
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
    test_files = discover_tests(project_root, full_path, func_names)
    tests = load_test_callables(test_files)

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
            None,
            budget_ms=budget_ms,
            max_per_category=max_per_category,
            passes=passes,
            category_order=cat_order,
        )
        results.append(sr.to_dict())

    return results


# ── Single-function profiling ──────────────────────────────────


def profile_function(
    project_root: str,
    source_file: str,
    function_name: str,
    budget_ms: float = 10000,
    max_per_category: int = 5,
    passes: int = 3,
    cached_state: dict | None = None,
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

    if func_node is None:
        return None

    cats = filter_categories(func_node)
    if not cats:
        return None

    priors = prioritize_categories(cats, cached_state)
    cat_order = [p.category for p in priors]

    test_files = discover_tests(project_root, full_path, func_names)
    tests = load_test_callables(test_files)

    rel = os.path.relpath(full_path, project_root)
    func_key = f"{rel}::{qualname}"

    result = run_function_converged(
        func_node,  # type: ignore[arg-type]
        func_key,
        cats,
        tests,
        None,
        budget_ms=budget_ms,
        max_per_category=max_per_category,
        passes=passes,
        category_order=cat_order,
    )
    return result.to_dict()


# ── Per-function result cache ──────────────────────────────────


def _code_hash(source: str) -> str:
    """Stable hash of source text for cache invalidation."""
    import hashlib
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _load_function_cache(project_root: str) -> dict:
    """Load per-function result cache from .wesker/function_cache.json.

    Returns a dict keyed by ``func_key:code_hash`` → result dict.
    Entries whose code_hash no longer matches are stale and will be
    ignored by callers.
    """
    cache_path = Path(project_root) / ".wesker" / "function_cache.json"
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return {}


def _save_function_cache(project_root: str, cache: dict) -> None:
    """Write per-function result cache."""
    cache_dir = Path(project_root) / ".wesker"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "function_cache.json").write_text(json.dumps(cache, indent=2))


def profile_function_cached(
    project_root: str,
    source_file: str,
    function_name: str,
    budget_ms: float = 10000,
    max_per_category: int = 5,
    passes: int = 3,
) -> dict | None:
    """Profile a function with per-function caching.

    Checks ``.wesker/function_cache.json`` for a valid cached result
    (matching code hash). On hit, returns the cached result immediately.
    On miss, profiles the function, writes the result to cache, and returns it.

    This is the preferred entry point for interactive/local use where
    the same function may be profiled repeatedly across sessions.
    """
    full_path = (
        os.path.join(project_root, source_file)
        if not os.path.isabs(source_file)
        else source_file
    )
    try:
        source = Path(full_path).read_text()
    except OSError:
        return None

    # Extract just the function's source for hashing
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    func_source = None
    for qn, node in walk_functions(tree):
        if qn == function_name or qn.split(".")[-1] == function_name:
            lines = source.splitlines(keepends=True)
            end = getattr(node, "end_lineno", None) or len(lines)
            func_source = "".join(lines[node.lineno - 1 : end])
            break

    if func_source is None:
        return None

    h = _code_hash(func_source)
    rel = os.path.relpath(full_path, project_root)
    # Find qualname for the cache key
    for qn, node in walk_functions(tree):
        if qn == function_name or qn.split(".")[-1] == function_name:
            cache_key = f"{rel}::{qn}:{h}"
            break
    else:
        return None

    # Check cache
    cache = _load_function_cache(project_root)
    if cache_key in cache:
        return cache[cache_key]

    # Cache miss — profile
    cached_state = _load_cached_state(project_root)
    result = profile_function(
        project_root, source_file, function_name,
        budget_ms=budget_ms, max_per_category=max_per_category,
        passes=passes, cached_state=cached_state,
    )

    if result is not None:
        cache[cache_key] = result
        _save_function_cache(project_root, cache)

    return result


# ── Codebase profiling with formatted output ─────────────────────


def profile_codebase(
    project_root: str,
    targets: list[str],
    budget_ms_per_file: float = 10000,
    max_per_category: int = 5,
    passes: int = 3,
    *,
    verbose: bool = True,
) -> dict:
    """Profile all functions across multiple files with multi-pass convergence.

    Automatically loads cached state from ``.wesker/mutation_report.json``
    (written by previous runs) to enable Layer 2 predictive priors. On
    first run, all category priors are uniform; subsequent runs prioritize
    categories with historically higher survival rates.

    Args:
        passes: Number of convergence passes per function. Each pass uses
            a different seed, sampling different mutants. Higher values give
            stronger statistical guarantees but cost more time.
        max_per_category: Mutants sampled per category per pass. Total unique
            mutants tested ≈ passes × max_per_category per category.
    """
    # Layer 2: load historical priors from previous run
    cached_state = _load_cached_state(project_root)
    if verbose and cached_state and cached_state.get("per_category"):
        n_cats = len(cached_state["per_category"])
        print(f"  {_DIM}(loaded {n_cats}-category priors from previous run){_RESET}")

    total_killed = 0
    total_mutants = 0
    total_equivalent = 0
    total_universe = 0
    total_functions = 0
    per_file: dict[str, dict] = {}
    global_cats: dict[str, dict] = {}
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
        total_killed += file_killed
        total_mutants += file_total
        total_equivalent += file_equiv
        total_universe += file_universe
        total_functions += len(results)

        # Aggregate per-category stats for the report (feeds next run's priors)
        for r in results:
            for cat_data in r.get("per_category", []):
                cat_name = cat_data.get("category", "")
                if not cat_name:
                    continue
                agg = global_cats.setdefault(
                    cat_name,
                    {"category": cat_name, "total": 0, "killed": 0, "survived": 0, "equivalent": 0},
                )
                agg["total"] += cat_data.get("total", 0)
                agg["killed"] += cat_data.get("killed", 0)
                agg["survived"] += cat_data.get("survived", 0)
                agg["equivalent"] += cat_data.get("equivalent", 0)

        if file_total > 0:
            effective_total = file_total - file_equiv
            kill_pct = round(100 * file_killed / effective_total) if effective_total > 0 else 100
            per_file[target] = {
                "functions": len(results),
                "killed": file_killed,
                "total": file_total,
                "equivalent": file_equiv,
                "universe": file_universe,
                "kill_pct": kill_pct,
                "elapsed_ms": round(file_ms),
            }
            if verbose:
                c = _pct_color(kill_pct)
                equiv_note = f" {_DIM}({file_equiv} equiv){_RESET}" if file_equiv else ""
                coverage = f" {_DIM}[{file_total}/{file_universe}]{_RESET}" if file_universe > file_total else ""
                print(f" {c}{file_killed}/{file_total}{_RESET}{equiv_note}{coverage}"
                      f" {_DIM}{file_ms:.0f}ms{_RESET}")
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
        "kill_pct": kill_pct,
        "total_functions": total_functions,
        "passes": passes,
        "elapsed_ms": round(elapsed),
        "per_file": per_file,
        "per_category": list(global_cats.values()),
    }
