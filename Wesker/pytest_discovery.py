"""pytest-driven test discovery — the robust discovery backend.

Wesker's original discovery (``ci.load_test_callables``) hand-rolls test
collection and only recognises a couple of conventions (bare ``test_*``
functions, ``Test*``-prefixed classes). Real suites use many more —
``<Name>Tests`` suffixes, ``unittest.TestCase`` mixins, parametrization,
conftest — and the hand-rolled loader silently drops most of them
(e.g. 16 of 283 tests on a real repo).

This backend delegates *collection* to pytest, which already understands every
convention, then resolves each collected item to an in-process callable Wesker
can run against a mutant:

  * ``unittest.TestCase`` items  -> a native runner (setUpClass once, then
    setUp / method / tearDown), which RAISES on failure so Wesker registers the
    kill;
  * function items -> the function, with any ``@parametrize`` values bound in
    from ``item.callspec.params`` (one bound zero-arg callable per case).

Tests requiring real runtime fixtures (monkeypatch, tmp_path, a custom fixture)
are skipped for now (a documented extension): pytest tears the session down
after collection, so supplying those in-process is a follow-up. This backend is
additive — the original loader stays as the fallback (see
``ci.discover_test_callables``).
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import os
import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from Wesker.session_manifest import PytestSessionManifest

# The regime the most recent collection ran under (Detective #58). A ContextVar rather than a
# return value because `collect_pytest_callables` has one caller contract — it returns
# CALLABLES — and threading a second value through every path would change every signature to
# carry something most callers do not read. Same session-state idiom as `ci._LIVE_SUITE`.
_LAST_MANIFEST: ContextVar[PytestSessionManifest | None] = ContextVar(
    "wesker_last_session_manifest", default=None
)


def last_session_manifest() -> PytestSessionManifest | None:
    """The manifest of the most recent collection in this context, or None.

    None is a REAL answer and callers must treat it as one: it means no collection has happened
    here, or the capture itself failed. Falling back to a re-derived regime is the exact
    substitution #58 exists to end — a consumer that cannot get the runner's own answer should
    say so, not quietly compute a lookalike.
    """
    return _LAST_MANIFEST.get()


def _build_callables(items: list[Any]) -> list[Callable[..., Any]]:
    import unittest

    cls_setup_done: set[type] = set()

    def make_tc_runner(cls: type, method: str) -> Callable[..., None]:
        def run() -> None:
            if cls not in cls_setup_done:
                setup_cls = getattr(cls, "setUpClass", None)
                if callable(setup_cls):
                    setup_cls()
                cls_setup_done.add(cls)
            inst = cls(method)  # unittest binds the method name
            inst.setUp()
            try:
                getattr(inst, method)()  # raises on assertion failure -> kill
            finally:
                inst.tearDown()

        run.__name__ = f"{cls.__name__}.{method}"
        return run

    def _tag_origin(runnable: Callable[..., Any], it: Any) -> Callable[..., Any]:
        # Every callable this builder hands out is a closure whose code object
        # lives in THIS module — the origin tag is the only way a consumer
        # (ci.callable_origin) can recover which test FILE it stands for.
        origin = getattr(it, "path", None) or getattr(it, "fspath", None)
        if origin is not None:
            with contextlib.suppress(Exception):
                # Dynamic metadata on a function object: valid Python, absent from
                # FunctionType's stub, so the checker needs telling.
                runnable.__wesker_origin__ = str(origin)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        return runnable

    callables: list[Callable[..., Any]] = []
    for it in items:
        cls = getattr(it, "cls", None)
        if isinstance(cls, type) and issubclass(cls, unittest.TestCase):
            method = (
                getattr(it, "originalname", None)
                or str(getattr(it, "name", "")).split("[")[0]
            )
            if method:
                callables.append(_tag_origin(make_tc_runner(cls, method), it))
            continue
        fn = getattr(it, "function", None)
        if not callable(fn):
            continue
        runnable = _bind_item(it, fn)
        if runnable is None:
            continue  # requires real runtime fixtures we can't supply in-process (v1)
        callables.append(_tag_origin(runnable, it))
    return callables


def _bind_item(item: Any, fn: Callable[..., Any]) -> Callable[..., Any] | None:
    """Resolve a collected pytest function item to a zero-arg in-process callable.

    A test's directly-required arguments are its signature parameters. ``@parametrize``
    supplies those as static values on ``item.callspec.params`` — bindable in-process,
    so a parametrized test becomes one bound callable per case. A required parameter
    that is NOT a parametrized value is a real runtime fixture (monkeypatch, tmp_path,
    a custom fixture) we cannot construct here, so that item is skipped (v1).
    """
    try:
        sig_params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        sig_params = []

    if not sig_params:
        if not getattr(fn, "__name__", None):
            with contextlib.suppress(Exception):
                # Rebinding __name__ on a live callable is how a parametrized case keeps its
                # nodeid identity (#16); the stub types it read-only.
                fn.__name__ = str(getattr(item, "name", "test"))  # ty: ignore[unresolved-attribute]
        return fn

    callspec = getattr(item, "callspec", None)
    params = dict(getattr(callspec, "params", {}) or {}) if callspec is not None else {}
    if any(p not in params for p in sig_params):
        return None  # a required arg is a real fixture, not a parametrized value

    bound = {p: params[p] for p in sig_params}

    def run() -> None:
        fn(**bound)

    run.__name__ = f"{getattr(fn, '__name__', 'test')}[{getattr(callspec, 'id', '')}]"
    # Honor the trace-cache identity contract `_make_item_callable` (the live-session path) already sets,
    # or this discovery-backend closure is mis-keyed. `trace_cache.test_fingerprint` hashes
    # `getsource(__wrapped__ or self) + __qualname__`; with BOTH unset here, every parametrized case in
    # every project hashed to ONE constant (this two-line `run` body + `_bind_item.<locals>.run`), so the
    # per-test trace cache served whichever case was traced first as the coverage for all of them. The
    # nodeid discriminates the cases; `__wrapped__` makes getsource() read the USER's test, so a source
    # edit still invalidates the entry (and distinct tests still differ).
    run.__qualname__ = str(getattr(item, "nodeid", run.__name__))
    run.__wrapped__ = fn  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute] — functools.wraps sets it; not in the stub
    return run


def collect_pytest_callables(
    project_root: str, paths: list[str] | None = None
) -> list[Callable[..., Any]] | None:
    """Collect runnable test callables via pytest's own collection.

    Returns the callables, or ``None`` if pytest is unavailable or collection
    fails/finds nothing — the caller then falls back to the legacy loader. So a
    project without pytest, or one pytest can't collect, behaves exactly as
    before.
    """
    try:
        import pytest
    except ImportError:
        return None

    class _Collect:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def pytest_collection_modifyitems(self, session, config, items) -> None:
            self.items = list(items)
            # Record the regime this collection RAN under, from the live Config/Session that
            # ran it (Detective #58). The hook already received all three arguments and used
            # one; the other two are the only authoritative description of the regime that
            # exists, and every consumer was otherwise re-deriving it from files on disk.
            # Never allowed to fail the collection: the measurement is the product, its
            # description is not.
            try:
                from Wesker.session_manifest import capture_manifest

                _LAST_MANIFEST.set(capture_manifest(session, config, items))
            except Exception:  # noqa: BLE001 — a manifest that raises breaks a working run
                pass

    # Evict already-imported test modules whose source lives under any collection
    # root so pytest re-imports the CURRENT on-disk file. Repeated in-process
    # collections otherwise serve a rewritten generated test file stale from
    # sys.modules — hiding freshly written killing tests as false survivors. Only
    # test_* modules under a root are dropped, so unrelated imports are untouched.
    # Extra roots (absolute paths in ``paths``) are included so tests written
    # OUT-OF-TREE — e.g. converge's --write-dir on a scratch dir — are re-imported
    # too; otherwise the out-of-tree fix would still serve stale survivors.
    roots = [os.path.abspath(project_root)]
    roots += [os.path.abspath(p) for p in (paths or []) if os.path.isabs(p)]
    for _name in list(sys.modules):
        _mod = sys.modules.get(_name)
        _f = getattr(_mod, "__file__", None)
        if _f and os.path.basename(_f).startswith("test_"):
            _fa = os.path.abspath(_f)
            if any(_fa.startswith(r) for r in roots):
                del sys.modules[_name]
    importlib.invalidate_caches()

    plugin = _Collect()
    # `--capture=sys` for the same reason as `pytest_runner.run_in_session`: pytest's default
    # fd-capture replaces descriptor 1, which for a STDIO consumer is the protocol channel, not a
    # terminal — collection output lands in the JSON-RPC frame and the client drops the
    # connection. Collection is quieter than a run, but "quieter" is not "silent", and a single
    # stray byte breaks a frame just as completely.
    args = ["--collect-only", "-q", "-p", "no:cacheprovider", "--capture=sys"]
    args += paths or ["."]
    cwd = os.getcwd()
    try:
        os.chdir(project_root)
        # Suppress pytest's --collect-only node-id dump so it doesn't pollute
        # Wesker's own output.
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            pytest.main(args, plugins=[plugin])
    except Exception:
        return None
    finally:
        try:
            os.chdir(cwd)
        except Exception:
            pass

    if not plugin.items:
        return None
    try:
        callables = _build_callables(plugin.items)
    except Exception:
        return None
    return callables or None
