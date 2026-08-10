"""Worker: run node IDs against mutants, isolated in this process (#19).

Spawned by `isolation` in its own process group with a hard timeout. Two protocols:

  * one-shot (``python -m Wesker._isolated_worker``): read ONE JSON spec on stdin, evaluate, and
    exit with pytest's code — the verdict IS the exit code.
  * server (``--serve``): read a SESSION spec (first line), then one mutant spec per line, writing
    ``{"rc": N}`` per line back — so many mutants reuse one interpreter, recycled by the parent
    after a bounded count or on drift. The result goes to the REAL stdout; pytest's own output is
    redirected away so it cannot corrupt the JSON-line protocol.

Either way the mutant is installed through pytest's NORMAL per-test lifecycle (a
``pytest_runtest_call`` hookwrapper reusing the exact ``_patch_mutant_into_test`` /
``_unpatch_mutant`` the in-process evaluator uses), compiled with the target module's globals in
scope (mirroring ``engine.evaluate_mutant``) so a mutation is measured, not a bare NameError.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
from typing import Any

import pytest


def _build_mutant(target_file: str, func_name: str, mutant_source: str) -> Any | None:
    """Compile the mutant with the target module's globals in scope (#19).

    Without them a function that calls a module-level helper raises ``NameError`` under EVERY
    mutant — a false all-crash 100% that hides whether the mutation is actually caught (the reason
    ``evaluate_mutant`` seeds the same namespace). Returns None if the file/name cannot be resolved
    or the mutant will not compile — the caller then patches nothing and the node runs the original.
    """
    spec = importlib.util.spec_from_file_location("_wesker_mutant_target", target_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        namespace: dict[str, Any] = dict(vars(module))
        exec(compile(mutant_source, "<mutant>", "exec"), namespace)  # noqa: S102
    except Exception:  # noqa: BLE001 — a mutant that will not compile installs nothing
        return None
    return namespace.get(func_name)


class _MutantPlugin:
    """Install the mutant into each test's namespace for the duration of its call (#19)."""

    def __init__(self, mutated_obj: Any, func_qualname: str) -> None:
        self._mutated = mutated_obj
        self._qualname = func_qualname
        self._func_name = func_qualname.split(".")[-1]

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: Any) -> Any:
        from Wesker.engine import (
            _execution_guard,
            _patch_mutant_into_test,
            _unpatch_mutant,
        )

        test_fn = getattr(item, "function", None)
        if test_fn is None or self._mutated is None:
            yield
            return
        with _execution_guard() as proof:
            patched, saved, target = _patch_mutant_into_test(
                proof, test_fn, self._qualname, self._mutated
            )
            try:
                yield
            finally:
                _unpatch_mutant(proof, patched, saved, target, self._func_name)


def _resolve_target(root: str, target_file: str) -> str:
    return (
        target_file if os.path.isabs(target_file) else os.path.join(root, target_file)
    )


def _evaluate_one(
    target_abspath: str, func_qualname: str, node_ids: list[str], mutant_source: str
) -> int:
    """Run the node IDs against ONE mutant and return pytest's exit code.

    pytest's own output is redirected to a sink so, in server mode, it cannot land in the
    JSON-line protocol on stdout; `--capture=sys` keeps the same isolation at the Python level
    without touching descriptor 1, the protocol channel.
    """
    mutated = _build_mutant(target_abspath, func_qualname.split(".")[-1], mutant_source)
    plugin = _MutantPlugin(mutated, func_qualname)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        rc = pytest.main(
            [*node_ids, "-p", "no:cacheprovider", "-q", "--no-header", "--capture=sys"],
            plugins=[plugin],
        )
    return int(rc)


def _setup(spec: dict[str, Any]) -> None:
    root = spec["project_root"]
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _serve() -> int:
    """Persistent protocol: session spec, then one mutant spec per line -> one `{"rc": N}` per line."""
    real_stdout = sys.stdout  # captured BEFORE any redirect: the protocol channel
    session = json.loads(sys.stdin.readline())
    _setup(session)
    target = _resolve_target(session["project_root"], session["target_file"])
    qualname = session["func_qualname"]
    node_ids = session["node_ids"]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        mutant_source = json.loads(line)["mutant_source"]
        rc = _evaluate_one(target, qualname, node_ids, mutant_source)
        real_stdout.write(json.dumps({"rc": rc}) + "\n")
        real_stdout.flush()
    return 0


def main() -> int:
    if "--serve" in sys.argv[1:]:
        return _serve()
    spec = json.loads(sys.stdin.read())
    _setup(spec)
    target = _resolve_target(spec["project_root"], spec["target_file"])
    return _evaluate_one(
        target, spec["func_qualname"], spec["node_ids"], spec["mutant_source"]
    )


if __name__ == "__main__":
    sys.exit(main())
