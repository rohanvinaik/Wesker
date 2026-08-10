"""Worker: run node IDs against ONE mutant, isolated in this process (#19).

Spawned by :func:`isolation.run_mutant_isolated` in its own process group with a hard timeout. It
reads a JSON spec on stdin, installs the mutant through the SAME ``_patch_mutant_into_test`` /
``_unpatch_mutant`` the in-process evaluator uses — via a pytest plugin, so the mutant runs through
pytest's NORMAL per-test lifecycle, exactly the guarantee this issue asks for. The exit code IS the
verdict: pytest 0 = every node passed under the mutant (survived), 1 = a node failed (the suite
detected the mutation), anything else is a harness/collection state that measures nothing.

The mutant is compiled into a namespace seeded with the target module's own globals, mirroring
``engine.evaluate_mutant`` (engine.py) so a mutation is measured rather than a bare NameError read
as an all-crash kill.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

import pytest


def _build_mutant(target_file: str, func_name: str, mutant_source: str) -> Any | None:
    """Compile the mutant with the target module's globals in scope (#19).

    The mutant resolves sibling helpers, module constants and imports through the target's own
    globals — without them a function that calls a module-level helper raises ``NameError`` under
    EVERY mutant, a false all-crash 100% that hides whether the mutation is actually caught (the
    exact reason ``evaluate_mutant`` seeds the same namespace). A fresh load of the target file
    supplies those globals; the mutant is patched into the TEST's namespace, so a second module
    object here changes nothing the test observes. Returns None if the file or name cannot be
    resolved — the caller then patches nothing and the node runs against the original.
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
        # Patch right before the test body and restore right after — the SAME window the in-process
        # evaluator uses, and the reason the mutant runs under the real pytest lifecycle rather than
        # a bare callable invocation. The execution lock is held only across the patch; a single
        # isolated worker has no concurrency, but `_patch_mutant_into_test` requires the proof.
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


def main() -> int:
    spec = json.loads(sys.stdin.read())
    root = spec["project_root"]
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    target = spec["target_file"]
    if not os.path.isabs(target):
        target = os.path.join(root, target)
    mutated = _build_mutant(
        target, spec["func_qualname"].split(".")[-1], spec["mutant_source"]
    )
    plugin = _MutantPlugin(mutated, spec["func_qualname"])
    return int(
        pytest.main(
            [*spec["node_ids"], "-p", "no:cacheprovider", "-q", "--no-header"],
            plugins=[plugin],
        )
    )


if __name__ == "__main__":
    sys.exit(main())
