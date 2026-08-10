"""Worker: run node IDs against mutants, isolated in this process (#19).

Spawned by `isolation` in its own process group with a hard timeout. Two protocols:

  * one-shot (``python -m Wesker._isolated_worker``): read ONE JSON spec on stdin, evaluate, and
    exit with pytest's code — the verdict IS the exit code.
  * server (``--serve``): read a SESSION spec (first line), then one mutant spec per line, writing
    ``{"rc": N, "killed_by": ..., "constructed": bool, "test_name": ...}`` per line back — so many
    mutants reuse one interpreter, recycled by the parent after a bounded count or on drift. The
    result goes to the REAL stdout; pytest's own output is redirected away so it cannot corrupt the
    JSON-line protocol.

Either way the mutant is installed through pytest's NORMAL per-test lifecycle (a
``pytest_runtest_call`` hookwrapper reusing the exact ``_patch_mutant_into_test`` /
``_unpatch_mutant`` the in-process evaluator uses), compiled with the target module's globals in
scope (mirroring ``engine.evaluate_mutant``) so a mutation is measured, not a bare NameError.

pytest's exit code alone answers only killed/survived. The kill VOCABULARY the in-process path
carries — ``assertion`` / ``exception`` (value pins) vs ``crash`` / ``timeout`` (run-only) — would
collapse to a single "failed" here, so the plugin inspects each node's ``excinfo`` and classifies it
through the SAME ladder ``engine._run_test_with_timeout`` uses, then reports ``killed_by`` so the
engine's ``value_killed`` split survives the crossing into an isolated process.
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

from Wesker.isolation import aggregate_kill_reason, classify_kill_reason


def _build_mutant(target_file: str, func_name: str, mutant_source: str) -> Any | None:
    """Compile the mutant with the target module's globals in scope (#19).

    Without them a function that calls a module-level helper raises ``NameError`` under EVERY
    mutant — a false all-crash 100% that hides whether the mutation is actually caught (the reason
    ``evaluate_mutant`` seeds the same namespace). Returns None if the file/name cannot be resolved
    or the mutant will not compile — the caller reports ``constructed=False`` so the engine scores it
    ``harness_error`` (outside the denominator), never a survivor.
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
    """Install the mutant into each test's namespace for the duration of its call, and CLASSIFY
    why each node failed (#19).

    The install/teardown reuse the engine's own ``_patch_mutant_into_test`` / ``_unpatch_mutant`` so
    the isolated lifecycle is bit-identical to the in-process one. The per-node classification reads
    the call's ``excinfo`` and names the reason with :func:`classify_kill_reason`, mirroring
    ``engine._run_test_with_timeout``'s ``except AssertionError`` / declared-failure / crash ladder —
    ``KeyboardInterrupt``/``SystemExit`` excluded exactly as it re-raises them.
    """

    def __init__(self, mutated_obj: Any, func_qualname: str) -> None:
        self._mutated = mutated_obj
        self._qualname = func_qualname
        self._func_name = func_qualname.split(".")[-1]
        self.reasons: list[str] = []
        self.first_failing_node: str | None = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: Any) -> Any:
        from Wesker.engine import (
            _execution_guard,
            _is_declared_failure,
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
                outcome = yield
            finally:
                _unpatch_mutant(proof, patched, saved, target, self._func_name)
        self._record(item, outcome, _is_declared_failure)

    def _record(self, item: Any, outcome: Any, is_declared_failure: Any) -> None:
        """Name why THIS node failed and remember it, or do nothing if it passed.

        ``outcome.excinfo`` is pluggy's ``(type, value, tb)`` for a hooked call that raised, or None
        when it returned — a pass is not a kill and records nothing. A skip surfaces as a non-Failed
        exception and lands on ``crash``, exactly as in-process; harmless, because pytest's exit code
        (not this list) decides the kill, and ``crash`` is outranked by any value pin in the
        aggregate. ``KeyboardInterrupt``/``SystemExit`` are excluded, matching the in-process re-raise.
        """
        excinfo = getattr(outcome, "excinfo", None)
        if excinfo is None:
            return
        exc = excinfo[1]
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            return
        reason = classify_kill_reason(
            isinstance(exc, AssertionError), bool(is_declared_failure(exc))
        )
        self.reasons.append(reason)
        if self.first_failing_node is None:
            self.first_failing_node = getattr(item, "nodeid", None)


def _resolve_target(root: str, target_file: str) -> str:
    return (
        target_file if os.path.isabs(target_file) else os.path.join(root, target_file)
    )


def _evaluate_full(
    target_abspath: str, func_qualname: str, node_ids: list[str], mutant_source: str
) -> dict[str, Any]:
    """Run the node IDs against ONE mutant; return its verdict payload.

    ``rc`` is pytest's exit code (the killed/survived authority); ``killed_by`` is the aggregated
    kill vocabulary the engine needs to keep its ``value_killed`` split; ``constructed`` is False
    when the mutant would not compile (→ ``harness_error``, never a survivor); ``test_name`` is the
    first failing node, for the kill matrix. pytest's own output is redirected to a sink so, in
    server mode, it cannot land in the JSON-line protocol on stdout; ``--capture=sys`` keeps the same
    isolation at the Python level without touching descriptor 1, the protocol channel.
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
    return {
        "rc": int(rc),
        "killed_by": aggregate_kill_reason(plugin.reasons) or None,
        "constructed": mutated is not None,
        "test_name": plugin.first_failing_node,
    }


def _trace_baseline_run(target_abspath: str, node_ids: list[str]) -> dict[str, Any]:
    """Run node IDs against the UNMUTATED target under a line tracer, in THIS fresh process (#19).

    The determinism check runs this twice in two fresh workers and compares. Here, from one fresh
    process, it captures the set of TARGET lines the covering tests execute and the run's outcome.
    The tracer mirrors ``line_coverage._trace_one_multi``'s dispatch: a global ``settrace`` whose
    dispatch installs the per-line recorder only on frames whose code lives in the target file —
    ``settrace``'s global hook still fires for every nested frame, so a test in another file that
    CALLS the target is recorded too. ``settrace`` binds the calling (main) thread, which is where
    pytest runs the node bodies.
    """
    from Wesker.isolation import isolated_test_outcome

    # `realpath`, not `abspath`: `python -m` prepends "" (cwd) to sys.path so an import can resolve to
    # a RELATIVE co_filename, AND on macOS the target dir is reached through the /var -> /private/var
    # SYMLINK — abspath leaves the symlink in, so the two spellings never == . realpath canonicalizes
    # both. A per-filename cache keeps the resolve off the hot per-line path; both runs normalize
    # identically, so the comparison is stable.
    target_real = os.path.realpath(target_abspath)
    covered: set[int] = set()
    _norm: dict[str, bool] = {}

    def _is_target(fn: str) -> bool:
        hit = _norm.get(fn)
        if hit is None:
            hit = os.path.realpath(fn) == target_real
            _norm[fn] = hit
        return hit

    def _local(frame: Any, event: str, _arg: Any) -> Any:
        if event == "line" and _is_target(frame.f_code.co_filename):
            covered.add(frame.f_lineno)
        return _local

    def _dispatch(frame: Any, event: str, _arg: Any) -> Any:
        if event == "call" and _is_target(frame.f_code.co_filename):
            return _local
        return None

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        sys.settrace(_dispatch)
        try:
            rc = pytest.main(
                [
                    *node_ids,
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "--no-header",
                    "--capture=sys",
                ]
            )
        finally:
            sys.settrace(None)
    return {"lines": sorted(covered), "outcome": isolated_test_outcome(int(rc), False)}


def _setup(spec: dict[str, Any]) -> None:
    root = spec["project_root"]
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _serve() -> int:
    """Persistent protocol: session spec, then one mutant spec per line -> one verdict dict per line.

    A mutant line may carry its OWN ``node_ids`` (per-mutant test scoping, so the isolated verdict
    matches the in-process scoped one); absent, the session's node_ids are used.
    """
    real_stdout = sys.stdout  # captured BEFORE any redirect: the protocol channel
    session = json.loads(sys.stdin.readline())
    _setup(session)
    target = _resolve_target(session["project_root"], session["target_file"])
    qualname = session["func_qualname"]
    session_nodes = session.get("node_ids", [])
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        spec = json.loads(line)
        nodes = spec.get("node_ids") or session_nodes
        payload = _evaluate_full(target, qualname, nodes, spec["mutant_source"])
        real_stdout.write(json.dumps(payload) + "\n")
        real_stdout.flush()
    return 0


def main() -> int:
    if "--serve" in sys.argv[1:]:
        return _serve()
    if "--baseline" in sys.argv[1:]:
        # Traced baseline run of the unmutated target, for the determinism check (#19). One fresh
        # process, one JSON line: {"lines": [...], "outcome": "..."}. The tracer's own redirect is
        # restored before this write, so the protocol line lands on the real stdout.
        real_stdout = sys.stdout
        spec = json.loads(sys.stdin.read())
        _setup(spec)
        target = _resolve_target(spec["project_root"], spec["target_file"])
        payload = _trace_baseline_run(target, spec["node_ids"])
        real_stdout.write(json.dumps(payload) + "\n")
        real_stdout.flush()
        return 0
    spec = json.loads(sys.stdin.read())
    _setup(spec)
    target = _resolve_target(spec["project_root"], spec["target_file"])
    # One-shot's verdict IS its exit code (`run_mutant_isolated` reads `mutant_verdict` off it); the
    # richer killed_by/constructed payload travels only over the server protocol the engine uses.
    return _evaluate_full(
        target, spec["func_qualname"], spec["node_ids"], spec["mutant_source"]
    )["rc"]


if __name__ == "__main__":
    sys.exit(main())
