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
from Wesker.mu_minus import begin_test
from Wesker.pytest_runner import FreshArguments, _install_cheap_failure_repr
from Wesker.write_authority import (
    contain_writes,
    declared_roots,
    install,
    session_write_roots,
)


def _build_mutant(target_file: str, func_name: str, mutant_source: str) -> Any | None:
    """Compile in the namespace pytest actually imported for this source.

    Relative imports, dataclasses and namespace packages use their real module identity.
    A flat module not imported during collection retains the standalone fallback, with
    normal sys.modules registration while its module body executes.
    """
    from types import ModuleType

    from Wesker.engine import _entry_probe

    target = os.path.realpath(target_file)
    # SNAPSHOT, not a redundant copy: the search below can trigger an import (and another thread
    # may import at any time), and mutating `sys.modules` while iterating its values raises
    # "dictionary changed size during iteration". Materialising first is what makes this safe.
    loaded_modules = list(sys.modules.values())
    module = next(
        (
            candidate
            for candidate in loaded_modules
            if isinstance(candidate, ModuleType)
            and isinstance(vars(candidate).get("__file__"), str)
            and os.path.realpath(vars(candidate)["__file__"]) == target
        ),
        None,
    )
    try:
        if module is None:
            name = "_wesker_mutant_target"
            spec = importlib.util.spec_from_file_location(name, target_file)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            previous = sys.modules.get(name)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        namespace: dict[str, Any] = dict(vars(module))
        exec(compile(mutant_source, "<mutant>", "exec"), namespace)  # noqa: S102
    # BLE001: a construction failure is UNMEASURED, never evidence of adequacy — the mutant could
    # not be built, so there is nothing to conclude about the suite that would have faced it.
    except Exception:  # noqa: BLE001
        return None
    obj = namespace.get(func_name)
    return _entry_probe(obj) if obj is not None else None


#: The pristine test arguments of this worker process (#55). Module-level on purpose: every
#: `pytest.main` here re-collects test modules that stay in `sys.modules`, so a module-level
#: `parametrize` row is the same object for every mutant. The snapshot is taken at the FIRST collection
#: this process makes (before any test body has run in it) and outlives every evaluation after it.
_ARGUMENTS = FreshArguments()


class _FreshArgumentsPlugin:
    """Give every evaluation fresh test arguments: the mutant's, and the original control's (#55).

    A mutant that writes into the argument it is given alters the object every later evaluation would
    be handed. ``FreshArguments`` snapshots each item's parameters once, at collection, and replaces
    them with a new deep copy before each item's setup, so no evaluation can change what another (or
    the control) receives. What a copy cannot reach is the original control's business.
    """

    def pytest_collection_finish(self, session: Any) -> None:
        for item in session.items:
            _ARGUMENTS.capture(item)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: Any) -> None:
        _ARGUMENTS.refresh(item)


def _node_failure_reason(outcome: Any, is_declared_failure: Any) -> str | None:
    """The kill-vocabulary word for one node's call outcome, or None (pass / not a failure).

    ``outcome.excinfo`` is pluggy's ``(type, value, tb)`` for a hooked call that raised, or None when
    it returned. ``KeyboardInterrupt``/``SystemExit`` are excluded exactly as the in-process
    evaluator re-raises them; ``MemoryError`` is returned as ``"memory"`` so the caller can name the
    cut. Everything else goes through :func:`classify_kill_reason`, the ladder
    ``engine._run_test_with_timeout`` uses.
    """
    excinfo = getattr(outcome, "excinfo", None)
    if excinfo is None:
        return None
    exc = excinfo[1]
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return None
    if isinstance(exc, MemoryError):
        return "memory"
    return classify_kill_reason(
        isinstance(exc, AssertionError), bool(is_declared_failure(exc))
    )


class _ControlPlugin(_FreshArgumentsPlugin):
    """Run nodes against the ORIGINAL and record how each fails (#55, the original control).

    Nothing is installed: the mutant's evaluation has already restored every binding, so the target
    is the original, in whatever process state that evaluation left (module globals, class
    attributes, files). A node that fails here the same way it failed under the mutant detected
    nothing: that is :func:`Wesker.engine.original_control_standing`, shared with the in-process loop.
    """

    def __init__(self) -> None:
        self.reasons: dict[str, str] = {}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: Any) -> Any:
        from Wesker.engine import _is_declared_failure

        outcome = yield
        reason = _node_failure_reason(outcome, _is_declared_failure)
        if reason is not None:
            self.reasons[getattr(item, "nodeid", "")] = reason


class _AuthoredControlPlugin(_ControlPlugin):
    """The original control on the AUTHORED objects: no fresh copy is handed out (#55)."""

    def pytest_runtest_setup(self, item: Any) -> None:
        return None


#: The nodes whose copies this process has already measured against the original (#55).
_MEASURED: set[str] = set()


class _MutantPlugin(_FreshArgumentsPlugin):
    """Install the mutant into each test's namespace for the duration of its call, and CLASSIFY
    why each node failed (#19).

    The install/teardown reuse the engine's own ``_patch_mutant_into_test`` / ``_unpatch_mutant`` so
    the isolated lifecycle is bit-identical to the in-process one. The per-node classification reads
    the call's ``excinfo`` and names the reason with :func:`classify_kill_reason`, mirroring
    ``engine._run_test_with_timeout``'s ``except AssertionError`` / declared-failure / crash ladder —
    ``KeyboardInterrupt``/``SystemExit`` excluded exactly as it re-raises them.
    """

    def __init__(
        self,
        mutated_obj: Any,
        func_qualname: str,
        target_file: str = "",
        mutant_source: str = "",
        step_caps: dict[str, int] | None = None,
        project_root: str = "",
        write_roots: tuple[str, ...] = (),
    ) -> None:
        self._target_file = target_file
        #: The write roots the spawning live session declared (#105): its basetemp and pytest's
        #: per-user root, which also holds this worker's own per-mutant session directories.
        self._write_roots = tuple(write_roots)
        self._mutant_source = mutant_source
        #: The loop budget per node, keyed by canonical selector (W#27). The parent counted the
        #: ORIGINAL on each node in its own process and sends the cap; an absent node is unbudgeted.
        self._step_caps = step_caps or {}
        self._project_root = project_root
        self._mutated = mutated_obj
        self._qualname = func_qualname
        self._func_name = func_qualname.split(".")[-1]
        self.reasons: list[str] = []
        self.first_failing_node: str | None = None
        #: Every failing node and its reason, in run order (#55): what the original control reads,
        #: and what `settle` rebuilds `reasons` / `first_failing_node` from once it has voided nodes.
        self.node_reasons: list[tuple[str, str]] = []
        #: Nodes whose failure the ORIGINAL reproduces in the same process state (#55): no kill.
        self.original_also_fails: list[str] = []
        #: Failures of a phase the call-phase ladder does not see (a fixture's setup, a teardown).
        #: They carry no reason; the control does not speak for them, so they keep their kill.
        self.other_failures = 0
        #: Nodes that reached their call phase, and every node collected, across the rounds of one
        #: evaluation (#55): a round that stops at a value failure the control voids is continued over
        #: the nodes it never ran, exactly as the in-process loop goes on to the next test.
        self.done: set[str] = set()
        self.collected: set[str] = set()
        #: Installation-and-entry proof (#18). `installed` is True once any node rebound an owner to
        #: the mutant; `ran` counts nodes that reached the call phase (so an empty run stays
        #: `unobserved`, never a false `not_entered`); `.entered` on the wrapped mutant records whether
        #: it was actually CALLED.
        self.installed = False
        self.ran = 0
        #: True when a node hit `MemoryError` under the worker's address-space cap (W#21) — a budget
        #: CUT, not a kill: the measurement could not complete, so the engine marks it non-gateable.
        self.memory_cut = False
        #: Nodes whose loop budget stopped the mutant (W#27): the named non-kill `diverged`, never a
        #: kill reason, whatever pytest's exit code says about the stopped node.
        self.diverged_nodes: list[str] = []
        self._mutant_codes: tuple[Any, ...] = ()
        #: The first node whose run against the mutant tried to write outside its write authority
        #: (Detective #105), and the paths that run was prevented from writing. Such a node records
        #: no kill reason: its failure is the guard's, never the mutant's.
        self.write_blocked: str | None = None
        self.blocked_writes: list[str] = []

    def pytest_runtest_logstart(self) -> None:
        """A test boundary (#49): a Form B wrapper's memory (stale) ends with the test that made it."""
        begin_test()

    def pytest_collection_modifyitems(self, items: list[Any]) -> None:
        """A continuation round runs only what an earlier round did not (#55)."""
        if self.done:
            items[:] = [i for i in items if getattr(i, "nodeid", "") not in self.done]

    def pytest_runtest_logreport(self, report: Any) -> None:
        self.done.add(str(getattr(report, "nodeid", "")))
        if getattr(report, "failed", False) and getattr(report, "when", "") != "call":
            self.other_failures += 1

    def pytest_collection_finish(self, session: Any) -> None:
        """Compile after pytest establishes the actual package/import regime."""
        super().pytest_collection_finish(session)
        from Wesker.engine import _step_codes_of

        self.collected.update(str(getattr(i, "nodeid", "")) for i in session.items)
        # Built ONCE per evaluation: a continuation round must keep the same mutant, whose `entered`
        # flag is the proof of entry the earlier rounds recorded.
        if self._mutant_source and self._mutated is None:
            self._mutated = _build_mutant(
                self._target_file, self._func_name, self._mutant_source
            )
        self._mutant_codes = (
            _step_codes_of(self._mutated) if self._mutated is not None else ()
        )

    def _step_arm_for(self, nodeid: str) -> tuple[tuple[Any, ...], int | None]:
        """The codes and cap one node's mutant observation runs under (W#27)."""
        from Wesker.isolation import isolated_selector_key

        cap = self._step_caps.get(isolated_selector_key(self._project_root, nodeid))
        if cap is None or not self._mutant_codes:
            return (), None
        return self._mutant_codes, cap

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: Any) -> Any:
        from Wesker.engine import (
            _execution_guard,
            _is_declared_failure,
            _patch_module_qualified,
            _patch_mutant_into_test,
            _StepArm,
            _unpatch_mutant,
        )

        test_fn = getattr(item, "function", None)
        if test_fn is None or self._mutated is None:
            yield
            return
        arm = _StepArm(*self._step_arm_for(getattr(item, "nodeid", "")))
        self.done.add(getattr(item, "nodeid", ""))
        with _execution_guard() as proof:
            module_saved = _patch_module_qualified(
                proof, self._func_name, self._mutated, self._target_file, self._qualname
            )
            patched, saved, target = _patch_mutant_into_test(
                proof, test_fn, self._qualname, self._mutated
            )
            if patched or module_saved:
                self.installed = True
            self.ran += 1
            # Armed only here, where the mutant is installed (#105) — never around `pytest.main`,
            # whose collection and basetemp creation are the harness's writes. This session's own
            # basetemp is read only if a fixture already made it: forcing one per mutant session costs
            # a numbered directory per mutant for a suite that never uses `tmp_path`.
            roots = (
                *self._write_roots,
                *session_write_roots(getattr(item, "config", None), create=False),
            )
            try:
                with (
                    arm,
                    declared_roots(roots),
                    contain_writes(_PROJECT_TREE or None) as blocked,
                ):
                    outcome = yield
            finally:
                _unpatch_mutant(proof, patched, saved, target, self._func_name)
                for owner, original in reversed(module_saved):
                    setattr(owner, self._func_name, original)
        if arm.stopped:
            # Budget-relative, never a kill (W#27): whatever pytest made of the stopped node.
            self.diverged_nodes.append(getattr(item, "nodeid", ""))
        if blocked and self.write_blocked is None:
            # The guard's failure, never the mutant's (Detective #105).
            self.write_blocked = getattr(item, "nodeid", None) or "?"
            self.blocked_writes = list(dict.fromkeys(blocked))[:3]
        # Both facts are recorded; the engine's per-mutant disposition ranks them.
        if arm.stopped or blocked:
            return
        self._record(item, outcome, _is_declared_failure)

    def _record(self, item: Any, outcome: Any, is_declared_failure: Any) -> None:
        """Name why THIS node failed and remember it, or do nothing if it passed.

        ``outcome.excinfo`` is pluggy's ``(type, value, tb)`` for a hooked call that raised, or None
        when it returned — a pass is not a kill and records nothing. A skip surfaces as a non-Failed
        exception and lands on ``crash``, exactly as in-process; harmless, because pytest's exit code
        (not this list) decides the kill, and ``crash`` is outranked by any value pin in the
        aggregate. ``KeyboardInterrupt``/``SystemExit`` are excluded, matching the in-process re-raise.
        """
        reason = _node_failure_reason(outcome, is_declared_failure)
        if reason is None:
            return
        if reason == "memory":
            # The address-space cap fired (W#21): the run exceeded its budget, so this is a CUT, not a
            # kill. Flag it and record no kill reason — pytest's rc will read `failed`, but the engine
            # checks the memory cut first and never scores it.
            self.memory_cut = True
            return
        self.node_reasons.append((getattr(item, "nodeid", ""), reason))
        self.settle(set())
        # Like the in-process evaluator, stop once a value kill is established.
        # Otherwise a later slow test can time out the worker and erase the already
        # observed assertion. shouldfail ends pytest with TESTS_FAILED, preserving it.
        if (
            reason in ("assertion", "exception")
            and getattr(item, "session", None) is not None
        ):
            item.session.shouldfail = "mutation distinguished by a value assertion"

    def settle(self, voided: set[str]) -> None:
        """Rebuild ``reasons`` and ``first_failing_node`` from the failing nodes not in ``voided``.

        Credit goes to the node supplying the strongest reason, never a preceding crash that would
        otherwise borrow a later assertion's value-kill attribution.
        """
        self.reasons = []
        self.first_failing_node = None
        for nodeid, reason in self.node_reasons:
            if nodeid in voided:
                continue
            previous = aggregate_kill_reason(self.reasons)
            self.reasons.append(reason)
            if (
                self.first_failing_node is None
                or aggregate_kill_reason(self.reasons) != previous
            ):
                self.first_failing_node = nodeid


class _CheapFailureRepr:
    """Give every item this worker collects the measurement session's cheap failure repr (W#34).

    EP-C1 (Detective docs/ENGINEERING_PASS_2026-09-26.md): every killed mutant is a failing test to
    pytest, and building its report formatted a source-annotated traceback — re-parsing source with
    ``ast`` per traceback entry — for text nothing reads. ``49f7e91`` stopped it for the in-process
    session; this process runs its OWN ``pytest.main`` per mutant (and per determinism baseline), so
    it kept paying, once per killed mutant. Nothing here reads a report's text either: the verdict is
    pytest's exit code, the kill reason comes from :class:`_MutantPlugin`'s raw ``excinfo``, and the
    terminal output goes to a sink. The SAME installer as the in-process session
    (``pytest_runner._install_cheap_failure_repr``), so the two cannot drift.
    """

    def pytest_collection_modifyitems(self, items: list[Any]) -> None:
        for item in items:
            _install_cheap_failure_repr(item)


def _resolve_target(root: str, target_file: str) -> str:
    return (
        target_file if os.path.isabs(target_file) else os.path.join(root, target_file)
    )


def _evaluate_full(
    target_abspath: str,
    func_qualname: str,
    node_ids: list[str],
    mutant_source: str,
    step_caps: dict[str, int] | None = None,
    project_root: str = "",
    write_roots: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run the node IDs against ONE mutant; return its verdict payload.

    ``rc`` is pytest's exit code (the killed/survived authority); ``killed_by`` is the aggregated
    kill vocabulary the engine needs to keep its ``value_killed`` split; ``constructed`` is False
    when the mutant would not compile (→ ``harness_error``, never a survivor); ``test_name`` is the
    first failing node, for the kill matrix. pytest's own output is redirected to a sink so, in
    server mode, it cannot land in the JSON-line protocol on stdout; ``--capture=sys`` keeps the same
    isolation at the Python level without touching descriptor 1, the protocol channel.
    """
    # W#27: per-node loop budgets arrive from the parent, which counted the ORIGINAL on each node in
    # its own process — never here, where a count pass would run nodes this mutant's first value
    # kill short-circuits, inside this request's own timeout.
    plugin = _MutantPlugin(
        None,
        func_qualname,
        target_abspath,
        mutant_source,
        step_caps,
        project_root,
        write_roots,
    )
    rcs: list[int] = []
    voided: set[str] = set()
    controlled = 0
    while True:
        rcs.append(_pytest_round(node_ids, plugin))
        # The ORIGINAL control (#55): every node that failed under the mutant is re-run against the
        # original, in the process state this evaluation left, and a failure it reproduces is not a
        # difference. One decision, `engine.original_control_standing`, shared with the in-process loop.
        fresh_failures = plugin.node_reasons[controlled:]
        controlled = len(plugin.node_reasons)
        sensitive: set[str] = set()
        if fresh_failures:
            newly_voided, sensitive = _voided_by_control(fresh_failures)
            voided |= newly_voided
            if sensitive:
                # These nodes failed under the mutant on COPIES of arguments their test cannot be
                # handed copies of (`identity_standing`): that failure says nothing. Forget it and
                # run them again on the authored objects.
                plugin.node_reasons = [
                    (n, r) for n, r in plugin.node_reasons if n not in sensitive
                ]
                controlled = len(plugin.node_reasons)
                plugin.done -= sensitive
                fresh_failures = [
                    (n, r) for n, r in fresh_failures if n not in sensitive
                ]
            plugin.settle(voided)
        # A round pytest stopped at a value failure that the control voided has not run the rest of the
        # suite against this mutant; go on with what it never reached, as the in-process loop does.
        stopped = any(
            reason in ("assertion", "exception") for _, reason in fresh_failures
        )
        value_standing = any(r in ("assertion", "exception") for r in plugin.reasons)
        if not sensitive and (
            not stopped
            or value_standing
            or not (plugin.collected - plugin.done)
            or plugin.memory_cut
        ):
            break
    rc = next((code for code in rcs if code != 0), 0)
    return {
        # W#27: a node whose loop budget stopped the mutant — the named non-kill `diverged`, read
        # by the engine BEFORE pytest's exit code (the stopped node reads as a failure there).
        "diverged": bool(plugin.diverged_nodes),
        "rc": int(rc),
        "killed_by": aggregate_kill_reason(plugin.reasons) or None,
        "constructed": plugin._mutated is not None,
        "test_name": plugin.first_failing_node,
        # Installation-and-entry proof (#18): the engine feeds these to `mutant_disposition` so an
        # installed-but-never-entered mutant scores `not_entered`, outside the denominator.
        "installed": plugin.installed,
        "ran": plugin.ran,
        "entered": bool(getattr(plugin._mutated, "entered", False)),
        # W#21: this mutant hit the worker's address-space cap — a budget CUT, not a kill.
        "memory_cut": plugin.memory_cut,
        # Detective #105: a node whose write was prevented, and what it was prevented from writing.
        "write_blocked": plugin.write_blocked,
        "blocked_writes": plugin.blocked_writes,
        # #55: the nodes whose failure the ORIGINAL reproduces, named only when nothing else failed
        # the mutant. A failure in a phase the call ladder does not see (a fixture's setup) is not
        # controlled and keeps pytest's verdict.
        "original_also_fails": (
            sorted(voided)
            if voided and not plugin.reasons and not plugin.other_failures
            else []
        ),
    }


def _pytest_round(node_ids: list[str], plugin: Any) -> int:
    """One ``pytest.main`` over ``node_ids`` with ``plugin``; its output goes to a sink.

    pytest's own output is redirected so, in server mode, it cannot land in the JSON-line protocol on
    stdout; ``--capture=sys`` keeps the same isolation at the Python level without touching descriptor
    1, the protocol channel.
    """
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return int(
            pytest.main(
                [
                    *node_ids,
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "--no-header",
                    "--capture=sys",
                ],
                plugins=[plugin, _CheapFailureRepr()],
            )
        )


def _voided_by_control(failures: list[tuple[str, str]]) -> tuple[set[str], set[str]]:
    """The failing nodes whose failure the ORIGINAL reproduces in this process state, and the ones that
    reproduce it only because they were handed copies (#55).

    Runs exactly those nodes once more with nothing installed (the mutant's evaluation restored every
    binding) and reads each pair of outcomes through ``engine.original_control_standing``. A node the
    original fails the same way is then run ONCE MORE on the authored objects, never twice in a
    process: if it passes there, ``identity_standing`` says the test depends on object identity, its
    arguments stay shared from now on and it is returned in the second set instead of being voided.
    A control that cannot run (a harness fault) voids nothing: absence of a control is not evidence of
    one.
    """
    from Wesker.engine import original_control_standing
    from Wesker.pytest_runner import identity_standing

    nodes = list(dict.fromkeys(nodeid for nodeid, _ in failures))
    control = _ControlPlugin()
    try:
        _pytest_round(nodes, control)
    # BLE001: a control that cannot run is no control; the failures keep the verdict they had
    except Exception:  # noqa: BLE001
        return set(), set()
    same = {
        nodeid
        for nodeid, reason in failures
        if original_control_standing(reason, control.reasons.get(nodeid))
        == "control_same"
    }
    unmeasured = sorted(same - _MEASURED)
    sensitive: set[str] = set()
    if unmeasured:
        _MEASURED.update(unmeasured)
        on_authored = _AuthoredControlPlugin()
        try:
            _pytest_round(unmeasured, on_authored)
        # BLE001: a measurement that cannot run measures nothing; the arguments keep their standing
        except Exception:  # noqa: BLE001
            return same, set()
        for nodeid in unmeasured:
            if (
                identity_standing(nodeid not in on_authored.reasons, False)
                == "identity_sensitive"
            ):
                _ARGUMENTS.mark_identity_sensitive(nodeid)
                sensitive.add(nodeid)
    return same - sensitive, sensitive


def _apply_mem_limit(mem_limit_bytes: Any) -> bool:
    """Apply the worker's address-space cap (W#21), returning whether the OS accepted it.

    A positive ``mem_limit_bytes`` caps ``RLIMIT_AS`` so a runaway mutant fails as a catchable
    ``MemoryError`` instead of taking the box down; Linux cooperates, macOS/Windows frequently do
    not, and the returned bool is what `memory_guard.memory_enforcement_standing` turns into an
    honest enforced/telemetry-only capability.
    """
    if not mem_limit_bytes:
        return False
    from Wesker.memory_guard import apply_address_limit

    return apply_address_limit(int(mem_limit_bytes))


def _trace_baseline_run(target_abspath: str, node_ids: list[str]) -> dict[str, Any]:
    """Run node IDs against the UNMUTATED target under a line tracer, in THIS fresh process (#19).

    The determinism check runs this twice in two fresh workers and compares. Here, from one fresh
    process, it captures the set of TARGET lines the covering tests execute and the run's outcome.
    Coordinated via sys.monitoring (W#32).
    """
    from Wesker.isolation import isolated_test_outcome
    from Wesker.monitoring import allocate_run_token, retire_run_token, subscribe

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

    token = allocate_run_token()

    def _on_line(
        code: Any,
        lineno: int,
        frame: Any = None,
        run_token: int | None = None,
    ) -> None:
        if _is_target(code.co_filename):
            covered.add(lineno)

    handle = subscribe(
        [],
        sys.monitoring.events.LINE,
        _on_line,
        run_token=token,
        selectors=[target_abspath],
    )

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        try:
            rc = pytest.main(
                [
                    *node_ids,
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "--no-header",
                    "--capture=sys",
                ],
                # The outcome is read off the exit code alone (W#34): no report text is needed here.
                plugins=[_CheapFailureRepr()],
            )
        finally:
            retire_run_token(token)
            handle.close()
    handle.check()
    return {"lines": sorted(covered), "outcome": isolated_test_outcome(int(rc), False)}


# The project tree this worker measures, recorded at setup (#105): the write authority protects it
# even after a fixture has changed directory, because the window arms only at the call phase.
_PROJECT_TREE = ""


def _setup(spec: dict[str, Any]) -> None:
    global _PROJECT_TREE
    # Before collection, so a temporary directory a test module or fixture creates is known (#105).
    install()
    root = spec["project_root"]
    _PROJECT_TREE = os.path.realpath(root)
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
    # Apply the address-space cap ONCE per worker, at session start, before any mutant runs (W#21).
    # `enforced` rides on every response so the parent reports an honest enforced/telemetry capability.
    enforced = _apply_mem_limit(session.get("mem_limit_bytes"))
    target = _resolve_target(session["project_root"], session["target_file"])
    qualname = session["func_qualname"]
    session_nodes = session.get("node_ids", [])
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        spec = json.loads(line)
        nodes = spec.get("node_ids") or session_nodes
        payload = _evaluate_full(
            target,
            qualname,
            nodes,
            spec["mutant_source"],
            step_caps=spec.get("step_caps"),
            project_root=session["project_root"],
            write_roots=tuple(session.get("write_roots") or ()),
        )
        payload["enforced"] = enforced
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
    # Past `mutant_source` every argument is named: W#27's budget and #105's write roots arrived as
    # separate optional groups, and a positional fifth argument bound write roots to `step_caps`.
    payload = _evaluate_full(
        target,
        spec["func_qualname"],
        spec["node_ids"],
        spec["mutant_source"],
        step_caps=spec.get("step_caps"),
        project_root=spec["project_root"],
        write_roots=tuple(spec.get("write_roots") or ()),
    )
    if payload["original_also_fails"] and not payload["killed_by"]:
        # The original reproduces every failure (#55): not a kill either. Same exit as a prevented write.
        return int(pytest.ExitCode.INTERNAL_ERROR)
    if payload["write_blocked"] and not payload["killed_by"]:
        # A prevented write is never a kill (#105). The exit code has no refusal to carry, so the
        # run reads as a harness outcome, outside the denominator, rather than the failure pytest saw.
        return int(pytest.ExitCode.INTERNAL_ERROR)
    return payload["rc"]


if __name__ == "__main__":
    sys.exit(main())
