"""Isolated worker execution — the gateable execution mode (#19).

In-process execution shares `sys.modules`, cwd, environment, patched attributes, plugin and
fixture lifecycle, and any singleton a test touches. It is CONDITIONALLY gateable: a hermetic test
measures honestly, but a test that leaves state behind, spawns a thread the interpreter cannot
join, or blocks in a subprocess/socket/C-extension cannot be contained in-process — a Python
thread cannot be killed, only asked to stop, and `interrupt.abandon` reports honestly when the ask
fails.

A separate PROCESS can be killed. This module runs the exact pytest node IDs in a child process
placed in its own process GROUP (`start_new_session`), so a timeout terminates the whole group —
the worker AND any child it spawned — with an uncatchable ``SIGKILL`` that a thread abandon
structurally cannot reach. Containment is therefore a REAL guarantee here, not a best-effort ask:
the group is reaped, or the run reports uncontained and the measurement is cut.

This is the parent-side primitive: spawn, feed a budget, terminate the group, classify the exit.
The mutant-installation and node-ID lifecycle that run INSIDE the worker build on it.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def isolated_test_outcome(returncode: int, timed_out: bool) -> str:
    """Map an isolated pytest run's exit to a typed outcome (#19, pure — pinned).

    pytest's exit code is the authoritative classifier, never a substring of its output (the same
    discipline `Detective.certify.pytest_status` keeps for the in-process verifier): parsing text
    is how "collected and failed" came to read as "could not collect". A ``timed_out`` run has no
    trustworthy code — the worker was killed mid-flight — so it is named ``timeout`` before the
    code is even consulted.

    pytest's codes: 0 all passed, 1 tests failed, 2 collection/usage error, 3 internal error, 4
    usage error, 5 no tests collected. Anything not 0/1/5 is ``error`` — an unknown non-green exit
    must never read as a pass, so the default is not-green.
    """
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "passed"
    if returncode == 1:
        return "failed"
    if returncode == 5:
        return "no_tests"
    return "error"


@dataclass(frozen=True)
class IsolatedRun:
    """The outcome of one isolated worker execution (#19)."""

    returncode: int
    timed_out: bool
    #: True only when the worker (and its process group) is CONFIRMED gone. A timeout that could
    #: not reap the group — a process wedged in an uninterruptible syscall — is ``False``, and a
    #: consumer must treat the measurement as uncontained/cut, exactly as the in-process path does.
    contained: bool
    stdout: str
    #: The kill vocabulary the worker classified for this mutant (assertion/exception/crash), or
    #: None. Set only on the server path, where a mutant's failure reasons cross back as data; the
    #: one-shot and timeout paths leave it None (a timeout's reason is named by `mutant_verdict`).
    killed_by: str | None = None
    #: False when the mutant would not compile — the worker installed nothing, so the outcome
    #: measures the harness, not the suite (#18). The engine scores that `harness_error`, outside
    #: the denominator, never a survivor; True (the default) preserves every existing construction.
    constructed: bool = True
    #: The first node that failed under the mutant, for the kill matrix; None when none did.
    test_name: str | None = None

    @property
    def outcome(self) -> str:
        return isolated_test_outcome(self.returncode, self.timed_out)


def _terminate_group(proc: subprocess.Popen[str]) -> bool:
    """SIGKILL the worker's whole process GROUP and confirm it is reaped. Returns contained.

    `os.killpg` on the group id reaches every process the worker started — the child a thread
    abandon leaves running. SIGKILL is uncatchable, so the group dies unless a process is wedged in
    an uninterruptible (D-state) syscall; the bounded `wait` distinguishes "reaped" (contained)
    from "could not confirm dead" (uncontained), rather than assuming the kill landed.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass  # already gone — the wait below confirms it
    try:
        proc.wait(timeout=5.0)
        return True
    except subprocess.TimeoutExpired:
        return False  # not confirmed dead — the honest answer is uncontained


def run_pytest_node_isolated(
    project_root: str,
    node_ids: Sequence[str],
    timeout_s: float,
    *,
    addopts_neutral: bool = False,
) -> IsolatedRun:
    """Run exact pytest ``node_ids`` in a killable child process group (#19).

    The child runs under the project's OWN pytest regime (its addopts), the same soundness the
    in-process verifier keeps (Detective #58): the certificate claims the project's real
    configuration, so verification must use it. ``addopts_neutral`` is an explicit opt-out for a
    caller that has already isolated the regime and only needs the node to run.

    On timeout the ENTIRE process group is terminated and reaped; ``contained`` says whether that
    was confirmed. The worker owns descriptor 1 for its own output — captured here, never streamed
    — so a caller speaking a protocol on stdout is untouched.
    """
    argv = [sys.executable, "-m", "pytest", *node_ids, "-p", "no:cacheprovider"]
    if addopts_neutral:
        argv += ["-o", "addopts="]
    proc = subprocess.Popen(
        argv,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # new session -> new process group -> killpg reaches children
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return IsolatedRun(proc.returncode, False, True, out or "")
    except subprocess.TimeoutExpired:
        contained = _terminate_group(proc)
        # Drain whatever the worker emitted before the kill, without blocking on a group that may
        # not be fully gone; the outcome is already `timeout`, this is only for diagnostics.
        try:
            out, _ = proc.communicate(timeout=1.0)
        except (subprocess.TimeoutExpired, ValueError):
            out = ""
        return IsolatedRun(-9, True, contained, out or "")


def mutant_verdict(outcome: str) -> str:
    """Map an isolated worker's pytest OUTCOME to a mutant verdict (#19, pure — pinned).

    The worker runs the covering node IDs against the mutant; pytest's exit is the verdict.

    * ``failed`` — a node FAILED under the mutant: the suite detected the mutation → ``killed``.
    * ``timeout`` — the mutant made a node hang past its budget: a run-only kill, detected by time
      → ``killed`` (the worker was terminated; containment travels separately on the run).
    * ``passed`` — every node passed under the mutant: nothing distinguished it → ``survived``.
    * ``no_tests`` / ``error`` — no node ran, or the run could not collect: this measures the
      HARNESS, not the suite, and belongs on NEITHER side of the denominator → ``harness``, the
      same discipline ``engine.mutant_disposition`` keeps for the in-process path so a collection
      failure never inflates a kill score.
    """
    if outcome in ("failed", "timeout"):
        return "killed"
    if outcome == "passed":
        return "survived"
    return "harness"


def classify_kill_reason(is_assertion: bool, is_declared_failure: bool) -> str:
    """Name WHY one isolated node failed, in the engine's kill vocabulary (#19, pure — pinned).

    The isolated worker runs real pytest, so a failure arrives as a report's ``excinfo`` rather
    than the caught exception ``engine._run_test_with_timeout`` sees in-process — but the RULE
    must be identical, or the isolated path's ``value_killed`` split silently disagrees with the
    in-process one. That ladder, from engine.py:

    * ``AssertionError`` → ``assertion`` — the test's assert pinned the return VALUE.
    * a pytest DECLARED failure (``pytest.raises`` violated / ``pytest.fail``; see
      ``engine._is_declared_failure``) → ``exception`` — a stated contract the mutant broke,
      the same strength as an assertion (both make ``value_killed`` count).
    * anything else that raised → ``crash`` — the mutant merely RAN differently; the value is
      not pinned, so it is a run-only kill (a value-survivor downstream).

    Assertion outranks a declared failure when both describe the same node, matching the
    in-process ``except AssertionError`` arm winning over the ``BaseException`` arm. The impure
    boundary supplies the two booleans off ``excinfo`` and NEVER calls this for a pass or a skip.
    """
    if is_assertion:
        return "assertion"
    if is_declared_failure:
        return "exception"
    return "crash"


def aggregate_kill_reason(reasons: list[str]) -> str:
    """Combine per-node kill reasons into ONE verdict for the mutant (#19, pure — pinned).

    The worker runs several nodes in one pytest invocation, so a mutant can be killed by more
    than one — one by assertion, another by crash. This is the same precedence
    ``evaluate_mutant``'s ``record_all_killers`` branch keeps (its inline twin, kept in step so a
    future refactor can share this): a VALUE PIN outranks a run-only kill, and among value pins
    ``assertion`` is named before ``exception``. Order-independent — a mutant ANY node kills by
    assertion is value-killed regardless of which node ran first. ``""`` when nothing killed
    (every node passed); the caller reads the kill itself from pytest's exit code, not from here.
    """
    if "assertion" in reasons:
        return "assertion"
    if "exception" in reasons:
        return "exception"
    if "crash" in reasons:
        return "crash"
    return ""


def run_mutant_isolated(
    project_root: str,
    node_ids: Sequence[str],
    target_file: str,
    func_qualname: str,
    mutant_source: str,
    timeout_s: float,
) -> IsolatedRun:
    """Evaluate ONE mutant against exact node IDs in a killable worker process (#19).

    The worker (`Wesker._isolated_worker`) installs the mutant through the REAL pytest lifecycle —
    the same `_patch_mutant_into_test` / `_unpatch_mutant` the in-process evaluator uses, from a
    plugin — and exits with pytest's code; `IsolatedRun.outcome` + :func:`mutant_verdict` name the
    result. A mutant that hangs is terminated with its whole process group, and ``contained`` says
    whether that was confirmed, so a runaway mutant cannot leave a live worker perturbing the next
    measurement — the containment the in-process thread path cannot guarantee.
    """
    payload = json.dumps(
        {
            "project_root": project_root,
            "node_ids": list(node_ids),
            "target_file": target_file,
            "func_qualname": func_qualname,
            "mutant_source": mutant_source,
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "Wesker._isolated_worker"],
        cwd=project_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(payload, timeout=timeout_s)
        return IsolatedRun(proc.returncode, False, True, out or "")
    except subprocess.TimeoutExpired:
        contained = _terminate_group(proc)
        try:
            out, _ = proc.communicate(timeout=1.0)
        except (subprocess.TimeoutExpired, ValueError):
            out = ""
        return IsolatedRun(-9, True, contained, out or "")


def should_recycle(evaluated: int, max_per_worker: int) -> bool:
    """Whether a persistent isolated worker has done enough mutants to recycle (#19, pure — pinned).

    A worker reused across mutants can accumulate application state a per-test lifecycle does not
    reset — a singleton, a registry, a module-level cache. Recycling to a fresh process after a
    bounded count discards that drift before it can perturb a verdict; ``max_per_worker <= 0`` means
    never recycle on count (a caller relying on other drift detection). The count is inclusive: at
    exactly the cap the worker is spent.
    """
    return max_per_worker > 0 and evaluated >= max_per_worker


def execution_mode_standing(execution_mode: str, measurement_gateable: bool) -> str:
    """The gateability TIER a profiling result earns from its execution mode (#19, pure — pinned).

    `measurement_gateable` is the result's existing measurement-level validity (`is_gateable`:
    contained, in-budget, unambiguous identity, exhaustive depth). The standing layers the mode on
    top of it, and it is INFORMATIONAL — it does not change `is_gateable`, so no in-process
    certificate is downgraded before the fast-mode shape check (increment 5) gives "conditional" its
    teeth.

    * ``cut`` — the measurement is not valid to gate on for ANY reason the existing conjunction
      already names (an uncontained worker, a cut budget, an ambiguous module identity). The mode
      cannot rescue an invalid measurement, so this is checked first.
    * ``gateable`` — measured under ``isolated``, where containment is a real SIGKILL guarantee: the
      counts may gate a downstream verdict outright.
    * ``conditional`` — measured under ``in_process``, where a runaway can only be ASKED to stop.
      The counts are valid, but the mode's containment is best-effort, so gating is conditional on
      the hermetic-shape check that increment 5 will require. Until then this is a label, not a gate.
    """
    if not measurement_gateable:
        return "cut"
    if execution_mode == "isolated":
        return "gateable"
    return "conditional"


def fast_mode_standing(
    spawns_subprocess: bool,
    starts_background_thread: bool,
    custom_collector: bool,
    signal_main_thread: bool,
    stateful_fixture: bool,
) -> str:
    """Whether the in_process FAST mode may be trusted for a test's SHAPE (#19, pure — pinned).

    in_process containment is a thread abandon — it ASKS a runaway to stop and cannot force it. So
    the fast mode is sound only for HERMETIC shapes; a test that escapes the interpreter or the
    per-test lifecycle must be REFUSED to the isolated mode rather than measured on a guarantee the
    mode cannot keep. Issue #19: "explicit warning/refusal for subprocess, background-thread, custom
    collector, signal/main-thread, or stateful fixture requirements ... never silently upgraded to
    the isolated guarantee." Each hazard is NAMED (a consumer reports which one refused), in the
    issue's own listing order; all-clear is ``hermetic``.

    Over-refusal is the safe direction: a shape wrongly flagged hazardous is merely routed to the
    always-sound isolated mode, while a hazard wrongly cleared would measure on a false containment —
    the one error a proof-facing tool must not make.
    """
    if spawns_subprocess:
        return "refuse_subprocess"
    if starts_background_thread:
        return "refuse_thread"
    if custom_collector:
        return "refuse_collector"
    if signal_main_thread:
        return "refuse_signal"
    if stateful_fixture:
        return "refuse_fixture"
    return "hermetic"


def baseline_determinism(
    coverage_a: list[int],
    outcome_a: str,
    coverage_b: list[int],
    outcome_b: str,
) -> str:
    """Whether two fresh-state baseline runs agree — the proof-facing nondeterminism check (#19,
    pure — pinned).

    A gateable measurement must be REPEATABLE: run the unmutated baseline twice from matched fresh
    state, and if the pass/fail outcome or the covered lines differ, the function is nondeterministic
    and no mutant verdict measured against it can be trusted. Issue #19: "execute baseline more than
    once from matched fresh state; classify inconsistent outcomes/coverage as nondeterministic."

    Outcome is checked before coverage because a flipped pass/fail is the louder signal, but either
    disagreement is decisive. Coverage is compared as a SET — order and repeats from the tracer are
    not signal, only WHICH lines ran.
    """
    if outcome_a != outcome_b:
        return "nondeterministic"
    if set(coverage_a) != set(coverage_b):
        return "nondeterministic"
    return "deterministic"


class IsolatedMutantWorker:
    """A PERSISTENT isolated worker evaluating many mutants in one interpreter (#19).

    The one-shot `run_mutant_isolated` pays pytest startup per mutant; this reuses one worker so a
    whole survivor set is measured for the cost of one import, and the caller recycles it (via
    `should_recycle` / a hang) rather than spawn per mutant. Every mutant is still installed and
    torn down per test, so the mutant itself never leaks between evaluations; recycling bounds the
    application-state drift a reused process can accumulate.

    A single mutant that hangs terminates the whole process GROUP — the worker AND any child — and
    marks this worker dead, so the caller recycles a fresh one; the hung mutant's result is a
    contained timeout kill.
    """

    def __init__(
        self,
        project_root: str,
        node_ids: Sequence[str],
        target_file: str,
        func_qualname: str,
    ) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "Wesker._isolated_worker", "--serve"],
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        self._alive = True
        self._evaluated = 0
        session = json.dumps(
            {
                "project_root": project_root,
                "node_ids": list(node_ids),
                "target_file": target_file,
                "func_qualname": func_qualname,
            }
        )
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(session + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, AssertionError):
            self._alive = False

    @property
    def evaluated(self) -> int:
        return self._evaluated

    @property
    def alive(self) -> bool:
        return self._alive and self._proc.poll() is None

    def evaluate(
        self,
        mutant_source: str,
        timeout_s: float,
        node_ids: Sequence[str] | None = None,
    ) -> IsolatedRun:
        """Evaluate ONE mutant on the reused worker. A hang kills the group and retires the worker.

        ``node_ids`` overrides the session's set for THIS mutant — per-mutant test scoping, so the
        isolated verdict matches the in-process scoped one (only the tests reaching the mutated line
        run); None uses the session default the worker was opened with.
        """
        if not self.alive:
            return IsolatedRun(-9, True, self._reap(), "")
        spec: dict[str, Any] = {"mutant_source": mutant_source}
        if node_ids is not None:
            spec["node_ids"] = list(node_ids)
        try:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(spec) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, AssertionError):
            self._alive = False
            return IsolatedRun(-9, True, self._reap(), "")
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout_s)
        if not ready:
            # The mutant hung the worker. Kill the whole group and retire it — a fresh worker takes
            # the next mutant, and this one is a contained timeout kill.
            contained = self._reap()
            self._alive = False
            return IsolatedRun(-9, True, contained, "")
        line = self._proc.stdout.readline()
        if not line:  # the worker died mid-evaluation
            self._alive = False
            return IsolatedRun(-9, True, self._reap(), "")
        self._evaluated += 1
        try:
            data = json.loads(line)
            rc = int(data.get("rc", 2))
        except (ValueError, TypeError):
            data, rc = (
                {},
                2,
            )  # unreadable line -> a collection-class code, never a silent pass
        return IsolatedRun(
            rc,
            False,
            True,
            "",
            killed_by=data.get("killed_by"),
            constructed=bool(data.get("constructed", True)),
            test_name=data.get("test_name"),
        )

    def _reap(self) -> bool:
        return _terminate_group(self._proc)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        if self._proc.poll() is None:
            self._reap()
        self._alive = False
