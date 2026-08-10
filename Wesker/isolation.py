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

import json
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass


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
