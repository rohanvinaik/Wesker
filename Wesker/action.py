"""The GitHub Action entry point — specification completeness as a CI check.

WHY THIS EXISTS SEPARATELY FROM ``cli.py``: the CLI is a tool a human drives and reads. A CI
check publishes a number to strangers, and that changes what "working" means. A human who sees
a suspicious 0% investigates; a badge that says 0% is simply believed. So this module's job is
mostly REFUSAL — it holds three gates, and each one turns a state that would produce a
plausible-but-wrong number into a loud failure with a one-line diagnosis.

    no live pytest session   -> the legacy runner cannot execute fixture-taking tests, so its
                                number is not a mutation score. Fail; do not quietly downgrade.
    the suite does not pass  -> a test that fails on UNMUTATED code cannot testify about any
                                mutant. If that is most of the suite, every mutant "survives"
                                and the run reports a confident 0% about a broken environment.
    the budget truncated     -> unevaluated mutants are missing from BOTH sides of the ratio,
                                so the result is a sample, not a measurement of completeness.

None of these is new information: the engine already computed every one of them. They were
simply never wired to anything that could act on them.

The reading is SPECIFICATION COMPLETENESS (``spec_pct``): of the behavioral dimensions the code
has, how many do the tests pin? Its denominator is derived from the AST, so it means the same
thing in every repo at every budget — which is the entire reason it is worth badging, and the
reason a kill rate is not. ``dof_pct`` is reported too, but it describes the ENGINE (did greedy
selection reach every dimension), and under the DOF budget it is ~always 100%. It is a proof
that the selection is optimal, not a claim about anyone's tests.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO

_DETECTIVE_PKG = "detective-spec"


# ── Diff scoping ─────────────────────────────────────────────────

# A git revision, per git-check-ref-format plus the range/parent operators a caller may pass:
# SHAs, `main`, `HEAD~1`, `v1.2^2`, `origin/main`. Notably it cannot begin with '-', so a value
# can never arrive at git's argv as an option instead of a revision.
_SAFE_REV = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._/~^-]*")

# A SARIF output path: a relative path of ordinary segments, ending in .sarif. The per-segment
# lookahead is what does the work — a plain `[A-Za-z0-9._-]+` class contains '.', so it happily
# matches '..' and `../../../etc/x.sarif` sails through. No leading '/' and no '..' segment
# means the path cannot leave the directory it is resolved against.
_SAFE_OUT_PATH = re.compile(r"(?:(?!\.\./)[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.sarif")


def _git(args: list[str], cwd: str) -> str | None:
    """Run git, returning stdout, or None if git is unavailable/failed.

    A missing base ref is ordinary (shallow clone, first commit on a new branch), so it must
    degrade to "cannot scope" rather than crash the check.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def changed_files(base_ref: str, project_root: str = ".") -> list[str] | None:
    """Python files that differ from ``base_ref``, relative to the repo root.

    Returns None when the diff cannot be computed at all — distinct from an empty list, which
    honestly means "this PR changed no Python". The caller must not silently treat the first
    as the second: "I could not tell what changed" and "nothing changed" imply opposite actions.
    """
    # ALLOWLIST, not a rejection test. argv does not distinguish a ref from a flag, so a
    # `base_ref` of `--upload-pack=...` reaches git as an OPTION rather than a revision. There
    # is no shell here (subprocess takes a list), so this is argument injection rather than
    # command injection — but the value is rebuilt from a matched pattern rather than merely
    # inspected, so nothing unvalidated reaches the command either way.
    #
    # The pattern is what git itself permits in a revision (git-check-ref-format), plus the
    # range/parent operators a caller may legitimately pass: SHAs, `main`, `HEAD~1`, `v1.2^2`,
    # `origin/main`. A ref may not begin with '-', and git forbids it too.
    match = _SAFE_REV.fullmatch(base_ref)
    if match is None:
        return None
    safe_ref = match.group(0)
    out = _git(["diff", "--name-only", f"{safe_ref}...HEAD"], project_root)
    if out is None:
        out = _git(["diff", "--name-only", safe_ref], project_root)
    if out is None:
        return None
    return [
        line.strip()
        for line in out.splitlines()
        if line.strip().endswith(".py") and Path(project_root, line.strip()).is_file()
    ]


# ── Gates ────────────────────────────────────────────────────────


def gate_execution_mode(report: dict) -> str | None:
    """Refuse anything but a pytest-judged run. Returns a failure reason, or None to proceed."""
    mode = report.get("execution_mode", "unknown")
    if mode == "pytest-live":
        return None
    return (
        f"execution mode is '{mode}', not 'pytest-live'.\n"
        "Wesker requires a working pytest session: mutants must be judged by pytest itself, "
        "with real fixtures and conftest, or the number is not a mutation score.\n"
        "Check that pytest is installed and that `pytest --collect-only` succeeds here."
    )


def gate_suite_health(report: dict, max_inert_pct: int = 50) -> str | None:
    """Refuse to measure a codebase whose own suite does not pass.

    A test failing on unmutated code is barred from kill attribution — correctly. But when that
    is true of the whole suite (an unmet dependency is the usual cause in CI), every mutant
    survives and the run reports a confident 0% specified. That is a statement about a broken
    environment wearing the costume of a statement about the code, and it is the single most
    likely way a first-time user gets a wrong number and rejects the tool.
    """
    suite = report.get("suite")
    if not suite:
        return None
    tests, inert = suite.get("tests", 0), suite.get("inert", 0)
    if tests == 0:
        return (
            "the live session collected 0 tests.\n"
            "There is nothing to measure specification against."
        )
    inert_pct = round(100 * inert / tests)
    if inert_pct > max_inert_pct:
        failing = suite.get("failing_on_baseline") or []
        named = (
            f"\nTests failing on unmutated code: {', '.join(failing[:5])}"
            if failing
            else ""
        )
        return (
            f"{inert}/{tests} tests ({inert_pct}%) fail against UNMUTATED code, so they cannot "
            "testify about any mutant.\n"
            "This is a broken environment, not a specification result — most often an "
            "unmet dependency. Every mutant would 'survive' and the score would read 0%."
            f"{named}"
        )
    return None


def gate_truncation(report: dict) -> str | None:
    """Refuse to call a CUT measurement a completeness measurement — cut by the budget, by a worker
    that could not be stopped, or by the memory cap (Wesker #13/#14/W#21).

    The refusal names each cut function under the remedy its cause needs. A count alone told the
    maintainer that something was cut and not what: the remedies differ, and for a worker that
    could not be stopped the budget the old message pointed at is the wrong fix. A report that
    predates the per-function record falls back to naming both remedies.
    """
    truncated = report.get("total_truncated", 0)
    if not truncated:
        return None
    from Wesker.ci import describe_truncation

    detail = describe_truncation(report.get("truncated_functions") or [])
    return (
        f"{truncated} function(s) were only PARTIALLY evaluated.\n"
        "Their unevaluated mutants are missing from both sides of the ratio, so this is a sample, "
        "not a completeness measurement.\n"
        + (
            detail
            or "Raise `budget` (or set [tool.wesker] max_per_category); an uncontained worker "
            "needs the target's blocking call bounded or run in a killable process."
        )
    )


def gate_unresolved(report: dict) -> str | None:
    """Refuse to call a run with no-verdict mutants a completeness measurement (#27).

    The sibling of `gate_truncation` for a run that was NOT cut: every mutant was evaluated, but
    some reached no verdict (a wall-clock bound fired, a deterministic step budget was exceeded, or
    a test's write outside its write authority was prevented; `ci.UNSCORED_ROLLUP` holds the list).
    Those questions are missing from both sides of the ratio, exactly as a truncated run's are, so
    the badge refuses — under their own remedy, never the "PARTIALLY evaluated" one, which is false
    of a run that measured everything. ``--allow-truncation`` covers it as it did when these mutants
    were reported as cuts.
    """
    unresolved = report.get("total_unresolved", 0)
    if not unresolved:
        return None
    from Wesker.ci import describe_unresolved

    detail = describe_unresolved(report.get("unresolved_functions") or [])
    return (
        f"{unresolved} function(s) were evaluated in full, but some mutants reached no verdict.\n"
        "Those mutants are missing from both sides of the ratio, so this is not a completeness "
        "measurement.\n" + detail
    )


def _completeness_refusal(report: dict, allow_sample: bool) -> str | None:
    """The refusal a cut or unresolved run earns, or None — under ``allow_sample``, warnings instead."""
    for refusal in (gate_truncation(report), gate_unresolved(report)):
        if not refusal:
            continue
        if not allow_sample:
            return refusal
        print(f"\nWARNING: {refusal}", file=sys.stderr)
    return None


# ── Reporting ────────────────────────────────────────────────────


def _summary_markdown(report: dict, scope_note: str) -> str:
    """The step summary — where the argument lives.

    The badge carries one number because a badge has room for one. This is the backend line
    item: it can afford to say WHY the number means something, and saying it here rather than
    on the front reads as conviction rather than salesmanship.
    """
    spec = report.get("spec_pct", 0)
    pinned = report.get("total_dof_pinned", 0)
    dof = report.get("total_dof", 0)
    mutants = report.get("total_mutants", 0)
    per_dim = round(mutants / dof, 2) if dof else 0
    survivors = report.get("survivors") or []

    lines = [
        "## Specification completeness",
        "",
        f"### {spec}%  —  {pinned} of {dof} behavioral dimensions pinned by tests",
        "",
        scope_note,
        "",
        "| | |",
        "|---|---|",
        f"| Dimensions in the code | **{dof}** |",
        f"| Pinned by a test | **{pinned}** |",
        f"| Unspecified | **{dof - pinned}** |",
        f"| Mutants run | {mutants} ({per_dim} per dimension) |",
        f"| Judged by | `{report.get('execution_mode', 'unknown')}` |",
        "",
        "<details><summary>Why this number is comparable and a kill rate is not</summary>",
        "",
        (
            "A mutation score's denominator is however many mutants a run happened to sample — "
            "change the budget and the number changes, so it cannot be compared across repos, "
            "configs, or tools."
        ),
        "",
        (
            f"This denominator ({dof}) is derived from the AST. It is a property of the code: the "
            "same on any machine, at any budget, today and tomorrow. That is what makes "
            f"{spec}% mean something to someone who has never seen this repo."
        ),
        "",
        (
            f"Selection reached {report.get('dof_pct', 0)}% of those dimensions using "
            f"{mutants} mutants — {per_dim} per dimension. Because every mutant's cover set is a "
            "singleton, greedy selection here is not merely within the usual `1 - 1/e` bound for "
            "submodular covers — it is *exactly* optimal at `min(m, D)`. That is machine-checked "
            "in Lean (`coverage_submodular`, `marginal_antitone`, `greedy_coverage_bound`), so the "
            "cost of this measurement is provably minimal, not just empirically low."
        ),
        "</details>",
        "",
    ]

    if survivors:
        lines += [
            f"### {len(survivors)} unspecified "
            + ("dimension" if len(survivors) == 1 else "dimensions"),
            "",
            (
                "Each line below is a *proof obligation*, not an opinion: Wesker changed the code "
                "and every test still passed. Write a test that fails against the change and it is "
                "discharged — verifiably, by re-running."
            ),
            "",
            "| Location | Dimension | No test distinguishes |",
            "|---|---|---|",
        ]
        for s in survivors[:25]:
            path = str(s.get("function_key", "")).split("::", 1)[0]
            line = s.get("mutated_line") or ""
            change = (s.get("change") or s.get("mutant") or "").replace("|", "\\|")
            lines.append(
                f"| `{path}:{line}` | `{s.get('dimension', '')}` | `{change}` |"
            )
        if len(survivors) > 25:
            # Never truncate silently: a capped list that does not say it is capped reads as
            # the complete set.
            lines.append(
                f"| … | | *{len(survivors) - 25} more — see the SARIF report* |"
            )
        lines += [
            "",
            "**Detective can write the tests that pin these:**",
            "",
            "```",
            f"pipx run {_DETECTIVE_PKG} converge <file>::<function>",
            "```",
            "",
            (
                "It synthesizes a complete, minimal suite for the function and Wesker re-runs to "
                "confirm the mutants die. No inference in the finding, none in the check."
            ),
            "",
        ]
    else:
        lines += [
            "### Every behavioral dimension is pinned",
            "",
            "No mutant survived. Nothing in this scope can be changed without a test noticing.",
            "",
        ]
    return "\n".join(lines)


def _write_github_file(env_var: str, content: str) -> None:
    path = os.environ.get(env_var)
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(content + "\n")


def _write_outputs(report: dict) -> None:
    """GITHUB_OUTPUT for downstream steps (badges, gates, custom reporting)."""
    outputs = {
        "spec-pct": report.get("spec_pct", 0),
        "dimensions-total": report.get("total_dof", 0),
        "dimensions-pinned": report.get("total_dof_pinned", 0),
        "dimensions-unspecified": report.get("total_dof", 0)
        - report.get("total_dof_pinned", 0),
        "dof-pct": report.get("dof_pct", 0),
        "kill-pct": report.get("kill_pct", 0),
        "total-mutants": report.get("total_mutants", 0),
        "execution-mode": report.get("execution_mode", "unknown"),
        "survivors": len(report.get("survivors") or []),
    }
    for key, value in outputs.items():
        _write_github_file("GITHUB_OUTPUT", f"{key}={value}")


def _annotate(report: dict) -> None:
    """Workflow-command annotations: the survivors, inline on the diff.

    Emitted in addition to SARIF because annotations need no extra permissions and show up
    even where code scanning is unavailable (most private repos on free plans).
    """
    for s in (report.get("survivors") or [])[:50]:
        path = str(s.get("function_key", "")).split("::", 1)[0]
        line = s.get("mutated_line")
        change = s.get("change") or s.get("mutant") or ""
        dim = s.get("dimension", "")
        loc = f"file={path}" + (f",line={line}" if line else "")
        msg = f"Unspecified dimension `{dim}` — no test distinguishes: {change}"
        print(f"::warning {loc}::{msg}")


# ── Entry point ──────────────────────────────────────────────────


# The process-global marker the outermost Action sets for as long as it owns the streams (W#52): its pid.
# It is read with `stack_dump_owner`, the same pure decision the stall dump reads, but is a DIFFERENT
# variable: the dump's marker exists only when a dump is armed, and a nested Action must find the stream
# owner whether or not one is.
_STREAM_OWNER_ENV = "WESKER_STREAM_OWNER"


def _is_nested() -> bool:
    """Whether an Action in THIS process already owns the streams (reads the marker; sets nothing)."""
    return (
        stack_dump_owner(os.environ.get(_STREAM_OWNER_ENV, ""), os.getpid()) == "nested"
    )


def _stream_reentry() -> Callable[[], None]:
    """Snapshot the REAL stdout/stderr fds; return the hand that re-enters them.

    Self-profiling runs mutants OF the fd-capture machinery in-process
    (line_coverage, pytest_runner), and a mutant that breaks fd restoration
    leaves fd 1 a corpse for the rest of the PROCESS — the engine's unpatch
    restores the function, not the file descriptor. Measured (2026-08-05,
    the first whole-codebase run after DATAFLOW grew that universe): the
    badge run completed its measurement and then died at the first
    ``::warning`` annotation with EBADF — taking the report down with it.
    Same belt-and-braces as ``ci._body`` re-entering the streams around the
    baseline, one layer up: the MEASUREMENT may fight hostile mutants — the
    REPORT must not. Per-file progress lines during the run go through a private
    descriptor (`_open_progress`); the GitHub-protocol lines (annotations, outputs,
    summary, banner) are the contract and are written only after re-entry.

    A NESTED Action (W#52) — a test of the Action run by the measurement inside the measuring
    process — owns no streams: it snapshots nothing and re-enters nothing, so its output lands where
    the calling test's capture put it, and a mutant of this very code cannot touch the descriptors the
    measuring process reports through. The hand it returns does nothing.
    """
    if _is_nested():
        return lambda: None
    saved_out, saved_err = os.dup(1), os.dup(2)

    def reenter() -> None:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        # The fd is necessary but not sufficient: sys.stdout is a Python
        # object wrapping the old fd state and may itself be closed. Rebind
        # both onto the restored descriptors, line-buffered so workflow
        # commands land as they are emitted.
        sys.stdout = io.TextIOWrapper(
            io.FileIO(1, "w", closefd=False), line_buffering=True
        )
        sys.stderr = io.TextIOWrapper(
            io.FileIO(2, "w", closefd=False), line_buffering=True
        )
        # W#52: rebinding drops the previous wrappers, and one a mutant built with `closefd=True`
        # closes the descriptor it wraps when it is collected, which is the one just restored. Let
        # every dropped wrapper die, then restore again: the descriptors the report speaks through
        # are the ones in place AFTER the last finalizer has run.
        gc.collect()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)

    return reenter


def progress_gap_notice(dropped: int) -> str:
    """The line that says this run's log is missing progress lines, or "" when it is not (W#52, pure — pinned).

    A progress line the run could not write is dropped so that the measurement survives. Dropping it
    silently would leave a log that reads as complete: the reader of a silent stretch cannot tell "the
    run printed nothing" from "the run could not print". The report itself is unaffected, and the
    notice says that too, so a hole in the log is never read as a hole in the measurement.
    """
    if dropped <= 0:
        return ""
    return (
        f"::warning::{dropped} progress line(s) of this run could not be written: the code under "
        "measurement closed or redirected the run's output. The measurement and this report are "
        "unaffected; the per-file progress above is incomplete."
    )


def _open_progress() -> IO[str]:
    """A line-buffered text stream on a PRIVATE duplicate of the real stdout, made before profiling (W#52).

    The measuring process's per-file progress is written here, never through ``sys.stdout`` or fd 1:
    a nested Action, or a mutant of the stream code, can close or redirect those, and the run then
    dies on its next ``[3/24]`` line. The duplicate is not a name the code under measurement uses.
    """
    return os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8", errors="replace")


def _profile_live(
    profile_codebase_live: Callable[..., dict | None],
    targets: list[str],
    budget: float,
    config: dict,
) -> tuple[dict | None, str]:
    """Run the live profile between the stream re-entry and the stack dump; ``(report, refusal)``.

    ``refusal`` is non-empty only for a lost worker (W#29), which ends the run with its own name. Every
    other outcome is the report's, as before, including None for no live session.
    """
    lost_worker = worker_lost_types()
    nested = _is_nested()
    reenter_streams = _stream_reentry()
    cancel_stack_dump = _stack_dump_from_env()
    # An owner reports progress through its private duplicate and publishes the marker that makes every
    # Action inside the measurement nested (W#52); a nested one does neither.
    progress = None if nested else _open_progress()
    kwargs = {} if progress is None else {"progress": progress}
    if not nested:
        os.environ[_STREAM_OWNER_ENV] = str(os.getpid())
    try:
        return (
            profile_codebase_live(
                ".",
                targets,
                budget_ms_per_file=budget,
                max_per_category=config.get("max_per_category"),
                passes=config.get("convergence_passes", 1),
                verbose=True,
                **kwargs,
            ),
            "",
        )
    except lost_worker as lost:
        return None, worker_lost_reason(lost)
    finally:
        cancel_stack_dump()
        if not nested:
            os.environ.pop(_STREAM_OWNER_ENV, None)
        # Everything from here down is the REPORT — gates, SARIF, annotations,
        # outputs, summary, banner — and must not inherit a corpse fd from a
        # hostile mutant of the capture machinery.
        reenter_streams()
        if progress is not None:
            # closing the private progress stream must never cost the report that follows
            with contextlib.suppress(Exception):
                progress.close()


def worker_lost_types() -> tuple[type[BaseException], ...]:
    """`interrupt.WorkerLost` from every loaded copy of Wesker (W#29).

    A self-profiling run measures through the private copy (`self_profile`), and that copy's exception
    is not the public class, so a handler naming only one would let the other through as a traceback.
    """
    from Wesker.self_profile import PRIVATE_PREFIX

    found: list[type[BaseException]] = []
    for name in ("Wesker.interrupt", f"{PRIVATE_PREFIX}.interrupt"):
        cls = getattr(sys.modules.get(name), "WorkerLost", None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            found.append(cls)
    return tuple(found)


def worker_lost_reason(lost: BaseException) -> str:
    """The refusal a lost worker earns (W#29): named, never a traceback, and never a re-run in a loop."""
    return (
        f"{lost}.\nA measurement worker's thread ended before it signalled its start, so the test it was to "
        "run never ran. Something raised into its bootstrap, or this process can no longer create a "
        "thread (a resource limit the measured code lowered is one measured cause). Nothing measured "
        "after it can be trusted. If it recurs, report it with this log."
    )


def _fail(reason: str) -> int:
    print("\n::error::Wesker could not produce a trustworthy number", file=sys.stderr)
    print(f"\nWESKER REFUSED TO REPORT: {reason}\n", file=sys.stderr)
    return 1


def stack_dump_setting(raw: str) -> tuple[str, int]:
    """Whether the run arms a periodic all-thread stack dump, and how often (pure — pinned).

    A whole-codebase run that stalls prints nothing until the CI job's own limit kills it, and
    that leaves no evidence of WHERE it stalled: every push of 2026-10-24 sat silent for six hours
    after "Whole codebase — 22 file(s)." ``WESKER_STACK_DUMP_S`` asks for a dump of every thread's
    stack at a fixed interval instead. Three codes, because absent and malformed are different
    facts and a typo must not read as "off":

    * ``off``     — unset or blank: nothing is armed, the default for every caller.
    * ``armed``   — a positive whole number of seconds: dump every that many seconds.
    * ``invalid`` — anything else (zero, a sign, a fraction, text): nothing is armed, and the run
      says so.
    """
    text = raw.strip()
    if not text:
        return "off", 0
    if not text.isdecimal():
        return "invalid", 0
    seconds = int(text)
    if seconds <= 0:
        return "invalid", 0
    return "armed", seconds


def stack_dump_owner(marker: str, pid: int) -> str:
    """Whether THIS process already owns an armed stack dump (W#52, pure — pinned).

    Under self-measurement, a test of the Action calls `main` inside the measuring process. Each such
    call armed its own dump and, on the way out, unregistered faulthandler's SIGURG handler, which is
    process-global. The run's own clock kept signalling a process with no handler, SIGURG is ignored
    by default, and the dump went silent: spec-full 40237240506 dumped at 22:04 and never again, through
    two and a half hours of silence. ``marker`` is ``WESKER_STACK_DUMP_OWNER``, which the arming process
    sets to its own pid; every module copy in the process sees it, the measured one included.

      "nested"  the marker names this process: an outer Action owns the dump, so arm nothing and cancel
                nothing
      "owner"   anything else (unset, or a marker inherited from another process): arm, and own it
    """
    return "nested" if marker == str(pid) else "owner"


# The process-global marker an armed dump's owner sets (W#52): its pid.
_DUMP_OWNER_ENV = "WESKER_STACK_DUMP_OWNER"
# How often the owner re-asserts its SIGURG registration (W#52). A mutant of this very code, run by a test
# in the measuring process, can unregister it; the owner restores it within this many seconds.
_REASSERT_S = 30.0


# The tick the dump clock sends. SIGURG is ignored by default, so if the code under measurement
# ever replaces or removes the handler, a stray tick is dropped instead of ending the run.
_DUMP_SIGNAL = "SIGURG"

# The dump clock: a child that signals its parent every `every` seconds until it is terminated
# or its parent is gone. Standard library only, and it holds none of the parent's pipes.
_DUMP_CLOCK = (
    "import os, sys, time\n"
    "parent, every, signum = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])\n"
    "while True:\n"
    "    time.sleep(every)\n"
    "    if os.getppid() != parent:\n"
    "        break\n"
    "    try:\n"
    "        os.kill(parent, signum)\n"
    "    except OSError:\n"
    "        break\n"
)


def _arm_stack_dump(seconds: int) -> Callable[[], None]:
    """Arm a periodic all-thread stack dump on a PRIVATE copy of stderr; return its cancel.

    The copy is the point. The live pytest session captures at the descriptor level, and
    self-profiling runs mutants of that capture machinery, so fd 2 can point at a capture file —
    or be closed — for long stretches of the run (see ``_stream_reentry``). A dump written to fd 2
    there vanishes with the capture. A descriptor duplicated before profiling starts still reaches
    the CI log. No exit: the job's own time limit ends the run, and successive dumps show whether
    a stall moves. A diagnostic must never break the run it observes, so a failure to arm
    degrades to a cancel that does nothing.

    THE CLOCK IS ITS OWN PROCESS (W#29). The first version armed
    ``faulthandler.dump_traceback_later``, which is ONE process-wide timer. pytest's own
    faulthandler plugin cancels that timer in ``pytest_exception_interact``, unconditionally, on
    every failing test (and on entering pdb). A self-measurement fails tests within seconds, so
    the dump armed before profiling was gone long before the stall. The 636 s local reproduction
    printed nothing, and sampling it showed no watchdog thread at all. Measured directly: a 1 s
    timer stops the moment pytest.main runs one failing test, and keeps going across a passing
    one. Now faulthandler dumps from a SIGURG handler (``faulthandler.register``, C-level, so it
    fires while every Python thread is deadlocked), and a separate child process sends the signal
    every ``seconds``. Nothing pytest or the measured code does to faulthandler's timer reaches it.
    """
    import faulthandler
    import signal

    signum = getattr(signal, _DUMP_SIGNAL, None)
    if signum is None or not hasattr(faulthandler, "register"):
        print(
            "WARNING: WESKER_STACK_DUMP_S needs POSIX signals; no stack dump is armed",
            file=sys.stderr,
        )
        return lambda: None
    try:
        stream = os.fdopen(os.dup(sys.stderr.fileno()), "w")
    # BLE001: a diagnostic degrades silently; it never fails the run it observes
    except Exception:  # noqa: BLE001
        return lambda: None
    try:
        faulthandler.register(signum, file=stream, all_threads=True, chain=False)
    # BLE001: a diagnostic degrades silently; it never fails the run it observes
    except Exception:  # noqa: BLE001
        stream.close()
        return lambda: None
    try:
        clock = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _DUMP_CLOCK,
                str(os.getpid()),
                str(seconds),
                str(signum),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # BLE001: a diagnostic degrades silently; it never fails the run it observes
    except Exception:  # noqa: BLE001
        faulthandler.unregister(signum)
        stream.close()
        return lambda: None

    stopped = threading.Event()

    def reassert() -> None:
        # W#52: a mutant of this code, run by a test in this process, can unregister the handler.
        while not stopped.wait(_REASSERT_S):
            try:
                faulthandler.register(
                    signum, file=stream, all_threads=True, chain=False
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
            ):  # what register raises: the diagnostic stops
                return

    threading.Thread(
        target=reassert, name="wesker-stack-dump-reassert", daemon=True
    ).start()

    def cancel() -> None:
        stopped.set()
        clock.terminate()
        try:
            clock.wait(timeout=5)
        except subprocess.TimeoutExpired:
            clock.kill()
            clock.wait()
        faulthandler.unregister(signum)
        stream.close()

    return cancel


def _stack_dump_from_env() -> Callable[[], None]:
    """Arm the dump ``WESKER_STACK_DUMP_S`` asks for and return its cancel (a no-op when off).

    Once per process (W#52): a nested Action, a test calling `main` in the measuring process, finds the
    owner's marker (`stack_dump_owner`) and neither arms nor cancels the process's dump."""
    code, seconds = stack_dump_setting(os.environ.get("WESKER_STACK_DUMP_S", ""))
    if code == "invalid":
        print(
            "WARNING: WESKER_STACK_DUMP_S must be a positive whole number of seconds; "
            "no stack dump is armed",
            file=sys.stderr,
        )
    if (
        code != "armed"
        or stack_dump_owner(os.environ.get(_DUMP_OWNER_ENV, ""), os.getpid())
        == "nested"
    ):
        return lambda: None
    cancel = _arm_stack_dump(seconds)
    os.environ[_DUMP_OWNER_ENV] = str(os.getpid())

    def release() -> None:
        cancel()
        os.environ.pop(_DUMP_OWNER_ENV, None)

    return release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m Wesker.action",
        description="Specification completeness as a CI check.",
    )
    parser.add_argument(
        "--base-ref",
        default="",
        help="Scope to files changed since this ref (the PR check). Omit for the whole codebase.",
    )
    parser.add_argument(
        "--targets", nargs="*", default=[], help="Explicit targets (overrides config)"
    )
    parser.add_argument("--sarif", default="", help="Write a SARIF 2.1.0 report here")
    parser.add_argument(
        "--threshold", type=int, default=0, help="Fail if spec %% is below this"
    )
    parser.add_argument(
        "--budget", type=float, default=15000, help="Budget per function, in ms"
    )
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Report a budget-limited sample instead of failing (the number is then not a "
        "completeness measurement)",
    )
    args = parser.parse_args(argv)

    from Wesker.metrics import _discover_targets, _load_config
    from Wesker.sarif import to_sarif
    from Wesker.self_profile import profiler_for_targets

    config = _load_config()
    targets = list(args.targets) or _discover_targets(config)
    if not targets:
        return _fail(
            "no source files found.\n"
            "Set [tool.wesker] source_dir in pyproject.toml, or pass targets explicitly."
        )

    scope_note = f"Whole codebase — {len(targets)} file(s)."
    if args.base_ref:
        changed = changed_files(args.base_ref)
        if changed is None:
            return _fail(
                f"could not diff against '{args.base_ref}'.\n"
                "For pull_request workflows set `fetch-depth: 0` on actions/checkout."
            )
        targets = [t for t in targets if t in set(changed)]
        if not targets:
            print("No Python source files changed — nothing to specify.")
            _write_github_file(
                "GITHUB_STEP_SUMMARY",
                "## Specification completeness\n\nNo Python source files changed in this pull "
                "request.",
            )
            return 0
        scope_note = f"Changed in this pull request — {len(targets)} file(s)."

    print(f"Wesker — {scope_note}")
    # Wesker profiling Wesker runs from a private copy of the package, so the modules being
    # mutated are not the ones executing the run. Any other project resolves to the public
    # engine here, exactly as before.
    report, lost = _profile_live(
        profiler_for_targets(targets), targets, args.budget, config
    )
    if lost:
        return _fail(lost)
    if report is None:
        return _fail(
            gate_execution_mode({"execution_mode": "legacy-direct-call"}) or ""
        )
    report["execution_mode"] = "pytest-live"

    for gate in (gate_execution_mode(report), gate_suite_health(report)):
        if gate:
            return _fail(gate)
    refusal = _completeness_refusal(report, args.allow_truncation)
    if refusal:
        return _fail(refusal)

    Path(".wesker").mkdir(exist_ok=True)
    Path(".wesker/mutation_report.json").write_text(json.dumps(report, indent=2))
    if args.sarif:
        # ALLOWLIST. `--sarif` names an output file, and an output file that can be
        # `../../../anywhere` is a path-traversal write with `parents=True` behind it. This
        # runs on a CI runner with a checkout and a token, so "the caller chose the path" is
        # not a reason to skip the check. The path is rebuilt from the matched pattern, so the
        # value reaching the filesystem is one this function constructed, not one it was given.
        sarif_match = _SAFE_OUT_PATH.fullmatch(args.sarif)
        if sarif_match is None:
            return _fail(
                f"--sarif must be a relative .sarif path inside the workspace: {args.sarif}"
            )
        sarif_path = Path(sarif_match.group(0))
        sarif_path.parent.mkdir(parents=True, exist_ok=True)
        sarif_path.write_text(
            json.dumps(to_sarif(report, version=_version()), indent=2)
        )
        print(f"SARIF: {sarif_path}")

    _annotate(report)
    gap = progress_gap_notice(report.get("progress_lines_dropped", 0))
    if gap:
        print(gap)
    _write_outputs(report)
    _write_github_file("GITHUB_STEP_SUMMARY", _summary_markdown(report, scope_note))

    spec = report.get("spec_pct", 0)
    print(
        f"\n{'=' * 60}\n"
        f"Specification completeness: {spec}% "
        f"({report.get('total_dof_pinned', 0)}/{report.get('total_dof', 0)} dimensions)\n"
        f"{'=' * 60}"
    )
    if args.threshold and spec < args.threshold:
        print(
            f"\n::error::Specification completeness {spec}% is below the "
            f"{args.threshold}% threshold",
            file=sys.stderr,
        )
        return 1
    return 0


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("Wesker")
    # BLE001: the version is a label on the report, and describing the run must not
    # fail the run
    except Exception:  # noqa: BLE001
        return ""


if __name__ == "__main__":
    sys.exit(main())
