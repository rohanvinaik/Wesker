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
import json
import os
import subprocess
import sys
from pathlib import Path

_DETECTIVE_PKG = "detective-spec"


# ── Diff scoping ─────────────────────────────────────────────────


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
    # A ref may not begin with '-': git forbids it, and argv does not distinguish a ref from a
    # flag. Without this, `--base-ref=--upload-pack=...` reaches git as an OPTION rather than a
    # revision. There is no shell here (subprocess takes a list), so this is argument
    # injection, not command injection — but the fix is the same and it costs one comparison.
    if base_ref.startswith("-"):
        return None
    out = _git(["diff", "--name-only", f"{base_ref}...HEAD"], project_root)
    if out is None:
        out = _git(["diff", "--name-only", base_ref], project_root)
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
    """Refuse to call a budget-limited sample a completeness measurement."""
    truncated = report.get("total_truncated", 0)
    if not truncated:
        return None
    return (
        f"{truncated} function(s) hit the per-file budget and were only partially evaluated.\n"
        "Their unevaluated mutants are missing from both sides of the ratio, so this is a "
        "sample, not a completeness measurement.\n"
        "Raise `budget` (or set [tool.wesker] max_per_category) before quoting the number."
    )


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
        "A mutation score's denominator is however many mutants a run happened to sample — "
        "change the budget and the number changes, so it cannot be compared across repos, "
        "configs, or tools.",
        "",
        f"This denominator ({dof}) is derived from the AST. It is a property of the code: the "
        "same on any machine, at any budget, today and tomorrow. That is what makes "
        f"{spec}% mean something to someone who has never seen this repo.",
        "",
        f"Selection reached {report.get('dof_pct', 0)}% of those dimensions using "
        f"{mutants} mutants — {per_dim} per dimension. Because every mutant's cover set is a "
        "singleton, greedy selection here is not merely within the usual `1 - 1/e` bound for "
        "submodular covers — it is *exactly* optimal at `min(m, D)`. That is machine-checked "
        "in Lean (`coverage_submodular`, `marginal_antitone`, `greedy_coverage_bound`), so the "
        "cost of this measurement is provably minimal, not just empirically low.",
        "</details>",
        "",
    ]

    if survivors:
        lines += [
            f"### {len(survivors)} unspecified "
            + ("dimension" if len(survivors) == 1 else "dimensions"),
            "",
            "Each line below is a *proof obligation*, not an opinion: Wesker changed the code "
            "and every test still passed. Write a test that fails against the change and it is "
            "discharged — verifiably, by re-running.",
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
            "It synthesizes a complete, minimal suite for the function and Wesker re-runs to "
            "confirm the mutants die. No inference in the finding, none in the check.",
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


def _fail(reason: str) -> int:
    print("\n::error::Wesker could not produce a trustworthy number", file=sys.stderr)
    print(f"\nWESKER REFUSED TO REPORT: {reason}\n", file=sys.stderr)
    return 1


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
        "--budget", type=float, default=15000, help="Per-file budget in ms"
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
    profile_codebase_live = profiler_for_targets(targets)
    report = profile_codebase_live(
        ".",
        targets,
        budget_ms_per_file=args.budget,
        max_per_category=config.get("max_per_category"),
        passes=config.get("convergence_passes", 1),
        verbose=True,
    )
    if report is None:
        return _fail(
            gate_execution_mode({"execution_mode": "legacy-direct-call"}) or ""
        )
    report["execution_mode"] = "pytest-live"

    for gate in (gate_execution_mode(report), gate_suite_health(report)):
        if gate:
            return _fail(gate)
    truncation = gate_truncation(report)
    if truncation and not args.allow_truncation:
        return _fail(truncation)
    if truncation:
        print(f"\nWARNING: {truncation}", file=sys.stderr)

    Path(".wesker").mkdir(exist_ok=True)
    Path(".wesker/mutation_report.json").write_text(json.dumps(report, indent=2))
    if args.sarif:
        # Kept inside the workspace. `--sarif` names an output file, and an output file that
        # can be `../../../anywhere` is a path-traversal write with `parents=True` behind it —
        # this runs on a CI runner with a checkout and a token, so "the caller chose the path"
        # is not a reason to skip the check.
        sarif_path = Path(args.sarif)
        workspace = Path.cwd().resolve()
        if not sarif_path.resolve().is_relative_to(workspace):
            return _fail(f"--sarif must stay inside the workspace: {args.sarif}")
        sarif_path.parent.mkdir(parents=True, exist_ok=True)
        sarif_path.write_text(
            json.dumps(to_sarif(report, version=_version()), indent=2)
        )
        print(f"SARIF: {sarif_path}")

    _annotate(report)
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
    except Exception:
        return ""


if __name__ == "__main__":
    sys.exit(main())
