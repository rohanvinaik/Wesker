"""Wesker CLI — mutation testing from the command line.

Usage:
    wesker src/                           # profile all .py files
    wesker src/core.py src/utils.py       # profile specific files
    wesker --mcdc src/scoring.py::func    # MC/DC on specific functions
    wesker --json                         # JSON output instead of terminal
    wesker --threshold 90                 # exit 1 if kill rate < 90%
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _discover_python_files(paths: list[str]) -> list[str]:
    """Expand directories into .py files, pass files through."""
    files: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".py":
            files.append(str(path))
        elif path.is_dir():
            for py in sorted(path.rglob("*.py")):
                if "__pycache__" not in str(py) and not py.name.startswith("test_"):
                    files.append(str(py))
    return files


def _parse_mcdc_targets(specs: list[str]) -> list[tuple[str, str]]:
    """Parse file.py::function specs into (file, function) tuples."""
    targets = []
    for spec in specs:
        if "::" in spec:
            f, func = spec.split("::", 1)
            targets.append((f, func))
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wesker",
        description="Wesker — in-process AST mutation testing for Python",
    )
    # There was no `--version` at all: `wesker --version` printed an argparse usage error, so
    # the only way to ask an INSTALLED engine what it was was to import it. That matters more
    # here than for a normal CLI — this engine decides the verdict (a kill classified `crash`
    # rather than `exception` changes what counts as specified at all), and its consumers key
    # their verdict caches on this number. A tool whose answer depends on its version has to
    # be able to say what it is. Sourced from `Wesker.__version__`, the one owner.
    from . import __version__

    parser.add_argument("--version", action="version", version=f"wesker {__version__}")
    parser.add_argument(
        "targets", nargs="*", default=["."], help="Files or directories to profile"
    )
    parser.add_argument(
        "--mcdc", nargs="*", metavar="FILE::FUNC", help="MC/DC verification targets"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="JSON output"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Minimum kill rate %% (exit 1 if below)",
    )
    parser.add_argument(
        "--budget", type=float, default=10000, help="Per-file budget in ms"
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Max mutants per category per pass "
        "(default: derived per function from its degrees of freedom; 0=exhaustive)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Convergence passes (extra passes deepen within covered dimensions)",
    )
    parser.add_argument("--exclude", nargs="*", default=[], help="Files to exclude")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--purge",
        action="store_true",
        help="delete regeneratable .wesker/ cruft from old runs and exit",
    )
    args = parser.parse_args(argv)

    if args.purge:
        from Wesker.memory_guard import purge_caches

        removed, reclaimed = purge_caches(".")
        if args.json_output:
            print(json.dumps({"removed": list(removed), "reclaimed_bytes": reclaimed}))
        elif removed:
            print(f"purged {len(removed)} file(s), reclaimed {reclaimed // 1024} KB")
        else:
            print("nothing to purge — clean state")
        return 0

    from Wesker.ci import profile_codebase, profile_codebase_live

    # Discover targets
    files = _discover_python_files(args.targets)
    exclude_set = set(args.exclude)
    files = [f for f in files if f not in exclude_set]

    if not files:
        print("No Python files found.", file=sys.stderr)
        return 1

    if not args.quiet and not args.json_output:
        print(f"Wesker — profiling {len(files)} files\n")

    # Run mutation profiling — pytest judges every mutant (real fixtures, conftest,
    # parametrization), so the kill rate is a standards-compliant mutation score.
    result = profile_codebase_live(
        ".",
        files,
        budget_ms_per_file=args.budget,
        max_per_category=args.max_per_category,
        passes=args.passes,
        verbose=not args.quiet and not args.json_output,
    )
    execution_mode = "pytest-live"
    if result is None:
        # Degrade only OUT LOUD, and on stderr so it survives --json-output being
        # piped: the legacy runner cannot execute fixture-taking tests, so its number
        # is not comparable to a standard mutation tester's.
        execution_mode = "legacy-direct-call"
        print(
            "WARNING: no live pytest session — falling back to the legacy direct-call "
            "runner.\n         Fixture-taking tests cannot run; this kill rate is NOT "
            "a standards-\n         compliant mutation score.",
            file=sys.stderr,
        )
        result = profile_codebase(
            ".",
            files,
            budget_ms_per_file=args.budget,
            max_per_category=args.max_per_category,
            passes=args.passes,
            verbose=not args.quiet and not args.json_output,
        )
    result["execution_mode"] = execution_mode

    # MC/DC if requested
    mcdc_result = None
    if args.mcdc:
        from Wesker.metrics import _verify_mcdc

        mcdc_targets = _parse_mcdc_targets(args.mcdc)
        if mcdc_targets:
            if not args.quiet and not args.json_output:
                print("\nMC/DC verification...")
            mcdc_result = _verify_mcdc(mcdc_targets)

    # Output
    if args.json_output:
        output = {"mutation": result}
        if mcdc_result:
            output["mcdc"] = mcdc_result
        print(json.dumps(output, indent=2))
    else:
        if not args.quiet:
            print(f"\n{'─' * 50}")
            print(
                f"Kill rate: {result['kill_pct']}% ({result['total_killed']}/{result['total_mutants']})"
            )
            # DOF coverage — the claim a bounded run can actually make. Under
            # singleton cover sets the greedy reaches min(picks, D) of D dimensions
            # exactly, so this is measured coverage, not an estimate.
            if result.get("total_dof"):
                print(
                    f"DOF coverage: {result['dof_pct']}% "
                    f"({result['total_dof_covered']}/{result['total_dof']} dimensions) "
                    f"via {result['total_mutants']}/{result['total_universe']} mutants"
                )
            print(f"Functions: {result['total_functions']}")
            if result.get("total_equivalent"):
                print(f"Equivalent: {result['total_equivalent']}")
            print(f"Elapsed: {result['elapsed_ms']}ms")
            if mcdc_result:
                status = "PASS" if mcdc_result["verified"] else "FAIL"
                print(
                    f"MC/DC: {status} ({mcdc_result['conditions_covered']}/{mcdc_result['conditions_total']})"
                )

    # Lightweight memory-telemetry footer (best-effort — never fails the run).
    if not args.json_output and not args.quiet:
        try:
            from Wesker.memory_guard import telemetry

            print(f"[{telemetry()}]")
        # BLE001: telemetry is advisory
        except Exception:  # noqa: BLE001
            pass

    # Threshold gate
    if args.threshold and result["kill_pct"] < args.threshold:
        if not args.json_output:
            print(
                f"\nFAIL: kill rate {result['kill_pct']}% < threshold {args.threshold}%",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
