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
    parser.add_argument("targets", nargs="*", default=["."], help="Files or directories to profile")
    parser.add_argument("--mcdc", nargs="*", metavar="FILE::FUNC", help="MC/DC verification targets")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument("--threshold", type=int, default=0, help="Minimum kill rate %% (exit 1 if below)")
    parser.add_argument("--budget", type=float, default=10000, help="Per-file budget in ms")
    parser.add_argument("--max-per-category", type=int, default=5, help="Max mutants per category per pass (0=exhaustive)")
    parser.add_argument("--passes", type=int, default=3, help="Convergence passes (different seed each)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Files to exclude")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args(argv)

    from Wesker.ci import profile_codebase

    # Discover targets
    files = _discover_python_files(args.targets)
    exclude_set = set(args.exclude)
    files = [f for f in files if f not in exclude_set]

    if not files:
        print("No Python files found.", file=sys.stderr)
        return 1

    if not args.quiet and not args.json_output:
        print(f"Wesker — profiling {len(files)} files\n")

    # Run mutation profiling
    result = profile_codebase(
        ".",
        files,
        budget_ms_per_file=args.budget,
        max_per_category=args.max_per_category,
        passes=args.passes,
        verbose=not args.quiet and not args.json_output,
    )

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
            print(f"Kill rate: {result['kill_pct']}% ({result['total_killed']}/{result['total_mutants']})")
            print(f"Functions: {result['total_functions']}")
            if result.get("total_equivalent"):
                print(f"Equivalent: {result['total_equivalent']}")
            print(f"Elapsed: {result['elapsed_ms']}ms")
            if mcdc_result:
                status = "PASS" if mcdc_result["verified"] else "FAIL"
                print(f"MC/DC: {status} ({mcdc_result['conditions_covered']}/{mcdc_result['conditions_total']})")

    # Threshold gate
    if args.threshold and result["kill_pct"] < args.threshold:
        if not args.json_output:
            print(f"\nFAIL: kill rate {result['kill_pct']}% < threshold {args.threshold}%", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
