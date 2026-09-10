#!/usr/bin/env python3
"""Fail if a built sdist carries a file git does not track.

A deliberate copy of Detective's ``scripts/check_sdist.py``, and the duplication is the point
rather than an oversight: this runs at BUILD time, before anything is installed, so it cannot
import from a package -- and Wesker is Detective's dependency, so taking the shared copy from
Detective would make the dependency circular. Two copies of a build gate, each owned by the
repo it gates. If the rule changes, both change; there is no third place to look.

Wesker has never shipped an untracked file. That is the reason to install the guard, not a
reason to skip it: it is currently clean because nothing large happens to be sitting in the
tree, which is a fact about today rather than a property of the configuration.

The defect this exists to catch, from Detective's 1.0.0 build: hatchling's VCS-ignore default
reads the ROOT ``.gitignore`` ONLY. A nested ``.gitignore`` is invisible to it (measured
2026-09-09 with a minimal probe project). So a directory can be genuinely gitignored -- absent
from ``git status``, absent from ``git ls-files``, invisible to every habit you have -- and be
packaged anyway. Detective shipped 6.9 GB of vendored Lean build output that way, and the size
was only the symptom: an sdist holding untracked files is a function of the BUILDER's working
tree rather than of the commit, so two people building the same sha get different tarballs.

The rule enforced here: an sdist member is either tracked by git, or synthesised by the build
backend (``PKG-INFO``). There is no third category.

Usage::

    python3 scripts/check_sdist.py [dist/<name>-<version>.tar.gz]

Exit codes follow the epistemic vocabulary rather than pass/fail: ``0`` reproducible from the
commit, ``1`` a measured defect, ``2`` the question could not be asked (no sdist built, not a
git checkout) -- a refusal is not a pass.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

# Members the build backend synthesises rather than copying out of the tree. Anything else in
# the sdist that git does not track came from the builder's working directory.
GENERATED = frozenset({"PKG-INFO"})


def _stem(name: str) -> str:
    """Strip the ``<name>-<version>/`` prefix every sdist member carries."""
    return name.split("/", 1)[1] if "/" in name else name


def sdist_verdict(members: list[str], tracked: list[str], generated: list[str]) -> str:
    """Classify an sdist against the repo's tracked set (pure -- pinned).

    Split out from the I/O so the judgement is expressible as ``--input``: opening the tar and
    shelling out to ``git ls-files`` are inexpressible, deciding the verdict is not. Returns a
    named code rather than a bool because "no members at all" and "members the commit does not
    contain" are different facts with different remedies, and a truthy check would collapse
    them into one.

    ``reproducible``      every member is tracked or build-generated
    ``carries_untracked`` at least one member came from the builder's working tree
    ``empty``             the archive listed no files to judge

    PIN RECEIPT (taken once, in Detective, on the byte-identical body): converged in isolation,
    ✓ COMPLETE, 12/13 killed modulo 1 unproven-equivalent and 2 crash-only; the hand-written
    intent suite at ``Detective/tests/test_sdist_guard_intent.py`` independently scores 11/13
    value-pinned and 100% killed. Not re-pinned here -- the same function pinned twice would be
    the same measurement twice, and the copy exists for the build-order reason in the module
    docstring, not because the decision differs.
    """
    if not members:
        return "empty"
    known = set(tracked) | set(generated)
    for name in members:
        if _stem(name) not in known:
            return "carries_untracked"
    return "reproducible"


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    archives = [Path(argv[0])] if argv else sorted((root / "dist").glob("*.tar.gz"))
    if not archives:
        print(
            "check_sdist: no sdist found -- build one first (`uv build`)",
            file=sys.stderr,
        )
        return 2

    try:
        listed = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"check_sdist: cannot read the tracked set: {exc}", file=sys.stderr)
        return 2
    tracked = sorted(set(listed.split("\n")) - {""})

    failed = False
    for archive in archives:
        with tarfile.open(archive) as tar:
            files = [m for m in tar.getmembers() if m.isfile()]
        members = [m.name for m in files]
        size = sum(m.size for m in files)
        verdict = sdist_verdict(members, tracked, sorted(GENERATED))
        if verdict == "reproducible":
            print(
                f"OK  {archive.name}: {len(members)} files, {size / 1e6:.2f} MB, all tracked"
            )
            continue
        failed = True
        if verdict == "empty":
            print(f"FAIL {archive.name}: contains no files", file=sys.stderr)
            continue
        known = set(tracked) | GENERATED
        extra = sorted(n for n in members if _stem(n) not in known)
        print(
            f"FAIL {archive.name}: {len(extra)} of {len(members)} files are not tracked by git.\n"
            "  This sdist is a function of the builder's working tree, not of the commit.\n"
            "  Add the offending path to [tool.hatch.build].exclude:",
            file=sys.stderr,
        )
        for name in extra[:20]:
            print(f"    {name}", file=sys.stderr)
        if len(extra) > 20:
            print(f"    ... and {len(extra) - 20} more", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
