"""A private, unmutatable copy of Wesker, so Wesker can profile Wesker.

THE PROBLEM. ``engine._patch_module_qualified`` installs a mutant into every module whose
binding of the target function has a matching ``co_filename``. That is exactly right when the
target is somebody else's code. When the target is ``Wesker/engine.py``, the module it patches
is the engine's own — so the engine's next global lookup resolves to the mutant and the run
dies mid-flight (``IndexError`` inside ``_patch_mutant_into_test``, reproducible today).

THE DESIGN. Import a SECOND copy of the package under a private name. The profiling engine runs
from the private copy; the public ``Wesker.*`` modules are mutated freely. Tests import
``Wesker.*`` and therefore see every mutant, direct and indirect, so the kill signal is
undiminished. Nothing is excluded and no third "unevaluated" outcome is invented.

WHY A LOADER AND NOT A FILE COPY. A copied module's ``from Wesker.engine import X`` still
resolves to the PUBLIC package, dragging the private copy back into mutated code — so the copy
has to be rewritten either way. Doing that rewrite ON DISK means shipping a function that
copies trees and writes files, whose destination is computed from its arguments. Mutate that
destination and it writes somewhere real; point it at a directory inside its own source and it
copies its own output forever. This module therefore rewrites the source IN MEMORY during the
import and never touches the filesystem: the only state it creates is entries in
``sys.modules``.

THE CONSEQUENCE, AND IT IS LOAD-BEARING. Because the private modules are compiled from the
REAL files, they carry the real ``co_filename`` — ``Wesker/engine.py`` for both copies. So
``_patch_module_qualified`` cannot tell them apart by filename, and would patch the private
copy too. The companion guard there must therefore match on the module's ``__name__``, not its
``co_filename``. The in-memory rewrite and the name-based guard are two halves of one design;
neither works alone.
"""

from __future__ import annotations

import os
import re
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

PUBLIC_PACKAGE = "Wesker"

# The private name. Anything a module-name guard can test with a cheap prefix check, and
# deliberately not importable by an ordinary `import _wesker_self` — the finder below is only
# installed when a self-profiling run asks for it.
PRIVATE_PREFIX = "_wesker_self"

# Anchored at line start so a mention inside a docstring or comment is left alone, and applied
# per line rather than per file because the imports that matter most are function-local:
# `ci.py` imports `_SESSION_BASELINE` from inside a function body, and a header-only rewrite
# would miss it — splitting the ContextVar across the two copies, which degrades the run to the
# legacy runner silently.
_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(rf"(?m)^(\s*)from {PUBLIC_PACKAGE}\.(\w+) import"),
        rf"\1from {PRIVATE_PREFIX}.\2 import",
    ),
    (
        re.compile(rf"(?m)^(\s*)from {PUBLIC_PACKAGE} import"),
        rf"\1from {PRIVATE_PREFIX} import",
    ),
    (
        re.compile(rf"(?m)^(\s*)import {PUBLIC_PACKAGE}\.(\w+)"),
        rf"\1import {PRIVATE_PREFIX}.\2",
    ),
)


def rewrite_source(source: str) -> str:
    """Point a module's intra-package imports at the private copy.

    Idempotent: a rewritten line has no `Wesker.` prefix left to match, so re-running is a
    no-op rather than a corruption.

    ``from Wesker.engine import X`` -> ``from _wesker_self.engine import X``. Only absolute
    intra-package imports are touched; relative imports already resolve against the importing
    module's package, which for the private copy IS the private package, so they need nothing.
    """
    for pattern, replacement in _REWRITES:
        source = pattern.sub(replacement, source)
    return source


class _PrivateLoader(SourceFileLoader):
    """A source loader that rewrites intra-package imports on the way in.

    ``get_data`` is the single seam every source import funnels through, so rewriting here
    catches the module body without re-implementing any of the import protocol. The path handed
    to ``super().__init__`` is the REAL file, so the compiled code carries the real
    ``co_filename`` — see the module docstring for why that matters.
    """

    def path_stats(self, path: str) -> dict:
        """Refuse the bytecode cache — ALWAYS compile from rewritten source.

        Raising OSError here is the documented way to tell ``SourceLoader.get_code`` it has no
        source timestamp, which makes it skip reading ``__pycache__`` AND skip writing it back.
        Both halves matter. A cached ``.pyc`` for ``Wesker/ci.py`` is compiled from the PUBLIC
        source, so consulting it would hand the private copy a module that imports the public
        engine — the exact silent degradation this module exists to prevent, arriving as a
        cache hit rather than an error. Writing one back would be worse: it would poison the
        public package's cache with privately-rewritten bytecode.
        """
        raise OSError("private copy is never cached")

    def get_data(self, path: str) -> bytes:
        return rewrite_source(super().get_data(path).decode("utf-8")).encode("utf-8")


class _PrivateFinder(MetaPathFinder):
    """Resolve ``_wesker_self[.submodule]`` to the public package's real source files."""

    def __init__(self, package_dir: Path) -> None:
        self._package_dir = package_dir

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != PRIVATE_PREFIX and not fullname.startswith(PRIVATE_PREFIX + "."):
            return None  # not ours — let the normal finders answer

        relative = fullname[len(PRIVATE_PREFIX) :].lstrip(".")
        parts = relative.split(".") if relative else []
        as_package = self._package_dir.joinpath(*parts)

        if as_package.is_dir():
            filename, search = as_package / "__init__.py", [str(as_package)]
        else:
            filename, search = (
                self._package_dir.joinpath(*parts).with_suffix(".py"),
                None,
            )
        if not filename.is_file():
            return None

        spec = ModuleSpec(
            fullname,
            _PrivateLoader(fullname, str(filename)),
            origin=str(filename),
            is_package=search is not None,
        )
        if search is not None:
            spec.submodule_search_locations = search  # type: ignore[assignment]
        return spec


def targets_own_package(
    targets: Sequence[str], package_dir: str, project_root: str = "."
) -> bool:
    """True when any target file lives inside ``package_dir`` — i.e. Wesker profiling Wesker.

    This is the whole trigger for the private copy: an ordinary project never matches, so an
    ordinary run never loads a second copy and never pays for one. Only a run that targets the
    profiler's own source needs the private engine, and it needs it for every target in the
    run, not just the matching ones — one mutant landing in the live engine is enough to kill
    it, so the decision is any(), not per-file.

    Pure: ``abspath`` normalises against the cwd without touching the filesystem, so this is a
    total function of its arguments and can be pinned with plain literals. The ``os.sep``
    suffix is what keeps ``/x/Weskerish/y.py`` from matching a package at ``/x/Wesker`` — a
    prefix test alone would match the wrong tree, the same segment-boundary bug
    ``_co_filename_matches`` guards against.
    """
    package = os.path.abspath(package_dir)
    for target in targets:
        resolved = os.path.abspath(os.path.join(project_root, target))
        if resolved == package or resolved.startswith(package + os.sep):
            return True
    return False


def install_finder(package_dir: str | Path | None = None) -> None:
    """Make ``_wesker_self`` importable. Idempotent."""
    if any(isinstance(f, _PrivateFinder) for f in sys.meta_path):
        return
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    sys.meta_path.insert(0, _PrivateFinder(root))


def load_private_wesker(package_dir: str | Path | None = None) -> ModuleType:
    """Import and return the private copy of the package. Idempotent."""
    install_finder(package_dir)
    if PRIVATE_PREFIX not in sys.modules:
        __import__(PRIVATE_PREFIX)
    return sys.modules[PRIVATE_PREFIX]


def profiler_for_targets(
    targets: Sequence[str], project_root: str = "."
) -> Callable[..., Any]:
    """The ``profile_codebase_live`` these targets should be profiled with.

    One seam, one decision: a run that targets Wesker's own source gets the PRIVATE engine, so
    the public modules it is mutating are not the ones executing the run. Every other run gets
    the public engine it has always used, on the same import line, with no private copy loaded
    and nothing else changed.

    The whole stack must come from one side. ``_LIVE_SUITE`` and ``_SESSION_BASELINE`` are
    ContextVars, and each copy of the package owns its own — so a live suite established by the
    public ``ci`` would be invisible to a private ``engine`` reading its own baseline. Returning
    the ENTRY POINT (rather than swapping pieces underneath it) is what keeps the halves from
    being mixed: everything downstream of this call resolves within one copy. If the two ever do
    get crossed, the session baseline goes missing and the run degrades to the legacy runner —
    which ``action.gate_execution_mode`` refuses rather than reports.
    """
    package_dir = str(Path(__file__).resolve().parent)
    if targets_own_package(targets, package_dir, project_root):
        load_private_wesker()
        from _wesker_self.ci import (  # type: ignore[import-not-found]
            profile_codebase_live,
        )

        return profile_codebase_live

    from Wesker.ci import profile_codebase_live

    return profile_codebase_live
