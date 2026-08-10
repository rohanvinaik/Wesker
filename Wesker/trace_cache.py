"""The session baseline, PERSISTED — because it is a measurement of a constant.

WHY: `build_session_baseline` runs the consumer's whole suite TWICE (once under a per-line
tracer, once plain) and the result lives in a ContextVar — one process. So every command
re-measures it from scratch, and `trace_suite`'s own docstring already says the expensive half
is function-INDEPENDENT: "what a test executes does not depend on which function we intend to
mutate. Only the final intersection with a function's executable lines is per-function."

That makes the cost `O(suite x invocations)` for a value that changed zero times. Measured on
Regenesis: 226 tests, ~3s each, an 11-MINUTE baseline for ONE function — and the next function
in the same file pays all 11 again. On LintGate's 14,000 tests the same pass is hours, and every
target re-pays it. The hoist to a suite-global baseline fixed `O(suite x functions)` WITHIN one
session; nothing carried it ACROSS one, and a CLI takes one target per invocation, so in
practice the hoist never paid.

WHAT IS KEYED, and why each part is load-bearing:

* the engine version — a baseline is this engine's measurement; another engine's is a different
  answer to the same question, and serving it silently would be the drift `verdict_cache` was
  written to end;
* each TARGET FILE's content — the map is `{test: {file: LINES}}`, and editing the target moves
  its line numbers. A stale map then points at lines that have moved, which reads downstream as
  coverage of code that is not there;
* each TEST's source, individually — NOT one digest over the suite. Per-test is what makes this
  survive a consumer whose product is WRITING tests: converge adds a file, and only that file's
  entries miss. A single suite-wide digest would invalidate all 14,000 because one arrived,
  which is exactly the total-invalidation defect `LazySessionBaseline.refresh` exists to avoid,
  reintroduced one layer down;
* the trace BUDGETS — they decide how much of the suite was measured at all. A budget-cut entry
  is under-counted by construction, and under-counted coverage is indistinguishable from "no
  test reaches this line".

WHAT IS NOT STORED: `inert`, keyed by `id()`. An id is a fact about one process's heap and means
nothing in the next. The TEST IDS are stored and the `id()`s rebuilt on load against the live
callables — the same information, addressed by something that survives a process boundary.
Those are `ci.callable_test_id` values since issue #16, not `__name__`s: matching on a name
readmitted every test that merely SHARED a name with an inert one, which is a false kill
attribution produced only on a warm cache.

ON DISK under `.wesker/`, and `memory_guard.purge_caches` owns its lifecycle — it targets
`trace_cache.json` by name (it did NOT for this file's whole early life, so `purge` reported "clean"
while a poisoned entry survived; that gap is closed). This is a regeneratable measurement, never a
product, and a user who distrusts it deletes it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable
from typing import Any

_CACHE_DIR = ".wesker"
_CACHE_FILE = "trace_cache.json"
_VERSION = (
    # 3 (issue #16): `failing` and `inert` are stored as TEST IDS, not `__name__`s. The two are
    # both bare strings, so a v2 file would LOAD cleanly and be read as ids — every entry
    # missing its match, every previously-inert test silently readmitted to kill attribution.
    # A misread is worse than a miss: the miss costs one cold trace, the misread is wrong and
    # says nothing. This is the case the version field exists for.
    #
    # 4 (issue #17): each per-file cell went from a bare `[lines]` to `{"lines": [...], "arcs":
    # [[a,b],...]}` so a warm cache carries branch edges, not only statements. A v3 cell is a
    # list, a v4 cell a dict, so a v3 file loaded under v4 would raise the moment `trace_suite`
    # read `cell["lines"]` — the version bump orphans every v3 entry so it re-traces once under
    # the new shape rather than crashing on the old one.
    4  # the on-disk shape; bump to orphan every prior entry rather than misread one
)


def _sha(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", "replace"), usedforsecurity=False
    ).hexdigest()[:16]


def test_fingerprint(fn: Callable[..., Any]) -> str:
    """A test's identity BY CONTENT — its source AND its parametrization, else its dotted name.

    Mirrors `Detective.verdict_cache`'s discipline deliberately: the same question deserves the
    same key everywhere, and two hashers that disagree are two caches that cannot warm each
    other. A live pytest item wraps the test as `__wrapped__`; unwrap first or every parametrized
    case fingerprints as the wrapper and the whole file collapses to one key.

    Unwrapping alone is NOT enough: a parametrized case shares its underlying function's SOURCE with
    every sibling case, so `getsource(real)` is byte-identical across them → one fingerprint → the
    trace cache serves case 0's coverage for case 1, and case 1's own branch lines silently vanish
    from the union. Since issue #16 `trace_suite` keys per ITEM (`ci.callable_test_id`) rather than
    on the base `__name__`, so the two agree because BOTH discriminate a case — where previously
    they agreed only because a coarse key and this fine one were deliberately paired, and that
    pairing was the reason a traced entry could not name which item produced it. Keyed on source
    alone they still disagree and a parametrized golden still profiles as one case. The live-item wrapper carries the
    full nodeid on `__qualname__` (`test[args0-…]`) — the per-case discriminator; fold it in so sibling
    cases fingerprint APART, while a source edit still invalidates every case (source stays in the
    hash) and the same nodeid still warms the cache across runs. Non-parametrized tests are unaffected:
    `getsource` already includes the `def` line, so distinct functions already have distinct sources.
    """
    # Contract accessors, not raw attribute reads (issue #6) — local import because ci
    # imports lazily in the other direction (same pattern as engine's trace_cache import).
    from Wesker.ci import callable_node_id, callable_source

    real = callable_source(fn)
    disc = callable_node_id(fn)
    try:
        return _sha(inspect.getsource(real) + "\x00" + disc)
    except (OSError, TypeError):
        # No readable source (a C callable, a closure built at runtime). Its NAME is not content,
        # so this cannot detect an edit — but it is stable, and the alternative is refusing to
        # cache the whole suite because one test is unreadable.
        return _sha(
            f"{getattr(real, '__module__', '?')}.{getattr(real, '__qualname__', repr(real))}\x00{disc}"
        )


def targets_fingerprint(target_files: set[str]) -> str:
    """The target files' CANONICAL PATH and CONTENT. Editing one moves its lines, so every
    entry naming it is void; and two different files must never share a key (issue #20).

    Keyed on `os.path.realpath`, not `os.path.basename`. A basename plus content digest is not
    an identity: two checkouts of the same repo, a vendored copy, or a `src/` and `build/` pair
    hold same-named files with byte-identical content, and every one of them collapsed onto one
    entry — so a trace measured against one file was served for another, silently, with line
    numbers that happen to look plausible because the content matched at the time it was cached.

    `realpath` is the right canonical form rather than merely the absolute path: symlink and
    case spellings of ONE file must still agree, which is the same identity `coverage_from_trace`
    resolves by `st_dev`/`st_ino` when reading a persisted trace back. Distinct files separate;
    distinct spellings of one file do not.

    The unreadable branch keeps the same canonical spelling. It previously fell back to the raw
    path while the success branch used a basename, so the FAILURE path was the more specific of
    the two — an inconsistency that would have masked this defect for any unreadable target.
    """
    parts: list[str] = []
    for f in sorted(target_files):
        try:
            canonical = os.path.realpath(f)
        except (
            OSError
        ):  # pragma: no cover — realpath is total on every supported platform
            canonical = f
        try:
            with open(f, "rb") as fh:
                parts.append(
                    f"{canonical}:{hashlib.sha256(fh.read()).hexdigest()[:16]}"
                )
        except OSError:
            parts.append(
                f"{canonical}:<unreadable>"
            )  # cannot vouch for it -> a key nothing will match
    return _sha("|".join(parts))


def _path(project_root: str) -> str:
    return os.path.join(project_root, _CACHE_DIR, _CACHE_FILE)


def load(
    project_root: str, targets: str, budgets: tuple[float | None, float | None]
) -> dict:
    """`{test_fingerprint: {file: [lines]}}` for entries still valid — {} when none are.

    Never raises and never partially answers: a cache is an optimisation, and one that can fail
    a run is a liability. Any doubt returns {} and the caller measures, which is what it would
    have done anyway.
    """
    try:
        with open(_path(project_root), encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(blob, dict) or blob.get("version") != _VERSION:
        return {}
    if blob.get("engine") != _engine_version():
        return {}
    if blob.get("targets") != targets or blob.get("budgets") != list(budgets):
        return {}
    entries = blob.get("entries")
    return entries if isinstance(entries, dict) else {}


def save(
    project_root: str,
    targets: str,
    budgets: tuple[float | None, float | None],
    entries: dict[str, dict[str, Any]],
    failing: list[str],
    inert_names: list[str],
) -> None:
    """Write the baseline. Best-effort: a cache that fails a run is worse than no cache."""
    try:
        os.makedirs(os.path.join(project_root, _CACHE_DIR), exist_ok=True)
        with open(_path(project_root), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": _VERSION,
                    "engine": _engine_version(),
                    "targets": targets,
                    "budgets": list(budgets),
                    "entries": entries,
                    "failing": sorted(set(failing)),
                    "inert_names": sorted(set(inert_names)),
                },
                fh,
            )
    except (OSError, TypeError, ValueError):
        return


def load_outcomes(project_root: str) -> tuple[list[str], list[str]]:
    """`(failing, inert_names)` from the same file `load` just validated — the SECOND pass.

    `build_session_baseline` runs the suite twice: traced, then plain, for `failing`/`inert`.
    Caching only the trace would leave half the bill standing, and the plain pass is the same
    constant measured the same redundant way.
    """
    try:
        with open(_path(project_root), encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return [], []
    return list(blob.get("failing") or []), list(blob.get("inert_names") or [])


def _engine_version() -> str:
    from Wesker import __version__

    return __version__
