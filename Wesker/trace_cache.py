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
import tempfile
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
    # 5 (issue #15): cached non-reach became routing evidence, so the key bound the whole
    # test/fixture context and pytest regime.
    # 7 (issue #15/#20): each observed outcome carries the same content fingerprint as its trace.
    # Reusing by TestId alone survived an edit under the same node ID and relabelled the old outcome
    # current even though the trace correctly missed.
    7  # the on-disk shape; bump to orphan every prior entry rather than misread one
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
    from Wesker.ci import (
        callable_fixture_origins,
        callable_node_id,
        callable_origin,
        callable_source,
    )

    real = callable_source(fn)
    disc = callable_node_id(fn)
    try:
        declared = inspect.getsource(real)
    except (OSError, TypeError):
        # No readable source (a C callable, a closure built at runtime). Its NAME is not content,
        # so this cannot detect an edit — but it is stable, and the alternative is refusing to
        # cache the whole suite because one test is unreadable.
        declared = f"{getattr(real, '__module__', '?')}.{getattr(real, '__qualname__', repr(real))}"

    # The function body is not the whole call path. A module helper or autouse/conftest fixture can
    # begin reaching the target while the test's own source stays byte-identical. Hash the complete
    # origin files governing this item before a cached non-reach may become routing evidence.
    origin = callable_origin(fn)
    context_files = {os.path.realpath(p) for p in callable_fixture_origins(fn) if p}
    if origin:
        origin = os.path.realpath(origin)
        context_files.add(origin)
        parent = os.path.dirname(origin)
        while parent and parent != os.path.dirname(parent):
            conftest = os.path.join(parent, "conftest.py")
            if os.path.isfile(conftest):
                context_files.add(os.path.realpath(conftest))
            parent = os.path.dirname(parent)

    context: list[str] = []
    for path in sorted(context_files):
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            # No content identity means no cross-run identity. A per-call nonce deliberately makes
            # this item uncacheable; reusing a stable "unreadable" sentinel would let two unknown
            # contexts compare equal and promote stale non-reach to impossible.
            digest = f"<unreadable:{os.urandom(8).hex()}>"
        context.append(f"{path}:{digest}")
    return _sha("\x00".join((declared, disc, *context)))


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
    project_root: str,
    targets: str,
    budgets: tuple[float | None, float | None],
    regime_digest: str = "",
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
    if (
        blob.get("targets") != targets
        or blob.get("budgets") != list(budgets)
        or blob.get("regime", "") != regime_digest
    ):
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
    regime_digest: str = "",
    outcomes_observed: list[str] | None = None,
    outcome_fingerprints: dict[str, str] | None = None,
) -> None:
    """Write the baseline. Best-effort: a cache that fails a run is worse than no cache."""
    temp_path = ""
    try:
        cache_dir = os.path.join(project_root, _CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".trace_cache.", dir=cache_dir, text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": _VERSION,
                    "engine": _engine_version(),
                    "targets": targets,
                    "budgets": list(budgets),
                    "regime": regime_digest,
                    "entries": entries,
                    "failing": sorted(set(failing)),
                    "inert_names": sorted(set(inert_names)),
                    "outcomes_observed": sorted(set(outcomes_observed or ())),
                    "outcome_fingerprints": dict(outcome_fingerprints or {}),
                },
                fh,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, _path(project_root))
    except (OSError, TypeError, ValueError):
        return
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _is_property_test(fn: Any) -> bool:
    """Is this a Hypothesis (``@given``) test — whose covered lines vary with the example seed?

    A cached non-reach for such a test is not safely replayable (§2.2 a-2): the same source and
    context can cover different lines on a re-run, so its cached negative must degrade to
    ``unknown`` and be re-traced. Hypothesis marks the wrapped test ``is_hypothesis_test`` /
    ``.hypothesis``; walk the ``__wrapped__`` chain because Wesker's own discovery wrapper may sit
    OUTSIDE the Hypothesis wrapper, and ``callable_source`` only unwraps one level.
    """
    seen: set[int] = set()
    cur: Any = fn
    while cur is not None and id(cur) not in seen:
        if getattr(cur, "is_hypothesis_test", False) or hasattr(cur, "hypothesis"):
            return True
        seen.add(id(cur))
        cur = getattr(cur, "__wrapped__", None)
    return False


def replayed_negative_admission(
    has_outcome: bool, fingerprint_matches: bool, is_property_test: bool
) -> str:
    """May a CACHED non-reach be replayed as evidence a test does not reach the target?
    (§2.2 option a, pure — pinned).

    A replayed negative is the only cached value that can SHRINK the search — the only one that
    could manufacture a false COMPLETE — so it is admitted under the strictest precondition and
    named, never a bool:

      "not_reached"  admissible — a COMPLETED baseline outcome exists at the SAME content+context
                     fingerprint, and the test's coverage is not example-dependent
      "unknown"      degrade and re-trace — no completed outcome, OR the fingerprint drifted, OR a
                     property (Hypothesis) test whose covered lines vary with the example seed
    """
    if not has_outcome or not fingerprint_matches:
        return "unknown"
    if is_property_test:
        return "unknown"
    return "not_reached"


def observed_function_reach(
    project_root: str,
    target_files: set[str],
    budgets: tuple[float | None, float | None],
    regime_digest: str,
    test_functions: list[Callable[..., Any]],
    target_file: str,
    executable_lines: set[int],
) -> dict[str, str]:
    """Prior per-TestId reach for ONE function; absent means genuinely unobserved (#15).

    This is routing evidence only. The admitted seed is fresh-traced before it can prove a mutant
    disposition. A complete cell with no intersecting line becomes negative evidence only after
    the same exact TestId also completed the baseline-outcome pass. A trace interrupted by an
    exception records the lines reached before it stopped; without outcome qualification, absence
    could mean "did not enter" or merely "did not finish" and must remain UNKNOWN.
    """
    from Wesker.ci import callable_test_id

    if not regime_digest:
        return {}
    cache = load(
        project_root,
        targets_fingerprint(target_files),
        budgets,
        regime_digest,
    )
    if not cache:
        return {}
    _failing, _inert, outcomes_observed, outcome_fingerprints = load_outcomes(
        project_root
    )
    outcomes = set(outcomes_observed)
    target = os.path.realpath(target_file)
    out: dict[str, str] = {}
    for test_fn in test_functions:
        test_id = callable_test_id(test_fn)
        fingerprint = test_fingerprint(test_fn)
        hit = cache.get(fingerprint)
        if hit is None:
            continue
        lines: set[int] = set()
        for path, cell in hit.items():
            if os.path.realpath(path) == target:
                lines.update(int(ln) for ln in cell.get("lines", ()))
        if executable_lines.intersection(lines):
            out[test_id] = "reached"
        elif (
            replayed_negative_admission(
                test_id in outcomes,
                outcome_fingerprints.get(test_id) == fingerprint,
                _is_property_test(test_fn),
            )
            == "not_reached"
        ):
            out[test_id] = "not_reached"
    return out


def load_outcomes(
    project_root: str,
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Outcome facts plus their TestId→content-fingerprint identity from the SECOND pass.

    `build_session_baseline` runs the suite twice: traced, then plain, for `failing`/`inert`.
    Caching only the trace would leave half the bill standing, and the plain pass is the same
    constant measured the same redundant way.
    """
    try:
        with open(_path(project_root), encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return [], [], [], {}
    return (
        list(blob.get("failing") or []),
        list(blob.get("inert_names") or []),
        list(blob.get("outcomes_observed") or []),
        dict(blob.get("outcome_fingerprints") or {}),
    )


def _engine_version() -> str:
    from Wesker import __version__

    return __version__
