"""Per-test line coverage of a target function — the second completeness axis.

Mutation testing answers "does a test *distinguish* a behavioral change?"; line
coverage answers the orthogonal "does any test *reach* this line at all?". A suite
can kill every killable mutant yet leave a line no test ever executes (a line no
mutant happened to touch), so a suite is only *complete* when it is both
mutant-complete AND line-complete, and only *minimal* when set-cover runs over the
union of both matrices.

This is measured in a single traced baseline pass over the UNMUTATED function —
the mutation loop stays untraced (and fast). ``executable_lines`` is the static
denominator (which lines *could* run); ``trace_line_coverage`` is the dynamic
numerator (which lines each test *did* run), keyed identically to the kill matrix
so the two feed the same set-cover.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import sys
import threading
import time
from collections.abc import Callable
from types import CodeType
from typing import Any

from Wesker.interrupt import Abandoned, JoinedWorker, MeasurementRefused, bounded_join
from Wesker.trace_evidence import reach_qualification


def _traceable_lines(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[int] | None:
    """Lines the compiled body can attribute an instruction to — ``co_lines`` union.

    The AST extent of a statement sweeps in lines that carry no bytecode: a bare
    ``else:`` or ``finally:`` (keyword-only lines with no AST node of their own),
    and comment or closing-bracket lines inside a multi-line statement. CPython can
    never emit a trace event for such a line, so leaving one in the denominator
    manufactures a line-coverage gap NO test can ever close — the caller then asks
    for input after input to reach a line that is not code. ``co_lines`` over the
    compiled function (and its nested code objects) is exactly the set the tracer
    can report, so intersecting with it removes every such phantom while keeping
    all sub-expression and mutant fire-site lines, which do carry instructions.

    Returns None when the body cannot compile standalone (e.g. ``nonlocal`` into
    an enclosing scope) — the caller then keeps the unfiltered extent rather than
    risk dropping real lines.
    """
    try:
        module = ast.Module(body=[func_node], type_ignores=[])
        top = compile(module, "<traceable-lines>", "exec")
    except SyntaxError:
        return None
    lines: set[int] = set()
    stack = [top]
    while stack:
        code = stack.pop()
        stack.extend(c for c in code.co_consts if isinstance(c, CodeType))
        lines.update(line for _, _, line in code.co_lines() if line is not None)
    return lines


def executable_lines(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    """Every line a function body's code occupies — the line-coverage denominator.

    This must span EVERY line a mutation can land on, not just the line a statement
    starts on. CPython reports a "line" event per executing sub-expression, so a
    multi-line statement traces at 1010, 1011, 1012, … while it *begins* only at
    1010; and a mutator records its fire site as the mutated NODE's own line
    (``_BaseMutator._mark_applied``), which is likewise a sub-expression line. A
    statement-start-only denominator drops those lines from the traced numerator
    (``_trace_one`` intersects with this set), leaving ~a quarter of all mutants
    keyed to a line no coverage entry can ever mention — and test-impact scoping
    then finds zero covering tests and reports them as survivors no matter how
    good the suite is. Spanning full statement extents keys the two together.

    The ``def`` line and a leading docstring are excluded: neither is behavior a
    test can meaningfully "reach". The spanned extent is then intersected with
    ``_traceable_lines`` so keyword-only lines (``else:``, ``finally:``) and other
    bytecode-free lines inside an extent cannot enter the denominator: they would
    read as a permanent, unclosable coverage gap.
    """
    body = list(func_node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]  # drop a leading docstring
    lines: set[int] = set()
    for stmt in body:
        for descendant in ast.walk(stmt):
            start = getattr(descendant, "lineno", None)
            if start is None:
                continue
            end = getattr(descendant, "end_lineno", None) or start
            lines.update(range(start, end + 1))
    traceable = _traceable_lines(func_node)
    if traceable is not None:
        lines &= traceable
    return lines


def _target_matcher(target_files: set[str]) -> Callable[[str], str | None]:
    """``co_filename`` -> the matching TARGET spelling, or None — by file identity.

    ``co_filename`` carries the spelling the module was imported under; the target
    path carries the spelling the caller typed. On a case-insensitive filesystem
    (macOS default: ``wesker/engine.py`` vs ``Wesker/engine.py``) or through a
    symlink, the two open the same file while comparing unequal — the dispatch then
    never attaches, every line of the target reads uncovered, and the kill
    measurement (which has no filename filter) proceeds normally: the report says
    "80/80 killed" and "18-line gap" about the same body. Identity ``(st_dev,
    st_ino)`` answers what string equality cannot. Deliberately NOT case-folding:
    spellings are never rewritten, and on a case-sensitive filesystem two paths
    differing only in case remain the two different files they are.

    The verdict is memoized per ``co_filename`` string, so the per-event dispatch
    stats each distinct filename once — never per line event.
    """
    by_id: dict[tuple[int, int], str] = {}
    for t in target_files:
        try:
            st = os.stat(t)
        except OSError:
            continue
        by_id[(st.st_dev, st.st_ino)] = t
    verdicts: dict[str, str | None] = {t: t for t in target_files}

    def match(filename: str) -> str | None:
        if filename in verdicts:
            return verdicts[filename]
        try:
            st = os.stat(filename)
        except OSError:
            found = None
        else:
            found = by_id.get((st.st_dev, st.st_ino))
        verdicts[filename] = found
        return found

    return match


# (#17) The audited operations that create a process. The child runs OUTSIDE this interpreter, where
# no monitoring tool can follow it, so a test that makes one has reach that is unknown. The hook is
# installed once (CPython has no way to remove one) and is a single dict lookup while no trace is
# active; it records only the creating THREAD, never blocks or alters the operation.
_PROCESS_CREATION_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "_posixsubprocess.fork_exec",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.forkpty",
    }
)
_spawn_watchers: dict[int, tuple[Callable[[int], bool], list[int]]] = {}
_spawn_hook_installed = False


def _spawn_audit_hook(event: str, _args: tuple) -> None:
    if event in _PROCESS_CREATION_EVENTS and _spawn_watchers:
        ident = threading.get_ident()
        for owns, spawned in list(_spawn_watchers.values()):
            if owns(ident):
                spawned.append(ident)


def _install_spawn_hook() -> None:
    global _spawn_hook_installed
    if not _spawn_hook_installed:
        sys.addaudithook(_spawn_audit_hook)
        _spawn_hook_installed = True


def _trace_one(
    test_fn: Callable[..., None],
    target_file: str,
    exec_lines: set[int],
    budget_s: float | None = None,
    reach: list[str] | None = None,
) -> tuple[set[int], bool, bool]:
    """Lines within ``exec_lines`` that ``test_fn()`` executes in ``target_file``, and whether the
    budget CUT this test (the trace's own report — never inferred from a clock by the caller).

    Owned local monitoring events follow code in the target file, so unrelated
    library frames contribute no coverage. A test that raises
    on the original still contributes the lines it reached before raising — partial
    coverage is real coverage.

    ``budget_s`` bounds ONE test's tracing wall-clock. It is the same concession applied to time
    that the paragraph above already applies to exceptions: a test cut at its budget contributes
    the lines it reached, because partial coverage is real coverage either way. Without it a
    single computationally-heavy test stalls the whole baseline with no output and no diagnosis —
    tracing costs a callback per line, so a hot combinatorial loop under trace runs orders of
    magnitude slower than it does untraced. ``None`` (the default) = unbounded = the historical
    behavior exactly. The bound is wall-clock in another thread (:func:`_trace_one_multi`), NOT a
    deadline ticked from the trace callback: the callback only fires for the TARGET file, so it
    could not bound the tests — nearly all of them — that never execute it.

    BOUNDARY: see :mod:`Wesker.interrupt`. A test blocked outside the interpreter cannot be
    preempted in-process, and is reported as cut and uncontained rather than pretended away.

    ``reach`` receives ``incomplete_thread`` when a thread
    the test started outlived its run (#17).
    """
    hits_map, truncated, contained, _ = _trace_one_multi(
        test_fn, {target_file}, budget_s=budget_s, capture_arcs=False, reach=reach
    )
    lines = hits_map.get(target_file, set())
    if not lines and hits_map:
        match = _target_matcher({target_file})
        for f, ls in hits_map.items():
            if match(f) == target_file:
                lines = ls
                break
    return lines & exec_lines, truncated, contained


def failing_on_baseline(
    test_functions: list[Callable[..., None]], original_func: Callable[..., Any]
) -> list[str]:
    """Test names whose assertion FAILS on the UNMUTATED function — a test that does
    not hold on correct code.

    Only an ``AssertionError`` counts: it means the test's own expectation is wrong
    for the current code (a stale golden, or a real regression the test is catching).
    Other exceptions (a missing fixture arg, an import error) are ambiguous under the
    direct-call contract and are NOT flagged, to avoid false accusations. Such a test
    is surfaced for a human to investigate — never proposed for automatic deletion,
    since it may be the only thing catching a genuine bug."""
    if getattr(original_func, "__code__", None) is None:
        return []
    # TEST IDS, not `__name__` (issue #16, missed on the first pass). This is the per-function
    # path; `build_session_baseline` computes the same list for the live-session path and moved
    # to ids there. Both write `ProfilingResult.failing_tests`, so leaving one behind meant that
    # ONE field spoke a different vocabulary depending on which path produced it — and a
    # consumer matching it against the traced map (which is id-keyed) silently matched nothing.
    from Wesker.ci import callable_test_id  # local: ci imports lazily the other way

    failing: list[str] = []
    # Isolate each discovered test's own stdout/stderr (argparse usage banners from
    # a `pytest.raises(SystemExit)` CLI test, prints, logging) so consumer-test
    # side-effects never pollute the engine's machine-readable output — the same
    # isolation evaluate_mutant's runner applies.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        for test_fn in test_functions:
            try:
                test_fn()
            except AssertionError:
                failing.append(callable_test_id(test_fn))
            # BLE001,S110: ambiguous (fixtures/imports); not a wrong assertion. `Abandoned` is
            # named because it is a BaseException subclass by design; a genuine
            # KeyboardInterrupt/SystemExit now ends the baseline instead of being absorbed (S5754).
            except (Exception, Abandoned):  # noqa: BLE001,S110
                pass
    return failing


def _trace_one_multi(
    test_fn: Callable[..., None],
    target_files: set[str],
    budget_s: float | None = None,
    capture_arcs: bool = False,
    reach: list[str] | None = None,
) -> tuple[dict[str, set[int]], bool, bool, dict[str, set[tuple[int, int]]]]:
    """Every line ``test_fn()`` executes in ANY of ``target_files``: ``({file: lines}, cut, contained, {file: arcs})``.

    Identical machinery to :func:`_trace_one` — the dispatch already decides per FRAME
    whether to trace, so watching N files costs the same single pass as watching one.
    Nothing is intersected with a function's executable lines here: that filter is the
    only per-function part, and it is a set operation over data already in hand.

    ``budget_s`` bounds this one test (see :func:`_trace_one` for the rationale and the
    outside-the-interpreter boundary). The third element reports whether the budget CUT this
    test, so the caller can name it: a truncated trace under-reports coverage, which reads
    downstream as "no test reaches this line" — a silent cap would turn a timing accident into a
    false completeness verdict. Suite-wide this matters more than per-function, since one heavy
    test stalls the single shared baseline every function then reuses.

    ``capture_arcs`` (#17): also record the ARC each line event completes — the ``(prev_line,
    cur_line)`` transition WITHIN a frame — so a consumer can distinguish the two edges of a
    conditional (``if x: a`` reaches ``a`` via one arc, the ``else`` body via another) that line
    coverage alone collapses. OFF by default and gated because it roughly doubles the per-line
    callback work on the engine's hottest path; a caller that needs branch obligations opts in and
    pays for a fresh trace (arc runs bypass the line cache), while ordinary profiling is untouched.
    The arc map is empty when ``capture_arcs`` is False.

    Threads the test starts are traced into the same maps while it runs (#17); ``reach`` receives
    ``incomplete_thread`` when one outlived the run. The maps returned are SNAPSHOTS: such a thread
    stops contributing at its next event, and the caller must not iterate a map it may still touch.
    """
    from Wesker.monitoring import (
        MonitoringObservationFailed,
        allocate_run_token,
        registered_thread,
        retire_run_token,
        subscribe,
    )

    hits: dict[str, set[int]] = {}
    arcs: dict[str, set[tuple[int, int]]] = {}
    match = _target_matcher(target_files)
    frame_last: dict[int, int] = {}

    alive_before = set(threading.enumerate())
    preexisting_hit: list[int] = []
    worker_holder: list[threading.Thread] = []
    token = allocate_run_token()

    def _owns_thread(thread_id: int) -> bool:
        if worker_holder and thread_id == worker_holder[0].ident:
            return True
        # current_thread() creates a DummyThread before bootstrap/after teardown.
        # Read CPython's existing registration without manufacturing an observer.
        registered = registered_thread(thread_id)
        return registered is not None and registered not in alive_before

    def _is_preexisting_thread(thread_id: int) -> bool:
        # A thread that was already alive when this test began, other than the one running it.
        registered = registered_thread(thread_id)
        return registered is not None and registered in alive_before

    def _on_foreign_line(
        code: CodeType,
        lineno: int,
        frame: Any = None,
        run_token: int | None = None,
    ) -> None:
        # (#17) WITNESS only: a pre-existing thread cannot be attributed to THIS test (a pool
        # warmed in a fixture serves whichever test submits), so its lines never enter the
        # test's reach. That it ran target code during the window is recorded, so the row can
        # say its reach is unknown instead of complete-and-empty.
        filename = code.co_filename
        if filename in target_files or match(filename) is not None:
            preexisting_hit.append(lineno)

    def _on_line(
        code: CodeType,
        lineno: int,
        frame: Any = None,
        run_token: int | None = None,
    ) -> None:
        filename = code.co_filename
        matched = filename if filename in target_files else match(filename)
        if matched is None:
            return
        hits.setdefault(matched, set()).add(lineno)
        if capture_arcs and frame is not None:
            fid = id(frame)
            prev = frame_last.get(fid)
            if prev is not None and prev != lineno:
                arcs.setdefault(matched, set()).add((prev, lineno))
            frame_last[fid] = lineno

    def _on_return(
        code: CodeType,
        offset: int,
        retval: Any,
        frame: Any = None,
        run_token: int | None = None,
    ) -> None:
        if capture_arcs and frame is not None:
            frame_last.pop(id(frame), None)

    def _on_unwind(
        code: CodeType,
        offset: int,
        exc: BaseException,
        frame: Any = None,
        run_token: int | None = None,
    ) -> None:
        if capture_arcs and frame is not None:
            frame_last.pop(id(frame), None)

    spawned: list[int] = []
    _install_spawn_hook()
    _spawn_watchers[token] = (_owns_thread, spawned)
    handles = []
    try:
        handles.append(
            subscribe(
                [],
                sys.monitoring.events.LINE,
                _on_foreign_line,
                run_token=token,
                selectors=tuple(target_files),
                thread_filter=_is_preexisting_thread,
            )
        )
        handles.append(
            subscribe(
                [],
                sys.monitoring.events.LINE,
                _on_line,
                run_token=token,
                selectors=tuple(target_files),
                thread_filter=_owns_thread,
            )
        )
        if capture_arcs:
            handles.append(
                subscribe(
                    [],
                    sys.monitoring.events.PY_RETURN,
                    _on_return,
                    run_token=token,
                    selectors=tuple(target_files),
                    thread_filter=_owns_thread,
                )
            )
            handles.append(
                subscribe(
                    [],
                    sys.monitoring.events.PY_UNWIND,
                    _on_unwind,
                    run_token=token,
                    selectors=tuple(target_files),
                    thread_filter=_owns_thread,
                )
            )
    except BaseException:
        _spawn_watchers.pop(token, None)
        retire_run_token(token)
        for h in handles:
            h.close()
        raise

    def _worker() -> None:
        worker_holder.append(threading.current_thread())
        try:
            test_fn()
        except (Exception, Abandoned, MeasurementRefused):  # noqa: BLE001, S110
            pass
        except (KeyboardInterrupt, SystemExit):
            raise
        # BLE001: pytest's own outcomes (`Skipped`, `XFailed`, `Failed`) derive from BaseException; a
        # test that raises one still contributes the lines it reached, and an unhandled one would
        # escape this thread as a thread-exception warning that pytest turns into a failure (#17)
        except BaseException:  # noqa: BLE001, S110
            pass
        finally:
            if reach is not None:
                code = reach_qualification(
                    any(
                        t not in alive_before and t is not threading.current_thread()
                        for t in threading.enumerate()
                    ),
                    bool(preexisting_hit),
                    bool(spawned),
                )
                if code != "complete":
                    reach.append(code)

    thread = JoinedWorker(target=_worker, daemon=True)
    try:
        thread.start()
        truncated, contained = bounded_join(
            thread, budget_s if budget_s and budget_s > 0 else None
        )
    finally:
        _spawn_watchers.pop(token, None)
        retire_run_token(token)
        for h in handles:
            h.close()

    for h in handles:
        try:
            h.check()
        except MonitoringObservationFailed as exc:
            # Non-quiescence is already explicit in this inadmissible result.
            # Preserve other observation failures rather than returning partial proof.
            if contained or exc.code != "observation_inflight":
                raise

    return (
        {f: set(lines) for f, lines in dict(hits).items()},
        truncated,
        contained,
        {f: set(edges) for f, edges in dict(arcs).items()},
    )


def trace_suite(
    test_functions: list[Callable[..., None]],
    target_files: set[str],
    budget_s: float | None = None,
    truncated: set[str] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    session_budget_s: float | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    uncontained: set[str] | None = None,
    arcs_out: dict[str, dict[str, set[tuple[int, int]]]] | None = None,
    replayed: set[str] | None = None,
    within_run: dict[str, dict[str, Any]] | None = None,
    incomplete_reach: dict[str, str] | None = None,
) -> dict[str, dict[str, set[int]]]:
    """Trace the WHOLE suite ONCE: ``{test_id: {file: lines}}``.

    WHY: ``trace_line_coverage`` traces the entire suite and then keeps only one
    function's lines, so profiling F functions traced the suite F times to answer F
    questions that one pass already answers. That is ``O(suite × functions)`` before a
    single mutant runs — invisible on a 0.3s suite, ruinous on a ten-minute one
    (measured: 28.6s of baseline per function on a 445-test suite, 89% of wall clock).

    The trace is function-INDEPENDENT: what a test executes does not depend on which
    function we intend to mutate. Only the final intersection with a function's
    executable lines is per-function, and that is free. So this is the same
    "refuse work that provably cannot change a result" reduction the engine already
    applies to mutants, applied to the baseline.

    Keyed by ``ci.callable_test_id`` — the pytest nodeid, or a namespaced ``legacy:`` id
    (issue #16). It was keyed by ``__name__``, which is NOT unique: two tests in different
    modules sharing a name, and every parametrized case of one test, collapsed onto one
    entry, and the union below existed to stop that collision LOSING an owner's coverage.
    Under a per-item id the collision cannot arise, so the union is now a structural
    no-op retained only for a backend that yields one id twice — it compensates for
    nothing. What this buys is the thing the union could not: an entry can name WHICH
    pytest item observed it, which is what a proof basis has to be able to say.

    ``budget_s`` bounds EACH test's tracing (None = unbounded = the historical behavior). This is
    the one place a budget is load-bearing rather than defensive: the whole point of tracing once
    is that every function reuses this baseline, so a single heavy test does not stall one
    profile — it stalls the session, before any mutant runs, with no output. Names of tests the
    budget cut are added to ``truncated`` when a set is passed, so the caller can report them:
    their coverage is under-counted, and unreported that reads as a real completeness gap.

    ``progress(done, total, elapsed_ms)`` is called per test, with the SAME signature the mutation
    loop's callback uses, so one reporter serves both phases. This phase is where a big suite
    spends its wall clock (the 89% above) and it runs BEFORE the first mutant — so with no
    callback here the mutation progress cannot print yet, and the engine is silent at exactly its
    slowest moment. Bounding the work made that silence finite; only reporting makes it legible.

    ``session_budget_s`` bounds the WHOLE pass, not one test. The two are independent limits: a
    per-test cap × N tests is still unbounded in aggregate, which on a large suite is a difference
    that matters (2000 tests × a 50s cap is a day). Once it is spent the remaining tests are left
    untraced and are named in ``truncated`` with everything else — they are under-counted for the
    same reason and must not read as covered.

    ``cache`` is ``{fingerprint: {file: lines}}`` (see :mod:`Wesker.trace_cache`) and carries the
    reduction above ACROSS invocations, which is where the O(suite) above actually lives. Hoisting
    made the trace once-per-SESSION; a CLI takes one target per session, so the hoist never paid —
    every command re-measured the same function-independent constant from zero. Measured on
    Regenesis: an 11-minute baseline for one function, and 11 minutes again for the next function
    in the same file. On LintGate's 14,000 tests it is hours, per target.

    Keyed per TEST, not per suite: a consumer whose product is writing tests (converge) adds one
    file, and only that file's entries may miss. One digest over the whole suite would void 14,000
    entries because one arrived — the total-invalidation defect `LazySessionBaseline.refresh`
    exists to avoid, reintroduced a layer down.

    Passed dict is MUTATED with the misses, so the caller persists the union without a second
    walk. A CUT trace is never stored: it is under-counted by construction, and a cache that
    remembers a truncation makes a timing accident permanent.

    ``within_run`` is a WITHIN-SESSION trace memo — same ``{fingerprint: cell}`` shape as ``cache``,
    but holding ONLY traces measured in THIS live session (a prior seed/widen pass). It is checked
    BEFORE ``cache`` and, on a hit, the reuse is ADMISSIBLE — the test is NOT marked ``replayed``,
    because it WAS measured this session; only the source-keyed DISK ``cache`` earns ``replayed``
    (#20, which guards against a cross-run helper edit the fingerprint cannot see). This is what lets
    a converge run — many profile passes over ONE unchanging target — trace each slow covering test
    ONCE instead of re-tracing it every pass, with no loss of proof admissibility. It must be created
    fresh per session (never a module global), or it degrades to exactly the stale cross-run cache
    the ``fresh`` bypass exists to refuse. Like ``cache``, a CUT trace is never stored in it.

    ``incomplete_reach`` (#17) collects ``{test_id: code}`` for every item whose reach the tracer
    could not fully observe (``incomplete_thread``: a thread it started outlived its run). Such a
    trace is stored in neither cache, for the reason a cut one is not: a replayed cell carries no
    such qualification, so a warm run would read the item's unobserved lines as never reached.
    """
    # Local imports: `ci` and `trace_cache` both import lazily in the other direction, and
    # the identity accessor must be the CONTRACT one — a raw attribute read here is exactly
    # the divergence issue #16 exists to remove.
    from Wesker.ci import callable_test_id
    from Wesker.trace_cache import test_fingerprint

    out: dict[str, dict[str, set[int]]] = {}
    if not target_files:
        return out
    total = len(test_functions)
    started = time.monotonic()
    session_deadline = (
        started + session_budget_s
        if session_budget_s and session_budget_s > 0
        else None
    )
    for i, test_fn in enumerate(test_functions):
        name = callable_test_id(test_fn)
        remaining = (
            session_deadline - time.monotonic()
            if session_deadline is not None
            else None
        )
        if remaining is not None and remaining <= 0:
            # Out of session budget: the REST go untraced. Name them — an untraced test's
            # coverage is absent, not zero, and the two are indistinguishable downstream.
            # Identified the same way as the traced ones: a `truncated` entry that could not
            # be matched against a `traced` key would report the cut against nothing.
            if truncated is not None:
                truncated.update(callable_test_id(t) for t in test_functions[i:])
            break
        test_budget = budget_s
        if remaining is not None:
            test_budget = (
                min(budget_s, remaining) if budget_s and budget_s > 0 else remaining
            )
        # Arcs travel WITH lines in the v4 cache cell (#17): `{file: {"lines": [...], "arcs":
        # [[a,b],...]}}`. Capture is ALWAYS on so a warm cache is always complete — bypassing the
        # cache to obtain arcs would re-trace the whole suite every session, the 11-minute cost
        # this cache exists to remove. Arcs are only EXPOSED when a caller passes `arcs_out`; the
        # cost of carrying them in the cell is the branch obligations #17 owns.
        fp = (
            test_fingerprint(test_fn)
            if (cache is not None or within_run is not None)
            else None
        )
        # WITHIN-SESSION memo first: a hit here was measured THIS session (a prior pass), so its reuse
        # is admissible. The source-keyed DISK cache is only consulted on a within-run miss, and only
        # IT earns `replayed` (#20).
        wr_hit = (
            within_run.get(fp) if (within_run is not None and fp is not None) else None
        )
        hit = (
            wr_hit
            if wr_hit is not None
            else (cache.get(fp) if (cache is not None and fp is not None) else None)
        )
        arc_file: dict[str, set[tuple[int, int]]] = {}
        if hit is not None:
            # Measured before, by this engine, on these target files, under these budgets. The
            # trace is function-independent, so re-running it cannot produce a different answer —
            # only the same one, slower. Cell is the v4 dict; `.get` on `arcs` tolerates a cell an
            # older path may have written without one.
            per_file = {f: set(cell["lines"]) for f, cell in hit.items()}
            arc_file = {
                f: {tuple(a) for a in cell.get("arcs", ())} for f, cell in hit.items()
            }
            was_cut = False
            # REPLAYED, not measured this session (#20): a DISK-cache hit came from a prior run, keyed
            # by source only, so the proof view must refuse it while routing still uses it. A
            # within-run hit is NOT replayed — it was measured this session and stays admissible.
            if wr_hit is None and replayed is not None:
                replayed.add(name)
        else:
            # The redirect isolates the TEST's own stdout/stderr and must wrap the test ONLY —
            # not the loop. Wrapped around the loop it also swallows `progress`, which reports on
            # stderr: the callback fires, writes into the StringIO, and the phase stays silent
            # exactly as if nothing were reporting at all.
            reach: list[str] = []
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                per_file, was_cut, contained, arc_file = _trace_one_multi(
                    test_fn, target_files, test_budget, capture_arcs=True, reach=reach
                )
            # See `trace_line_coverage`: an unstoppable worker is not an ordinary cut (#19).
            if not contained and uncontained is not None:
                uncontained.add(name)
            if reach and incomplete_reach is not None:
                incomplete_reach[name] = reach[0]
            if fp is not None and not was_cut and not reach:
                # NOT when cut: a truncated trace is under-counted, and downstream that is
                # indistinguishable from "no test reaches this line". Storing it would make one
                # slow afternoon a permanent false gap. NOT when its reach is incomplete either
                # (#17): the cell has no field to say so, so a replay would turn "not observed" into
                # "not reached". One cell serves both stores: the disk cache (cross-run routing
                # evidence) and the within-session memo (this run's later passes).
                cell = {
                    f: {
                        "lines": sorted(per_file[f]),
                        "arcs": sorted(list(a) for a in arc_file.get(f, ())),
                    }
                    for f in per_file
                }
                if within_run is not None:
                    within_run[fp] = cell
                if cache is not None:
                    cache[fp] = cell
        if was_cut and truncated is not None:
            truncated.add(name)
        bucket = out.setdefault(name, {})
        for f, lines in per_file.items():
            bucket[f] = bucket.get(f, set()) | lines
        if arcs_out is not None:
            abucket = arcs_out.setdefault(name, {})
            for f, aset in arc_file.items():
                abucket[f] = abucket.get(f, set()) | aset
        if progress is not None:
            progress(i + 1, total, (time.monotonic() - started) * 1000.0)
    return out


def admissible_coverage(
    observed: dict[str, list[int]], inadmissible: list[str]
) -> dict[str, list[int]]:
    """The PROOF view of a coverage map: observed reach minus the entries whose owner may not
    discharge an obligation (issue #17).

    Two views, deliberately, and they must not be collapsed:

    * OBSERVED is what `_build_test_scope` scopes mutants with. Routing wants it conservative —
      including a test that cannot kill costs a little time, excluding one that can turns a
      real kill into a reported gap. A baseline-failing test still REACHES the line, so it
      still belongs there.
    * ADMISSIBLE is what a completeness claim may rest on. The same failing test proves nothing
      about the line it touched, because it fails on the unmutated program too.

    Returning a filtered MAP rather than a union is the point of #17's "do not union before
    outcome qualification": once the per-test structure is flattened, the reason an entry was
    dropped is gone, and a consumer can no longer name which item owns an obligation — which is
    exactly what a proof basis has to be able to do.

    Entries are dropped whole, never emptied to ``[]``. An empty list is a positive claim that
    the test reached nothing; absence says the engine holds no admissible observation from it.
    """
    barred = set(inadmissible)
    return {tid: lines for tid, lines in observed.items() if tid not in barred}


def coverage_from_trace(
    traced: dict[str, dict[str, set[int]]], target_file: str, exec_lines: set[int]
) -> dict[str, list[int]]:
    """One function's view of a :func:`trace_suite` result — the per-function filter.

    Returns exactly what :func:`trace_line_coverage` would have returned for that
    function, so it is a drop-in for callers holding a suite trace.
    """
    if not target_file or not exec_lines:
        return {}

    # A persisted suite trace may key this file under another same-file spelling
    # (case-insensitive filesystem, symlink, or a cache written by a caller who typed it
    # differently). WHICH spellings those are is a property of the trace, not of any one
    # test, so resolve it ONCE here. Doing it inside the per-test lookup meant every test
    # that did not hit the target — i.e. nearly all of them, which is the normal shape of a
    # scoped run — walked that test's whole key set before returning empty. Spellings are
    # never rewritten; the alias only ever adds a place to look.
    aliases: list[str] = []
    try:
        st = os.stat(target_file)
    except OSError:
        st = None
    if st is not None:
        target_id = (st.st_dev, st.st_ino)
        checked: set[str] = {target_file}
        for files in traced.values():
            for key in files:
                if key in checked:
                    continue
                checked.add(key)
                # One stat per distinct key, against an identity read once above —
                # `samefile` would re-stat the target for every comparison.
                try:
                    kst = os.stat(key)
                except OSError:
                    continue
                if (kst.st_dev, kst.st_ino) == target_id:
                    aliases.append(key)

    def _lines_for(files: dict[str, set[int]]) -> frozenset[int] | set[int]:
        got = files.get(target_file)
        if got is not None:
            return got
        for key in aliases:  # empty in the ordinary case — one spelling, one file
            got = files.get(key)
            if got is not None:
                return got
        return frozenset()

    return {
        name: sorted(_lines_for(files) & exec_lines) for name, files in traced.items()
    }


def arcs_from_trace(
    traced_arcs: dict[str, dict[str, set[tuple[int, int]]]],
    target_file: str,
    exec_lines: set[int],
) -> dict[str, list[tuple[int, int]]]:
    """One function's view of a :func:`trace_suite` ARC result (#17) — the per-function filter.

    The arc sibling of :func:`coverage_from_trace`: same file-identity resolution (a persisted
    trace may key the target under a symlink/case spelling), then keep only the arcs BOTH of whose
    endpoints lie in the function's executable lines. A cross-boundary arc — into or out of the
    function via a call or return — is not one of ITS branch edges, so it is dropped rather than
    attributed to the function it merely entered or left.
    """
    if not target_file or not exec_lines:
        return {}
    aliases: list[str] = []
    try:
        st = os.stat(target_file)
    except OSError:
        st = None
    if st is not None:
        target_id = (st.st_dev, st.st_ino)
        checked: set[str] = {target_file}
        for files in traced_arcs.values():
            for key in files:
                if key in checked:
                    continue
                checked.add(key)
                try:
                    kst = os.stat(key)
                except OSError:
                    continue
                if (kst.st_dev, kst.st_ino) == target_id:
                    aliases.append(key)

    def _arcs_for(files: dict[str, set[tuple[int, int]]]) -> set[tuple[int, int]]:
        got = files.get(target_file)
        if got is not None:
            return got
        for key in aliases:
            got = files.get(key)
            if got is not None:
                return got
        return set()

    return {
        name: sorted(
            arc
            for arc in _arcs_for(files)
            if arc[0] in exec_lines and arc[1] in exec_lines
        )
        for name, files in traced_arcs.items()
    }


def trace_line_coverage(
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any],
    exec_lines: set[int],
    budget_s: float | None = None,
    truncated: set[str] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    session_budget_s: float | None = None,
    uncontained: set[str] | None = None,
    incomplete_reach: dict[str, str] | None = None,
) -> dict[str, list[int]]:
    """Map each test id to the target lines it covers, over the UNMUTATED function.

    Keyed by ``ci.callable_test_id`` (issue #16) — and the kill matrix's test VALUES are
    keyed the same way, because a caller runs set-cover over ``kill_matrix`` and this
    together and two different vocabularies would silently intersect to nothing. That
    coupling is the reason the two must move in one commit, not two. The target file is taken from
    the original function's own code object — authoritative and absolute, the same
    identity ``evaluate_mutant`` patches against — so coverage attributes to the
    real function under test, not a same-named sibling. Empty when the function's
    code object is unavailable (degrades to "no line data", never an error).

    ``budget_s`` bounds EACH test's tracing and ``truncated`` collects the names the budget cut
    (both default to the historical unbounded behavior); see :func:`_trace_one` for why partial
    coverage is the right thing to keep, and for the outside-the-interpreter boundary.
    ``session_budget_s`` bounds the whole pass (a per-test cap × N tests is not an aggregate
    bound), and ``progress(done, total, elapsed_ms)`` reports per test in the same shape the
    mutation loop uses — this pass runs BEFORE the first mutant, so without it the engine is
    silent through the part that costs the most. See :func:`trace_suite`, which does both the same
    way for the suite-global pass — and collects ``incomplete_reach`` the same way (#17).
    """
    code = getattr(original_func, "__code__", None)
    target_file = getattr(code, "co_filename", None)
    if not target_file or not exec_lines:
        return {}
    from Wesker.ci import callable_test_id  # local: ci imports lazily the other way

    coverage: dict[str, list[int]] = {}
    total = len(test_functions)
    started = time.monotonic()
    session_deadline = (
        started + session_budget_s
        if session_budget_s and session_budget_s > 0
        else None
    )
    for i, test_fn in enumerate(test_functions):
        name = callable_test_id(test_fn)
        remaining = (
            session_deadline - time.monotonic()
            if session_deadline is not None
            else None
        )
        if remaining is not None and remaining <= 0:
            if (
                truncated is not None
            ):  # the rest go untraced — say so, never imply "covered"
                truncated.update(callable_test_id(t) for t in test_functions[i:])
            break
        test_budget = budget_s
        if remaining is not None:
            test_budget = (
                min(budget_s, remaining) if budget_s and budget_s > 0 else remaining
            )
        # Isolate consumer-test stdout/stderr during the traced baseline pass (see
        # failing_on_baseline) so a test's prints/argparse banners never leak into the
        # engine's output. Wraps the TEST, not the loop: around the loop it ALSO swallows
        # `progress` (which reports on stderr), so the callback fires into a StringIO and the
        # phase stays as silent as if nothing were reporting — the bug this progress exists to fix.
        reach: list[str] = []
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            covered, was_cut, contained = _trace_one(
                test_fn, target_file, exec_lines, test_budget, reach
            )
        if was_cut and truncated is not None:
            truncated.add(name)
        if reach and incomplete_reach is not None:
            incomplete_reach[name] = reach[0]
        # A cut that could not be STOPPED is a different fact from a cut (#19): the worker is
        # still running, so every later measurement in this process shares it. Reported by name
        # for the same reason `truncated` is — silently folding it in makes a live runaway
        # indistinguishable from an ordinary budget trim.
        if not contained and uncontained is not None:
            uncontained.add(name)
        # Union across duplicate test names (parametrized cases share a __name__).
        merged = set(coverage.get(name, ())) | covered
        coverage[name] = sorted(merged)
        if progress is not None:
            progress(i + 1, total, (time.monotonic() - started) * 1000.0)
    return coverage
