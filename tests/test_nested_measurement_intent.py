"""Intent pins for the engine-inside-the-engine lock cycle: refused by name, never waited into (#28).

THE DEFECT, measured 2026-09-27 (Detective `docs/ENGINEERING_PASS_2026-09-26.md` §7). The execution
lock is held by the thread running `evaluate_mutant` for the WHOLE evaluation, and that thread joins
the worker running each test. A test that itself runs the engine in-process — most of Wesker's own
engine tests do — makes the worker's nested acquire wait on the holder while the holder waits on the
worker. A wait-for cycle. The holder's timeout fires, `abandon` cannot land on a thread parked in a
C-level acquire, the holder returns a FALSE `timeout` kill marked uncontained and releases, the
worker's acquire then succeeds, the pending injection lands before the ownership record is written,
and the worker dies OWNING the lock. Every later evaluation reads `orphaned` and the run is refused.
Reproduced against the pre-fix code with exactly the shape `_engine_running_test` below has: outer
result `killed=True, killed_by="timeout", contained=False`; the lock repr naming a dead owner; the
next acquire refused `orphaned`. The same cycle forms under the BASELINE guard, which holds the lock
while its workers run each test.

Detection was sound; prevention was missing. The fix (D1 of the issue) recognises the start/join edge
BEFORE any blocking acquire and raises a named control exception instead of waiting, and carries the
status on the worker itself — because a BaseException alone is caught twice on its way out (pytest's
call wrapper, then `_run_test_with_timeout`'s own `except BaseException`, which reads it "crash").

Each schedule below is forced by construction or by events, never by sleeps: the outer evaluation
holds the lock by definition while it runs a test, so the nested acquire always meets a held lock.
These are written from what the engine must do; a generated suite would pin today's answer, right
or wrong.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import threading

import pytest

from Wesker import engine as E
from Wesker.engine import (
    _SESSION_BASELINE,
    SCORED_DISPOSITIONS,
    LazySessionBaseline,
    MutationCategory,
    _acquire_execution_lock,
    _baseline_failures,
    _execution_guard,
    _run_test_with_timeout,
    baseline_probe_disposition,
    build_session_baseline,
    evaluate_mutant,
    generate_mutants,
    mutant_disposition,
    nested_acquire_disposition,
    run_function_converged,
    run_function_profiling,
)
from Wesker.filter import filter_categories
from Wesker.interrupt import (
    JoinedWorker,
    MeasurementRefused,
    mark_refusal,
    starter_chain,
)

_SRC = "def compute(n):\n    return n * 2\n"

MAIN, WORKER, OTHER = 1000, 2000, 3000


# THE PURE DECISION FIRST, deliberately. When this file is consulted to pin the decision itself,
# its mutants are installed in the engine that evaluates them, and a mutant that disables the
# refusal makes every engine-running test below wait out its allowance; tests run in file order,
# so the assertions that kill those mutants must come before anything that takes the lock.

# ── the pure decision (#28, pinned) ──────────────────────────────────────────────────────────


def test_owning_the_lock_is_reentry_whatever_the_chain_says():
    assert nested_acquire_disposition(WORKER, 0, WORKER, (MAIN,)) == "reentrant"
    assert nested_acquire_disposition(0, WORKER, WORKER, (MAIN,)) == "reentrant"


def test_an_upstream_owner_is_a_cycle_on_either_channel():
    """The record and the repr are independent: the measured orphan lived exactly in the window
    where the record is blind, so the repr alone must suffice — and vice versa."""
    assert nested_acquire_disposition(MAIN, 0, WORKER, (MAIN,)) == "nested_measurement"
    assert nested_acquire_disposition(0, MAIN, WORKER, (MAIN,)) == "nested_measurement"
    assert (
        nested_acquire_disposition(MAIN, 0, WORKER, (OTHER, MAIN))
        == "nested_measurement"
    ), "two joins upstream is still upstream"


def test_a_free_lock_or_an_unrelated_owner_is_an_ordinary_acquire():
    assert nested_acquire_disposition(0, 0, WORKER, (MAIN,)) == "acquire"
    assert nested_acquire_disposition(OTHER, OTHER, WORKER, (MAIN,)) == "acquire"
    assert nested_acquire_disposition(MAIN, MAIN, WORKER, ()) == "acquire", (
        "no chain, no cycle the edge can show — the existing bounds decide"
    )


def test_zero_is_never_matched_as_a_thread():
    """0 means "no owner recorded"; an upstream chain cannot be made to contain it by accident."""
    assert nested_acquire_disposition(0, 0, WORKER, (0,)) == "acquire"


def test_owning_the_lock_outranks_even_an_upstream_record():
    """Self-ownership is decided first: a caller that holds the lock waits for nothing, so no
    cycle can exist whatever else the channels say (a chain cannot contain its own head, but the
    decision must not depend on that)."""
    assert nested_acquire_disposition(MAIN, MAIN, MAIN, (MAIN,)) == "reentrant"


def test_a_record_naming_a_stranger_does_not_hide_an_upstream_repr_owner():
    """The two channels disagreeing is exactly the measured window (the record stale or unwritten,
    the repr truthful): either one naming an upstream owner is the cycle."""
    assert (
        nested_acquire_disposition(OTHER, MAIN, WORKER, (MAIN,)) == "nested_measurement"
    )


def test_a_stranger_owner_with_no_readable_repr_is_an_ordinary_acquire():
    assert nested_acquire_disposition(OTHER, 0, WORKER, (MAIN,)) == "acquire"


def test_a_refusal_is_its_own_disposition_in_the_phase_order():
    """Construction and installation come first (nothing ran); the refusal comes before entry and
    containment, because the refused test may be the one that would have entered the mutant."""
    nm = "nested_measurement"
    assert mutant_disposition(True, True, None, True, False, nm) == nm
    assert mutant_disposition(True, True, False, True, False, nm) == nm, (
        "ahead of not_entered"
    )
    assert mutant_disposition(True, True, True, False, False, nm) == nm, "ahead of cut"
    assert mutant_disposition(False, True, None, True, False, nm) == "harness_error"
    assert mutant_disposition(True, False, None, True, False, nm) == "not_installed"


def test_a_kill_outranks_a_refusal():
    """A kill another test earned is a verdict on its own; the refusal withheld nothing."""
    assert (
        mutant_disposition(True, True, True, True, True, "nested_measurement")
        == "killed_after_entry"
    )


def test_without_a_refusal_the_precedence_is_unchanged():
    table = {
        (False, False, False, False, False): "harness_error",
        (True, False, False, False, False): "not_installed",
        (True, True, False, False, False): "not_entered",
        (True, True, True, False, False): "cut",
        (True, True, True, True, True): "killed_after_entry",
        (True, True, True, True, False): "survived_after_entry",
        # Unobserved entry (no test ran) stays a scored survivor — the #18 rule, unchanged.
        (True, True, None, True, False): "survived_after_entry",
    }
    for args, expected in table.items():
        assert mutant_disposition(*args) == expected
        assert mutant_disposition(*args, "") == expected


def test_the_baseline_probe_names_a_refusal_as_its_own_fourth_state():
    """Neither usable (not a pass) nor inert (never observed to fail): filing it inert would bar the
    test and leave every mutant only it covers to be scored a plain survivor."""
    assert baseline_probe_disposition(None) == "usable"
    assert baseline_probe_disposition("uncontained") == "uncontained"
    assert baseline_probe_disposition("nested_measurement") == "nested_measurement"
    for outcome in ("assertion", "exception", "crash", "timeout", "", "unheard_of"):
        assert baseline_probe_disposition(outcome) == "inert"


@pytest.fixture
def target(tmp_path):
    """A real module on disk under its own name, so the module-qualified patch can match it."""
    path = tmp_path / "nested_mod.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("nested_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nested_mod"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod, str(path)
    finally:
        sys.modules.pop("nested_mod", None)


def _mutants():
    node = ast.parse(_SRC).body[0]
    return generate_mutants(
        node, filter_categories(node, True), max_per_category=0, pass_index=0
    )


class _RecordingLock:
    """Forwards to the real RLock and records WHICH thread asked to acquire it.

    The structural witness for "never entered the C wait": a refused thread must not appear here
    at all. The repr is forwarded verbatim, so the repr-owner channel reads exactly what it would.
    """

    def __init__(self, inner):
        self._inner = inner
        self.acquirers: list[int] = []

    def acquire(self, *args, **kwargs):
        self.acquirers.append(threading.get_ident())
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        return self._inner.release()

    def _is_owned(self):
        return self._inner._is_owned()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def __repr__(self):
        return repr(self._inner)


@pytest.fixture
def recording_lock(monkeypatch):
    lock = _RecordingLock(E._EXECUTION_LOCK)
    monkeypatch.setattr(E, "_EXECUTION_LOCK", lock)
    return lock


def _engine_running_test(mod, path, seen: dict | None = None):
    """A test that drives the engine in-process before it reaches the target, as Wesker's own
    engine tests do. It records its own thread so a test can ask what that thread did."""
    inner = _mutants()[1]

    def plain():
        assert mod.compute(2) == 4

    def test_runs_the_engine():
        if seen is not None:
            seen["thread"] = threading.get_ident()
        evaluate_mutant(
            inner, [plain], mod.compute, qualname="compute", source_path=path
        )
        assert mod.compute(1) == 2

    return test_runs_the_engine


def _no_live_joined_workers() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if isinstance(t, JoinedWorker)]


# ── the cycle, prevented ─────────────────────────────────────────────────────────────────────


def test_a_test_that_runs_the_engine_is_refused_by_name_never_killed(target):
    """THE regression. Before: a false `timeout` kill, uncontained. Now: neither killed nor
    survived — named, and kept out of the denominator under its own name."""
    mod, path = target
    outer = _mutants()[0]
    result = evaluate_mutant(
        outer,
        [_engine_running_test(mod, path)],
        mod.compute,
        timeout_ms=5000,
        qualname="compute",
        source_path=path,
    )
    assert result.killed is False, f"a refused measurement was scored a kill: {result}"
    assert result.killed_by is None
    assert result.contained is True, (
        "no worker was left parked, so nothing is uncontained"
    )
    assert result.refusal == "nested_measurement"
    assert result.refused_test and result.refused_test.endswith("test_runs_the_engine")
    disposition = mutant_disposition(
        result.constructed,
        result.installed,
        result.entered,
        True,
        result.killed,
        result.refusal,
    )
    assert disposition == "nested_measurement"
    assert disposition not in SCORED_DISPOSITIONS, (
        "a refusal must never read as survival"
    )


def test_the_refused_worker_never_enters_the_lock_wait(target, recording_lock):
    """Prevention, not detection: the worker never ASKS the lock. A thread that never calls
    `acquire` cannot be parked in it, so no injection can be pending when it returns."""
    mod, path = target
    seen: dict = {}
    evaluate_mutant(
        _mutants()[0],
        [_engine_running_test(mod, path, seen)],
        mod.compute,
        qualname="compute",
        source_path=path,
    )
    assert "thread" in seen, "the engine-running test never ran"
    assert threading.get_ident() in recording_lock.acquirers, (
        "the outer evaluation must hold the lock — the precondition of the cycle"
    )
    assert seen["thread"] not in recording_lock.acquirers, (
        "the measurement's own worker reached the C-level acquire: the cycle was entered"
    )


def test_after_a_refusal_patches_are_restored_no_worker_lives_and_the_lock_is_free(
    target,
):
    """The four after-states the issue names, all at once: the original body is back, no
    measurement worker is alive (none was abandoned), and the next acquire takes the lock at once
    (no orphan) — re-acquired by THIS thread inside a bound that a probe-length wait would blow."""
    mod, path = target
    evaluate_mutant(
        _mutants()[0],
        [_engine_running_test(mod, path)],
        mod.compute,
        qualname="compute",
        source_path=path,
    )
    assert mod.compute(3) == 6, "a mutant was left installed"
    assert _no_live_joined_workers() == []
    assert E._lock_owner_from_repr() == 0, "the lock is still owned by someone"
    _acquire_execution_lock(probe_s=0.5, wait_s=0.0)
    E._EXECUTION_LOCK.release()


def test_a_broad_except_in_the_nested_engine_cannot_swallow_the_refusal(target):
    """`except Exception` cannot catch it (a BaseException); `except BaseException` can swallow the
    EXCEPTION but not the RECORD. Here the test swallows it outright and then passes — under a
    channel that rode on the exception this would be a clean pass, i.e. survival evidence."""
    mod, path = target
    inner = _mutants()[1]

    def plain():
        assert mod.compute(2) == 4

    caught: dict = {}

    def broad_exception_handler():
        try:
            evaluate_mutant(
                inner, [plain], mod.compute, qualname="compute", source_path=path
            )
        # BLE001: the broad handler is what is under test
        except Exception as exc:  # noqa: BLE001
            caught["exception"] = exc
        assert mod.compute(1) == 2

    def swallows_everything():
        try:
            evaluate_mutant(
                inner, [plain], mod.compute, qualname="compute", source_path=path
            )
        # BLE001: the broadest handler is what is under test
        except BaseException as exc:  # noqa: BLE001
            caught["base"] = exc
        assert mod.compute(1) == 2

    for test in (broad_exception_handler, swallows_everything):
        result = evaluate_mutant(
            _mutants()[0], [test], mod.compute, qualname="compute", source_path=path
        )
        assert result.killed is False
        assert result.refusal == "nested_measurement", (
            f"{test.__name__}: the refusal was lost — {result}"
        )
    assert "exception" not in caught, "`except Exception` caught a control exception"
    assert isinstance(caught.get("base"), MeasurementRefused)
    assert caught["base"].code == "nested_measurement"


def test_a_kill_by_another_test_outranks_the_refusal(target):
    """A refusal withholds a verdict only where nothing else gives one: an assertion kill by a
    different test is a full verdict on its own and must not be downgraded."""
    mod, path = target

    def kills():
        assert mod.compute(3) == 6

    killed = 0
    for mutant in _mutants():
        result = evaluate_mutant(
            mutant,
            [_engine_running_test(mod, path), kills],
            mod.compute,
            qualname="compute",
            source_path=path,
        )
        if result.killed:
            killed += 1
            assert result.refusal == "", "a refusal was carried on a kill"
            assert result.killed_by == "assertion"
    assert killed >= 1, "no mutant was killed, so the precedence was never exercised"


def test_a_refusal_two_joins_deep_reaches_the_outer_measurement(target):
    """The cycle through an intermediate measurement: the outer evaluation holds the lock, its
    worker starts its OWN bounded worker (unlocked), and THAT one runs a test that takes the lock.
    The owner is two joins upstream; the outer measurement must still learn of it."""
    mod, path = target

    def takes_the_lock():
        with _execution_guard():
            pass

    def runs_an_inner_measurement():
        # The intermediate level: not holding the lock, starting a bounded worker of its own.
        inner = _run_test_with_timeout(takes_the_lock, None, True, 5000)
        assert inner == "nested_measurement", inner
        assert mod.compute(1) == 2

    result = evaluate_mutant(
        _mutants()[0],
        [runs_an_inner_measurement],
        mod.compute,
        qualname="compute",
        source_path=path,
    )
    assert result.refusal == "nested_measurement", result
    assert _no_live_joined_workers() == []


# ── what must NOT change ─────────────────────────────────────────────────────────────────────


def test_same_thread_reentrancy_is_unchanged():
    """The RLock's own re-entry — every patch site under `evaluate_mutant`, and the isolated
    worker's hookwrapper — and the same inside a measurement worker that owns the lock itself."""
    with _execution_guard() as outer, _execution_guard() as inner:
        assert outer is not inner

    def owns_then_reenters():
        # This worker's starter does NOT hold the lock, so this is ordinary ownership.
        with _execution_guard(), _execution_guard(timeout_s=0.05):
            _acquire_execution_lock(probe_s=0.05, wait_s=0.0)
            E._EXECUTION_LOCK.release()

    assert _run_test_with_timeout(owns_then_reenters, None, True, 5000) is None


def test_an_unrelated_holder_is_waited_for_not_refused():
    """Serialisation between threads whose waits do not depend on each other is what the lock is
    FOR. A measurement worker whose chain does not contain the owner must wait and then get in."""
    held = threading.Event()
    asking = threading.Event()
    release = threading.Event()
    got_in = threading.Event()
    outcome: dict = {}

    def unrelated_holder():
        with _execution_guard():
            held.set()
            release.wait(timeout=30)

    def worker_body():
        asking.set()
        try:
            with _execution_guard():
                got_in.set()
        # BLE001: any refusal is recorded for the assertion below
        except BaseException as exc:  # noqa: BLE001
            outcome["exc"] = exc

    holder = threading.Thread(target=unrelated_holder)
    holder.start()
    assert held.wait(timeout=30)
    worker = JoinedWorker(target=worker_body)
    worker.start()
    assert asking.wait(timeout=30)
    # Deterministic, not timed: the holder still owns the lock, so the worker cannot be in.
    assert not got_in.is_set()
    release.set()
    worker.join(timeout=30)
    holder.join(timeout=30)
    assert "exc" not in outcome, f"an unrelated holder was refused: {outcome}"
    assert got_in.is_set()


# ── the baseline guard's workers ─────────────────────────────────────────────────────────────


def test_the_baseline_guard_refuses_instantly_and_keeps_the_test(
    target, recording_lock
):
    """The second site of the cycle: `_baseline_failures` holds the guard while its worker runs
    each test. The worker must be refused without waiting — and the test KEPT: neither inert (that
    would leave the mutants only it covers to be read as plain survivors) nor compromised (nothing
    was left running)."""
    mod, path = target
    seen: dict = {}
    engine_test = _engine_running_test(mod, path, seen)

    def plain():
        assert mod.compute(2) == 4

    inert, compromised = _baseline_failures(
        [engine_test, plain], mod.compute, "compute"
    )
    assert id(engine_test) not in inert
    assert compromised == set()
    assert seen["thread"] not in recording_lock.acquirers
    assert E._lock_owner_from_repr() == 0


# ── through the profiling paths, end to end ─────────────────────────────────────────────────


def _profile_node():
    return ast.parse(_SRC).body[0]


@pytest.mark.parametrize("path_name", ["profiling", "converged"])
def test_a_profile_whose_only_test_runs_the_engine_names_every_mutant(
    target, path_name
):
    """Both loops report the same quantity to the same consumers: every mutant unscored under
    `nested_measurement`, each named with the test to act on, none in the survivor records, and the
    profile not gateable — a floor with its reason, never `orphaned`, never a weak-suite verdict."""
    mod, path = target
    node = _profile_node()
    tests = [_engine_running_test(mod, path)]
    if path_name == "profiling":
        res = run_function_profiling(
            node,
            "nested_mod.py::compute",
            filter_categories(node, True),
            tests,
            mod.compute,
            max_per_category=0,
        )
    else:
        res = run_function_converged(
            node,
            "nested_mod.py::compute",
            filter_categories(node, True),
            tests,
            mod.compute,
            budget_ms=60000,
        )
    unscored = sum(
        cr.unscored_by.get("nested_measurement", 0) for cr in res.per_category
    )
    assert unscored >= 1, res.to_dict()
    assert res.total_killed == 0
    assert res.survivor_records == [], "a refused mutant was reported as a survivor"
    assert not res.is_gateable
    named = [
        r for r in res.unscored_records if r["disposition"] == "nested_measurement"
    ]
    assert len(named) == unscored
    assert all(r["test"].endswith("test_runs_the_engine") for r in named)
    assert res.to_dict()["unscored_records"] == res.unscored_records
    assert mod.compute(3) == 6
    assert _no_live_joined_workers() == []


_WIDEN_SRC = "def scoreit(a, b, flag):\n    if flag:\n        return a * 2 + b\n    return a - b\n"
_WIDEN_CATS = {
    MutationCategory.VALUE,
    MutationCategory.ARITHMETIC,
    MutationCategory.SWAP,
    MutationCategory.BOUNDARY,
}


def _widen_matrix(res):
    return {
        "total_mutants": res.total_mutants,
        "total_killed": res.total_killed,
        "kill_matrix": {m: sorted(k) for m, k in res.kill_matrix.items()},
        "survivors": sorted(r["mutant_id"] for r in res.survivor_records),
        "unscored": sorted(
            (r["mutant_id"], r["disposition"]) for r in res.unscored_records
        ),
    }


@pytest.mark.parametrize("runner", [run_function_profiling, run_function_converged])
def test_a_refused_mutant_stays_open_until_the_widen_finds_its_killer(runner):
    """Seed + widen must equal the full run (`test_widen_matches_full`'s oracle) with a refusal in
    the seed. Seeded with ONLY the engine-running test, every `flag=True` mutant is refused; the
    killer arrives as a widened unknown. If a refused mutant stopped being an open obligation, the
    widen would end — or skip it — and report `nested_measurement` where the full run reports a kill."""
    node = ast.parse(_WIDEN_SRC).body[0]
    ns: dict = {}
    # S102: test fixture source
    exec(compile(ast.parse(_WIDEN_SRC), "<nested-widen>", "exec"), ns)  # noqa: S102
    original = ns["scoreit"]

    def test_runs_the_engine():
        with _execution_guard():
            pass
        # F821: bound into this module's globals below, as `test_widen_matches_full` does
        assert scoreit(1, 2, True) == 4  # noqa: F821

    def test_true_kills():
        assert scoreit(1, 2, True) == 4  # noqa: F821

    def test_false_kills():
        assert scoreit(5, 3, False) == 2  # noqa: F821

    tests = [test_runs_the_engine, test_true_kills, test_false_kills]
    for t in tests:
        t.__globals__["scoreit"] = original
    target_files = {original.__code__.co_filename}

    def build(subset=None, fresh=False, **_kw):
        return build_session_baseline(
            tests if subset is None else list(subset), target_files
        )

    def run(holder, **kw):
        token = _SESSION_BASELINE.set(holder)
        try:
            return runner(
                node,
                f"{original.__code__.co_filename}::scoreit",
                _WIDEN_CATS,
                tests,
                original,
                max_per_category=0,
                **kw,
            )
        finally:
            _SESSION_BASELINE.reset(token)

    full = run(LazySessionBaseline(build))
    holder = LazySessionBaseline(build)
    holder.seed([test_runs_the_engine])
    seeded = run(holder, widen_tests=[test_true_kills, test_false_kills])

    assert full.total_killed >= 1
    assert not full.unscored_records, "the killer outranks the refusal in the full run"
    assert _widen_matrix(seeded) == _widen_matrix(full)


# ── the start/join edge itself ──────────────────────────────────────────────────────────────


def test_the_chain_names_every_live_joiner_nearest_first():
    seen: dict = {}

    def leaf():
        seen["chain"] = starter_chain()

    def middle():
        inner = JoinedWorker(target=leaf)
        inner.start()
        inner.join(timeout=30)

    mid_thread = JoinedWorker(target=middle)
    mid_thread.start()
    mid_thread.join(timeout=30)
    assert seen["chain"] == (mid_thread.ident, threading.get_ident())
    assert starter_chain() == (), "the main thread is nobody's worker"


def test_a_dead_starter_ends_the_chain():
    """A finished starter waits on nothing, and its ident may already be someone else's."""
    go = threading.Event()
    seen: dict = {}
    holder: dict = {}

    def orphan_worker():
        go.wait(timeout=30)
        seen["chain"] = starter_chain()

    def short_lived_starter():
        holder["t"] = JoinedWorker(target=orphan_worker)
        holder["t"].start()

    starter = threading.Thread(target=short_lived_starter)
    starter.start()
    starter.join(timeout=30)
    go.set()
    holder["t"].join(timeout=30)
    assert seen["chain"] == ()


def test_the_refusal_is_recorded_on_every_worker_below_the_owner_only():
    seen: dict = {}
    marked: dict = {}

    def leaf():
        mark_refusal("nested_measurement", threading.main_thread().ident)
        seen["leaf"] = threading.current_thread()

    def middle():
        inner = JoinedWorker(target=leaf)
        inner.start()
        inner.join(timeout=30)
        marked["leaf"] = inner.refusal

    mid_thread = JoinedWorker(target=middle)
    mid_thread.start()
    mid_thread.join(timeout=30)
    assert marked["leaf"] == "nested_measurement"
    assert mid_thread.refusal == "nested_measurement"

    def owner_below():
        # The owner is THIS worker's starter: only this worker is below it.
        inner = JoinedWorker(target=lambda: mark_refusal("nested_measurement", me[0]))
        inner.start()
        inner.join(timeout=30)
        marked["inner"] = inner.refusal

    me: list[int] = []
    top = JoinedWorker(target=lambda: (me.append(threading.get_ident()), owner_below()))
    top.start()
    top.join(timeout=30)
    assert marked["inner"] == "nested_measurement"
    assert top.refusal == "", (
        "a worker at or above the owner was never part of the cycle"
    )


def test_the_record_survives_whatever_the_test_made_of_the_exception():
    """The side channel against both interceptions at once: the worker catches BaseException
    (would read "crash"), and here the test itself re-raises a DIFFERENT exception, as pytest's
    wrapper does when it reports a failure it captured."""

    def pytest_like_wrapper():
        try:
            with _execution_guard():
                pass
        # BLE001: imitating pytest's interception, which catches everything
        except BaseException:  # noqa: BLE001
            raise AssertionError("pytest reported failure for nodeid") from None

    with _execution_guard():
        outcome = _run_test_with_timeout(pytest_like_wrapper, None, True, 5000)
    assert outcome == "nested_measurement"


def test_the_refusal_survives_a_real_pytest_session(tmp_path):
    """Through the REAL interception: a live pytest session's item wrapper (`runtestprotocol`
    catches the BaseException into a report, `_ExcCapture` re-raises it) — once with the test
    failing on the refusal, once with the test swallowing it and PASSING."""
    from Wesker.ci import discover_test_callables, run_with_live_suite

    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "nestlive_mod.py").write_text("def target(x):\n    return x * 2\n")
    (root / "tests" / "test_nestlive.py").write_text(
        "from Wesker.engine import _execution_guard\n"
        "from nestlive_mod import target\n\n"
        "def test_fails_on_the_refusal():\n"
        "    with _execution_guard():\n"
        "        pass\n"
        "    assert target(3) == 6\n\n"
        "def test_swallows_the_refusal():\n"
        "    try:\n"
        "        with _execution_guard():\n"
        "            pass\n"
        "    except BaseException:\n"
        "        pass\n"
        "    assert target(3) == 6\n"
    )
    for name in ("nestlive_mod", "test_nestlive", "tests.test_nestlive"):
        sys.modules.pop(name, None)
    outcomes: dict = {}

    def body():
        for test in discover_test_callables(str(root), "nestlive_mod.py", ["target"]):
            with _execution_guard():
                outcomes[test.__name__] = _run_test_with_timeout(test, None, True, 5000)

    cwd = os.getcwd()
    try:
        os.chdir(root)
        run_with_live_suite(str(root), body, paths=[str(root / "tests")])
    finally:
        os.chdir(cwd)
        for name in ("nestlive_mod", "test_nestlive", "tests.test_nestlive"):
            sys.modules.pop(name, None)
    assert outcomes == {
        "test_fails_on_the_refusal": "nested_measurement",
        "test_swallows_the_refusal": "nested_measurement",
    }, outcomes
