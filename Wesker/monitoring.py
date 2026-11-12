"""The one owner of ``sys.monitoring`` in the pair (EP-B3), and the deterministic step budget.

Detective's EP-B3 (docs/ENGINEERING_PASS_2026-09-26.md): three owners of process-global interpreter
hooks, each with its own install/restore, on the legacy ``settrace``/``setprofile`` slots. With 3.12 as
the floor, ``sys.monitoring`` (PEP 669) is on every supported interpreter, and this module is where the
pair takes a tool id and registers its callbacks — once, instead of per call site.

OWNED SUBSCRIPTION MODEL (W#32):
To prevent destructive event bitmask overwrites, thread-safety races, and stack distortion across
consumers (such as step budgeting, input harvesting, and line coverage), this module serves as the
single process-level subscription broker:
1. Subscriptions register required events (`PY_START`, `PY_RETURN`, `JUMP`, `LINE`) per `CodeType`.
2. Local event bitmasks are derived as the bitwise union of active subscriptions on that code.
3. Callbacks dispatch with target stack frame injection (`sys._getframe(1)` verified against the
   monitored code object) so subscribers receive the authentic execution frame without stack guessing.
4. Ownership is enforced via thread identity and per-run tokens: retired tokens suppress late callbacks
   from abandoned worker threads, freezing evidence.
5. Registration is transactional with rollback on failure; close is idempotent; teardown is version-aware.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import sys
import threading
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, Self

from .interrupt import acquire_unstranded

TOOL_NAME = "wesker"
# Ids 3 and 4 have no assigned role (0 debugger, 1 coverage, 2 profiler, 5 optimizer): tried first.
_PREFERRED_IDS = (3, 4, 2, 5, 0, 1)

WITHIN = "within"
EXCEEDED = "exceeded"
UNBOUNDED = "unbounded"

SUBSCRIPTION_OVERLAP_REFUSED = "subscription_overlap_refused"


class StepBudgetExceeded(BaseException):
    """Raised INTO a monitored frame when its loop iterations pass the budget.

    A ``BaseException``, like ``interrupt.Abandoned``: a target or a test that wraps its work in a broad
    ``except Exception`` must not be able to swallow the stop.
    """

    def __init__(self, iterations: int, cap: int):
        super().__init__(iterations, cap)
        self.iterations = iterations
        self.cap = cap


class MonitoringUnavailable(RuntimeError):
    """Every ``sys.monitoring`` tool id is held by another tool, so nothing can be counted this way."""


class MonitoringObservationFailed(MonitoringUnavailable):
    """An observation failed; raised at its owning boundary, never into the target."""

    def __init__(self, reason: str, subscription_id: int):
        self.code, _, self.detail = reason.partition(":")
        self.subscription_id = subscription_id
        super().__init__(f"{reason} (subscription={subscription_id})")


class SubscriptionOverlapRefused(RuntimeError):
    """An exclusive subscription (such as a step budget) conflicts with an existing subscription."""


class ActiveSubscriptionsError(RuntimeError):
    """Cannot release monitoring tool while active subscriptions exist."""


def step_cap(original_iterations: int, factor: int, floor: int) -> int:
    """The loop budget a mutant gets where its original made ``original_iterations`` (pure — pinned).

    ``factor`` times what the original did there, plus ``floor`` so an original that made no iterations
    still leaves a mutant room for a few. A negative count (never produced, but not representable as
    "fewer than none") counts as none.
    """
    return max(0, original_iterations) * factor + floor


#: The factor and floor every mutant observation's loop budget is built from (W#27): a mutant may
#: make STEP_BUDGET_FACTOR times the loop iterations its ORIGINAL made on the same test, plus
#: STEP_BUDGET_FLOOR. Chosen from measurement, not taste — see
#: docs/STEP_BUDGET_CALIBRATION_2026-10-25.md for the corpus, the distribution of terminating
#: mutants' ratios, and the margin rule. In short: the furthest any of 3,176 terminating mutant
#: observations went was 4.44x a large original (3,272 -> 14,523) and 1,075 iterations from an
#: original of 3 (a ``//`` -> ``/`` halving loop running the float-underflow tail). 32 is the
#: smallest power of two that keeps 5x headroom over the first, and 10,000 the smallest 1-2-5 step
#: that keeps it over the second. Both values are part of the mutation policy's observation clause,
#: so changing either moves the policy id.
STEP_BUDGET_FACTOR = 32
STEP_BUDGET_FLOOR = 10_000
#: The loop budget an ORIGINAL gets on a SYNTHESIZED probe input (W#27), where no test fixes a
#: reference count: the survivor-equivalence probes run original and mutant on boundary inputs the
#: engine invented, and a function that does not terminate on one (``collatz(0)``) used to hang the
#: whole run. An input the original cannot finish inside this budget is no evidence either way.
#: About 26 times the largest count any calibration original made on a whole test (7,723).
STEP_PROBE_CAP = 200_000


#: An upper bound on what one COUNTED loop iteration costs, in microseconds (W#27). Measured on the
#: calibration corpus, per counted iteration on an M1 Max: p90 0.92 µs on its performance cores and
#: 3.81 µs on its efficiency cores (``taskpolicy -b``, W#27's slow-machine simulation); this allows
#: about four times the slower p90. It sizes the wall a budgeted observation gets, so the
#: deterministic budget decides before the wall can; a body slower than this (exponential bignum
#: growth measured 11-33 µs) may still meet the wall first — `undetermined`.
STEP_ITERATION_US = 16.0


def armed_wall_ms(
    cap: int, original_ms: float, wall_ms: float, ceiling_ms: float | None
) -> float:
    """The wall bound a step-budgeted observation runs under (W#27, pure — pinned).

    The wall clock bounds LIVENESS and must never decide what the deterministic budget decides.
    So a budgeted observation's wall covers what reaching its cap costs — the ORIGINAL's own
    counted run plus ``cap`` iterations at :data:`STEP_ITERATION_US` — and is:

    * never less than the wall it already had (``wall_ms``);
    * never past ``ceiling_ms``, the aggregate deadline that bounds the whole run (#13);
    * not extended at all when no ceiling is known: without one there is nothing to stay inside.
    """
    if ceiling_ms is None:
        return wall_ms
    need = original_ms + cap * STEP_ITERATION_US / 1000.0
    return max(wall_ms, min(need, ceiling_ms))


#: The compute budget of one mutant observation's wall, in REFERENCE seconds (W#27): the work a mutant
#: may do that the loop budget cannot count (a loop in a helper, a C builtin, another thread) before
#: the wall calls it `undetermined`. 10.0 is the floor the isolated wall always had (it covers pytest
#: startup in a fresh worker); what changed is the unit. It is seconds on the reference machine, and
#: the bound on any other machine is this times how much slower that machine runs the reference
#: workload, so the same function reaches the same verdict on a slow machine as on a fast one.
#: The user sets it with ``WESKER_COMPUTE_BUDGET_S``.
LIVENESS_BUDGET_S = 10.0
#: The slowdown a bound will follow, at most. Past it the machine is not slow, it is starved, and a
#: bound that grows without limit is no bound; the code ``capped`` names that.
LIVENESS_MAX_SLOWDOWN = 64
#: Iterations of the reference workload (a pure-Python integer loop, the shape of the work no counter
#: sees) and the time it takes the reference machine: 35.7 ms, the fastest of nine runs on the
#: development container while its load average was 4.7 (a second set of nine agreed to 0.1 ms), so an idle machine of that class is no
#: slower. A machine at least this fast has slowdown 1.0: the bound never shrinks below the budget.
LIVENESS_REFERENCE_ITERATIONS = 1_000_000
LIVENESS_REFERENCE_NS = 35_700_000
COMPUTE_BUDGET_ENV = "WESKER_COMPUTE_BUDGET_S"


def harness_slowdown(
    measured_ns: int, reference_ns: int, max_slowdown: int
) -> tuple[float, str]:
    """How much slower than the reference machine this one runs, and why that number (pure — pinned).

    ``measured_ns`` is this machine's time on the reference workload, ``reference_ns`` the reference
    machine's. The result is ``(factor, code)``:

    * ``unmeasured`` — no usable measurement or reference (zero or negative): factor 1.0. Absence is
      not a value: the code says the bound is the reference bound because nothing was measured.
    * ``reference``  — at least as fast as the reference: factor 1.0. A faster machine does not
      shrink the bound; the budget is a floor.
    * ``slower``     — slower than the reference: the ratio, so the bound grows with the machine.
    * ``capped``     — slower than ``max_slowdown`` times: that cap, never more.
    """
    if measured_ns <= 0 or reference_ns <= 0 or max_slowdown < 1:
        return 1.0, "unmeasured"
    if measured_ns <= reference_ns:
        return 1.0, "reference"
    ratio = measured_ns / reference_ns
    if ratio > max_slowdown:
        return float(max_slowdown), "capped"
    return ratio, "slower"


def liveness_bound_ms(
    requested_ms: float, budget_ms: float, armed_cost_ms: float, slowdown: float
) -> tuple[float, str]:
    """The wall that bounds LIVENESS of one isolated mutant observation (W#27, pure — pinned).

    ``requested_ms`` is the caller's per-mutant ask and ``budget_ms`` the compute budget, both in
    reference time; ``slowdown`` is :func:`harness_slowdown`'s factor, so both are stretched by how
    slow this machine is. ``armed_cost_ms`` is what reaching every armed node's cap costs, already
    measured on THIS machine (the original's own counted run plus ``cap`` iterations at
    :data:`STEP_ITERATION_US`), so it is added unscaled: scaling it again would count the machine's
    slowness twice. The bound is the largest term, and the code names which:

    * ``armed``     — the budget plus the cost of reaching the caps decided: a budgeted observation
      must reach its deterministic stop before the wall can.
    * ``requested`` — the caller asked for more than the budget.
    * ``budget``    — the compute budget decided (the usual case).
    """
    scale = max(slowdown, 1.0)
    base = max(requested_ms, budget_ms) * scale
    if armed_cost_ms > 0 and budget_ms * scale + armed_cost_ms >= base:
        return budget_ms * scale + armed_cost_ms, "armed"
    if requested_ms > budget_ms:
        return base, "requested"
    return base, "budget"


def liveness_action(timed_out: bool, contained: bool, attempts: int) -> str:
    """What a fired liveness wall asks for next (W#27, pure — pinned).

    A wall that fires once may have met a stall (another process took the cores) and not a mutant
    that never ends, so it is tried once more; a wall that fires again names the observation
    `undetermined`. Never a kill, never survival evidence.

    * ``accept``       — the wall did not fire: the run stands.
    * ``retry``        — the first fire, and the stopped worker is confirmed dead: run it once more.
    * ``undetermined`` — it fired again, or the runaway could not be stopped (running it again
      beside a live runaway would measure the runaway): no verdict.
    """
    if not timed_out:
        return "accept"
    if attempts < 2 and contained:
        return "retry"
    return "undetermined"


def parse_compute_budget(raw: str | None) -> tuple[float, str]:
    """The compute budget a setting names, in reference seconds (W#27, pure — pinned).

    * ``default``  — nothing set (None or blank): :data:`LIVENESS_BUDGET_S`.
    * ``set``      — a finite positive number.
    * ``invalid``  — anything else (not a number, zero, negative, infinite, NaN): the default
      value, but named, so a caller can refuse a setting that was ignored instead of obeying it
      silently.
    """
    if raw is None or not raw.strip():
        return LIVENESS_BUDGET_S, "default"
    try:
        value = float(raw)
    except ValueError:
        return LIVENESS_BUDGET_S, "invalid"
    if not (value > 0.0) or value == float("inf"):
        return LIVENESS_BUDGET_S, "invalid"
    return value, "set"


def reference_workload(iterations: int) -> int:
    """The fixed workload the harness is calibrated on: a pure-Python integer loop.

    It is the shape of work the loop budget cannot count and the wall must bound, and its value is
    returned so nothing can optimise it away.
    """
    acc = 0
    i = 0
    while i < iterations:
        acc = (acc + i * 3) & 0xFFFF
        i += 1
    return acc


_harness_ns: int | None = None


def measure_harness_ns(repeats: int = 5) -> int:
    """This machine's median wall time on the reference workload, in ns (impure).

    Wall time, not CPU time: the bound it sizes is a wall bound, and a machine that is busy with
    other work is exactly the slowness the bound must follow.
    """
    samples = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter_ns()
        reference_workload(LIVENESS_REFERENCE_ITERATIONS)
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def machine_slowdown(refresh: bool = False) -> tuple[float, str]:
    """The calibrated slowdown of THIS machine, measured once per process (impure).

    The first call measures; later calls reuse it. ``refresh=True`` measures again and keeps the
    SLOWER of the two: it is asked for after a wall fired, because a machine that became busy since
    the first measurement is the one cause a retry can cure, and a bound must not shrink back.
    """
    global _harness_ns
    if _harness_ns is None or refresh:
        now = measure_harness_ns()
        _harness_ns = now if _harness_ns is None else max(_harness_ns, now)
    return harness_slowdown(_harness_ns, LIVENESS_REFERENCE_NS, LIVENESS_MAX_SLOWDOWN)


def compute_budget_s() -> float:
    """The compute budget in force, from ``WESKER_COMPUTE_BUDGET_S`` (impure).

    An invalid setting is refused rather than ignored: a bound the user thinks they raised and did
    not is a verdict they cannot account for.
    """
    raw = os.environ.get(COMPUTE_BUDGET_ENV)
    value, code = parse_compute_budget(raw)
    if code == "invalid":
        raise ValueError(
            f"{COMPUTE_BUDGET_ENV}={raw!r} is not a finite positive number of seconds"
        )
    return value


def mutant_step_cap(
    original_outcome: str, original_iterations: int, factor: int, floor: int
) -> tuple[str, int | None]:
    """Whether one mutant observation of a test is step-budgeted, and its cap (W#27, pure — pinned).

    The cap rests on the ORIGINAL's loop count on the SAME test, counted under the SAME
    instrumentation the mutant will run under. Only an original that PASSED that test under the
    count is a basis; every other state leaves the observation bounded by the wall clock alone,
    exactly as before the budget existed:

    * ``armed`` — the original passed with ``original_iterations`` counted: the cap is
      ``step_cap(original_iterations, factor, floor)``.
    * ``original_failed`` — the original failed or crashed under the count: a baseline failure,
      or a test whose own instrumentation cannot run beside ours (a step budget the test opens
      itself does not nest). A cap there would turn our instrumentation into a verdict.
    * ``original_undetermined`` — the original's own counted run reached its wall bound.
    * ``original_uncounted`` — the original passed but no count was taken (every monitoring tool
      id held, or a budget already open), so there is no number to scale.
    * ``unmeasured`` — no original observation exists for this test.
    """
    if original_outcome == "passed":
        return "armed", step_cap(original_iterations, factor, floor)
    if original_outcome == "failed":
        return "original_failed", None
    if original_outcome == "undetermined":
        return "original_undetermined", None
    if original_outcome == "uncounted":
        return "original_uncounted", None
    return "unmeasured", None


def original_step_observation(outcome: str, counted: bool) -> str:
    """Name what the ORIGINAL's counted run of one test established (W#27, pure — pinned).

    ``outcome`` is ``_run_test_with_timeout``'s return spelled as a string (``"passed"`` for its
    None). A pass is a basis only if the count was actually taken; a wall-bound or uncontained run
    established nothing; anything else — a kill reason, a declared failure, a refused nested
    measurement, an exceeded budget — is a run the original did not pass under the count.
    """
    if outcome == "passed":
        return "passed" if counted else "uncounted"
    if outcome in ("undetermined", "uncontained"):
        return "undetermined"
    return "failed"


def step_budget_verdict(iterations: int, cap: int | None) -> str:
    """What a finished count says against its budget (pure — pinned).

    ``unbounded`` — no cap was set: the count is a measurement only (the original's own run).
    ``exceeded``  — more iterations than the cap: the monitored function diverged from its budget.
    ``within``    — at most the cap.
    """
    if cap is None:
        return UNBOUNDED
    return EXCEEDED if iterations > cap else WITHIN


@dataclass
class StepCount:
    """The live tally of one budget: how many loop iterations, and whether the cap stopped them."""

    iterations: int = 0
    cap: int | None = None
    stopped: bool = False

    @property
    def verdict(self) -> str:
        return step_budget_verdict(self.iterations, self.cap)


# ── Run-token manager ─────────────────────────────────────────────────────────


def registered_thread(thread_id: int) -> threading.Thread | None:
    """Read CPython's existing thread registration without creating a DummyThread.

    This broker owns the private runtime bridge. current_thread() creates an object
    during bootstrap/teardown; the observer must leave the observed population alone.
    CPython's registry is not declared by the public threading type stub.
    """
    registry: dict[int, threading.Thread] = vars(threading)["_active"]
    return registry.get(thread_id)


# Callbacks run in workers that may receive an asynchronous Abandoned exception,
# and abandonment can strand a lock taken with `with` (interrupt's STRANDED LOCK).
# Run tokens therefore take no lock at all: for integer tokens these CPython
# built-ins finish before the next Python bytecode boundary, and no user-defined
# hash/equality or observer runs inside them. The broker's one lock is taken only
# through `acquire_unstranded`, released in the next `try`'s `finally` (W#29).
_token_seq = itertools.count(1)
_active_run_tokens: set[int] = set()


def allocate_run_token() -> int:
    """Allocate a fresh monotonically increasing run token."""
    tok = next(_token_seq)
    _active_run_tokens.add(tok)
    return tok


def is_run_token_active(token: int | None) -> bool:
    """Check if a run token is currently active."""
    if token is None:
        return True
    return token in _active_run_tokens


def retire_run_token(token: int | None) -> None:
    """Retire a run token so late callbacks from abandoned workers are discarded."""
    if token is None:
        return
    _active_run_tokens.discard(token)


# ── Subscription registry & broker ───────────────────────────────────────────


@dataclass
class _ObservationState:
    failure: str = ""
    inflight: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class Subscription:
    sub_id: int
    codes: tuple[CodeType, ...]
    event: int
    callback: Callable[..., Any]
    thread_id: int | None = None
    run_token: int | None = None
    exclusive: bool = False
    selectors: tuple[str, ...] | None = None
    selector_descriptors: (
        tuple[tuple[str, tuple[int, int] | None, bool], ...] | None
    ) = None
    observation: _ObservationState = field(
        default_factory=_ObservationState, compare=False
    )
    thread_filter: Callable[[int], bool] | None = None


_sub_id_seq = 0
_delivery_seq = itertools.count(1)
# Never `with _subscriptions_lock:`. PY_START discovery takes it inside a callback, on every
# thread, including a test's runaway that a bounded join is abandoning; under `with`, 3.14
# stranded it there and deadlocked Wesker's own whole-codebase run (W#29). Every site uses
# `acquire_unstranded(_subscriptions_lock)` followed at once by `try: ... finally: release()`.
_subscriptions_lock = threading.RLock()
# Maps CodeType -> list of active subscriptions on that code
_subscriptions_by_code: dict[CodeType, list[Subscription]] = {}
# All active subscriptions by sub_id
_active_subscriptions: dict[int, Subscription] = {}
# Active subscriptions that have selectors
_subscriptions_with_selectors: dict[int, Subscription] = {}
# Dynamically discovered codes per sub_id
_discovered_codes_by_sub: dict[int, set[CodeType]] = {}
# Set of sub_ids attached per CodeType
_discovered_for_code: dict[CodeType, set[int]] = {}

# File normalization & selector matching caches
_norm_file_cache: dict[str, tuple[str, tuple[int, int] | None]] = {}
_selector_match_cache: dict[tuple[int, str], bool] = {}
# Discovery's negative cache (W#29): a file no live selector names, mapped to the selector
# generation it was checked under. Every selector subscribe and close bumps the generation and
# clears the map, so an entry written by a decision that raced a change carries the old number
# and can only cause a re-check, never a missed discovery. Keyed by file, not code, so it never
# keeps a code object (a mutant's included) alive, and stays bounded by the files that ran.
_selector_generation = 0
_unselected_files: dict[str, int] = {}


def _file_info(filename: str) -> tuple[str, tuple[int, int] | None]:
    info = _norm_file_cache.get(filename)
    if info is None:
        real = os.path.realpath(filename)
        try:
            st = os.stat(real)
            dev_ino = (st.st_dev, st.st_ino)
        except OSError:
            dev_ino = None
        info = (real, dev_ino)
        _norm_file_cache[filename] = info
    return info


def _matches_selectors(
    sub_id: int,
    filename: str,
    descriptors: tuple[tuple[str, tuple[int, int] | None, bool], ...],
) -> bool:
    cache_key = (sub_id, filename)
    cached = _selector_match_cache.get(cache_key)
    if cached is not None:
        return cached

    real, dev_ino = _file_info(filename)
    matched = False
    for sel_path, sel_dev_ino, sel_isdir in descriptors:
        if sel_dev_ino is not None and dev_ino is not None and sel_dev_ino == dev_ino:
            matched = True
            break
        if real == sel_path:
            matched = True
            break
        if sel_isdir and real.startswith(sel_path + os.sep):
            matched = True
            break

    _selector_match_cache[cache_key] = matched
    return matched


def _extract_nested_codes(code: CodeType) -> list[CodeType]:
    """Recursively extract all nested code objects within a code object's constants."""
    res = [code]
    seen = {code}
    stack = [code]
    while stack:
        c = stack.pop()
        for const in c.co_consts:
            if isinstance(const, CodeType) and const not in seen:
                seen.add(const)
                res.append(const)
                stack.append(const)
    return res


_tool: int | None = None


def tool_id() -> int:
    """This pair's ``sys.monitoring`` tool id, claimed on first use from the free ones."""
    global _tool
    mon = sys.monitoring
    if _tool is not None and mon.get_tool(_tool) == TOOL_NAME:
        return _tool
    for candidate in _PREFERRED_IDS:
        if mon.get_tool(candidate) is None:
            mon.use_tool_id(candidate, TOOL_NAME)
            # Register top-level dispatchers on the claimed tool
            mon.register_callback(candidate, mon.events.JUMP, _dispatch_jump)
            mon.register_callback(candidate, mon.events.PY_START, _dispatch_py_start)
            mon.register_callback(candidate, mon.events.PY_RETURN, _dispatch_py_return)
            mon.register_callback(candidate, mon.events.LINE, _dispatch_line)
            mon.register_callback(candidate, mon.events.PY_UNWIND, _dispatch_py_unwind)
            mon.register_callback(
                candidate, mon.events.INSTRUCTION, _dispatch_instruction
            )
            _tool = candidate
            return candidate
    raise MonitoringUnavailable(
        "every sys.monitoring tool id is in use by other tools: "
        + ", ".join(f"{i}={mon.get_tool(i)!r}" for i in range(6))
    )


def _recompute_mask(tool: int, code: CodeType) -> None:
    """Derive and set the local event bitmask for a code object from active subscriptions."""
    subs = _subscriptions_by_code.get(code, [])
    mask = 0
    for s in subs:
        if s.event != sys.monitoring.events.PY_UNWIND:
            mask |= s.event
    sys.monitoring.set_local_events(tool, code, mask)


def _recompute_global_events(tool: int) -> None:
    """Derive and set global interpreter events for subscriptions needing process-wide dispatch."""
    mask = 0
    if _subscriptions_with_selectors:
        mask |= sys.monitoring.events.PY_START
    for sub in _active_subscriptions.values():
        if sub.event == sys.monitoring.events.PY_UNWIND:
            mask |= sys.monitoring.events.PY_UNWIND
    sys.monitoring.set_events(tool, mask)


# ── Callback dispatchers with target frame injection ──────────────────────────


def _thread_matches(sub: Subscription, thread_id: int) -> bool:
    """Admit the owning thread before materializing a monitored frame."""
    if sub.thread_id is not None and sub.thread_id != thread_id:
        return False
    if sub.thread_filter is None:
        return True
    try:
        return sub.thread_filter(thread_id)
    except Exception as exc:  # noqa: BLE001
        sub.observation.failure = (
            sub.observation.failure or f"ownership_filter_failed:{type(exc).__name__}"
        )
        return False


def _notify(sub: Subscription, code: CodeType, *args: Any, frame: Any) -> None:
    """Deliver one observation with admission and failure owned by the subscription.

    Reserve before checking the token: a retired owner can detect an already
    admitted callback that has not finished, rather than freeze mutable evidence.
    """
    state = sub.observation
    delivery = next(_delivery_seq)
    try:
        state.inflight.add(delivery)
        if (
            state.failure
            or _active_subscriptions.get(sub.sub_id) is not sub
            or not is_run_token_active(sub.run_token)
        ):
            return
        if frame.f_code is not code:
            state.failure = "target_frame_unavailable"
            return
        sub.callback(code, *args, frame=frame, run_token=sub.run_token)
    except Exception as exc:  # noqa: BLE001
        # Observer failure cannot be a target crash or a mutant kill. Its owner
        # checks this state after retirement and cleanup, before using evidence.
        state.failure = state.failure or f"observer_failed:{type(exc).__name__}"
    finally:
        state.inflight.discard(delivery)


def _dispatch_jump(
    code: CodeType, instruction_offset: int, destination_offset: int
) -> object:
    if destination_offset > instruction_offset:
        return None  # Forward jump: control flow, not an iteration
    cur_thread = threading.get_ident()
    subs = _subscriptions_by_code.get(code)
    if not subs:
        return None
    for sub in list(subs):
        if sub.event != sys.monitoring.events.JUMP:
            continue
        if not _thread_matches(sub, cur_thread):
            continue
        if sub.run_token is not None and not is_run_token_active(sub.run_token):
            continue
        # JUMP callbacks are control callbacks (e.g. step budget)
        res = sub.callback(code, instruction_offset, destination_offset)
        if res is not None:
            return res
    return None


def _discovery_pending(code: CodeType, thread_id: int) -> list[Subscription]:
    """The selector subscriptions that should attach to ``code`` now, decided WITHOUT the lock.

    Before W#29 this decision ran under the broker lock for every code no selector had attached,
    and a code whose file no selector names is never attached. So every PY_START of every
    unselected function in the process took the lock while a trace was live: about 2.4 us of
    overhead per call, measured with line_coverage's three subscriptions. That was also the
    contention that put an abandoned runaway in the lock's wait. It is read-only over the registry:
    the selector match is cached per (subscription, file) and the attached set is only read. A
    subscription closed after this read is dropped under the lock by :func:`_attach_discovered`.

    A file that no live selector names is recorded in ``_unselected_files`` under the generation
    read BEFORE deciding, so the dispatcher skips this whole decision for it with one lookup.
    Thread ownership is never cached: a selected file started on a thread the subscription does
    not own is decided again on its next start.
    """
    generation = _selector_generation
    attached = _discovered_for_code.get(code, ())
    pending: list[Subscription] = []
    named = False
    live = list(
        _subscriptions_with_selectors.values()
    )  # one C call: other threads subscribe and close
    for sub in live:
        if sub.sub_id in attached:
            named = True
            continue
        if not sub.selector_descriptors or not _matches_selectors(
            sub.sub_id, code.co_filename, sub.selector_descriptors
        ):
            continue
        named = True
        if not _thread_matches(sub, thread_id):
            continue
        if sub.run_token is not None and not is_run_token_active(sub.run_token):
            continue
        pending.append(sub)
    if not named:
        _unselected_files[code.co_filename] = generation
    return pending


def _discovery_failed(exc: Exception) -> None:
    """Record that discovery itself failed on every live selector subscription (W#29).

    Discovery runs inside a PY_START callback, on every thread, for every function that starts while
    a selector subscription is live. An exception it let escape was raised into whatever code had just
    started. That could be a test, where it read as a crash kill, or the measuring engine itself, where
    it ended the run: measured 2026-10-25, Wesker's local self-measurement died 27 minutes in with
    `ValueError: too many values to unpack` out of `_matches_selectors`, raised into the engine's own
    `inspect.getmodule`. `_notify` and `_dispatch_instruction` already hold the rule this applies
    here: an observer failure is never the observed code's failure. Every selector subscription could
    have missed a code here, so each is marked; `SubscriptionHandle.check` then refuses its evidence
    as `discovery_failed:<type>` at the owning boundary. A BaseException (an `Abandoned` injection, a
    budget stop) is not caught, and still unwinds the thread as it must.
    """
    reason = f"discovery_failed:{type(exc).__name__}"
    live = list(
        _subscriptions_with_selectors.values()
    )  # one C call: other threads subscribe and close
    for sub in live:
        sub.observation.failure = sub.observation.failure or reason


def _selectors_changed() -> None:
    """Invalidate discovery's negative cache; called under the broker lock AFTER the set of live
    selector subscriptions changes. A decision reads the generation before it snapshots the
    subscriptions, so one tagged with the new generation has seen the change."""
    global _selector_generation
    _selector_generation += 1
    _unselected_files.clear()


def _attach_discovered(code: CodeType, pending: list[Subscription]) -> None:
    """Attach ``pending`` to ``code`` and every code nested in it, under the broker lock.

    The lock is taken only here, and only through :func:`acquire_unstranded` (W#29). A
    subscription closed since :func:`_discovery_pending` read it is skipped, so discovery never
    re-attaches what ``close`` has already detached.
    """
    nested_codes = _extract_nested_codes(code)
    acquire_unstranded(_subscriptions_lock)
    try:
        for sub in pending:
            if _active_subscriptions.get(sub.sub_id) is not sub:
                continue
            for c in nested_codes:
                subs_for_c = _subscriptions_by_code.setdefault(c, [])
                if sub not in subs_for_c:
                    subs_for_c.append(sub)
                    _discovered_codes_by_sub.setdefault(sub.sub_id, set()).add(c)
                    _discovered_for_code.setdefault(c, set()).add(sub.sub_id)
                    if _tool is not None:
                        _recompute_mask(_tool, c)
    finally:
        _subscriptions_lock.release()


def _dispatch_py_start(code: CodeType, offset: int) -> None:
    # Dynamic discovery for subscriptions with path selectors: decided without the lock, which
    # only an actual attachment takes, and skipped outright for a file no live selector names.
    if (
        _subscriptions_with_selectors
        and _unselected_files.get(code.co_filename) != _selector_generation
    ):
        try:
            pending = _discovery_pending(code, threading.get_ident())
            if pending:
                _attach_discovered(code, pending)
        # BLE001: an observer failure is never the observed code's failure; it is recorded on the
        # observations that relied on discovery, and the owning boundary refuses them by name
        except Exception as exc:  # noqa: BLE001
            _discovery_failed(exc)

    subs = _subscriptions_by_code.get(code)
    if not subs:
        return
    cur_thread = threading.get_ident()
    matching_subs = []
    for sub in subs:
        if sub.event != sys.monitoring.events.PY_START or sub.observation.failure:
            continue
        if not _thread_matches(sub, cur_thread):
            continue
        if sub.run_token is not None and not is_run_token_active(sub.run_token):
            continue
        matching_subs.append(sub)
    if not matching_subs:
        return

    target_frame = sys._getframe(1)
    for sub in matching_subs:
        _notify(sub, code, offset, frame=target_frame)


def _dispatch_py_return(code: CodeType, offset: int, retval: Any) -> None:
    subs = _subscriptions_by_code.get(code)
    if not subs:
        return
    cur_thread = threading.get_ident()
    matching_subs = []
    for sub in subs:
        if sub.event != sys.monitoring.events.PY_RETURN or sub.observation.failure:
            continue
        if not _thread_matches(sub, cur_thread):
            continue
        if sub.run_token is not None and not is_run_token_active(sub.run_token):
            continue
        matching_subs.append(sub)
    if not matching_subs:
        return

    target_frame = sys._getframe(1)
    for sub in matching_subs:
        _notify(sub, code, offset, retval, frame=target_frame)


def _dispatch_line(code: CodeType, line_number: int) -> None:
    subs = _subscriptions_by_code.get(code)
    if not subs:
        return
    cur_thread = threading.get_ident()
    matching_subs = []
    for sub in subs:
        if sub.event != sys.monitoring.events.LINE or sub.observation.failure:
            continue
        if not _thread_matches(sub, cur_thread):
            continue
        if sub.run_token is not None and not is_run_token_active(sub.run_token):
            continue
        matching_subs.append(sub)
    if not matching_subs:
        return

    target_frame = sys._getframe(1)
    for sub in matching_subs:
        _notify(sub, code, line_number, frame=target_frame)


def _dispatch_py_unwind(code: CodeType, offset: int, exc: BaseException) -> None:
    subs = _subscriptions_by_code.get(code)
    if not subs:
        return
    cur_thread = threading.get_ident()
    matching_subs = []
    for sub in subs:
        if sub.event != sys.monitoring.events.PY_UNWIND or sub.observation.failure:
            continue
        if not _thread_matches(sub, cur_thread):
            continue
        if sub.run_token is not None and not is_run_token_active(sub.run_token):
            continue
        matching_subs.append(sub)
    if not matching_subs:
        return

    target_frame = sys._getframe(1)
    for sub in matching_subs:
        _notify(sub, code, offset, exc, frame=target_frame)


def _dispatch_instruction(code: CodeType, offset: int) -> object:
    subs = _subscriptions_by_code.get(code)
    if not subs:
        return None
    cur_thread = threading.get_ident()
    for sub in subs:
        if sub.event != sys.monitoring.events.INSTRUCTION or sub.observation.failure:
            continue
        if not _thread_matches(sub, cur_thread):
            continue
        if sub.run_token is not None and not is_run_token_active(sub.run_token):
            continue
        try:
            res = sub.callback(code, offset)
            if res is sys.monitoring.DISABLE:
                return sys.monitoring.DISABLE
        except Exception as exc:  # noqa: BLE001
            sub.observation.failure = (
                sub.observation.failure or f"observer_failed:{type(exc).__name__}"
            )
    return None


# ── Subscription API ──────────────────────────────────────────────────────────


class SubscriptionHandle:
    """An active owned monitoring subscription."""

    def __init__(self, sub: Subscription, close_fn: Callable[[], None]):
        self.sub = sub
        self._close_fn = close_fn
        self._closed = False

    @property
    def sub_id(self) -> int:
        return self.sub.sub_id

    def close(self) -> None:
        """Idempotently close this subscription and update event bitmasks."""
        if not self._closed:
            self._close_fn()
            self._closed = True

    def check(self) -> None:
        """Refuse faulty or still-changing evidence at the owning boundary."""
        if not self._closed:
            raise MonitoringObservationFailed("subscription_active", self.sub_id)
        state = self.sub.observation
        if state.inflight:
            state.failure = state.failure or "observation_inflight"
        if state.failure:
            raise MonitoringObservationFailed(state.failure, self.sub_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
        if not args or args[0] is None:
            self.check()


def subscribe(
    codes: Sequence[CodeType],
    event: int,
    callback: Callable[..., Any],
    *,
    thread_id: int | None = None,
    run_token: int | None = None,
    exclusive: bool = False,
    selectors: Sequence[str] | None = None,
    thread_filter: Callable[[int], bool] | None = None,
) -> SubscriptionHandle:
    """Register an owned monitoring subscription across ``codes`` for ``event``.

    Supports bounded dynamic discovery for late-loaded or dynamically imported
    code when ``selectors`` are provided. Derives event masks atomically, supports
    multiplexed observers, and enforces control exclusion where ``exclusive=True``.
    Raises :class:`SubscriptionOverlapRefused` on incompatible overlap.
    """
    global _sub_id_seq
    tool = tool_id()
    codes_tuple = tuple(codes)

    # Expand any initial codes to include nested code objects in co_consts
    expanded_codes: list[CodeType] = []
    for c in codes_tuple:
        for nested in _extract_nested_codes(c):
            if nested not in expanded_codes:
                expanded_codes.append(nested)
    expanded_tuple = tuple(expanded_codes)

    # Precompute selector descriptors if selectors are provided
    descriptors: tuple[tuple[str, tuple[int, int] | None, bool], ...] | None = None
    selectors_tuple: tuple[str, ...] | None = None
    if selectors:
        selectors_tuple = tuple(selectors)
        desc_list: list[tuple[str, tuple[int, int] | None, bool]] = []
        for s in selectors_tuple:
            real_path, dev_ino = _file_info(s)
            desc_list.append((real_path, dev_ino, os.path.isdir(real_path)))
        descriptors = tuple(desc_list)

    acquire_unstranded(_subscriptions_lock)
    try:
        # Check compatibility across all requested code objects
        for c in expanded_tuple:
            existing = _subscriptions_by_code.get(c, [])
            for s in existing:
                if s.event == event:
                    threads_overlap = (
                        s.thread_id is None
                        or thread_id is None
                        or s.thread_id == thread_id
                    )
                    if (exclusive or s.exclusive) and threads_overlap:
                        raise SubscriptionOverlapRefused(
                            f"conflicting exclusive subscription on event {event} "
                            f"(existing={s.sub_id}, requested_exclusive={exclusive})"
                        )

        _sub_id_seq += 1
        sub_id = _sub_id_seq
        sub = Subscription(
            sub_id=sub_id,
            codes=expanded_tuple,
            event=event,
            callback=callback,
            thread_id=thread_id,
            run_token=run_token,
            exclusive=exclusive,
            selectors=selectors_tuple,
            selector_descriptors=descriptors,
            thread_filter=thread_filter,
        )

        # Apply masks with transactional rollback on failure
        applied: list[CodeType] = []
        try:
            _active_subscriptions[sub_id] = sub
            if descriptors:
                _subscriptions_with_selectors[sub_id] = sub
                _selectors_changed()
            # Install discovery before local instrumentation. With a foreign
            # PY_START callback returning DISABLE, reversing this order loses
            # INSTRUCTION events on the code's first invocation (3.12–3.14).
            # Reapplying an unchanged local mask does not repair that cold call.
            _recompute_global_events(tool)
            for c in expanded_tuple:
                _subscriptions_by_code.setdefault(c, []).append(sub)
                _discovered_for_code.setdefault(c, set()).add(sub_id)
                applied.append(c)
                _recompute_mask(tool, c)
        except BaseException:
            # (#32) Registry first, masks after, and a failing mask must not stop the rest: the
            # fault that brought us here (a code object the interpreter refuses to instrument) can
            # recur in the rollback itself, and a rollback that raises half way leaves this
            # subscription registered in a dict nothing will ever clean.
            _active_subscriptions.pop(sub_id, None)
            selectors_changed = (
                _subscriptions_with_selectors.pop(sub_id, None) is not None
            )
            for c in applied:
                subs = _subscriptions_by_code.get(c, [])
                if sub in subs:
                    subs.remove(sub)
                if not subs:
                    _subscriptions_by_code.pop(c, None)
                sub_set = _discovered_for_code.get(c)
                if sub_set:
                    sub_set.discard(sub_id)
                    if not sub_set:
                        _discovered_for_code.pop(c, None)
            for c in applied:
                # BLE001: the original failure is the one the caller must see
                try:
                    _recompute_mask(tool, c)
                except Exception:  # noqa: BLE001, S110
                    pass
            try:
                if selectors_changed:
                    _selectors_changed()
                _recompute_global_events(tool)
            # BLE001: as above
            except Exception:  # noqa: BLE001, S110
                pass
            raise

        def _do_close() -> None:
            acquire_unstranded(_subscriptions_lock)
            try:
                _active_subscriptions.pop(sub_id, None)
                if _subscriptions_with_selectors.pop(sub_id, None) is not None:
                    _selectors_changed()
                discovered = _discovered_codes_by_sub.pop(sub_id, set())
                all_codes = expanded_tuple + tuple(discovered)
                for c in all_codes:
                    subs = _subscriptions_by_code.get(c, [])
                    if sub in subs:
                        subs.remove(sub)
                    if not subs:
                        _subscriptions_by_code.pop(c, None)
                    sub_set = _discovered_for_code.get(c)
                    if sub_set:
                        sub_set.discard(sub_id)
                        if not sub_set:
                            _discovered_for_code.pop(c, None)
                    _recompute_mask(tool, c)
                # Purge match cache for this subscription. Snapshot the keys in one C call:
                # discovery fills this cache without the lock, and a Python-level walk of the
                # live dict would raise "changed size during iteration" mid-close.
                cached_keys = list(_selector_match_cache)
                keys_to_remove = [k for k in cached_keys if k[0] == sub_id]
                for k in keys_to_remove:
                    _selector_match_cache.pop(k, None)
                _recompute_global_events(tool)
            finally:
                _subscriptions_lock.release()

        return SubscriptionHandle(sub, _do_close)
    finally:
        _subscriptions_lock.release()


def release_monitoring_tool() -> None:
    """Safely release the claimed sys.monitoring tool ID.

    Requires zero active subscriptions. Explicitly clears all registered callbacks,
    global events, and local event masks before calling ``free_tool_id`` (ensuring
    leak-free teardown on Python 3.12 where ``free_tool_id`` does not auto-clear).
    """
    global _tool
    acquire_unstranded(_subscriptions_lock)
    try:
        if _active_subscriptions or _subscriptions_by_code:
            raise ActiveSubscriptionsError(
                f"cannot release monitoring tool: {len(_active_subscriptions)} active subscriptions"
            )
        mon = sys.monitoring
        if _tool is not None:
            tid = _tool
            # Clear callbacks
            for ev in (
                mon.events.JUMP,
                mon.events.PY_START,
                mon.events.PY_RETURN,
                mon.events.LINE,
                mon.events.PY_UNWIND,
                mon.events.INSTRUCTION,
            ):
                with contextlib.suppress(ValueError):
                    mon.register_callback(tid, ev, None)
            mon.set_events(tid, 0)
            mon.free_tool_id(tid)
            _tool = None
    finally:
        _subscriptions_lock.release()


# ── Step budget implementation on top of owned subscriptions ──────────────────


@dataclass
class _Budget:
    thread: int
    count: StepCount
    handle: SubscriptionHandle | None = None


_active: _Budget | None = None


def is_budget_active() -> bool:
    """True iff a step budget is currently open on any thread."""
    return _active is not None


@contextlib.contextmanager
def step_budget(
    codes: Sequence[CodeType], cap: int | None
) -> Generator[StepCount, None, None]:
    """Count the loop iterations ``codes`` make on THIS thread while the block runs; with a ``cap``,
    raise :class:`StepBudgetExceeded` into the looping frame the moment the count passes it.

    ``cap=None`` measures without bounding (the ORIGINAL's run, whose count sets the mutant's cap).
    Budgets do not nest: one is open at a time, which the engine's execution lock already guarantees
    for every caller, and a second open is refused rather than silently merged.
    """
    global _active
    count = StepCount(cap=cap)
    tid = threading.get_ident()

    if _active is not None:
        raise RuntimeError("step budgets do not nest: one is already open")

    def _budget_on_jump(
        _c: CodeType, instruction_offset: int, destination_offset: int
    ) -> object:
        if destination_offset > instruction_offset:
            return None
        count.iterations += 1
        if count.cap is not None and count.iterations > count.cap:
            count.stopped = True
            raise StepBudgetExceeded(count.iterations, count.cap)
        return None

    try:
        handle = subscribe(
            codes,
            sys.monitoring.events.JUMP,
            _budget_on_jump,
            thread_id=tid,
            exclusive=True,
        )
    except SubscriptionOverlapRefused as err:
        raise RuntimeError("step budgets do not nest: one is already open") from err

    budget = _Budget(thread=tid, count=count, handle=handle)
    _active = budget
    try:
        yield count
    finally:
        _active = None
        handle.close()
