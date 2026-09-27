"""The one owner of ``sys.monitoring`` in the pair (EP-B3), and the deterministic step budget.

Detective's EP-B3 (docs/ENGINEERING_PASS_2026-09-26.md): three owners of process-global interpreter
hooks, each with its own install/restore, on the legacy ``settrace``/``setprofile`` slots. With 3.12 as
the floor, ``sys.monitoring`` (PEP 669) is on every supported interpreter, and this module is where the
pair takes a tool id and registers its callbacks — once, instead of per call site.

TWO BOUNDS, TWO JOBS. ``interrupt.py`` bounds work by WALL-CLOCK, which is the only bound that reaches
code anywhere (a test that never touches the target, a callee, a fixture). That bound is liveness
only: when it fires, nothing was established. This module bounds something else — how many LOOP
ITERATIONS a monitored function makes — and that count is the same on every machine at every load,
so exceeding it is EVIDENCE: the function ran past the budget its original set, and the verdict does
not depend on how busy the machine was (DETERMINISTIC_SICP law 6: efficiency is a deterministic
budget, never wall-clock). The stop is deterministic too: the exception is raised from the monitoring
callback INTO the looping frame, at its back-edge, so a pure-Python runaway unwinds on the spot
instead of being waited out.

WHAT IT COUNTS: backward jumps (JUMP events whose destination precedes their source) in the given code
objects, on the thread that opened the budget — one per loop iteration. Forward jumps are control
flow, not iteration. Recursion is not counted: a runaway recursion ends in ``RecursionError``, a
deterministic crash already.

WHAT IT CANNOT SEE, stated: a loop in a CALLEE of the monitored code, another thread, or code blocked
outside the interpreter. Those stay the wall bound's business, and a wall bound that fires reads
``undetermined`` — never as divergence, never as a kill.

TOOL IDS: ``sys.monitoring`` has six, and coverage.py takes id 1 when a suite runs under it (measured
on 3.14.7), so the id is chosen from the free ones at first use, never assumed.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from types import CodeType

TOOL_NAME = "wesker"
# Ids 3 and 4 have no assigned role (0 debugger, 1 coverage, 2 profiler, 5 optimizer): tried first.
_PREFERRED_IDS = (3, 4, 2, 5, 0, 1)

WITHIN = "within"
EXCEEDED = "exceeded"
UNBOUNDED = "unbounded"


class StepBudgetExceeded(BaseException):
    """Raised INTO a monitored frame when its loop iterations pass the budget.

    A ``BaseException``, like ``interrupt.Abandoned``: a target or a test that wraps its work in a broad
    ``except Exception`` must not be able to swallow the stop."""

    def __init__(self, iterations: int, cap: int):
        super().__init__(iterations, cap)
        self.iterations = iterations
        self.cap = cap


class MonitoringUnavailable(RuntimeError):
    """Every ``sys.monitoring`` tool id is held by another tool, so nothing can be counted this way."""


def step_cap(original_iterations: int, factor: int, floor: int) -> int:
    """The loop budget a mutant gets where its original made ``original_iterations`` (pure — pinned).

    ``factor`` times what the original did there, plus ``floor`` so an original that made no iterations
    still leaves a mutant room for a few. A negative count (never produced, but not representable as
    "fewer than none") counts as none.
    """
    return max(0, original_iterations) * factor + floor


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


@dataclass
class _Budget:
    thread: int
    count: StepCount
    codes: tuple[CodeType, ...] = field(default_factory=tuple)


_tool: int | None = None
_active: _Budget | None = None


def _on_jump(
    _code: CodeType, instruction_offset: int, destination_offset: int
) -> object:
    budget = _active
    if budget is None or destination_offset > instruction_offset:
        return None  # no budget open, or a forward jump: control flow, not an iteration
    if threading.get_ident() != budget.thread:
        return None  # another thread running the same code: not this budget's work
    count = budget.count
    count.iterations += 1
    if count.cap is not None and count.iterations > count.cap:
        count.stopped = True
        raise StepBudgetExceeded(count.iterations, count.cap)
    return None


def tool_id() -> int:
    """This pair's ``sys.monitoring`` tool id, claimed on first use from the free ones."""
    global _tool
    mon = sys.monitoring
    if _tool is not None and mon.get_tool(_tool) == TOOL_NAME:
        return _tool
    for candidate in _PREFERRED_IDS:
        if mon.get_tool(candidate) is None:
            mon.use_tool_id(candidate, TOOL_NAME)
            mon.register_callback(candidate, mon.events.JUMP, _on_jump)
            _tool = candidate
            return candidate
    raise MonitoringUnavailable(
        "every sys.monitoring tool id is in use by other tools: "
        + ", ".join(f"{i}={mon.get_tool(i)!r}" for i in range(6))
    )


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
    tool = tool_id()
    count = StepCount(cap=cap)
    budget = _Budget(thread=threading.get_ident(), count=count, codes=tuple(codes))
    # No lock here, deliberately. This code runs inside test threads that the engine may ABANDON with
    # an injected exception (`interrupt.abandon`); a plain `threading.Lock` taken in that window can be
    # left held, and a thread then blocked in its C-level acquire is exactly what injection cannot stop.
    # Budgets are opened one at a time by construction (the engine's execution lock), so the check
    # below guards misuse, not a race.
    if _active is not None:
        raise RuntimeError("step budgets do not nest: one is already open")
    _active = budget
    events = sys.monitoring.events.JUMP
    try:
        for code in budget.codes:
            sys.monitoring.set_local_events(tool, code, events)
        yield count
    finally:
        for code in budget.codes:
            sys.monitoring.set_local_events(tool, code, 0)
        _active = None
