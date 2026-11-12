"""Stopping a runaway test thread — the one primitive both timed paths need.

Wesker bounds two things by wall-clock: a MUTANT's test run (`engine._run_test_with_timeout`) and
a test's TRACED baseline pass (`line_coverage._trace_one`). Both face the same problem, and it has
exactly one honest in-process answer, so it lives here once rather than being re-derived on each
side (`engine` imports `line_coverage`, so a shared home is also the only way both can use it).

THE PROBLEM: `thread.join(timeout)` bounds the WAIT, not the WORK. A daemon thread left running is
reclaimed at PROCESS exit — i.e. never, within a run — so each timeout leaks a live thread that
burns a core, and later timeouts are then CAUSED by earlier ones still running. The failure
compounds, and its writes can surface on stdout long after the caller moved on.

THE ANSWER: CPython's async-exception injection — the thread raises at its next BYTECODE boundary.
In-process and stdlib-only, so it costs Wesker neither its architecture nor its zero-dependency
promise. It is also the only mechanism that bounds work wherever it happens: a budget that rides a
trace callback can only fire while the traced code is executing, which in a real suite is a small
minority of the time (most tests never touch the one file under analysis, so they are not traced
at all — and therefore not bounded at all).

THE BOUNDARY, stated honestly and NOT fixable in-process: a thread blocked OUTSIDE the interpreter
— `subprocess.run`, a C extension, a socket — executes no bytecode, so the injection cannot land
until that call returns on its own. `abandon` returns False there rather than claiming a stop it
did not make. Bounding that for real needs process isolation, which is a different engine than an
in-process one.

WHAT IS STILL STOPPABLE THERE (W#29). Most of what a test blocks on outside the interpreter in this
pair's own suite is a CHILD PROCESS: ``subprocess.run`` parked in ``selector.poll``. That wait ends the
moment the child does, and the injection already pending then lands, so the thread unwinds. A worker
found parked in such a wait has its children killed (:func:`release_child_waits`), and only a worker
that is STILL alive after that is reported uncontained. What remains (a lock, a socket, a sleep, a C
extension) is recorded with the place it is parked (:func:`note_abandoned`), so a refusal can name it,
and its standing is a fact about a LIVE thread (:func:`abandoned_standing`): it lapses when the thread
ends.

THE STRANDED LOCK (W#29). The injection lands at the target's next eval-breaker check, and the
check right after a call returns is one of them. A thread parked in a CONTENDED acquire releases
the GIL, so the injection is made while it waits; the lock is then handed to it, the acquire
returns, and the exception is raised at that call's own check, with the lock held and before any
``try`` or ``with`` that would release it. The thread unwinds and exits owning a lock nobody can
release, and every later acquirer in every thread waits forever. ``with lock:`` is exposed on
3.14, which compiles the statement to ``LOAD_SPECIAL __enter__; CALL 0`` and only then begins
the protected range (3.11–3.13's single ``BEFORE_WITH`` opcode had no check there). A
``@contextmanager`` wrapper, or a bare ``acquire()`` written before its ``try``, is exposed on
every supported version: the acquire is its own call. Code that abandonment can reach takes its
locks through :func:`acquire_unstranded` and releases them in the very next ``try``'s
``finally``.

THE START/JOIN EDGE (#28). One blocking call that boundary covers is ENTIRELY the pair's own: the
execution lock. A test that runs the engine, measured inside a held evaluation, parks its worker in
the lock's C-level acquire while the lock's owner — the very thread that started the worker and is
joining it — waits for the worker. A wait-for cycle; the injection cannot land; the worker dies
owning the lock when the holder finally releases. Unlike a socket, this wait is KNOWABLE before it
begins: every worker the pair bounds is a :class:`JoinedWorker`, which records the thread that
started it, so a worker can ask whether the lock's owner is upstream of it and refuse instead of
waiting (:class:`MeasurementRefused`). A thread the code UNDER TEST starts records nothing, so a
cycle routed through one stays outside what can be seen here — it is still detected and refused by
the lock's orphan channels, never prevented.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import threading
import weakref
from typing import Any


class Abandoned(BaseException):
    """Injected into a timed-out thread to unwind it.

    A ``BaseException``, not an ``Exception``, deliberately: a test that wraps its body in a broad
    ``except Exception`` — an ordinary, innocent-looking thing for a test to do — would otherwise
    SWALLOW the injection and keep running, leaving the leak in place exactly where the code looks
    safest. The interrupt has to outrank the interrupted code's own error handling to be one.
    """


class MeasurementRefused(BaseException):
    """Raised INSTEAD of a blocking wait that could only complete a wait-for cycle (#28).

    ``code`` names why (``"nested_measurement"``: the execution lock is held by a thread upstream of
    this one on the start/join edge, so waiting for it waits for itself); ``owner_tid`` is the owner
    it would have waited on.

    A ``BaseException``, like :class:`Abandoned`: the engine nested inside a test is ordinary code
    full of ``except Exception`` handlers (and one ``except ExecutionLockUnavailable``), and any of
    them catching the refusal would let it carry on into the next acquire — or worse, into a result.
    Outranking them unwinds the nested engine at once. That is NOT what makes the outcome survive,
    though: pytest's call wrapper and the measurement's own worker both catch ``BaseException``, so
    the status travels on :attr:`JoinedWorker.refusal`, written BEFORE this is raised
    (:func:`mark_refusal`). Swallowing the exception cannot unwrite the record.
    """

    def __init__(self, code: str, owner_tid: int) -> None:
        super().__init__(f"{code} (owner_tid={owner_tid})")
        self.code = code
        self.owner_tid = owner_tid


class WorkerLost(BaseException):
    """A :class:`JoinedWorker`'s thread ended before it signalled its start (W#29).

    ``Thread.start()`` waits on ``_started`` with no timeout. A bootstrap that ends before setting it,
    because an exception was raised into it before its first line ran, leaves that wait unbounded. It
    was measured on the whole-codebase CI run: two runs sat 180 minutes each at the same
    ``_run_test_with_timeout`` start, with no other thread alive. It was then reproduced on Linux
    CPython 3.14 under a diagnostic (2026-10-25). The thread is gone, so the test it was to run never
    ran: no kill, no pass. And the process that ended it is not one to keep measuring in.
    :meth:`JoinedWorker.start` raises this instead of waiting, so every starter names it.

    A ``BaseException``, like :class:`Abandoned` and :class:`MeasurementRefused`, so that no broad
    ``except Exception`` between a start and its handler can turn a lost worker into a result.
    """

    code = "worker_lost"

    def __init__(self, thread: threading.Thread) -> None:
        super().__init__(
            f"worker_lost: {thread.name} ended before it signalled its start"
        )
        self.thread_name = thread.name


# How often a start's wait re-checks whether the thread has already ended. A healthy start returns
# the moment `_started` is set, so the slice costs nothing unless the bootstrap died.
_START_SLICE_S = 0.05


def _run_is_over(thread: threading.Thread, entry_released: list[object]) -> bool:
    """Whether ``thread``'s OS-level run has ended (W#29).

    On 3.13+ the thread's own handle says so: CPython's thread runner notifies it as the very last
    thing it does, whatever ended the run. 3.12 has no handle, so there the signal is the release of
    the entry callable CPython's bootstate held: a bound method, which no traceback frame references,
    so a hook that keeps the dead bootstrap's exception (pytest's unraisable plugin does) cannot
    withhold the release.
    """
    handle = getattr(thread, "_os_thread_handle", None) or getattr(
        thread, "_handle", None
    )
    if handle is not None:
        return bool(handle.is_done())
    return bool(entry_released)


def _forget_unstarted(thread: threading.Thread) -> None:
    """Take a thread whose bootstrap died out of ``threading._limbo`` (W#29).

    ``Thread.start()`` put it there, and only its bootstrap would have taken it out. Left behind,
    ``threading.enumerate()`` would list a dead thread forever, and every later trace would read
    ``incomplete_thread``. One ``dict.pop``, which completes before the next bytecode boundary, so
    no lock is taken that an injected exception could strand (the module docstring's STRANDED LOCK).
    """
    vars(threading)["_limbo"].pop(thread, None)


class JoinedWorker(threading.Thread):
    """A worker its starter waits on under a bounded join — the start/join edge, recorded (#28).

    ``starter`` is the thread that constructed it, which in every site that uses this class is the
    thread that starts it and then waits in :func:`bounded_join` for it. ``refusal`` is the side
    channel :func:`mark_refusal` writes: empty until something this worker ran was refused, then the
    refusal's code. The starter reads it after the join, where neither pytest's exception capture nor
    the worker's own ``except BaseException`` can have erased it.

    :meth:`start` never waits forever on a bootstrap that died: it raises :class:`WorkerLost` (W#29).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.starter = threading.current_thread()
        self.refusal = ""
        self._boot_released: list[object] = []
        self._boot_ref: weakref.ref | None = None

    def start(self) -> None:
        """``Thread.start()``, except that a bootstrap which ended before it signalled raises
        :class:`WorkerLost` instead of leaving the start waiting forever (W#29).

        The end is OBSERVED, not timed (`_run_is_over`). On 3.13+ it is the thread's own handle. On
        3.12 it is the release of the entry callable, recorded by a weakref whose callback is the
        C-level ``list.append``, so no Python code runs in the dying thread to record it.

        The entry is the bound ``self._bootstrap`` that ``Thread.start()`` would read anyway, held on
        the instance for that one read so its release can be watched. A bound method is C-level: it
        leaves no frame in a dead bootstrap's traceback, so a hook that keeps that exception (pytest's
        unraisable plugin does) keeps no reference to it. The instance's reference is dropped before
        the wait polls, so CPython's bootstate is the only holder left.

        ``Thread.start()`` ends in ``self._started.wait()``. Shadowing that one bound method on the
        instance turns the unbounded wait into slices that check between them. Both shadows are removed
        before this returns, and a start that succeeds returns exactly as ``Thread.start()`` does.
        """
        started = self._started
        wait = started.wait
        entry = self._bootstrap
        self._boot_ref = weakref.ref(entry, self._boot_released.append)
        vars(self)["_bootstrap"] = entry
        del entry

        def _wait(
            *_unbounded: object,
        ) -> bool:  # Thread.start() calls it with no timeout
            vars(self).pop("_bootstrap", None)  # the C starter holds the entry now
            while not wait(_START_SLICE_S):
                if _run_is_over(self, self._boot_released) and not started.is_set():
                    _forget_unstarted(self)
                    raise WorkerLost(self)
            return True

        started.wait = _wait  # type: ignore[method-assign]
        try:
            super().start()
        finally:
            del started.wait
            vars(self).pop("_bootstrap", None)


def starter_chain() -> tuple[int, ...]:
    """Idents of the live threads waiting, through bounded joins, on the CURRENT thread — nearest first.

    Empty for any thread that is not a :class:`JoinedWorker` (the main thread, a thread the code
    under test started), which is the common case and costs one ``isinstance``. The walk stops at the
    first starter that is no longer alive: a dead thread waits on nothing, and its ident may already
    belong to an unrelated thread, so naming it would forge an edge. Bounded by a seen-set as well,
    because a chain is only acyclic by construction and this must terminate even if constructed wrong.
    """
    chain: list[int] = []
    seen: set[int] = set()
    current = threading.current_thread()
    while isinstance(current, JoinedWorker) and id(current) not in seen:
        seen.add(id(current))
        starter = current.starter
        if starter.ident is None or not starter.is_alive():
            break
        chain.append(starter.ident)
        current = starter
    return tuple(chain)


def mark_refusal(code: str, owner_tid: int) -> None:
    """Record ``code`` on every :class:`JoinedWorker` from the current thread up to — not including —
    the thread ``owner_tid`` names (#28).

    Every level below the owner learns it, because each is a measurement whose result now contains a
    refused nested measurement; the owner's own measurement is the one the refusal protects, and a
    worker ABOVE it was never part of the cycle. The first code recorded on a worker is kept.
    """
    seen: set[int] = set()
    current = threading.current_thread()
    while (
        isinstance(current, JoinedWorker)
        and current.ident != owner_tid
        and id(current) not in seen
    ):
        seen.add(id(current))
        if not current.refusal:
            current.refusal = code
        current = current.starter


# How long to let an injected thread unwind before conceding it is blocked outside the interpreter.
# Only ever paid on a timeout (already the slow path), and only to keep the concession honest.
UNWIND_S = 0.1


def injection_landed(marked: int) -> bool:
    """Whether an async-exception injection took, given how many threads CPython says it marked.

    Exactly 1 is the good case. 0 means the thread already finished (a benign race against
    ``is_alive``). >1 must never happen — CPython documents it as "you're in trouble" — and the
    caller must UNDO it: we cannot know which other threads were poisoned, so it can never count
    as success.
    """
    return marked == 1


def abandon(thread: Any) -> bool:
    """Best-effort stop of a runaway thread. True when the thread is confirmed gone.

    Verified against all three cases that matter: it unwinds a pure-Python runaway, it survives a
    test's broad ``except Exception``, and it honestly reports False for a thread blocked outside
    the interpreter (see the module docstring's BOUNDARY).
    """
    tid = getattr(thread, "ident", None)
    if tid is None:
        return False
    marked = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(Abandoned)
    )
    if not injection_landed(marked):
        if (
            marked > 1
        ):  # pragma: no cover — CPython: "you're in trouble"; undo the over-broad set
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return False
    thread.join(timeout=UNWIND_S)
    return not thread.is_alive()


# How long a worker whose child processes were just killed is given to unwind. The kill ends the wait
# the instant the kernel delivers it and the injection already pending lands at the next bytecode, so
# this is a ceiling for a loaded machine, paid only on a timeout and only when a child wait was found.
CHILD_WAIT_GRACE_S = 2.0


def process_tree_members(
    roots: list[int], children_of: dict[int, list[int]], protected: list[int]
) -> list[int]:
    """The process ids to stop for a worker parked on ``roots``: each root and everything beneath it (#29, pure — pinned).

    A child the worker waits on may have children of its own (a test that runs a whole pytest session
    in a subprocess does), and one that inherited the pipe keeps the wait alive after its parent dies,
    so the root alone is not enough. Split from :func:`release_child_waits` because the process table
    cannot be supplied as a literal; this takes the table as a ``dict``.

    Never a process this decision must not name: ``protected`` (the measuring process itself, whose id
    a stale or forged table could place under a root), ids ``<= 1`` (init, or a group id passed by
    mistake), and anything beneath a protected id. A cycle in the table ends the walk instead of
    looping. Sorted, so the same table always gives the same answer.
    """
    guarded = set(protected)
    seen: set[int] = set()
    frontier = list(roots)
    # Generation by generation, with a bound no table can exceed (every id is entered once, so the
    # depth is at most the number of ids): a malformed table ends the walk, and a mutant of the
    # guards below cannot make it loop without end.
    for _ in range(len(children_of) + len(roots) + 1):
        following: list[int] = []
        for pid in frontier:
            if pid <= 1 or pid in guarded or pid in seen:
                continue
            seen.add(pid)
            following.extend(children_of.get(pid, ()))
        frontier = following
    return sorted(seen)


def child_wait_disposition(found: int, gone: bool) -> str:
    """What stopping a worker's child processes achieved (#29, pure — pinned).

      "no_child_wait"  the worker was parked on no child process, so there was nothing to stop and
                       the worker is exactly as uncontained as it was
      "released"       its children were stopped and the worker has since ended
      "survived"       its children were stopped and the worker is STILL alive: it is blocked on
                       something else as well (or on the child's pipe, held by a process outside the
                       tree), and stays uncontained

    ``found`` is how many child processes the worker's own frames named; ``gone`` whether the worker
    had ended after they were stopped. Three states rather than a bool: "nothing to release" and "I
    released and it did not help" have different remedies, and only the second says that the tree
    walk missed a holder.
    """
    if found <= 0:
        return "no_child_wait"
    return "released" if gone else "survived"


def _popen_pids(thread: Any) -> list[int]:
    """Process ids of the unreaped ``subprocess.Popen`` objects the thread's own frames hold.

    A worker parked in ``subprocess.run`` / ``communicate`` / ``wait`` holds its child in a local of
    ``run`` and as ``self`` of ``communicate``, so the frames name exactly the children THIS worker
    is waiting on and no other process: not a sibling's, not the measuring process's own helpers.
    A child already reaped (``returncode`` set) is skipped, since its id may belong to someone else.
    Best-effort: a frame that will not give up its locals names nothing.
    """
    frame = sys._current_frames().get(getattr(thread, "ident", None))
    pids: list[int] = []
    while frame is not None:
        try:
            values = list(frame.f_locals.values())
        # BLE001: a frame that cannot be read names no child; it must not fail the stop
        except Exception:  # noqa: BLE001
            values = []
        for value in values:
            # `pid` is None until the spawn returns: a Popen caught mid-construction names no child
            if (
                isinstance(value, subprocess.Popen)
                and value.returncode is None
                and isinstance(value.pid, int)
                and value.pid not in pids
            ):
                pids.append(value.pid)
        frame = frame.f_back
    return pids


def _children_map() -> dict[int, list[int]]:
    """``{parent: [children]}`` for every process, from ``/proc`` or else ``ps``; empty when neither answers."""
    parents: dict[int, int] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        names = []
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        # the command name sits in parentheses and may hold spaces; the fields after it do not
        tail = data.rsplit(b")", 1)[-1].split()
        if len(tail) > 1 and tail[1].isdigit():
            parents[int(name)] = int(tail[1])
    if not parents:
        try:
            listing = subprocess.run(
                ["ps", "-A", "-o", "pid=,ppid="],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        for line in listing.splitlines():
            fields = line.split()
            if len(fields) == 2 and all(f.isdigit() for f in fields):
                parents[int(fields[0])] = int(fields[1])
    children: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        children.setdefault(ppid, []).append(pid)
    return children


def release_child_waits(thread: Any, grace_s: float = CHILD_WAIT_GRACE_S) -> bool:
    """Stop the child processes ``thread`` is parked on, and say whether the thread then ended (#29).

    Called after :func:`abandon` has left an exception pending on a thread that did not unwind. A wait
    on a child ends when the child does, and the pending exception then lands, so killing the children
    is what actually delivers the stop ``abandon`` could not. True only when the thread is confirmed
    gone; ``child_wait_disposition`` names the other two outcomes. The kill is ``SIGKILL`` on the
    worker's own children and everything beneath them (``process_tree_members``), never the measuring
    process.
    """
    roots = _popen_pids(thread)
    members = process_tree_members(
        roots, _children_map() if roots else {}, [os.getpid()]
    )
    sig = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
    for pid in members:
        with contextlib.suppress(OSError):
            os.kill(pid, sig)
    if roots:
        thread.join(timeout=grace_s)
    return child_wait_disposition(len(roots), not thread.is_alive()) == "released"


def blocked_site(thread: Any) -> str:
    """Where ``thread`` is parked, as ``file:line in function`` for its innermost frame, and the
    innermost frame OUTSIDE the standard library and site-packages (the test's own line) when that is
    a different one; "" when the thread has no frame (#29)."""
    frame = sys._current_frames().get(getattr(thread, "ident", None))
    if frame is None:
        return ""
    stdlib = os.path.dirname(os.__file__)
    inner = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno} in {frame.f_code.co_name}"
    while frame is not None:
        path = frame.f_code.co_filename
        if not path.startswith(stdlib) and "site-packages" not in path:
            own = f"{path}:{frame.f_lineno} in {frame.f_code.co_name}"
            return inner if own.endswith(inner) else f"{inner}, called from {own}"
        frame = frame.f_back
    return inner


class _Abandoned:
    """One worker a bounded join gave up on while it was still running (#29)."""

    __slots__ = ("label", "thread", "where")

    def __init__(self, thread: Any, where: str, label: str) -> None:
        self.thread = thread
        self.where = where
        self.label = label


# Every worker a bounded join left running, oldest first. Never pruned: an entry whose thread has
# since ended is how `abandoned_standing` tells "it ended" from "it was never observed".
_ABANDONED: list[_Abandoned] = []


def note_abandoned(thread: Any, where: str, label: str = "") -> None:
    """Record a worker left running past its stop, with the place it is parked (#29).

    ``label`` names the test it was running when the caller knows it; the baseline trace's join does
    not, and :func:`claim_abandoned` labels those after the fact.
    """
    _ABANDONED.append(_Abandoned(thread, where, label))


def claim_abandoned(label: str) -> None:
    """Give ``label`` to every recorded worker that has none yet (#29).

    Called by the owner of a loop that runs ONE test at a time (the baseline trace), straight after
    each test: whatever was left running since the last call is that test's.
    """
    for record in _ABANDONED:
        if not record.label:
            record.label = label


def abandoned_worker_standing(recorded: int, alive: int) -> str:
    """Whether the workers left running under one label are still a hazard (#29, pure — pinned).

    "unobserved"  no worker was recorded under the label: nothing is known about it, so it is
                  treated as live. Absence of a record is not evidence that it ended.
    "live"        at least one is still alive: it shares this process's working directory,
                  environment, standard streams and patched module attributes, and runs the
                  unwinding of its test (monkeypatch undo, fixture teardown) whenever its blocking
                  call returns. No measurement taken while it lives is clean.
    "ended"       every recorded worker has since ended: the hazard lapsed. A measurement that
                  began after the last one ended shares nothing with it.
    """
    if recorded <= 0:
        return "unobserved"
    return "live" if alive > 0 else "ended"


def abandoned_standing(label: str) -> str:
    """:func:`abandoned_worker_standing` for the workers recorded under ``label``, read now."""
    mine = [r for r in _ABANDONED if r.label == label]
    return abandoned_worker_standing(
        len(mine), sum(1 for r in mine if r.thread.is_alive())
    )


def abandoned_site(label: str) -> str:
    """Where the (most recent) worker recorded under ``label`` was parked, or "" when none was."""
    for record in reversed(_ABANDONED):
        if record.label == label:
            return record.where
    return ""


def bounded_join(
    thread: Any, timeout_s: float | None, unwind_s: float = 0.0, label: str = ""
) -> tuple[bool, bool]:
    """Wait for ``thread`` at most ``timeout_s`` (``None`` = unbounded) and, on EVERY way out, leave
    no runaway behind. Returns ``(timed_out, contained)``: ``timed_out`` when the wait expired with
    the thread still running; ``contained`` when it is confirmed gone afterwards (``abandon``'s
    honest answer, then :func:`release_child_waits` for a worker parked on a child process — False
    for a thread blocked outside the interpreter on anything else). ``unwind_s`` is extra
    settling time the caller grants after the injection, on top of ``abandon``'s own. A worker that
    is still running is recorded (:func:`note_abandoned`) under ``label`` with where it is parked.

    THE ORPHAN (measured 2026-09-06, `detective converge` on its own engine): a bounded join is often
    NESTED — the classifier's `_outcome` joins a runaway mutant from inside a test that the traced
    baseline is itself running in a worker, and the baseline's budget can abandon THAT worker while
    it is parked in this join. The injected exception lands at the first bytecode after the join
    returns — before the line that would have abandoned the runaway — so the runaway was orphaned:
    a live thread hogging the GIL for the rest of the process (two of them in the faulthandler
    dump, thread numbers past 4,600), and every later phase crawled. Written as `join; if alive:
    abandon`, every one of the four bounded-join sites in the pair had this hole. The `finally` is
    the fix: the caller may leave by any exception, its own abandonment included, and the thread it
    was bounding is stopped on the way out. What `abandon` cannot reach (the BOUNDARY above) is
    still reported, never claimed.
    """
    timed_out = False
    contained = True
    try:
        thread.join(timeout=timeout_s)
    finally:
        timed_out = thread.is_alive()
        if timed_out:
            contained = abandon(thread)
            if not contained:
                where = blocked_site(thread)
                contained = release_child_waits(thread)
                if unwind_s > 0 and not contained:
                    thread.join(timeout=unwind_s)
                    contained = not thread.is_alive()
                if not contained:
                    note_abandoned(thread, where, label)
    return timed_out, contained


def acquire_unstranded(lock: Any, timeout: float = -1) -> bool:
    """Take an RLock so that an injected :class:`Abandoned` cannot strand it (W#29). The caller
    writes ``try: … finally: lock.release()`` on the next line, never ``with lock:``.

    MEASURED 2026-10-25: Wesker's whole-codebase CI run sat silent for six hours on 10 pushes of
    10. Its all-thread dump showed exactly two threads, both parked acquiring the monitoring
    broker's lock. An RLock never blocks its own owner, so neither thread held it: the owner was
    a test's runaway thread, abandoned while it was taking that lock, and by then dead (the
    module docstring's STRANDED LOCK). On a deterministic harness (lock contended, injection made
    while the taker waits, lock then handed over), ``with lock:`` strands 50/50 on 3.14 and 0/50
    on 3.12 and 3.13. A ``@contextmanager`` strands 40/40 on all three. This function strands
    0/40 on all three.

    THE CLOSE: the acquire runs INSIDE the ``try``, so an exception raised at the check after it
    returns lands in the handler, which undoes exactly this acquire. The test is the recursion
    DEPTH, not ``_is_owned``: a re-entrant entry already owns the lock, so only the depth says
    whether this acquire took (``_recursion_count`` is on the C RLock in every supported version;
    ``test_rlock_recursion_count_still_reports_depth`` fails loudly if CPython drops it).
    Returning here and then entering the caller's ``try`` passes no eval-breaker check, so
    nothing can land between the two.

    One injection per thread is the model ``abandon`` provides. A second asynchronous exception
    arriving inside the handler itself could still strand the lock; nothing in the pair sends
    one.
    """
    depth = lock._recursion_count()
    try:
        return lock.acquire(timeout=timeout)
    except BaseException:
        if lock._recursion_count() > depth:
            lock.release()
        raise
