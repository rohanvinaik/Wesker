"""A capacity-aware memory guard for the mutation engine.

Mutation profiling materializes a function's whole mutant set and accumulates its
kill/survivor records in RAM. For one function that is small, but nothing bounds it
in principle — a pathologically large function, or a long-lived host that never
releases between calls, could grow without a ceiling. This guard gives that ceiling
a GUARANTEE rather than trusting scope and process-exit:

  * the budget is auto-set from the machine's own capacity — a modest, non-intrusive
    fraction of total RAM, so a run never dominates the system;
  * it is a value the user can select (``WESKER_MEM_BUDGET_MB``, or an explicit
    argument), overriding the default;
  * when a run crosses the budget it stops accumulating and reclaims — the caller
    dumps transient state instead of climbing past the ceiling.

Stdlib only (``os`` + ``resource``); no psutil dependency. ``resource`` is Unix-only,
so it is imported defensively: on Windows the RSS self-check degrades to a no-op and the
memory guarantee rests on static worker-count admission (see ``worker_count``), which needs
only arithmetic and is identical on every OS.
"""

from __future__ import annotations

import gc
import os
import sys

try:
    import resource  # Unix only — absent on Windows.
except ImportError:  # pragma: no cover — exercised only on Windows
    # SUPPRESSED DELIBERATELY, and this is the one place in either repo where the suppression is
    # the fix rather than an evasion. The design fact is real: `resource` is genuinely optional
    # at runtime, `None` is genuinely the fallback, and every consumer here guards on
    # `if resource is None: return` before touching it (lines 94, 163).
    #
    # The alternatives were measured and are worse. Annotating `ModuleType | None` makes the six
    # `resource.getrlimit` / `RLIMIT_AS` / `RUSAGE_SELF` reads unresolvable on `ModuleType`,
    # trading one honest diagnostic for six false ones. Branching on `sys.platform` instead of
    # `ImportError` typechecks cleanly but changes WHAT IS BEING ASKED — "is this Windows"
    # rather than "is this module importable" — and this is a memory SAFETY guard, so degrading
    # on a stripped or embedded build must stay driven by the import that actually failed.
    resource = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

_MB = 1024 * 1024
_GB = 1024 * _MB

# Non-intrusive default: an eighth of system RAM, clamped so it is neither trivially
# small on a tiny box nor greedy on a large one. The user overrides this per their
# machine; the fraction is only the sensible starting point.
_DEFAULT_FRACTION = 8
_DEFAULT_FLOOR = 256 * _MB
_DEFAULT_CEILING = 2 * _GB

# Parallel profiling budgets the WHOLE fleet, not one process, so it may claim a larger
# slice of RAM (a quarter, higher ceiling) than the single-process default — still a
# minority of a big box. ``per_worker_peak`` is the conservative RSS a worker may reach;
# the fleet is sized so ``workers × peak <= budget`` BY CONSTRUCTION (the portable
# guarantee), independent of any OS resource limit.
_PARALLEL_FRACTION = 4
_PARALLEL_CEILING = 8 * _GB
_DEFAULT_WORKER_PEAK = 512 * _MB


def system_memory_bytes() -> int:
    """Total physical RAM, or a conservative 4 GB fallback when it cannot be read
    (so the budget is never accidentally unbounded)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 4 * _GB


def default_budget_bytes(system_bytes: int | None = None) -> int:
    """The sensible, non-intrusive default budget: ``system_RAM / 8``, clamped to
    [256 MB, 2 GB]. Pure in ``system_bytes`` so it is testable without reading the
    host."""
    total = system_bytes if system_bytes is not None else system_memory_bytes()
    return max(_DEFAULT_FLOOR, min(total // _DEFAULT_FRACTION, _DEFAULT_CEILING))


def resolve_budget(explicit_mb: int | None = None) -> int:
    """The active budget in bytes, most-specific source winning: an explicit
    argument, else ``WESKER_MEM_BUDGET_MB`` from the environment, else the
    capacity-derived default. A non-positive selection means "unbounded"
    (``sys.maxsize``) — the user opting out, on purpose."""
    if explicit_mb is not None:
        return explicit_mb * _MB if explicit_mb > 0 else sys.maxsize
    env = os.environ.get("WESKER_MEM_BUDGET_MB")
    if env is not None:
        try:
            value = int(env)
            return value * _MB if value > 0 else sys.maxsize
        except ValueError:
            pass
    return default_budget_bytes()


def process_rss_bytes() -> int:
    """This process's peak resident set size. ``ru_maxrss`` is bytes on macOS and
    kilobytes on Linux — normalized to bytes. Peak (not instantaneous) is the
    conservative signal: once the peak crosses the budget the run has already
    demanded that much, so stopping there is the guarantee. Returns 0 where
    ``resource`` is unavailable (Windows) — the self-check simply never fires and the
    static worker-count admission carries the memory guarantee instead."""
    if resource is None:
        return 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def current_rss_bytes() -> int:
    """This process's INSTANTANEOUS resident set size, or 0 when unobtainable.

    `ru_maxrss` is a lifetime peak and never falls, so it cannot answer "how much is resident
    NOW". `resource` offers no current figure, and this module is stdlib-only by contract (no
    psutil), so the platform routes are `/proc/self/statm` on Linux and `mach_task_basic_info`
    via ctypes on macOS.

    MEASURED, and the measurement corrects W#21's premise. The issue predicts that after a
    release the current figure drops while the peak stays high (`current ~20 MB, guard ~116 MB`).
    On macOS/CPython 3.14 it does NOT:

        start                    resident   13.3 MB    resident_max   13.3 MB
        holding 200MB            resident  213.3 MB    resident_max  213.3 MB
        released + gc.collect    resident  213.3 MB    resident_max  213.3 MB

    The allocator keeps the pages, so current and peak are equally "poisoned" and swapping one
    for the other would have fixed nothing. What made the number usable was measuring GROWTH
    within a run (see `run_growth_bytes`): cycles 2 and 3 of the same probe reallocated 200MB and
    resident stayed flat at 213.3, because the retained pages were reused — a later run that
    demands nothing new from the OS correctly shows zero growth.
    """
    if sys.platform == "darwin":
        return _darwin_resident_bytes()
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _darwin_resident_bytes() -> int:
    """`mach_task_basic_info().resident_size` — the only stdlib route to current RSS on macOS."""
    try:
        import ctypes

        class _Info(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time_s", ctypes.c_int32),
                ("user_time_us", ctypes.c_int32),
                ("system_time_s", ctypes.c_int32),
                ("system_time_us", ctypes.c_int32),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
        libc.task_info.restype = ctypes.c_int
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_").value
        info = _Info()
        count = ctypes.c_uint(ctypes.sizeof(_Info) // ctypes.sizeof(ctypes.c_uint))
        # 20 == MACH_TASK_BASIC_INFO
        if libc.task_info(task, 20, ctypes.byref(info), ctypes.byref(count)) != 0:
            return 0
        return int(info.resident_size)
    except Exception:  # noqa: BLE001 — an unavailable probe is not an error; it is degraded mode
        return 0


def rss_capability() -> str:
    """What this platform can actually MEASURE — named, because a guarantee must not be claimed
    on a platform that only observes (W#21).

    ``current``     — instantaneous RSS is readable, so per-run growth is enforceable.
    ``peak_only``   — only ``ru_maxrss``. Growth within a run is not separable from history.
    ``unavailable`` — neither. The static worker-count admission carries the guarantee instead,
                      and no memory figure may be presented as a bound.
    """
    if current_rss_bytes() > 0:
        return "current"
    if process_rss_bytes() > 0:
        return "peak_only"
    return "unavailable"


def memory_budget_standing(
    growth_bytes: int, budget_bytes: int, capability: str
) -> str:
    """Whether THIS RUN has exhausted its memory budget (W#21, pure — pinned).

    Judged on GROWTH SINCE THE RUN BEGAN, never on an absolute process figure. The defect W#21
    names is that one historical spike makes every later low-budget run in a long-lived MCP
    process read as permanently over budget: `ru_maxrss` never falls, so the guard answers a
    question about the process's whole history when it was asked about this run.

    Growth is the honest quantity for a further reason the probe measured: after a release macOS
    keeps the pages, so a later run that reallocates the same amount grows by ZERO. It demanded
    nothing new from the OS, and a budget is about demand.

    Three states, and the third is why this is not a bool:

    ``within``        — growth is inside the budget.
    ``exhausted``     — growth crossed it. A graceful cut, owned by us.
    ``unmeasurable``  — the platform reports no RSS at all, so there is nothing to compare. It
      must NOT read as ``within``: "we looked and it is fine" and "we cannot look" are different
      facts, and collapsing them is how an absent guard comes to read as a passed one. The
      caller keeps running (refusing every run on Windows would be absurd) but may not describe
      the result as memory-bounded.
    """
    if capability == "unavailable":
        return "unmeasurable"
    if budget_bytes <= 0:
        return "within"
    return "exhausted" if growth_bytes > budget_bytes else "within"


def memory_enforcement_standing(applied: bool) -> str:
    """Whether an isolated worker's memory budget is ENFORCED or only observed (W#21, pure — pinned).

    A hard budget is a promise only if the OS keeps it. `apply_address_limit` sets ``RLIMIT_AS`` so a
    runaway allocation fails as a catchable ``MemoryError`` — but Linux cooperates, macOS often
    rejects lowering the limit, and Windows has no ``resource`` module at all, so the cap frequently
    does not land. When it did (``applied``), the boundary is ``enforced``: a mutant cannot grow the
    worker past the cap. When it did not, the run is ``telemetry_only`` — memory is measured, not
    bounded, and a consumer must NOT describe the result as memory-guaranteed. Naming the two apart is
    the whole point of W#21's "do not describe a guarantee when the platform offers only observation":
    collapsing them is how an unenforced run comes to read as a bounded one.
    """
    return "enforced" if applied else "telemetry_only"


def run_baseline_bytes() -> int:
    """The RSS a run starts from. Capture once, before the work, and pass it to `over_budget`.

    Without it the guard compares an ABSOLUTE process figure against a per-run budget, which is
    W#21's defect: in a long-lived MCP process, one earlier spike leaves `ru_maxrss` permanently
    high and every later low-budget run reads as exhausted before it allocates anything.
    """
    return current_rss_bytes() or process_rss_bytes()


def run_growth_bytes(baseline_bytes: int) -> int:
    """RSS growth since a run's baseline, floored at zero.

    Floored because a shrink is not negative demand — it is the allocator returning pages, which
    a budget has no opinion about.
    """
    now = current_rss_bytes() or process_rss_bytes()
    return max(0, now - baseline_bytes)


def over_budget(
    budget_bytes: int | None = None, baseline_bytes: int | None = None
) -> bool:
    """True when THIS RUN has crossed its memory budget and must stop growing.

    `baseline_bytes` is what makes the answer about this run rather than about the process's
    whole history — see `run_baseline_bytes`. Passing None keeps the historical absolute
    comparison, which is retained ONLY so an external caller that never established a run is
    not silently changed; every caller inside Wesker now supplies one.
    """
    budget = budget_bytes if budget_bytes is not None else resolve_budget()
    if baseline_bytes is None:
        return process_rss_bytes() > budget
    return (
        memory_budget_standing(
            run_growth_bytes(baseline_bytes), budget, rss_capability()
        )
        == "exhausted"
    )


def reclaim() -> None:
    """Force a collection to release whatever transient analysis the caller just
    dropped — the "dump" half of the guard, made explicit rather than left to the
    collector's own schedule."""
    gc.collect()


def parallel_budget_bytes(system_bytes: int | None = None) -> int:
    """Total RAM the whole worker fleet may claim — ``system_RAM / 4`` clamped to
    [256 MB, 8 GB]. Larger than the single-process default (it budgets many workers) but
    still a minority of a big box. Pure in ``system_bytes`` for testability."""
    total = system_bytes if system_bytes is not None else system_memory_bytes()
    return max(_DEFAULT_FLOOR, min(total // _PARALLEL_FRACTION, _PARALLEL_CEILING))


def available_cores() -> int:
    """Usable CPU count, leaving 2 for the OS + the parent; at least 1."""
    return max(1, (os.cpu_count() or 2) - 2)


def worker_count(
    per_worker_peak: int | None = None,
    cores: int | None = None,
    budget_bytes: int | None = None,
) -> int:
    """The PORTABLE memory guarantee: how many workers fit without exceeding the fleet
    budget, ``min(cores, ⌊budget / per_worker_peak⌋)``, at least 1.

    Because each worker is admitted only if ``workers × per_worker_peak <= budget``, the
    fleet's total memory is bounded BY CONSTRUCTION — no OS resource limit required, so the
    guarantee holds identically on Mac, Windows and Linux. Deterministic per machine: the
    same box always plans the same fleet. ``resolve_budget`` env/explicit override still
    applies (a user capping ``WESKER_MEM_BUDGET_MB`` shrinks the fleet accordingly)."""
    peak = per_worker_peak or _DEFAULT_WORKER_PEAK
    cores = available_cores() if cores is None else max(1, cores)
    # Honour an explicit/env budget override, else the parallel (fleet) budget.
    explicit = resolve_budget()
    budget = (
        budget_bytes
        if budget_bytes is not None
        else (
            explicit if explicit != default_budget_bytes() else parallel_budget_bytes()
        )
    )
    by_mem = max(1, budget // max(1, peak))
    return max(1, min(cores, by_mem))


def apply_address_limit(peak_bytes: int | None = None) -> bool:
    """Best-effort per-process address-space cap (``RLIMIT_AS``) — a runaway allocation
    then fails as a catchable ``MemoryError`` (a deterministic resource-guard kill) rather
    than an OOM. Returns whether the limit was actually applied.

    This is a Linux BONUS, not the guarantee: macOS often rejects lowering ``RLIMIT_AS``
    and Windows has no ``resource`` module, so it degrades to a no-op there. The portable
    guarantee is the static ``worker_count`` admission; this only hardens it where the OS
    cooperates. Never raises — a platform that refuses the limit is not an error."""
    if resource is None:
        return False
    cap = peak_bytes or _DEFAULT_WORKER_PEAK
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = hard if hard != resource.RLIM_INFINITY and hard < cap else cap
        resource.setrlimit(resource.RLIMIT_AS, (cap, new_hard))
        return True
    except (ValueError, OSError, AttributeError):
        return False


def telemetry(budget_bytes: int | None = None) -> str:
    """A one-line, commonsense memory report for a CLI footer: what this run used,
    against the budget and the machine, with an offload hint when it runs hot. Cheap
    (one getrusage) and never fails — visibility, not enforcement."""
    budget = budget_bytes if budget_bytes is not None else resolve_budget()
    rss = process_rss_bytes()
    total = system_memory_bytes()
    unbounded = budget >= sys.maxsize
    if unbounded:
        return f"mem: {rss // _MB} MB used · budget OFF · system {total // _GB} GB"
    pct = round(100 * rss / budget) if budget else 0
    hot = pct >= 80
    hint = (
        "  ⚠ near budget — offload: `purge` to clear caches, or raise WESKER_MEM_BUDGET_MB"
        if hot
        else ""
    )
    return f"mem: {rss // _MB} MB / {budget // _MB} MB budget ({pct}%) · system {total // _GB} GB{hint}"


def purge_caches(project_root: str) -> tuple[tuple[str, ...], int]:
    """Delete regeneratable analysis cruft under ``project_root`` — the mutation/mcdc
    reports AND the per-test ``trace_cache.json`` a prior run left in ``.wesker/``.

    ``trace_cache.json`` is a target because it is content-keyed and therefore CAN be poisoned:
    a stale or collapsed entry (e.g. a parametrized case mis-keyed so one case's coverage is served
    for all) survives a code change, and `purge` is the documented recovery path — one that must
    actually remove the file it claims to. It was absent from this list for its whole existence, so
    `purge` reported "a clean state" while the bad entry stayed; that is now closed.

    Returns ``(removed_paths, reclaimed_bytes)``. Generated TEST files are the
    product, not cruft, and are never touched. Everything removed here is rebuilt on
    the next run from the current code, so purging can only cost recomputation, never
    correctness — which is the point: a clean restart that guarantees no stale state
    lingers.

    ONLY ``.wesker/``. This cannot purge a consumer's own caches — it does not know they
    exist — so a consumer that keeps state of its own must purge it alongside this rather
    than delegate and assume. Detective's `purge` had delegated here and reported "a clean
    state" over 3.1 MB of its own untouched cache.

    ``function_cache.json`` was dropped from the targets when the subsystem that wrote it
    was removed in 0.6.0: nothing outside its own tests ever called it, and it invalidated
    on the function's hash but NOT its tests', so editing a test served a stale verdict.
    Detective's `verdict_cache` had already rebuilt the same idea keyed on both."""
    wesker_dir = os.path.join(project_root, ".wesker")
    targets = ("mutation_report.json", "mcdc_report.json", "trace_cache.json")
    removed: list[str] = []
    reclaimed = 0
    for name in targets:
        path = os.path.join(wesker_dir, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        try:
            os.remove(path)
        except OSError:
            continue
        removed.append(path)
        reclaimed += size
    return tuple(removed), reclaimed
