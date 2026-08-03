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
    resource = None  # type: ignore[assignment]

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


def over_budget(budget_bytes: int | None = None) -> bool:
    """True when this process has crossed the memory budget and must stop growing."""
    budget = budget_bytes if budget_bytes is not None else resolve_budget()
    return process_rss_bytes() > budget


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
