"""Under `--isolated` the measuring process does not run a mutant uncapped (Wesker #45 / #39).

THE DEFECT, measured 2026-11-12 on the fixture `test_memory_standing_intent` already uses
(`hog.py::collect`, `python -m Wesker hog.py --isolated --mem-limit-mb 64`). The worker is cut at its
cap, correctly, and a named `memory` cut is recorded. The engine then ran `check_equivalent` on the
survivors and on that CUT mutant, in the PARENT, on inputs it invented, where no cap applies: the
parent went from 57 MB to 6105 MB resident, stopped only by an outer `ulimit -v 6000000`. The rule the
write and process-effect windows already follow applies: code the engine runs on its own initiative must
not be able to take the measuring process down.

THE FIX. A mutant the worker already cut for memory is not run again in the parent
(`equivalence_probe_standing`), and every probe that does run is inside a soft address-space window sized
by the worker's cap (`bounded_address_space`, `speculative_address_cap`); an allocation past it is a
REFUSAL of that mutant (`_speculative_call`), never an exception to compare and never evidence of
equivalence. Every test that allocates runs in a subprocess under an outer address-space limit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from Wesker.engine import (
    _speculative_call,
    _StepArm,
    equivalence_probe_standing,
)
from Wesker.memory_guard import speculative_address_cap

_MB = 1024 * 1024
INF = -1

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="RLIMIT_AS is measured on Linux"
)

# ── the pure decisions (pinned) ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("vm", "extra", "soft", "hard", "expected"),
    [
        (1000, 500, INF, INF, 1500),
        (1000, 500, 1200, INF, 1200),
        (1000, 500, INF, 1100, 1100),
        (1000, 500, 1300, 1100, 1100),
        (1000, 500, 5000, 6000, 1500),
        (0, 500, INF, INF, INF),
        (-1, 500, INF, INF, INF),
        (1000, 0, INF, INF, INF),
        (1000, -5, INF, INF, INF),
    ],
)
def test_the_window_never_widens_a_limit_and_sets_nothing_it_cannot_size(
    vm, extra, soft, hard, expected
):
    assert speculative_address_cap(vm, extra, soft, hard, INF) == expected


@pytest.mark.parametrize(
    ("isolated", "cut", "expected"),
    [
        (True, True, "refused_memory_cut"),
        (True, False, "probe"),
        (False, True, "probe"),
        (False, False, "probe"),
    ],
)
def test_a_mutant_the_worker_cut_for_memory_is_not_run_again_in_the_parent(
    isolated, cut, expected
):
    assert equivalence_probe_standing(isolated, cut) == expected


def test_running_out_of_memory_is_a_refusal_not_an_observation():
    def hog(_n):
        raise MemoryError

    result, exc, refused = _speculative_call(hog, (1,), _StepArm((), None), ([],))
    assert (result, exc, refused) == (None, None, True)


# ── the window, measured (subprocess, outer limit) ──────────────────────────────────────────


def _run(script: str, outer_mb: int = 6000) -> subprocess.CompletedProcess:
    import resource

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (outer_mb * _MB, outer_mb * _MB))

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
        preexec_fn=cap,
    )


@linux_only
def test_the_window_stops_an_allocation_past_its_bound_and_is_restored_after():
    done = _run(
        """
        import resource
        from Wesker.memory_guard import bounded_address_space
        before = resource.getrlimit(resource.RLIMIT_AS)
        with bounded_address_space(64 * 1024 * 1024) as bounded:
            assert bounded is True
            try:
                junk = [bytearray(8 * 1024 * 1024) for _ in range(40)]
            except MemoryError:
                print("stopped")
        assert resource.getrlimit(resource.RLIMIT_AS) == before, "restored"
        ok = bytearray(300 * 1024 * 1024)
        print("after", len(ok))
        with bounded_address_space(None) as bounded:
            assert bounded is False
            assert resource.getrlimit(resource.RLIMIT_AS) == before
        """
    )
    assert done.returncode == 0, done.stderr[-1500:]
    assert "stopped" in done.stdout and "after 314572800" in done.stdout


_HOG_SRC = textwrap.dedent(
    """
    def collect(n):
        out = []
        i = 0
        while i < n:
            out.append(bytearray(8 * 1024 * 1024))
            i += 1
        return len(out)
    """
)


def _hog_probe(mem_limit: str) -> subprocess.CompletedProcess:
    return _run(
        f"""
        import ast, resource
        from Wesker import engine as E
        node = ast.parse({_HOG_SRC!r}).body[0]
        muts = E.generate_mutants(node, set(E.MutationCategory), max_per_category=0)
        # the mutant that grows without bound on the probe's own inputs: `i += 1` -> `i -= 1`
        hog = [m for m in muts if "replace arithmetic operator" in m.description
               and "i -= 1" in ast.unparse(m.mutated_node)][0]
        verdict = E.check_equivalent(node, hog, {mem_limit})
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        print(verdict, peak)
        """,
        outer_mb=3000,
    )


@linux_only
def test_a_probe_bounded_by_the_workers_cap_refuses_the_hog_and_the_parent_stays_small():
    done = _hog_probe("64")
    assert done.returncode == 0, done.stderr[-1500:]
    verdict, peak = done.stdout.split()
    assert verdict == "False", "a refused mutant is never equivalent"
    assert int(peak) < 600, f"the parent grew to {peak} MB under a 64 MB bound"


@linux_only
def test_without_a_bound_the_same_probe_takes_the_parent_to_the_outer_limit():
    """The control, and the measurement of what the bound prevents (in-process mode's standing:
    memory unbounded, and the page says so)."""
    done = _hog_probe("None")
    assert done.returncode == 0, done.stderr[-1500:]
    _verdict, peak = done.stdout.split()
    assert int(peak) > 1000, (
        f"the control no longer hogs ({peak} MB): this test proves nothing"
    )


# ── end to end: the real command on the hog fixture ─────────────────────────────────────────


@linux_only
def test_the_hog_fixture_under_isolated_keeps_the_parent_small_and_exits_clean(
    tmp_path,
):
    import test_memory_standing_intent as fixture

    (tmp_path / "tests").mkdir()
    (tmp_path / "hog.py").write_text(fixture._HOG)
    (tmp_path / "tests" / "test_hog.py").write_text(fixture._TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npythonpath = .\n"
    )
    driver = textwrap.dedent(
        """
        import resource, sys
        from Wesker import cli
        rc = cli.main(["hog.py", "--isolated", "--mem-limit-mb", "64", "--json"])
        sys.stderr.write("PEAK_MB %d\\n" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024))
        sys.exit(rc)
        """
    )
    import resource

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (6000 * _MB, 6000 * _MB))

    done = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=tmp_path,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
        preexec_fn=cap,
    )
    assert done.returncode == 0, done.stderr[-1500:]
    peak = int(done.stderr.rsplit("PEAK_MB", 1)[1].split()[0])
    assert peak < 1000, f"the parent peaked at {peak} MB"
    mutation = json.loads(done.stdout)["mutation"]
    cuts = {t["function_key"]: t["cause"] for t in mutation["truncated_functions"]}
    assert cuts == {"hog.py::collect": "memory"}
    assert mutation["total_equivalent"] == 0, (
        "a cut mutant is never declared equivalent"
    )
