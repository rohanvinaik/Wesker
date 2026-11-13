"""An in-process mutant that allocates without bound is one mutant with no verdict, not a dead runner (Wesker #29).

THE DEFECT, from the first sharded badge run (GitHub run 43315433429 on 8fcadfe): shard 4/4 was killed by the
kernel (`Killed`, exit 137) while it was evaluating a mutant of `interrupt.py`. The badge run is not
`--isolated`, so the mutant runs in the measuring process, and only the isolated worker had an address-space
cap: `memory_guard.apply_address_limit` and the `memory_cut` disposition were reachable from `--isolated`
alone, and the CLI's own help said so ("Without it NO per-mutant memory bound exists").

Measured on this fixture (`span(hi, lo)` returns `len(b"a" * (hi - lo))`; the mutant `hi - lo` -> `hi + lo`
allocates 5 GB where the original allocates 100 bytes) under an outer `RLIMIT_AS` of 6000 MB, which stands in
for the runner's RAM: the process peaked at 4809 MB and the mutant was scored a kill. With the cap it peaks
at the baseline and the mutant is `memory_cut`.

THE RULING pinned here. In process a mutant may grow the measuring process by the run's memory budget
(`WESKER_MEM_BUDGET_MB`, else a capacity-derived default; a non-positive budget is the user opting out). An
allocation past it raises `MemoryError`, and a mutant that does so, whoever catches it, is the unscored
outcome `memory_cut` that the isolated worker's cap already produces: one mutant with no verdict, never a
kill, a survivor or an equivalent, while the function's other mutants are still evaluated. This is the
in-process counterpart of the isolated path's disposition, not a second name for it. On a platform that will
not take the limit (macOS often does not) nothing is capped and nothing is claimed.

Every test that allocates runs in a subprocess under an outer address-space limit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from Wesker.engine import MEMORY_CUT, memory_cut_run_standing
from Wesker.memory_guard import inprocess_cap_mb, inprocess_cap_standing

_MB = 1024 * 1024

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="RLIMIT_AS is measured on Linux"
)

# ── the pure decisions ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("explicit", "budget", "expected"),
    [
        (64, 2048 * _MB, "explicit"),
        (1, sys.maxsize, "explicit"),
        (0, 2048 * _MB, "budget"),
        (0, 1, "budget"),
        (0, sys.maxsize, "unbounded"),
        (-5, sys.maxsize, "unbounded"),
        (-5, 100 * _MB, "budget"),
        (0, 1 << 62, "unbounded"),
        (0, (1 << 62) - 1, "budget"),
        (0, 0, "budget"),
    ],
)
def test_where_the_in_process_cap_comes_from(explicit, budget, expected):
    assert inprocess_cap_standing(explicit, budget) == expected


@pytest.mark.parametrize(
    ("failed", "raised", "expected"),
    [
        (True, True, "cap_failed_node"),
        (True, False, "cap_failed_node"),
        (False, True, "cap_swallowed"),
        (False, False, "no_cap"),
    ],
)
def test_the_in_process_decision_is_the_isolated_workers(failed, raised, expected):
    from Wesker._isolated_worker import memory_cut_standing

    assert memory_cut_run_standing(failed, raised) == expected
    assert memory_cut_standing(failed, raised) == expected


def test_the_cap_follows_the_budget_and_an_explicit_cap_wins(monkeypatch):
    monkeypatch.setenv("WESKER_MEM_BUDGET_MB", "300")
    assert inprocess_cap_mb() == 300
    assert inprocess_cap_mb(64) == 64
    monkeypatch.setenv("WESKER_MEM_BUDGET_MB", "0")
    assert inprocess_cap_mb() is None, "a non-positive budget is the opt-out"


# ── the command, measured ────────────────────────────────────────

_SPAN = textwrap.dedent(
    """\
    def span(hi, lo):
        return len(b"a" * (hi - lo))
    """
)

# `except Exception: return -1` swallows the cap's MemoryError; the test then fails on the VALUE -1.
_SWALLOW = textwrap.dedent(
    """\
    def span(hi, lo):
        try:
            return len(b"a" * (hi - lo))
        except Exception:
            return -1
    """
)

_TEST = textwrap.dedent(
    """\
    from span import span


    def test_span():
        assert span(500_000_100, 500_000_000) == 100
        assert span(7, 7) == 0
    """
)

_DRIVER = textwrap.dedent(
    """\
    import resource, sys
    resource.setrlimit(resource.RLIMIT_AS, (3000 * 1024 * 1024,) * 2)
    from Wesker import cli
    rc = cli.main(["span.py", "--json"])
    sys.stderr.write("PEAK_MB %d\\n" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024))
    sys.exit(rc)
    """
)


def _measure(tmp_path, source: str, budget_mb: str):
    (tmp_path / "tests").mkdir()
    (tmp_path / "span.py").write_text(source)
    (tmp_path / "tests" / "test_span.py").write_text(_TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npythonpath = .\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUNBUFFERED",)}
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=os.pathsep.join(sys.path),
        WESKER_MEM_BUDGET_MB=budget_mb,
    )
    done = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-1500:]
    peak = int(done.stderr.rsplit("PEAK_MB", 1)[1].split()[0])
    return json.loads(done.stdout)["mutation"], peak


def _cut(mutation) -> dict:
    named = {u["function_key"]: u for u in mutation["unresolved_functions"]}
    assert set(named) == {"span.py::span"}, named
    return named["span.py::span"]


@linux_only
def test_a_mutant_that_allocates_past_the_budget_is_memory_cut_and_the_process_stays_small(
    tmp_path,
):
    mutation, peak = _measure(tmp_path, _SPAN, "256")
    assert peak < 700, f"the measuring process grew to {peak} MB under a 256 MB budget"
    cut = _cut(mutation)
    assert cut["cause"] == MEMORY_CUT
    assert len(cut["mutants"]) == 1
    # One mutant without a verdict: the function is not cut, and its other mutants were evaluated.
    assert mutation["truncated_functions"] == []
    assert mutation["total_mutants"] >= 2
    assert mutation["total_killed"] >= 1
    survivors = {s["mutant_id"] for s in mutation["survivors"]}
    assert not set(cut["mutants"]) & survivors, "a cut mutant is not a survivor"


@linux_only
def test_a_memory_error_the_code_under_test_swallows_is_still_the_cap_not_a_kill(
    tmp_path,
):
    mutation, peak = _measure(tmp_path, _SWALLOW, "256")
    assert peak < 700
    cut = _cut(mutation)
    assert cut["cause"] == MEMORY_CUT, (
        "the handler's -1 failed the test on a value, and that failure is the cap's"
    )


@linux_only
def test_a_non_positive_budget_is_the_users_opt_out_and_caps_nothing(tmp_path):
    """The control: with the budget off the same mutant allocates, and is killed on its value."""
    mutation, peak = _measure(tmp_path, _SPAN, "0")
    assert mutation["unresolved_functions"] == []
    assert peak > 700, (
        f"the control no longer allocates ({peak} MB): this proves nothing"
    )
