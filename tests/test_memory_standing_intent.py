"""Intent: what bounds memory on a run is REPORTED, and a cap the OS refused is never silent (W#45, W#39).

THE DEFECT. `--mem-limit-mb` is a cap the OS can refuse (macOS: `apply_address_limit` returns False and
the standing is `telemetry_only`). `memory_standing` was computed per function and sat in
`ProfilingResult.to_dict`, but the rollup the CLI prints and serializes had no such field, so a run
over an unapplied cap read like a bounded one. Worse, the DEFAULT path (`run_function_converged` via
`profile_file`, no `--isolated`) has no per-mutant memory bound at all, and `--mem-limit-mb` (default
512) was accepted and silently ignored there.

THE RULING (Detective tasks 14.3a, B4): a memory-cap stop is a NAMED CUT, never a kill. The rollup
carries `isolated`, `memory_limit_mb`, `memory_standing` (one code) and `memory_standings` (functions
per standing); the summary prints a Memory line; a memory cut is named per function with its remedy; the
help text says the cap applies only with `--isolated`.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys

import pytest

from Wesker import cli
from Wesker.ci import (
    describe_memory,
    describe_unresolved,
    memory_rollup_standing,
    profile_codebase,
)

# ── the pure decision ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("isolated", "limit", "standings", "code"),
    [
        (False, 0, [], "unbounded"),
        (False, 0, ["n/a", "n/a"], "unbounded"),
        (False, 64, ["n/a"], "limit_ignored"),
        (True, 0, ["telemetry_only"], "no_limit"),
        (True, -5, ["enforced"], "no_limit"),
        (True, 64, ["enforced", "enforced"], "enforced"),
        (True, 64, ["cut"], "enforced"),
        (True, 64, ["enforced", "cut"], "enforced"),
        (True, 64, ["telemetry_only"], "telemetry_only"),
        (True, 64, ["enforced", "telemetry_only"], "partial"),
        (True, 64, ["cut", "telemetry_only"], "partial"),
        (True, 64, [], "none_measured"),
        (True, 64, ["n/a"], "none_measured"),
    ],
)
def test_the_rollup_standing_code(isolated, limit, standings, code):
    assert memory_rollup_standing(isolated, limit, standings) == code


def test_every_code_renders_a_different_line_and_a_missing_record_is_not_a_code():
    rendered = {}
    for code in (
        "unbounded",
        "limit_ignored",
        "no_limit",
        "enforced",
        "telemetry_only",
        "partial",
        "none_measured",
    ):
        rendered[code] = describe_memory(
            {
                "memory_standing": code,
                "memory_limit_mb": 64,
                "memory_standings": {"telemetry_only": 1},
            }
        )
    assert len(set(rendered.values())) == 7
    assert "NOT applied on this platform" in rendered["telemetry_only"]
    assert "can exhaust this machine" in rendered["telemetry_only"]
    assert "NOT bounded" in rendered["unbounded"]
    assert "--isolated" in rendered["unbounded"]
    assert "NOT applied" in rendered["limit_ignored"]
    assert "never a kill" in rendered["enforced"]
    missing = describe_memory({"kill_pct": 90})
    assert "not recorded" in missing and missing not in rendered.values()


# ── the rollup consumes the per-function standing ─────────────────────────────────────────────


def _fn(key: str, standing: str) -> dict:
    return {"function_key": key, "memory_standing": standing, "total_mutants": 1}


def _rollup(monkeypatch, results, **kw):
    monkeypatch.setattr("Wesker.ci.profile_file", lambda *a, **k: results)
    monkeypatch.setattr("Wesker.ci._load_cached_state", lambda root: None)
    return profile_codebase(".", ["m.py"], verbose=False, **kw)


def test_rollup_carries_the_standing_the_functions_computed(monkeypatch):
    report = _rollup(
        monkeypatch,
        [_fn("m.py::a", "telemetry_only"), _fn("m.py::b", "telemetry_only")],
        isolated=True,
        worker_mem_limit_mb=64,
    )
    assert report["isolated"] is True
    assert report["memory_limit_mb"] == 64
    assert report["memory_standing"] == "telemetry_only"
    assert report["memory_standings"] == {"telemetry_only": 2}
    assert "NOT applied on this platform" in describe_memory(report)


def test_rollup_on_the_default_path_says_unbounded_not_enforced(monkeypatch):
    report = _rollup(monkeypatch, [_fn("m.py::a", "n/a")])
    assert report["memory_standing"] == "unbounded"
    assert report["memory_limit_mb"] == 0
    assert report["memory_standings"] == {"n/a": 1}


def test_a_cap_given_without_isolated_is_reported_ignored(monkeypatch):
    report = _rollup(monkeypatch, [_fn("m.py::a", "n/a")], worker_mem_limit_mb=64)
    assert report["memory_standing"] == "limit_ignored"
    assert "--mem-limit-mb 64 was NOT applied" in describe_memory(report)


# ── a memory cut is named per function with its remedy, and is a cut ──────────────────────────


def test_a_memory_cut_is_named_with_its_remedy_and_never_as_a_kill():
    """W#45: a mutant the cap stopped is ONE mutant with no verdict, named under `unresolved` like
    `diverged`. It is no longer a cut of the function (the old `memory` truncation cause said the
    function's other mutants were not evaluated, which was the defect)."""
    text = describe_unresolved(
        [
            {
                "function_key": "m.py::hog",
                "cause": "memory_cut",
                "mutants": ["STMT_a", "BOUNDARY_b"],
            }
        ]
    )
    assert "--mem-limit-mb" in text
    assert "neither killed nor survived" in text
    assert "m.py::hog — STMT_a, BOUNDARY_b" in text
    assert "every other mutant of the function was evaluated" in text
    assert "could not stop" not in text


# ── the CLI: flag truth ───────────────────────────────────────────────────────────────────────


def _run_cli(monkeypatch, tmp_path, argv):
    (tmp_path / "foo.py").write_text("def f(): return 1\n")
    seen: dict = {}

    def fake(root, files, **kwargs):
        seen.update(kwargs)
        return {
            "kill_pct": 100,
            "total_killed": 1,
            "total_mutants": 1,
            "total_functions": 1,
            "elapsed_ms": 1,
        }

    import Wesker.ci

    monkeypatch.setattr(Wesker.ci, "profile_codebase_live", fake)
    monkeypatch.chdir(tmp_path)
    assert cli.main([*argv, "--quiet", "foo.py"]) == 0
    return seen


def test_isolated_without_a_limit_gets_the_default_cap(monkeypatch, tmp_path):
    seen = _run_cli(monkeypatch, tmp_path, ["--isolated"])
    assert seen["isolated"] is True
    assert seen["worker_mem_limit_mb"] == 512


def test_no_flag_passes_no_cap_so_the_report_cannot_claim_one(monkeypatch, tmp_path):
    seen = _run_cli(monkeypatch, tmp_path, [])
    assert seen["isolated"] is False
    assert seen["worker_mem_limit_mb"] is None


def test_an_explicit_cap_without_isolated_still_reaches_the_rollup(
    monkeypatch, tmp_path
):
    seen = _run_cli(monkeypatch, tmp_path, ["--mem-limit-mb", "128"])
    assert seen["isolated"] is False
    assert seen["worker_mem_limit_mb"] == 128


def test_the_help_text_says_the_truth(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert (
        "WESKER_MEM_BUDGET_MB" in out
    )  # W#29: the in-process bound is the run's budget
    assert "ONLY with --isolated" in out
    assert "named cut, never a kill" in out


# ── Linux interception, measured through the real command ─────────────────────────────────────

_HOG = """\
def ok(a):
    return a + 1


def collect(n):
    out = []
    i = 0
    while i < n:
        out.append(bytearray(8 * 1024 * 1024))
        i += 1
    return len(out)
"""
_TEST = """\
from hog import ok, collect


def test_ok():
    assert ok(1) == 2
    assert ok(5) == 6


def test_collect():
    assert collect(2) == 2
    assert collect(0) == 0
"""


def _outer_cap():
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))


@pytest.mark.skipif(
    sys.platform != "linux", reason="RLIMIT_AS interception is measured on Linux"
)
def test_a_runaway_mutant_under_the_cap_is_a_named_memory_cut_and_the_parent_survives(
    tmp_path,
):
    """The issue's missing measurement: a mutant that allocates without bound under
    `--isolated --mem-limit-mb 64` ends as a `memory` cut for ITS function; the parent exits 0 and the
    other function's result is intact. (An outer 6 GB address-space limit keeps a failure of the
    cap from taking the machine with it.)"""
    (tmp_path / "tests").mkdir()
    (tmp_path / "hog.py").write_text(_HOG)
    (tmp_path / "tests" / "test_hog.py").write_text(_TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npythonpath = .\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "Wesker",
            "hog.py",
            "--isolated",
            "--mem-limit-mb",
            "64",
            "--json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
        preexec_fn=_outer_cap,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    mutation = json.loads(done.stdout)["mutation"]
    # not a cut of the function (W#45): its other mutants were evaluated; the capped ones are named
    assert mutation["truncated_functions"] == []
    assert mutation["total_truncated"] == 0
    named = {u["function_key"]: u["cause"] for u in mutation["unresolved_functions"]}
    assert named == {"hog.py::collect": "memory_cut"}
    assert mutation["memory_standing"] == "enforced"
    assert mutation["memory_standings"] == {"cut": 1, "enforced": 1}
    assert mutation["total_functions"] == 2
    # the cut mutant is neither killed nor survived: the ratio holds what the other function measured
    assert mutation["total_mutants"] >= 1
