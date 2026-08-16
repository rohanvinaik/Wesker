"""Hermetic-first kill-loop ordering: a fast test that kills a mutant runs BEFORE the slow one.

`evaluate_mutant` short-circuits on the first assertion/exception kill, so the ORDER of a mutant's
covering tests decides whether an expensive shape-hazardous test (subprocess/thread/signal — a 50s
live-game system test in the wild) runs at all. `_build_test_scope` used to hand them over in
discovery order, so the slow test could run BEFORE a fast killer and its cost was paid on every
mutant even though a cheap test would have killed it. Now covering tests are ordered hermetic-first
(the resilient shape check, source-scanned on the live path where the stamp is absent), so the
short-circuit spares the slow test on every mutant a hermetic test kills.

Pinned through the REAL live-session profiling path: with a hermetic killer and a shape-hazardous
test both covering the target, the shaped test runs STRICTLY FEWER times than the hermetic one — it
is short-circuited past on every mutant the hermetic test kills, and runs only for the equivalents
no test can kill. Reordering is verdict-independent, so nothing is lost by running it less.
"""

from __future__ import annotations

import os

import pytest

Detective = pytest.importorskip("Detective")
from Detective import engine as deng  # noqa: E402
from Wesker import engine as weng  # noqa: E402
from Wesker.ci import callable_test_id, run_with_live_suite  # noqa: E402

pytestmark = pytest.mark.skipif(
    not getattr(deng, "_WESKER_TARGET_FIRST", False),
    reason="installed Wesker has no target-first seed/widen path",
)


def _repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\nmarkers = ['detective: generated']\n"
    )
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    # No branches: a single assertion kills EVERY mutant of `x * 2`, so there are no survivors that
    # would force the shaped test to run anyway.
    (pkg / "mod.py").write_text("def target(x):\n    return x * 2\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_fast.py").write_text(
        "from pkg.mod import target\n\ndef test_fast_kills():\n    assert target(3) == 6\n"
    )
    # Covers + kills identically, but is SHAPE-HAZARDOUS (subprocess) -> must be ordered LAST.
    (tests / "test_shaped.py").write_text(
        "import subprocess\n"
        "from pkg.mod import target\n\n"
        "def test_shaped_kills():\n"
        "    subprocess.run(['true'])\n"
        "    assert target(3) == 6\n"
    )
    return str(tmp_path)


def test_shaped_covering_test_is_not_run_when_a_hermetic_test_kills_first(
    tmp_path, monkeypatch
):
    root = _repo(tmp_path)
    runs: dict[str, int] = {}
    real = weng._run_test_with_timeout

    def spy(test_fn, mutated_obj, patched, remaining_ms):
        tid = callable_test_id(test_fn)
        runs[tid] = runs.get(tid, 0) + 1
        return real(test_fn, mutated_obj, patched, remaining_ms)

    monkeypatch.setattr(weng, "_run_test_with_timeout", spy)

    result = {}

    def body():
        result["r"] = deng.profile(
            "pkg/mod.py", "target", root, scope_tests=True, use_cache=False
        )

    run_with_live_suite(
        root,
        body,
        target_files=[os.path.join(root, "pkg", "mod.py")],
        paths=[os.path.join(root, "tests")],
    )

    shaped_runs = sum(n for tid, n in runs.items() if "shaped" in tid)
    hermetic_runs = sum(n for tid, n in runs.items() if "fast" in tid)
    r = result["r"]
    # The hermetic killer runs once per mutant; the shaped test, ordered LAST, runs STRICTLY FEWER
    # times — every mutant the hermetic test kills short-circuits the loop before the shaped test's
    # turn. The shaped test runs ONLY for mutants no hermetic test kills (equivalents/survivors),
    # never as a redundant re-check of a killed one. Were the ordering broken (shaped discovered
    # first), it would run at least as often as the hermetic test.
    assert hermetic_runs >= 1, runs
    assert r.total_killed >= 1, (
        r.total_killed
    )  # kills happened, so the short-circuit had something to spare
    assert shaped_runs < hermetic_runs, runs
