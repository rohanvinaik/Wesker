"""Within-session trace memo: a converge run traces each covering test ONCE, admissibly.

A converge run makes many seed/widen passes over ONE unchanging target, each `fresh=True` (bypassing
the cross-run disk cache so the trace stays proof-admissible). Without a within-session memo, every
pass re-traces the same covering tests — measured on a real target: a ~50s live-game test re-traced
11× in one converge (~447s). `run_with_live_suite` now carries a per-session `within_run` memo: a
test traced in an earlier pass is REUSED in later passes of the SAME session, and — critically —
that reuse is ADMISSIBLE (it was measured this session), NOT `replayed` like a cross-run disk hit.

Pinned here through the REAL live-session path (two profile passes over one target in one session):
the second pass performs ZERO new line-traces, and the target still gates (the reused coverage
entered the proof basis, so it was not silently demoted to routing-only).
"""

from __future__ import annotations

import os

import pytest

Detective = pytest.importorskip("Detective")
from Detective import engine as deng  # noqa: E402
from Wesker import line_coverage as lc  # noqa: E402
from Wesker.ci import run_with_live_suite  # noqa: E402

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
    (pkg / "mod.py").write_text(
        "def target(x):\n    if x > 0:\n        return x * 2\n    return -x\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_t.py").write_text(
        "from pkg.mod import target\n\n"
        "def test_pos():\n    assert target(3) == 6\n\n"
        "def test_neg():\n    assert target(-4) == 4\n"
    )
    return str(tmp_path)


def test_second_pass_reuses_the_first_passes_traces_within_one_session(
    tmp_path, monkeypatch
):
    root = _repo(tmp_path)
    calls = {"n": 0}
    real = lc._trace_one_multi

    def spy(fn, tf, budget, capture_arcs=True):
        calls["n"] += 1
        return real(fn, tf, budget, capture_arcs=capture_arcs)

    monkeypatch.setattr(lc, "_trace_one_multi", spy)

    seen = {}

    def body():
        deng.profile("pkg/mod.py", "target", root, scope_tests=True, use_cache=False)
        seen["after_pass1"] = calls["n"]
        r2 = deng.profile(
            "pkg/mod.py", "target", root, scope_tests=True, use_cache=False
        )
        seen["after_pass2"] = calls["n"]
        seen["gateable2"] = bool(getattr(r2, "is_gateable", True))

    run_with_live_suite(
        root,
        body,
        target_files=[os.path.join(root, "pkg", "mod.py")],
        paths=[os.path.join(root, "tests")],
    )

    # Pass 1 traced the covering tests; pass 2, in the SAME session, adds ZERO new line-traces.
    assert seen["after_pass1"] >= 1, seen
    assert seen["after_pass2"] == seen["after_pass1"], seen
    # And the reuse was ADMISSIBLE — the target still gates on pass 2, so the reused coverage entered
    # the proof basis rather than being demoted to routing-only (`replayed`).
    assert seen["gateable2"] is True, seen
