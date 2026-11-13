"""A failure that escapes the measurement ends the shard with a NAMED record, never with no record (Wesker #29).

THE DEFECT, from the first sharded badge run (GitHub run 43315433429 on 8fcadfe). Shard 1/4 died on a
traceback out of the baseline trace (`MonitoringObservationFailed: observation_inflight`), the Action
wrote no `.wesker/shard_report.json`, the upload step failed with "No files were found", and the
aggregation could say only `report_missing`: "no report for planned shard(s)", which names no cause. Every
exception that escapes the profile now earns the refusal `measurement_failed`, recorded with the
exception's class, its message and the frame it came from, so the merge states the shard's real cause.

Every Action here runs in a subprocess, as `test_shard_merge_intent.py` does: the Action owns the process's
streams, and the nesting guard (W#52) is for the measuring process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from Wesker.action import escaped_exception_reason
from Wesker.shards import SHARD_REFUSALS, record_standing

REPO = Path(__file__).resolve().parents[1]

DRIVER = textwrap.dedent(
    """\
    import sys
    import Wesker.ci as ci
    from Wesker.action import main

    def _boom(*args, **kwargs):
        raise {exc}

    ci.profile_codebase = _boom
    sys.exit(main(sys.argv[1:]))
    """
)


def _project(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/m.py").write_text("def f(x):\n    return x + 1\n")
    (root / "tests").mkdir()
    (root / "tests/test_m.py").write_text(
        "from pkg.m import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )
    (root / "home").mkdir()


def _run(root: Path, driver: str | None, args: list[str]):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GITHUB_", "WESKER_", "PYTEST_"))
        and k != "PYTHONUNBUFFERED"
    }
    env["HOME"] = str(root / "home")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), *filter(None, [os.environ.get("PYTHONPATH", "")])]
    )
    env["GITHUB_SHA"] = "e83b5fafafc9ef5a10e1c8ab0c41dfe3686c76d7"
    argv = (
        [sys.executable, "-c", driver]
        if driver
        else [sys.executable, "-m", "Wesker.action"]
    )
    return subprocess.run(
        [*argv, *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.fixture
def root(tmp_path):
    _project(tmp_path)
    return tmp_path


# ── The pure reason ──────────────────────────────────────────────


def test_the_reason_names_the_class_the_message_and_the_frame():
    reason = escaped_exception_reason(
        "MonitoringObservationFailed",
        "observation_inflight (subscription=4765)",
        "Wesker/monitoring.py:912 in check",
    )
    assert "MonitoringObservationFailed" in reason
    assert "observation_inflight (subscription=4765)" in reason
    assert "Wesker/monitoring.py:912 in check" in reason


def test_a_blank_message_and_a_blank_frame_are_said_to_be_unavailable():
    reason = escaped_exception_reason("KeyError", "  ", "")
    assert "(no message)" in reason
    assert "(frame unavailable)" in reason


def test_the_refusal_is_one_the_merge_can_restate():
    assert "measurement_failed" in SHARD_REFUSALS


# ── The Action, end to end ───────────────────────────────────────


@pytest.mark.parametrize(
    "exc, shown",
    [
        ('RuntimeError("boom in the profile")', "RuntimeError: boom in the profile"),
        ("SystemExit(7)", "SystemExit: 7"),
    ],
)
def test_an_exception_out_of_the_profile_writes_a_named_shard_record(root, exc, shown):
    proc = _run(
        root,
        DRIVER.format(exc=exc),
        ["--targets", "pkg/m.py", "--shard", "1/1", "--budget", "20000"],
    )
    assert proc.returncode == 1, proc.stderr
    record_path = root / ".wesker/shard_report.json"
    assert record_path.exists(), (
        "no record: the aggregation would read `report_missing`"
    )
    record = json.loads(record_path.read_text())
    assert record_standing(record) == "ok"
    assert record["refusal"] == "measurement_failed"
    assert record["report"] is None
    assert record["targets"] == ["pkg/m.py"]
    assert shown.split(":")[0] in record["refusal_detail"]
    assert shown.split(": ")[1] in record["refusal_detail"]
    # The traceback is still in the log: the record names the cause, it does not replace the evidence.
    assert "Traceback" in proc.stderr
    assert "WESKER REFUSED TO REPORT" in proc.stderr


def test_the_merge_states_the_shards_cause_not_report_missing(root):
    plan = root / "plan.json"
    planned = _run(
        root,
        None,
        ["--plan-shards", "1", "--plan-out", str(plan), "--targets", "pkg/m.py"],
    )
    assert planned.returncode == 0, planned.stderr
    shard = _run(
        root,
        DRIVER.format(exc='RuntimeError("boom in the profile")'),
        ["--targets", "pkg/m.py", "--shard", "1/1", "--budget", "20000"],
    )
    assert shard.returncode == 1
    merged = _run(
        root,
        None,
        ["--merge", "--plan", str(plan), str(root / ".wesker/shard_report.json")],
    )
    assert merged.returncode == 1
    assert "report_missing" not in merged.stderr
    assert "no report for planned shard" not in merged.stderr
    assert (
        "measurement_failed" in merged.stderr or "boom in the profile" in merged.stderr
    )
    assert "boom in the profile" in merged.stderr
    assert "pkg/m.py" in merged.stderr
