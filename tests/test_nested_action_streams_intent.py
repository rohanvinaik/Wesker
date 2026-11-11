"""A test that calls the Action inside the measuring process never takes the run's output away (W#52).

THE DEFECT, measured 2026-10-26. Spec-full run 40254151337 measured `action.py` (109/324 in 51 s). Its
next progress line, `print("[3/24] ci.py", flush=True)`, then raised `OSError: [Errno 9] Bad file
descriptor` on the measuring process's own stdout, and the whole run crashed with exit 1. Self-measuring
`action.py` runs eight test files that call `action.main` INSIDE the measuring process, and the Action's
stream re-entry is itself under mutation: a mutant that closes or misdirects fd 1 (for example
`closefd=False -> True` on the rebound `FileIO(1)`, which closes fd 1 when the wrapper is collected)
leaves the OUTER process's stdout dead. The same nested Actions printed their own refusals into the job
log (`WESKER REFUSED TO REPORT: worker_lost`), and left `Bad file descriptor` finalizer warnings at exit.

THE RULING, pinned here:
- the measuring process's progress and final report reach the real stdout whatever a nested Action, or a
  mutant of the stream code, does to fd 1/2 and to `sys.stdout`/`sys.stderr`: progress goes through a
  descriptor duplicated BEFORE profiling, and the owner's re-entry restores the descriptors after the last
  wrapper a mutant built has been collected;
- a nested Action owns no streams: it neither snapshots nor re-enters them (the owner marker, read with
  the same pure `stack_dump_owner` the stall dump reads), so its output stays inside the calling test's
  capture;
- the owner publishes the marker for exactly as long as it owns the streams.

The descriptor behaviour is process-global, so the wiring cases run in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from Wesker import action, ci

REPO = Path(__file__).resolve().parents[1]

# What a nested Action, or a mutant of the stream code it runs, can do to the process's streams.
_DAMAGE = {
    "closefd_true_rebinding": "sys.stdout = io.TextIOWrapper(io.FileIO(1, 'w', closefd=True)); "
    "sys.stderr = io.TextIOWrapper(io.FileIO(2, 'w', closefd=True))",
    "close_fd1": "os.close(1)",
    "close_fd1_and_fd2": "os.close(1); os.close(2)",
    "close_sys_stdout": "sys.stdout.close(); sys.stderr.close()",
    "fd1_to_devnull": "n = os.open(os.devnull, os.O_WRONLY); os.dup2(n, 1); os.dup2(n, 2)",
    "sys_streams_replaced": "sys.stdout = io.StringIO(); sys.stderr = io.StringIO()",
}

_OUTER = """
    import io, os, sys
    from Wesker import action, ci
    import Wesker.self_profile as sp

    sp.profiler_for_targets = lambda targets, project_root=".": (lambda *a, **k: None)

    def damage():
        {damage}

    calls = []
    def fake_profile_file(root, target, **kw):
        calls.append(target)
        if len(calls) == 1:
            damage()        # what a nested Action under a stream-code mutant does to the process
            try:
                action.main(["--targets", "x.py"])   # and the nested Action itself
            except BaseException:
                pass    # a test that dies on the damage is that test's failure, not the run's
        return []
    ci.profile_file = fake_profile_file

    def live(root, targets, **kw):
        return ci.profile_codebase(root, targets, **kw)

    action._profile_live(live, ["a.py", "b.py", "c.py"], 1000, {})
    print("REPORT-REACHED-STDOUT", flush=True)
    sys.stderr.write("REPORT-REACHED-STDERR\\n")
"""


def _run(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]),
        HOME=str(tmp_path),
    )
    for name in (
        "WESKER_STREAM_OWNER",
        "WESKER_STACK_DUMP_S",
        "WESKER_STACK_DUMP_OWNER",
    ):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
    )


@pytest.mark.parametrize("damage", sorted(_DAMAGE))
def test_the_measuring_process_output_survives_a_nested_action(damage, tmp_path):
    proc = _run(_OUTER.replace("{damage}", _DAMAGE[damage]), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Progress for every file, written after the damage, and the report written after the profile.
    for line in ("[1/3] a.py", "[2/3] b.py", "[3/3] c.py", "REPORT-REACHED-STDOUT"):
        assert line in proc.stdout, (line, proc.stdout, proc.stderr)
    assert "REPORT-REACHED-STDERR" in proc.stderr, proc.stderr
    assert "Bad file descriptor" not in proc.stderr, proc.stderr


def test_the_nested_actions_own_stream_code_under_the_closefd_mutant(tmp_path):
    """The mutant the issue names, applied to the real source: `closefd=False -> True`."""
    script = """
        import inspect, types
        from Wesker import action
        src = inspect.getsource(action).replace("closefd=False", "closefd=True")
        assert "closefd=True" in src
        mut = types.ModuleType("action_mut")
        mut.__file__ = action.__file__
        exec(compile(src, action.__file__, "exec"), mut.__dict__)
        import Wesker.self_profile as sp
        sp.profiler_for_targets = lambda targets, project_root=".": (lambda *a, **k: None)
        from Wesker import ci
        n = []
        def fake_profile_file(root, target, **kw):
            n.append(target)
            if len(n) == 1:
                mut.main(["--targets", "x.py"])
            return []
        ci.profile_file = fake_profile_file
        action._profile_live(
            lambda root, targets, **kw: ci.profile_codebase(root, targets, **kw),
            ["a.py", "b.py", "c.py"], 1000, {},
        )
        print("REPORT-REACHED-STDOUT", flush=True)
    """
    proc = _run(script, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[3/3] c.py" in proc.stdout and "REPORT-REACHED-STDOUT" in proc.stdout
    assert "Bad file descriptor" not in proc.stderr, proc.stderr


def test_reentry_restores_fd1_after_a_wrapper_that_closes_it_is_collected(tmp_path):
    """The order matters: the descriptor the report speaks through is the one in place AFTER finalizers."""
    script = """
        import io, sys
        from Wesker.action import _stream_reentry
        reenter = _stream_reentry()
        sys.stdout = io.TextIOWrapper(io.FileIO(1, "w", closefd=True), line_buffering=True)
        reenter()
        print("survives", flush=True)
    """
    proc = _run(script, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "survives" in proc.stdout
    assert "Bad file descriptor" not in proc.stderr


def _open_fds() -> set[str]:
    return set(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else set()


def test_a_nested_action_owns_no_streams(monkeypatch):
    monkeypatch.setenv(action._STREAM_OWNER_ENV, str(os.getpid()))
    before_fds, before_out, before_err = _open_fds(), sys.stdout, sys.stderr
    reenter = action._stream_reentry()
    assert _open_fds() == before_fds  # it snapshotted nothing
    reenter()
    assert (sys.stdout, sys.stderr) == (before_out, before_err)  # it rebound nothing


def test_a_marker_from_another_process_does_not_make_an_action_nested(monkeypatch):
    monkeypatch.setenv(action._STREAM_OWNER_ENV, str(os.getpid() + 1))
    before = _open_fds()
    reenter = action._stream_reentry()
    try:
        if before:
            assert len(_open_fds()) == len(before) + 2  # an owner snapshots both
    finally:
        reenter()


def test_a_nested_actions_output_stays_in_the_calling_tests_capture(
    tmp_path, monkeypatch, capsys
):
    from Wesker import self_profile

    (tmp_path / "m.py").write_text("def f(x):\n    return x\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        self_profile, "profiler_for_targets", lambda targets: lambda *a, **k: None
    )
    monkeypatch.setenv(action._STREAM_OWNER_ENV, str(os.getpid()))
    out_before = sys.stdout
    assert action.main(["--targets", "m.py"]) == 1
    assert sys.stdout is out_before  # nothing rebound it onto the real descriptor
    captured = capsys.readouterr()
    assert "Whole codebase" in captured.out
    assert "WESKER REFUSED TO REPORT" in captured.err


def test_the_owner_publishes_the_marker_only_while_it_owns_the_streams(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(action._STREAM_OWNER_ENV, raising=False)
    seen: dict[str, object] = {}

    def profiler(*args, **kwargs):
        seen["marker"] = os.environ.get(action._STREAM_OWNER_ENV)
        seen["progress"] = kwargs.get("progress")

    action._profile_live(profiler, ["m.py"], 1.0, {})
    assert seen["marker"] == str(os.getpid())
    assert seen["progress"] is not None
    assert action._STREAM_OWNER_ENV not in os.environ


def test_the_marker_is_released_when_the_profile_raises(monkeypatch):
    monkeypatch.delenv(action._STREAM_OWNER_ENV, raising=False)

    def profiler(*args, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        action._profile_live(profiler, ["m.py"], 1.0, {})
    assert action._STREAM_OWNER_ENV not in os.environ


def test_a_nested_profile_gets_no_progress_channel_and_keeps_the_owners_marker(
    monkeypatch,
):
    monkeypatch.setenv(action._STREAM_OWNER_ENV, str(os.getpid()))
    seen: dict[str, object] = {}

    def profiler(*args, **kwargs):
        seen["progress"] = kwargs.get("progress", "absent")

    action._profile_live(profiler, ["m.py"], 1.0, {})
    assert seen["progress"] == "absent"
    assert os.environ[action._STREAM_OWNER_ENV] == str(
        os.getpid()
    )  # not its to release


def _one_target_profile(monkeypatch, **kwargs):
    monkeypatch.setattr(ci, "profile_file", lambda *a, **k: [])
    return ci.profile_codebase(".", ["pkg/mod.py"], **kwargs)


def test_progress_lines_go_to_the_stream_they_were_given(monkeypatch, capsys, tmp_path):
    sink = (tmp_path / "progress.txt").open("w")
    _one_target_profile(monkeypatch, progress=sink)
    sink.close()
    assert "[1/1] mod.py" in (tmp_path / "progress.txt").read_text()
    assert capsys.readouterr().out == ""


def test_progress_defaults_to_stdout_as_it_stands(monkeypatch, capsys):
    _one_target_profile(monkeypatch)
    assert "[1/1] mod.py" in capsys.readouterr().out


def test_progress_that_cannot_be_written_never_costs_the_measurement(
    monkeypatch, tmp_path
):
    sink = (tmp_path / "p.txt").open("w")
    sink.close()  # a dead progress stream: ValueError on write
    report = _one_target_profile(monkeypatch, progress=sink)
    assert report["total_functions"] == 0
