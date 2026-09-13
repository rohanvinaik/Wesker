"""Intent tests: an isolated worker leaves no open pipe behind, however it ends.

THE DEFECT, found 2026-09-13 by running this suite under `-W error` while evaluating pytest's
`filterwarnings = ["error"]`: 17 tests in test_isolation.py failed with
`ResourceWarning: unclosed file <_io.TextIOWrapper name=13>`. `IsolatedMutantWorker.close()` closed
the worker's stdin but never its stdout, and did not reap a worker that had already exited, so the
stdout pipe object lived until garbage collection. The one-shot runners had the same gap on their
timeout paths, where `communicate()` never completes. Every engine run that recycles or finishes an
isolated worker leaked one descriptor.

These assert the parent's pipe ends are closed after each way a worker can end: a normal close, a
hang that retires it, and a one-shot run that times out.
"""

from __future__ import annotations

import subprocess

from Wesker.isolation import IsolatedMutantWorker, run_baseline_traced_isolated


def _tiny_project(tmp_path) -> None:
    (tmp_path / "t.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "test_t.py").write_text(
        "from t import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )


def _pipes_closed(proc: subprocess.Popen) -> bool:
    return all(s is None or s.closed for s in (proc.stdin, proc.stdout, proc.stderr))


def test_close_after_normal_use_closes_every_pipe(tmp_path):
    _tiny_project(tmp_path)
    worker = IsolatedMutantWorker(str(tmp_path), ["test_t.py::test_f"], "t.py", "f")
    worker.evaluate("def f(x):\n    return x - 1\n", 30.0)
    worker.close()
    assert _pipes_closed(worker._proc)


def test_a_hang_retires_the_worker_with_its_pipes_closed(tmp_path):
    _tiny_project(tmp_path)
    worker = IsolatedMutantWorker(str(tmp_path), ["test_t.py::test_f"], "t.py", "f")
    try:
        run = worker.evaluate("def f(x):\n    while True:\n        pass\n", 2.0)
        assert run.outcome == "timeout"
        assert not worker.alive
        assert _pipes_closed(worker._proc)
    finally:
        worker.close()


def test_a_one_shot_baseline_timeout_closes_its_pipes(tmp_path, monkeypatch):
    (tmp_path / "h.py").write_text("def f():\n    while True:\n        pass\n")
    (tmp_path / "test_h.py").write_text("from h import f\n\n\ndef test_f():\n    f()\n")
    created: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", spy)
    lines, outcome, _ = run_baseline_traced_isolated(
        str(tmp_path), ["test_h.py::test_f"], "h.py", 1.0
    )
    assert (lines, outcome) == ([], "timeout")
    assert len(created) == 1
    assert _pipes_closed(created[0])
