"""#29 — a pytest failure AFTER the measurement finished does not throw the measurement away, and a missing session names why.

THE DEFECT, from the first sharded badge run (GitHub run 43315433429 on 8fcadfe, shard 2/4). The shard
measured `ci.py` for 45 minutes, printed every file's result, and ended with

    WESKER REFUSED TO REPORT: execution mode is 'legacy-direct-call', not 'pytest-live'.
    Check that pytest is installed and that `pytest --collect-only` succeeds here.

`run_in_session` hosts the measurement inside `pytest_runtestloop`. Its `except Exception` around
`pytest.main` returned None for ANY exception pytest raised, including one raised AFTER the body had
returned its report (the terminal summary, a `pytest_sessionfinish` hook, a stdout that a mutant of the
capture machinery left dead). The Action read None as "no live session" and printed the one sentence it had,
which is also what a project whose suite cannot be collected gets. Reproduced without CI: a conftest whose
`pytest_sessionfinish` raises makes `run_in_session(root, body)` return None with `reason: pytest_crashed`
after `body` returned its dict.

THE RULING pinned here.
- A crash before or during the body still returns None (there is no result), and says which.
- A crash after the body returned keeps the result, and the diagnostic says `pytest_crashed_after_result`
  with the exception, so the Action prints a warning instead of hiding it.
- A missing session names its cause (`no_live_session_reason`): the crash, the collection errors, an empty
  collection, a missing pytest; the old sentence remains only for "no cause was recorded", and says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from Wesker.action import no_live_session_reason
from Wesker.pytest_runner import crash_standing, describe_crash

REPO = Path(__file__).resolve().parents[1]

# ── the pure decisions ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("started", "finished", "expected"),
    [
        (False, False, "no_session"),
        (True, False, "session_lost"),
        (True, True, "result_kept"),
        (False, True, "result_kept"),
    ],
)
def test_what_a_pytest_crash_means_for_the_measurement_it_hosted(
    started, finished, expected
):
    assert crash_standing(started, finished) == expected


@pytest.mark.parametrize(
    ("standing", "words"),
    [
        ("no_session", "before the measurement started"),
        ("session_lost", "while the measurement was running"),
        ("result_kept", "after the measurement had finished; its result was kept"),
        ("anything_else", "before the measurement started"),
    ],
)
def test_the_crash_sentence_follows_the_standing(standing, words):
    said = describe_crash("RuntimeError", "boom", standing, "  File x\n")
    assert said.startswith(f"pytest raised RuntimeError {words}: boom")
    assert said.endswith("File x")


def test_a_blank_message_is_said_to_be_blank():
    assert "(no message)" in describe_crash("KeyError", " ", "no_session", "")


@pytest.mark.parametrize(
    ("reason", "detail", "says"),
    [
        ("pytest_crashed", "RuntimeError: x", "ended in a crash"),
        ("collection_errors", "tests/t.py: ImportError", "could not collect the suite"),
        ("collection_errors", "", "pytest --collect-only"),
        ("empty_collection", "", "collected no test"),
        ("pytest_missing", "", "pytest is not installed"),
        ("", "", "no cause was recorded"),
        ("something_new", "", "no cause was recorded"),
    ],
)
def test_a_missing_session_names_its_cause(reason, detail, says):
    text = no_live_session_reason(reason, detail)
    assert says in text
    assert detail in text


def test_only_an_unrecorded_cause_reads_as_the_old_sentence():
    old = "execution mode is 'legacy-direct-call'"
    assert old in no_live_session_reason("", "")
    for reason in (
        "pytest_crashed",
        "collection_errors",
        "empty_collection",
        "pytest_missing",
    ):
        assert old not in no_live_session_reason(reason, "d")


# ── the real seams (subprocess: the Action owns the streams) ─────

_PROJECT = {
    "pkg/__init__.py": "",
    "pkg/m.py": "def f(x):\n    return x + 1\n",
    "tests/test_m.py": "from pkg.m import f\n\n\ndef test_f():\n    assert f(1) == 2\n",
}

_FINISH_CRASH = textwrap.dedent(
    """\
    def pytest_sessionfinish(session, exitstatus):
        raise RuntimeError("summary failed")
    """
)


def _project(root: Path, extra: dict[str, str] | None = None) -> None:
    for rel, text in {**_PROJECT, **(extra or {})}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "home").mkdir(exist_ok=True)


def _env(root: Path) -> dict[str, str]:
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
    return env


def _run(root: Path, argv: list[str], code: str | None = None):
    cmd = (
        [sys.executable, "-c", code]
        if code
        else [sys.executable, "-m", "Wesker.action"]
    )
    return subprocess.run(
        [*cmd, *argv],
        cwd=root,
        env=_env(root),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


_SESSION = textwrap.dedent(
    """\
    import json
    from Wesker.pytest_runner import run_in_session
    diag = {}
    result = run_in_session(".", lambda calls, session: {"measured": len(calls)}, diagnostic=diag)
    print(json.dumps({"result": result, "diag": diag}))
    """
)


def test_a_crash_after_the_body_keeps_the_result_and_says_so(tmp_path):
    _project(tmp_path, {"conftest.py": _FINISH_CRASH})
    done = _run(tmp_path, [], _SESSION)
    assert done.returncode == 0, done.stderr[-1500:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["result"] == {"measured": 1}
    assert out["diag"]["reason"] == "pytest_crashed_after_result"
    assert "RuntimeError" in out["diag"]["detail"]
    assert "summary failed" in out["diag"]["detail"]


def test_a_crash_before_the_body_is_still_no_result(tmp_path):
    _project(
        tmp_path,
        {
            "conftest.py": textwrap.dedent(
                """\
                def pytest_configure(config):
                    raise RuntimeError("configuration failed")
                """
            )
        },
    )
    done = _run(tmp_path, [], _SESSION)
    assert done.returncode == 0, done.stderr[-1500:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["result"] is None
    assert out["diag"]["reason"] in ("pytest_crashed", "collection_errors")


def test_the_action_keeps_a_finished_report_when_pytest_fails_after_it(tmp_path):
    _project(tmp_path, {"conftest.py": _FINISH_CRASH})
    done = _run(tmp_path, ["--targets", "pkg/m.py", "--budget", "20000"])
    assert done.returncode == 0, done.stderr[-1500:]
    assert "legacy-direct-call" not in done.stderr
    assert "::warning::pytest raised after the measurement had finished" in done.stderr
    assert "summary failed" in done.stderr
    record = json.loads((tmp_path / ".wesker/shard_report.json").read_text())
    assert record["refusal"] == ""
    assert record["report"]["total_mutants"] > 0


def test_a_project_with_no_tests_is_refused_for_that_cause(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/m.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "home").mkdir()
    done = _run(tmp_path, ["--targets", "pkg/m.py", "--budget", "20000"])
    assert done.returncode == 1
    assert "WESKER REFUSED TO REPORT" in done.stderr
    assert "pytest collected no test" in done.stderr
    record = json.loads((tmp_path / ".wesker/shard_report.json").read_text())
    assert record["refusal"] == "no_live_session"
    assert "pytest collected no test" in record["refusal_detail"]
