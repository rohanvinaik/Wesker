"""A mutant over the worker's memory cap is ONE mutant with no verdict, and its outcome is its own (Wesker #45 / #39).

THE DEFECT, measured on `main` (7b1afe4) with `python -m Wesker hog.py --isolated --mem-limit-mb 512 --json`
on the hog fixture (`test_memory_standing_intent._HOG`), four consecutive runs of one command on one tree:

    total_mutants 10 killed 8 equiv 1 | collect: 6/23 evaluated, hit the cap: STMT_0de1c53a
    total_mutants  8 killed 7 equiv 0 | collect: 4/23 evaluated, hit the cap: STMT_0de1c53a
    total_mutants  7 killed 6 equiv 0 | collect: 3/23 evaluated, hit the cap: ARITHMETIC_79b8b80a
    total_mutants  8 killed 7 equiv 0 | collect: 4/23 evaluated, hit the cap: BOUNDARY_ff392566

and CI failing `a cut mutant is never declared equivalent` (`assert 1 == 0`). Four mechanisms:

1. A capped mutant was `contained=False`, which the loops read as "a worker is still alive": the function
   stopped after that mutant, listed it under `containment_lost` and was a `cut`. The cap contained it.
2. Category order followed SET iteration order (`prioritize_categories` kept it on ties), so the order
   mutants ran in changed with the process, and with it how many were reached before the cap stopped the
   function.
3. The worker kept running after a cap with the runaway's allocations still held (the failing test's
   traceback keeps its frames), so the NEXT mutants read as memory cuts for allocations they never made
   (`return len(out)` -> `return None` was an assertion kill in one run and a memory cut in the next).
4. A `MemoryError` caught by the code under test (`except Exception: return -1`) became an ASSERTION kill:
   the cap manufactured a kill.
The `equiv 1` was none of these being equivalent: it was a real survivor (`out.append(bytearray(..))` ->
`out.append(<int>)`, which `collect` cannot tell apart) reached only in runs where the order got that far.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import textwrap

import pytest

from Wesker.ci import UNSCORED_ROLLUP, describe_unresolved, unresolved_causes
from Wesker.engine import (
    MEMORY_CUT,
    NO_VERDICT_KILLED_BY,
    SCORED_DISPOSITIONS,
    _isolated_result,
    generate_mutants,
    mutant_disposition,
    worker_succession,
)
from Wesker.filter import filter_categories, prioritize_categories
from Wesker.isolation import IsolatedRun

_MB = 1024 * 1024
linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="RLIMIT_AS is measured on Linux"
)

# ── the pure decisions ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("alive", "memory_cut", "expected"),
    [
        (True, False, "keep"),
        (True, True, "replace_memory_cut"),
        (False, False, "replace_dead"),
        (False, True, "replace_dead"),
    ],
)
def test_a_worker_a_cap_stopped_is_replaced_never_reused(alive, memory_cut, expected):
    assert worker_succession(alive, memory_cut) == expected


def test_memory_cut_is_unscored_unresolved_and_never_a_verdict():
    disposition = mutant_disposition(
        True, True, None, True, False, "", killed_by=MEMORY_CUT
    )
    assert disposition == MEMORY_CUT
    assert disposition not in SCORED_DISPOSITIONS
    assert dict(UNSCORED_ROLLUP)[MEMORY_CUT] == "unresolved"
    assert MEMORY_CUT in NO_VERDICT_KILLED_BY
    assert unresolved_causes({"memory_cut": 2, "diverged": 1}) == (
        "diverged",
        "memory_cut",
    )
    # killed is never True for it, and the disposition outranks entry: it is not `not_entered`
    assert (
        mutant_disposition(True, True, False, True, False, "", killed_by=MEMORY_CUT)
        == MEMORY_CUT
    )


def test_the_engine_result_for_a_capped_mutant_is_contained_unkilled_and_named():
    import ast

    node = ast.parse("def f(x):\n    return x + 1\n").body[0]
    mutant = generate_mutants(node, filter_categories(node))[0]
    res = _isolated_result(
        mutant, IsolatedRun(1, False, True, "", memory_cut=True), 0.0
    )
    assert (res.killed, res.contained, res.killed_by) == (False, True, MEMORY_CUT)
    assert res.equivalent is False


def test_ties_between_categories_break_by_name_not_by_set_order():
    from Wesker.engine import MutationCategory

    cats = set(MutationCategory)
    order = [p.category.value for p in prioritize_categories(cats)]
    assert order == sorted(order)
    # a cached prior still outranks the name
    cached = {"per_category": [{"category": "VALUE", "total": 10, "survived": 9}]}
    assert prioritize_categories(cats, cached)[0].category.value == "VALUE"


def test_the_order_does_not_depend_on_the_hash_seed():
    code = (
        "from Wesker.engine import MutationCategory;"
        "from Wesker.filter import prioritize_categories;"
        "print([p.category.value for p in prioritize_categories(set(MutationCategory))])"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={
                **os.environ,
                "PYTHONHASHSEED": str(seed),
                "PYTHONPATH": os.pathsep.join(sys.path),
            },
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in (1, 2, 3, 4)
    }
    assert len(outs) == 1, outs


def test_the_remedy_names_the_cap_and_says_the_function_was_still_evaluated():
    text = describe_unresolved(
        [{"function_key": "m.py::hog", "cause": MEMORY_CUT, "mutants": ["A", "B"]}]
    )
    assert "neither killed nor survived" in text
    assert "every other mutant of the function was evaluated" in text


# ── the worker, measured (a real cap) ───────────────────────────────────────────────────────────


def _tiny_project(root) -> None:
    (root / "t.py").write_text("def f(x):\n    return x + 1\n")
    (root / "test_t.py").write_text(
        "from t import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )


def _in_capped_child(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a child under an outer address-space limit, so a failing cap cannot take the
    machine with it."""

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (6000 * _MB, 6000 * _MB))

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        preexec_fn=cap,
    )


@linux_only
def test_a_memory_error_the_code_under_test_swallows_is_the_cap_not_a_value_kill(
    tmp_path,
):
    _tiny_project(tmp_path)
    swallowing = (
        "def f(x):\n"
        "    try:\n"
        "        junk = [bytearray(64 * 1024 * 1024) for _ in range(40)]\n"
        "    except Exception:\n"
        "        return -1\n"
        "    return x + 1\n"
    )
    done = _in_capped_child(
        f"""
        import json
        from Wesker.isolation import IsolatedMutantWorker
        w = IsolatedMutantWorker({str(tmp_path)!r}, ["test_t.py::test_f"], "t.py", "f",
                                 mem_limit_bytes=300 * 1024 * 1024)
        try:
            r = w.evaluate({swallowing!r}, 60.0)
            print(json.dumps([r.memory_cut, r.killed_by]))
        finally:
            w.close()
        """
    )
    assert done.returncode == 0, done.stderr[-1500:]
    assert json.loads(done.stdout.splitlines()[-1]) == [True, None], (
        "the swallowed MemoryError made the node fail with an assertion, and that failure was read "
        "as the suite's"
    )


@linux_only
def test_a_value_failure_with_no_memory_error_is_still_a_kill(tmp_path):
    _tiny_project(tmp_path)
    done = _in_capped_child(
        f"""
        import json
        from Wesker.isolation import IsolatedMutantWorker
        w = IsolatedMutantWorker({str(tmp_path)!r}, ["test_t.py::test_f"], "t.py", "f",
                                 mem_limit_bytes=300 * 1024 * 1024)
        try:
            r = w.evaluate("def f(x):\\n    return x - 1\\n", 60.0)
            print(json.dumps([r.memory_cut, r.killed_by]))
        finally:
            w.close()
        """
    )
    assert done.returncode == 0, done.stderr[-1500:]
    assert json.loads(done.stdout.splitlines()[-1]) == [False, "assertion"]


# ── end to end: the same command, many times, fresh directories ─────────────────────────────────

RUNS = 5


def _run_fixture(tmp_path, seed: int, mem_mb: int = 512) -> dict:
    import test_memory_standing_intent as fixture

    (tmp_path / "tests").mkdir()
    (tmp_path / "hog.py").write_text(fixture._HOG)
    (tmp_path / "tests" / "test_hog.py").write_text(fixture._TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npythonpath = .\n"
    )

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (6000 * _MB, 6000 * _MB))

    env = {
        k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"
    }  # CI does not set it
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=os.pathsep.join(sys.path),
        # a different string-hash seed per run: the order a SET of categories iterates in
        PYTHONHASHSEED=str(seed),
    )
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "Wesker",
            "hog.py",
            "--isolated",
            "--mem-limit-mb",
            str(mem_mb),
            "--json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
        preexec_fn=cap,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)["mutation"]


def _shape(mutation: dict) -> dict:
    """What a run says per mutant, as ids are known to THIS interpreter (compared within it only)."""
    (unresolved,) = mutation["unresolved_functions"]
    return {
        "totals": {
            k: mutation[k]
            for k in (
                "total_mutants",
                "total_killed",
                "total_equivalent",
                "total_truncated",
                "total_unresolved",
                "total_universe",
            )
        },
        "cut": sorted(unresolved["mutants"]),
        "cause": unresolved["cause"],
        "survivors": sorted(
            (s["mutant_id"], bool(s.get("equivalent"))) for s in mutation["survivors"]
        ),
        "truncated": mutation["truncated_functions"],
        "memory": (mutation["memory_standing"], mutation["memory_standings"]),
    }


@linux_only
def test_the_fixture_command_gives_the_same_dispositions_every_time(tmp_path):
    shapes = []
    for run in range(RUNS):
        d = tmp_path / f"run{run}"
        d.mkdir()
        shapes.append(_shape(_run_fixture(d, seed=run + 1)))
    assert all(shape == shapes[0] for shape in shapes), json.dumps(shapes, indent=1)
    first = shapes[0]
    assert first["cause"] == MEMORY_CUT
    assert first["truncated"] == []
    # more than the handful the old stop reached, and every capped mutant named, none a survivor
    assert first["totals"]["total_mutants"] > 10
    # exactly the three runaways (`i += 1` -> `i -= 1`, -> `pass`, and `while i < n` -> `while True`):
    # a worker kept after a cap made `while i <= n` a fourth, for an allocation it never made
    assert len(first["cut"]) == 3
    assert not set(first["cut"]) & {m for m, _ in first["survivors"]}
    assert [eq for _m, eq in first["survivors"] if eq] == [True]
    assert first["totals"]["total_equivalent"] == 1
