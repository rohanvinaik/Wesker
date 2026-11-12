"""The badge run is a matrix of file groups plus one merge, and the merge equals the unsharded run (Wesker #29, B1).

THE DEFECT. `Specification (full)` measured all 24 files in ONE job: 146 minutes when it finished
(run 43102482321 on fb2f8b4, `engine.py` alone 88), past the 180-minute limit when it did not
(185985c). The law (Detective ARCHITECTURE section 11) is that the unit of measurement is one function, so anything
that scales with the codebase is split; a longer timeout is not the fix.

THE RULING pinned here, on a fixture project with real tests, a function that survives and a file no test
reaches:

- the plan (`--plan-shards N`) uses the unsharded run's own target resolution and its groups are
  deterministic, disjoint and complete, with no empty group;
- each shard, run as the ordinary Action with `--shard i/N`, writes a record naming its files, commit,
  policy id, Wesker version, shard index/count and whether it ended in a refusal (and which);
- the merge (`--merge --plan P reports...`) gives the SAME `dimensions-pinned`, `dimensions-total`,
  `spec-pct`, kill rate, survivors and exit code as the unsharded run, for 1, 2 and 3 shards and more shards
  than files;
- the merge REFUSES, with no number and the cause named, for a missing shard, shards that disagree on
  commit / policy id / Wesker version, a file in two shards, a union that is not the planned set, and an
  unreadable or foreign report; a shard that ended in a refusal is that same named refusal, naming the shard
  and its files, and is never averaged into a number.

Every Action here runs in a subprocess: the Action owns the process's streams and the nesting guard
(W#52) is for the measuring process, so a test does not call `action.main` in-process.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from Wesker.shards import (
    completeness_code,
    merge_standing,
    partition,
    plan_standing,
    record_standing,
    shard_setting,
)

REPO = Path(__file__).resolve().parents[1]
SHA = "e83b5fafafc9ef5a10e1c8ab0c41dfe3686c76d7"

FILES = {
    "pkg/__init__.py": "",
    "pkg/shapes.py": textwrap.dedent(
        """\
        def area(w, h):
            if w < 0 or h < 0:
                raise ValueError("negative")
            return w * h


        def classify(n):
            if n > 10:
                return "big"
            if n > 0:
                return "small"
            return "none"
        """
    ),
    "pkg/text.py": textwrap.dedent(
        """\
        def clip(s, n):
            if len(s) > n:
                return s[:n]
            return s


        def shout(s):
            return s.upper() + "!"
        """
    ),
    # A function with a survivor: the tests never pin `pct > 50`.
    "pkg/money.py": textwrap.dedent(
        """\
        def total(prices, tax):
            s = 0
            for p in prices:
                s += p
            return s + s * tax


        def discount(x, pct):
            if pct > 50:
                pct = 50
            return x - x * pct // 100
        """
    ),
    # A file no test reaches.
    "pkg/untested.py": "def lonely(x):\n    if x:\n        return x + 1\n    return 0\n",
    "tests/test_all.py": textwrap.dedent(
        """\
        import pytest
        from pkg.shapes import area, classify
        from pkg.text import clip, shout
        from pkg.money import total, discount


        def test_area():
            assert area(2, 3) == 6
            with pytest.raises(ValueError):
                area(-1, 3)


        def test_classify():
            assert classify(50) == "big"
            assert classify(5) == "small"
            assert classify(-1) == "none"


        def test_clip():
            assert clip("hello", 3) == "hel"
            assert clip("hi", 3) == "hi"


        def test_shout():
            assert shout("a") == "A!"


        def test_total():
            assert total([1, 2], 0) == 3


        def test_discount():
            assert discount(100, 10) == 90
        """
    ),
    "pyproject.toml": '[project]\nname = "fx"\nversion = "0.0.1"\n'
    '[tool.wesker]\nsource_dir = "pkg"\n'
    '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
}
ALL = [
    "pkg/money.py",
    "pkg/shapes.py",
    "pkg/text.py",
    "pkg/untested.py",
]
KEYS = (
    "spec-pct",
    "dimensions-total",
    "dimensions-pinned",
    "dimensions-unspecified",
    "dof-pct",
    "kill-pct",
    "total-mutants",
    "survivors",
)


def _run(args: list[str], cwd: Path, *, sha: str | None = SHA, extra=None):
    """Run the Action in `cwd`; return (exit code, GITHUB_OUTPUT as a dict, stdout, stderr)."""
    out_file = cwd / "gh_output.txt"
    out_file.write_text("")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GITHUB_", "WESKER_", "PYTEST_"))
    }
    env["HOME"] = str(cwd / "home")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), *filter(None, [os.environ.get("PYTHONPATH", "")])]
    )
    env["GITHUB_OUTPUT"] = str(out_file)
    if sha is not None:
        env["GITHUB_SHA"] = sha
    env.update(extra or {})
    proc = subprocess.run(
        [sys.executable, "-m", "Wesker.action", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    outputs = dict(
        line.split("=", 1) for line in out_file.read_text().splitlines() if "=" in line
    )
    return proc.returncode, outputs, proc.stdout, proc.stderr


def _project(root: Path) -> Path:
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "home").mkdir()
    return root


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The unsharded run, and for 1, 2, 3 and 9 shards the plan, every shard's record and the merge."""
    root = _project(tmp_path_factory.mktemp("fx"))
    whole = _run(["--budget", "20000"], root)
    shutil_report = json.loads((root / ".wesker/mutation_report.json").read_text())
    runs: dict[int, dict] = {}
    for n in (1, 2, 3, 9):
        plan_path = root / f"plan{n}.json"
        code, _, matrix, err = _run(
            ["--plan-shards", str(n), "--plan-out", str(plan_path)], root
        )
        assert code == 0, err
        plan = json.loads(plan_path.read_text())
        records = []
        for entry in plan["include"]:
            index = entry["shard"].split("/")[0]
            c, _, _, e = _run(
                ["--budget", "20000", "--shard", entry["shard"], "--targets"]
                + entry["targets"].split(),
                root,
            )
            assert c == 0, e
            rec = root / f"rec{n}_{index}.json"
            rec.write_text((root / ".wesker/shard_report.json").read_text())
            records.append(rec)
        merged = _run(["--merge", "--plan", str(plan_path), *map(str, records)], root)
        runs[n] = {
            "plan": plan,
            "plan_path": plan_path,
            "matrix": matrix,
            "records": records,
            "merged": merged,
            "merged_report": json.loads(
                (root / ".wesker/mutation_report.json").read_text()
            ),
        }
    return {"root": root, "whole": whole, "whole_report": shutil_report, "runs": runs}


# ── Equality is the acceptance ───────────────────────────────────


def test_the_fixture_has_a_survivor_and_an_untested_file(world):
    code, outputs, _, _ = world["whole"]
    assert code == 0
    assert 0 < int(outputs["dimensions-pinned"]) < int(outputs["dimensions-total"])
    assert "pkg/untested.py" not in world["whole_report"]["per_file"]
    assert world["whole_report"]["survivors"]


@pytest.mark.parametrize("n", [1, 2, 3, 9])
def test_the_merged_numbers_equal_the_unsharded_run(world, n):
    """THE ACCEPTANCE: same dimensions, same percentages, same exit code."""
    whole_code, whole_out, _, _ = world["whole"]
    code, out, _, err = world["runs"][n]["merged"]
    assert code == whole_code == 0, err
    for key in KEYS:
        assert out[key] == whole_out[key], key


@pytest.mark.parametrize("n", [1, 2, 3, 9])
def test_the_merged_report_carries_the_same_facts_as_the_unsharded_report(world, n):
    whole = world["whole_report"]
    merged = world["runs"][n]["merged_report"]
    assert merged["per_file"].keys() == whole["per_file"].keys()
    for path, row in whole["per_file"].items():
        mrow = merged["per_file"][path]
        for field in (
            "functions",
            "killed",
            "total",
            "equivalent",
            "dof",
            "dof_pinned",
            "spec_pct",
        ):
            assert mrow[field] == row[field], (path, field)
    key = lambda s: (
        s["function_key"],
        s.get("mutated_line"),
        s.get("dimension"),
        s.get("change"),
    )
    assert sorted(map(key, merged["survivors"])) == sorted(map(key, whole["survivors"]))
    for field in (
        "total_killed",
        "total_mutants",
        "total_equivalent",
        "total_functions",
        "total_truncated",
        "total_unresolved",
        "total_harness_mutants",
        "harness_mutants",
        "memory_standing",
    ):
        assert merged[field] == whole[field], field


@pytest.mark.parametrize("n", [1, 2, 3, 9])
def test_the_merge_threshold_is_the_unsharded_runs_exit_code(world, n):
    root = world["root"]
    run = world["runs"][n]
    args = ["--merge", "--plan", str(run["plan_path"]), *map(str, run["records"])]
    code, _, _, err = _run([*args, "--threshold", "90"], root)
    whole_code, _, _, _ = _run(["--budget", "20000", "--threshold", "90"], root)
    assert code == whole_code == 1
    assert "below the 90% threshold" in err


def test_the_merge_writes_the_step_summary_the_single_run_writes(world, tmp_path):
    root = world["root"]
    run = world["runs"][3]
    summary = tmp_path / "summary.md"
    _run(
        ["--merge", "--plan", str(run["plan_path"]), *map(str, run["records"])],
        root,
        extra={"GITHUB_STEP_SUMMARY": str(summary)},
    )
    text = summary.read_text()
    assert "## Specification completeness" in text
    assert "27 of 55 behavioral dimensions pinned" in text


# ── The plan ─────────────────────────────────────────────────────


def test_the_plan_covers_exactly_the_files_an_unsharded_run_measures(world):
    for n, run in world["runs"].items():
        flat = [f for g in run["plan"]["groups"] for f in g]
        assert sorted(flat) == ALL, n
        assert len(flat) == len(set(flat))
        assert (
            run["plan"]["shard_count"] == len(run["plan"]["groups"]) == min(n, len(ALL))
        )
        assert all(run["plan"]["groups"])


def test_the_matrix_is_one_line_of_include_entries_and_nothing_else(world):
    matrix = world["runs"][2]["matrix"]
    assert matrix.count("\n") == 1
    assert list(json.loads(matrix)) == ["include"]


def test_more_shards_than_files_plans_one_group_per_file_and_no_empty_group(world):
    groups = world["runs"][9]["plan"]["groups"]
    assert len(groups) == len(ALL)
    assert all(len(g) == 1 for g in groups)


def test_a_plan_with_no_files_refuses_instead_of_emitting_an_empty_matrix(tmp_path):
    root = _project(tmp_path / "p")
    (root / "pkg/shapes.py").unlink()
    for f in list((root / "pkg").glob("*.py")):
        f.unlink()
    code, _, out, err = _run(["--plan-shards", "3"], root)
    assert code == 1
    assert out.strip() == ""
    assert "no source files found" in err


def test_a_plan_of_zero_shards_refuses(tmp_path):
    root = _project(tmp_path / "p")
    code, _, out, err = _run(["--plan-shards", "0"], root)
    assert code == 1
    assert "bad_shard_count" in err
    assert out.strip() == ""


def test_a_directory_target_is_refused_not_silently_expanded(tmp_path):
    root = _project(tmp_path / "p")
    code, _, _, err = _run(["--plan-shards", "2", "--targets", "pkg"], root)
    assert code == 1
    assert "not_a_file" in err


def test_a_path_with_a_space_is_refused_because_targets_is_space_separated(tmp_path):
    root = _project(tmp_path / "p")
    (root / "pkg/two words.py").write_text("x = 1\n")
    code, _, _, err = _run(
        ["--plan-shards", "2", "--targets", "pkg/two words.py", "pkg/text.py"], root
    )
    assert code == 1
    assert "unsplittable_path" in err


def test_a_shard_record_names_what_the_merge_needs(world):
    rec = json.loads(world["runs"][2]["records"][0].read_text())
    plan = world["runs"][2]["plan"]
    assert rec["kind"] == "wesker-shard-report"
    assert rec["shard_index"] == 1 and rec["shard_count"] == 2
    assert rec["targets"] == plan["groups"][0]
    assert rec["commit"] == SHA
    assert rec["policy_id"]
    assert rec["wesker_version"]
    assert rec["refusal"] == "" and rec["report"]["total_dof"] >= 0


# ── The merge refuses ────────────────────────────────────────────


def _merge(world, mutate, n=3, drop=None, tmp_path=None, extra_files=()):
    """Merge the n-shard records after `mutate(list_of_records)`; return the run tuple."""
    run = world["runs"][n]
    records = [json.loads(p.read_text()) for p in run["records"]]
    mutate(records)
    paths = []
    for i, rec in enumerate(records):
        if rec is None:
            continue
        p = tmp_path / f"m{i}.json"
        p.write_text(json.dumps(rec))
        paths.append(str(p))
    paths += list(extra_files)
    return _run(["--merge", "--plan", str(run["plan_path"]), *paths], world["root"])


def _refused(result, code):
    exit_code, outputs, _, err = result
    assert exit_code == 1
    assert "spec-pct" not in outputs, "a refusal must publish no number"
    assert f"WESKER REFUSED TO REPORT: {code}:" in err
    return err


def test_a_missing_shard_report_refuses_and_names_the_shard_and_its_files(
    world, tmp_path
):
    err = _refused(
        _merge(world, lambda r: r.__setitem__(1, None), tmp_path=tmp_path),
        "report_missing",
    )
    assert "shard 2/3" in err
    assert " ".join(world["runs"][3]["plan"]["groups"][1]) in err


def test_no_reports_at_all_refuses(world):
    run = world["runs"][2]
    result = _run(["--merge", "--plan", str(run["plan_path"])], world["root"])
    _refused(result, "no_reports")


def test_a_missing_plan_refuses(world):
    result = _run(["--merge", *map(str, world["runs"][2]["records"])], world["root"])
    _refused(result, "plan_missing")


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("commit", "0" * 40, "commit_disagree"),
        ("policy_id", "9.other", "policy_id_disagree"),
        ("wesker_version", "0.0.0", "wesker_version_disagree"),
        ("commit", "", "commit_missing"),
        ("policy_id", "", "policy_id_missing"),
        ("wesker_version", "", "wesker_version_missing"),
    ],
)
def test_shards_that_disagree_or_do_not_say_what_they_measured_refuse(
    world, tmp_path, field, value, code
):
    def mutate(records):
        records[1][field] = value

    err = _refused(_merge(world, mutate, tmp_path=tmp_path), code)
    assert "shard" in err


def test_a_file_measured_by_two_shards_refuses(world, tmp_path):
    def mutate(records):
        records[1]["targets"].append(records[0]["targets"][0])

    err = _refused(_merge(world, mutate, tmp_path=tmp_path), "file_in_two_shards")
    assert records_first_target(world) in err


def records_first_target(world):
    return world["runs"][3]["plan"]["groups"][0][0]


def test_a_shard_that_measured_something_it_was_not_asked_to_refuses(world, tmp_path):
    def mutate(records):
        records[0]["report"]["per_file"]["pkg/elsewhere.py"] = {
            "dof": 1,
            "dof_pinned": 1,
        }

    err = _refused(_merge(world, mutate, tmp_path=tmp_path), "shard_files_differ")
    assert "pkg/elsewhere.py" in err


def test_a_shard_that_measured_fewer_files_than_planned_refuses(world, tmp_path):
    def mutate(records):
        records[2]["targets"] = records[2]["targets"][:-1] or ["pkg/x.py"]

    _refused(_merge(world, mutate, tmp_path=tmp_path), "shard_files_differ")


def test_a_report_cut_for_another_shard_count_refuses(world, tmp_path):
    def mutate(records):
        records[0]["shard_count"] = 4

    _refused(_merge(world, mutate, tmp_path=tmp_path), "shard_count_differs")


def test_two_reports_for_one_shard_refuse(world, tmp_path):
    def mutate(records):
        records[1]["shard_index"] = 1

    _refused(_merge(world, mutate, tmp_path=tmp_path), "duplicate_shard")


def test_an_unreadable_report_refuses(world, tmp_path):
    bad = tmp_path / "garbage.json"
    bad.write_text("{not json")
    result = _merge(
        world,
        lambda r: r.__setitem__(2, None),
        tmp_path=tmp_path,
        extra_files=[str(bad)],
    )
    _refused(result, "report_unreadable")


def test_a_foreign_json_file_is_not_a_shard_report(world, tmp_path):
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"spec_pct": 100}))
    result = _merge(
        world,
        lambda r: r.__setitem__(2, None),
        tmp_path=tmp_path,
        extra_files=[str(foreign)],
    )
    _refused(result, "not_a_shard_report")


def test_a_record_with_no_report_and_no_refusal_is_malformed(world, tmp_path):
    def mutate(records):
        records[0]["report"] = None

    _refused(_merge(world, mutate, tmp_path=tmp_path), "malformed")


@pytest.mark.parametrize(
    "code", ["truncated", "unresolved", "worker_lost", "suite_health"]
)
def test_a_shard_that_refused_makes_the_merge_that_same_refusal(world, tmp_path, code):
    """A refusal is never averaged away: the other shards' numbers do not buy a number back."""

    def mutate(records):
        records[1]["refusal"] = code
        records[1]["refusal_detail"] = "WHY-THE-SHARD-REFUSED"
        records[1]["report"] = None

    err = _refused(_merge(world, mutate, tmp_path=tmp_path), code)
    assert "shard 2/3" in err
    assert " ".join(sorted(world["runs"][3]["plan"]["groups"][1])) in err
    assert "WHY-THE-SHARD-REFUSED" in err


def test_a_truncated_report_is_refused_by_the_merge_even_if_the_shard_let_it_through(
    world, tmp_path
):
    """The merged gate consumes the per-function facts: a cut function in any shard's report refuses."""

    def mutate(records):
        records[0]["report"]["total_truncated"] = 1
        records[0]["report"]["truncated_functions"] = [
            {
                "function_key": "pkg/shapes.py::area",
                "cause": "budget",
                "elapsed_ms": 1.0,
                "tested": 1,
                "universe": 9,
                "containment_lost": [],
            }
        ]

    err = _refused(_merge(world, mutate, tmp_path=tmp_path), "truncated")
    assert "pkg/shapes.py::area" in err


def test_a_real_cut_shard_is_recorded_as_its_refusal_and_the_merge_repeats_it(
    world, tmp_path
):
    """End to end: a shard run on a 1 ms budget is cut, exits 1, records `truncated`, and the merge says so."""
    root = world["root"]
    plan = world["runs"][3]["plan"]
    entry = plan["include"][0]
    code, _, _, err = _run(
        [
            "--budget",
            "1",
            "--shard",
            entry["shard"],
            "--targets",
            *entry["targets"].split(),
        ],
        root,
    )
    rec = json.loads((root / ".wesker/shard_report.json").read_text())
    if code == 0:
        pytest.skip("the 1 ms budget did not cut on this machine")
    assert code == 1 and "WESKER REFUSED TO REPORT" in err
    assert rec["refusal"] in {"truncated", "unresolved"}
    assert rec["refusal_detail"]
    cut = tmp_path / "cut.json"
    cut.write_text(json.dumps(rec))
    others = [str(p) for p in world["runs"][3]["records"][1:]]
    result = _run(
        ["--merge", "--plan", str(world["runs"][3]["plan_path"]), str(cut), *others],
        root,
    )
    _refused(result, rec["refusal"])


def test_a_run_with_no_commit_to_name_records_none_and_the_merge_refuses(tmp_path):
    root = _project(tmp_path / "p")
    code, _, _, err = _run(["--budget", "20000", "--shard", "1/1"], root, sha=None)
    assert code == 0, err
    rec = json.loads((root / ".wesker/shard_report.json").read_text())
    assert rec["commit"] == ""
    code, _, _, _ = _run(
        ["--plan-shards", "1", "--plan-out", "plan.json"], root, sha=None
    )
    result = _run(
        ["--merge", "--plan", "plan.json", ".wesker/shard_report.json"], root, sha=None
    )
    # one shard of one, but nothing says what it measured
    exit_code, _, _, merr = result
    assert exit_code == 1 and "commit_missing" in merr


def test_an_invalid_shard_argument_refuses_before_measuring(tmp_path):
    root = _project(tmp_path / "p")
    for bad in ("0/3", "4/3", "x", "1/0", "-1/2", "1.5/2", "2"):
        code, _, _, err = _run([f"--shard={bad}"], root)
        assert code == 1, bad
        assert "--shard must be i/N" in err
        assert not (root / ".wesker").exists()


# ── The pure decisions ───────────────────────────────────────────


def test_shard_setting_codes():
    assert shard_setting("") == ("whole", 1, 1)
    assert shard_setting("  ") == ("whole", 1, 1)
    assert shard_setting("2/4") == ("shard", 2, 4)
    assert shard_setting("1/1") == ("shard", 1, 1)
    for bad in ("0/3", "4/3", "x", "1/0", "-1/2", "1.5/2", "2", "/", "1/", "/2"):
        assert shard_setting(bad)[0] == "invalid", bad


def test_plan_standing_codes():
    assert plan_standing(["a.py"], 0) == "bad_shard_count"
    assert plan_standing([], 3) == "no_files"
    assert plan_standing([], 0) == "bad_shard_count"
    assert plan_standing(["a b.py"], 2) == "unsplittable_path"
    assert plan_standing([""], 2) == "unsplittable_path"
    assert plan_standing(["a.py", "a.py"], 2) == "duplicate_file"
    assert plan_standing(["a.py", "b.py"], 5) == "planned"


def test_partition_is_complete_disjoint_sorted_and_never_empty():
    weights = {"e.py": 900, "a.py": 100, "b.py": 100, "c.py": 50, "d.py": 50}
    groups = partition(weights, 3)
    assert groups == [["e.py"], ["a.py", "c.py"], ["b.py", "d.py"]]
    assert sorted(f for g in groups for f in g) == sorted(weights)
    assert partition(weights, 1) == [sorted(weights)]
    assert partition(weights, 99) == [
        [f] for f in ("e.py", "a.py", "b.py", "c.py", "d.py")
    ]
    assert partition({}, 3) == []
    assert partition(weights, 0) == []


def test_partition_is_independent_of_input_order_and_hash_seed():
    weights = {f"m{i}.py": (i * 37) % 11 for i in range(20)}
    forward = partition(weights, 4)
    assert partition(dict(reversed(list(weights.items()))), 4) == forward
    code = (
        "import json;from Wesker.shards import partition;"
        "w={f'm{i}.py':(i*37)%11 for i in range(20)};print(json.dumps(partition(w,4)))"
    )
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            },
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert json.loads(out) == forward


def test_partition_is_the_same_literal_on_every_interpreter():
    """The expected value is written out: run this file under 3.12 and 3.14 and both must meet it."""
    weights = {
        "Wesker/engine.py": 561736,
        "Wesker/ci.py": 141893,
        "Wesker/pytest_runner.py": 50742,
        "Wesker/line_coverage.py": 44142,
        "Wesker/isolation.py": 43492,
        "Wesker/monitoring.py": 41724,
        "Wesker/action.py": 39495,
        "Wesker/_isolated_worker.py": 38115,
    }
    assert partition(weights, 4) == [
        ["Wesker/engine.py"],
        ["Wesker/ci.py"],
        [
            "Wesker/_isolated_worker.py",
            "Wesker/monitoring.py",
            "Wesker/pytest_runner.py",
        ],
        ["Wesker/action.py", "Wesker/isolation.py", "Wesker/line_coverage.py"],
    ]


def test_completeness_code_decisions():
    assert completeness_code(0, 0, False) == "none"
    assert completeness_code(2, 0, False) == "truncated"
    assert completeness_code(0, 3, False) == "unresolved"
    assert completeness_code(2, 3, False) == "truncated"
    assert completeness_code(2, 3, True) == "none"


def _good_record():
    return {
        "kind": "wesker-shard-report",
        "schema": 1,
        "shard_index": 1,
        "shard_count": 2,
        "targets": ["a.py"],
        "commit": "c",
        "policy_id": "p",
        "wesker_version": "v",
        "refusal": "",
        "refusal_detail": "",
        "report": {},
    }


def test_record_standing_codes():
    good = _good_record()
    assert record_standing(good) == "ok"
    assert record_standing({}) == "not_a_shard_report"
    assert record_standing({**good, "schema": 2}) == "wrong_schema"
    assert record_standing({**good, "commit": None}) == "malformed"
    assert record_standing({**good, "shard_index": True}) == "malformed"
    assert record_standing({**good, "shard_count": "2"}) == "malformed"
    assert record_standing({**good, "targets": "a.py"}) == "malformed"
    assert record_standing({**good, "targets": [1]}) == "malformed"
    assert record_standing({**good, "report": None}) == "malformed"
    assert record_standing({**good, "report": [1]}) == "malformed"
    assert record_standing({**good, "report": None, "refusal": "truncated"}) == "ok"


def _summaries(n=2):
    plan = [["a.py"], ["b.py"], ["c.py"]][:n]
    sums = [
        {
            "index": i,
            "count": n,
            "targets": list(plan[i - 1]),
            "stray": [],
            "commit": "c",
            "policy_id": "p",
            "wesker_version": "v",
            "refusal": "",
            "detail": "",
        }
        for i in range(1, n + 1)
    ]
    return plan, sums


def test_merge_standing_codes():
    plan, sums = _summaries(3)
    assert merge_standing(plan, sums) == "merged"
    assert merge_standing([], sums) == "plan_empty"
    assert merge_standing([[], ["b.py"]], sums) == "plan_empty"
    assert merge_standing(plan, []) == "no_reports"
    assert merge_standing(plan, sums[:2]) == "report_missing"

    def with_(i, **kw):
        out = copy.deepcopy(sums)
        out[i].update(kw)
        return out

    assert merge_standing(plan, with_(0, count=4)) == "shard_count_differs"
    assert merge_standing(plan, with_(1, index=1)) == "duplicate_shard"
    assert merge_standing(plan, with_(1, index=7)) == "shard_unplanned"
    assert merge_standing(plan, with_(1, index=0)) == "shard_unplanned"
    assert merge_standing(plan, with_(1, commit="x")) == "commit_disagree"
    assert merge_standing(plan, with_(1, commit="")) == "commit_missing"
    assert merge_standing(plan, with_(1, policy_id="x")) == "policy_id_disagree"
    assert merge_standing(plan, with_(1, policy_id="")) == "policy_id_missing"
    assert (
        merge_standing(plan, with_(1, wesker_version="x")) == "wesker_version_disagree"
    )
    assert merge_standing(plan, with_(1, wesker_version="")) == "wesker_version_missing"
    assert (
        merge_standing(plan, with_(1, targets=["b.py", "a.py"])) == "file_in_two_shards"
    )
    assert merge_standing(plan, with_(1, targets=["z.py"])) == "shard_files_differ"
    assert merge_standing(plan, with_(1, stray=["z.py"])) == "shard_files_differ"
    assert merge_standing(plan, with_(2, refusal="worker_lost")) == "worker_lost"
    # the lowest-index refusal names the result
    two = with_(2, refusal="unresolved")
    two[0]["refusal"] = "truncated"
    assert merge_standing(plan, two) == "truncated"
    # structure is judged before a refusal: a wrong commit is not hidden by a shard's refusal
    broken = with_(1, commit="x")
    broken[2]["refusal"] = "truncated"
    assert merge_standing(plan, broken) == "commit_disagree"
