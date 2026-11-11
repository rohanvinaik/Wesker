"""Intent: every disposition the engine leaves unscored has a named owner in the rollup, and a mutant
whose test's write was prevented is counted and named, never dropped (W#51).

THE DEFECT, measured 2026-10-25 on Wesker 6ba533d and again on 7e61cd3, with Detective #105's own
end-to-end fixture (`store.py::save` writes through `atomic_write_text`; its test writes under
`tmp_path`). Detective named the prevented SWAP mutant per mutant:

    ⚠ no verdict         1 — SWAP_6dd88cc8 (write_outside_authority): not kills, not survivors

`python -m Wesker store.py --json` (exit 0) named it nowhere:

    "total_unresolved": 0, "unresolved_functions": []

`save`'s record was `unscored_by == {"write_outside_authority": 1}`. The rollup read two dispositions
by name (`undetermined`, then `diverged`), so the third left the denominator and was reported by
nothing. And it read only the FIRST of those: a function holding both was listed under one cause, with
that cause's mutants only.

THE RULING:
- `ci.UNSCORED_ROLLUP` maps every unscored disposition to its owner:
  - "unresolved": counted, named and refused by the badge under its own remedy;
  - "truncated": `truncation_cause` owns it;
  - "harness": it measures the harness, not the suite.
- Totality is tested against `mutant_disposition` itself.
- A disposition the table does not know is unresolved under its own name.
- Every cause a function holds gets its own entry, and `total_unresolved` still counts functions.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

import pytest

from Wesker.action import gate_unresolved
from Wesker.ci import (
    _UNRESOLVED_REMEDIES,
    UNSCORED_ROLLUP,
    completeness_warnings,
    describe_unresolved,
    unresolved_causes,
)
from Wesker.engine import (
    ORIGINAL_ALSO_FAILS,
    SCORED_DISPOSITIONS,
    mutant_disposition,
)
from Wesker.write_authority import WRITE_OUTSIDE_AUTHORITY

# ── the table is total over what the engine can return ────────────────────────────────────────


def _engine_dispositions() -> set[str]:
    """Every value `mutant_disposition` returns over its whole input grid, every refusal code included."""
    seen: set[str] = set()
    for (
        constructed,
        installed,
        entered,
        contained,
        killed,
        refusal,
        killed_by,
    ) in itertools.product(
        (True, False),
        (True, False),
        (True, False, None),
        (True, False),
        (True, False),
        ("", "nested_measurement", WRITE_OUTSIDE_AUTHORITY, ORIGINAL_ALSO_FAILS),
        (
            None,
            "assertion",
            "exception",
            "crash",
            "diverged",
            "undetermined",
            "timeout",
        ),
    ):
        seen.add(
            mutant_disposition(
                constructed,
                installed,
                entered,
                contained,
                killed,
                refusal,
                killed_by=killed_by,
            )
        )
    return seen


def test_every_unscored_disposition_the_engine_returns_has_exactly_one_owner():
    owners = dict(UNSCORED_ROLLUP)
    assert len(owners) == len(UNSCORED_ROLLUP), (
        "a disposition listed twice has two owners"
    )
    unscored = _engine_dispositions() - set(SCORED_DISPOSITIONS)
    assert unscored == set(owners), (
        "unowned: "
        + repr(sorted(unscored - set(owners)))
        + " / unreachable: "
        + repr(sorted(set(owners) - unscored))
    )
    assert set(owners.values()) == {"unresolved", "truncated", "harness"}


def test_every_unresolved_owner_has_a_remedy_and_every_remedy_an_owner():
    unresolved = [d for d, owner in UNSCORED_ROLLUP if owner == "unresolved"]
    assert [cause for cause, _ in _UNRESOLVED_REMEDIES] == unresolved


# ── the decision ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("unscored_by", "causes"),
    [
        ({WRITE_OUTSIDE_AUTHORITY: 1}, (WRITE_OUTSIDE_AUTHORITY,)),
        (
            {WRITE_OUTSIDE_AUTHORITY: 2, "diverged": 1, "undetermined": 1},
            ("undetermined", "diverged", WRITE_OUTSIDE_AUTHORITY),
        ),
        ({"cut": 3, "harness_error": 1, "not_installed": 1}, ()),
        ({"not_entered": 4, "nested_measurement": 2}, ()),
        ({WRITE_OUTSIDE_AUTHORITY: 0}, ()),
        # Unknown to the table: unresolved under its own name, after the known causes, never dropped.
        ({"zz_new_outcome": 1, "diverged": 1}, ("diverged", "zz_new_outcome")),
        ({"b_new": 1, "a_new": 1}, ("a_new", "b_new")),
    ],
)
def test_each_cause_a_function_holds_is_named_and_an_unknown_one_is_never_dropped(
    unscored_by, causes
):
    assert unresolved_causes(unscored_by) == causes


def test_the_badge_refuses_a_prevented_write_under_its_own_remedy():
    report = {
        "total_unresolved": 1,
        "unresolved_functions": [
            {
                "function_key": "store.py::save",
                "cause": WRITE_OUTSIDE_AUTHORITY,
                "mutants": ["SWAP_a"],
            }
        ],
    }
    refusal = gate_unresolved(report)
    assert refusal is not None
    assert "write outside its write authority" in refusal
    assert "store.py::save — SWAP_a" in refusal
    assert "cause not recorded" not in describe_unresolved(
        report["unresolved_functions"]
    )


def test_the_shared_warning_renderer_names_each_missing_cause_and_is_silent_without_one():
    assert completeness_warnings({}) == ""
    assert completeness_warnings({"total_truncated": 0, "total_unresolved": 0}) == ""
    text = completeness_warnings(
        {
            "total_truncated": 1,
            "truncated_functions": [
                {
                    "function_key": "m.py::t",
                    "cause": "budget",
                    "tested": 1,
                    "universe": 2,
                }
            ],
            "total_unresolved": 1,
            "unresolved_functions": [
                {
                    "function_key": "store.py::save",
                    "cause": WRITE_OUTSIDE_AUTHORITY,
                    "mutants": ["SWAP_a"],
                }
            ],
        }
    )
    assert "1 function(s) were only PARTIALLY evaluated" in text
    assert "m.py::t" in text
    assert "1 function(s) hold mutants with no verdict" in text
    assert "store.py::save — SWAP_a" in text


# ── end to end through the real CLI: #105's own fixture ─────────────────────────────────────────

_STORE = (
    "import os\n\n\n"
    "def atomic_write_text(path, text):\n"
    '    staging = f"{path}.tmp-{os.getpid()}"\n'
    '    with open(staging, "w") as fh:\n'
    "        fh.write(text)\n"
    "    os.replace(staging, path)\n\n\n"
    "def save(path, text):\n"
    "    atomic_write_text(path, text)\n"
    "    return path\n"
)
_TEST = (
    "from store import save\n\n\n"
    "def test_save_writes_the_file(tmp_path):\n"
    "    target = tmp_path / 'store.json'\n"
    "    assert save(str(target), '{\"a\": 1}') == str(target)\n"
    "    assert target.exists()\n"
    "    assert target.read_text() == '{\"a\": 1}'\n"
)


def _wesker(cwd, *argv):
    return subprocess.run(
        [sys.executable, "-m", "Wesker", *argv],
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_a_prevented_write_is_counted_and_named_by_the_cli_rollup(tmp_path):
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "store.py").write_text(_STORE)
    (project / "tests" / "test_store.py").write_text(_TEST)

    proc = _wesker(project, "store.py", "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    mutation = json.loads(proc.stdout)["mutation"]
    assert mutation["total_unresolved"] == 1, mutation
    [entry] = [
        u
        for u in mutation["unresolved_functions"]
        if u["cause"] == WRITE_OUTSIDE_AUTHORITY
    ]
    assert entry["function_key"] == "store.py::save", entry
    assert len(entry["mutants"]) == 1 and entry["mutants"][0].startswith("SWAP_"), entry

    text = _wesker(project, "store.py")
    assert "hold mutants with no verdict" in text.stderr, text.stderr
    assert "write outside its write authority" in text.stderr, text.stderr
    assert "store.py::save — SWAP_" in text.stderr, text.stderr
    assert sorted(os.listdir(tmp_path)) == ["proj"], (
        "a profiled mutant wrote beside the project"
    )
