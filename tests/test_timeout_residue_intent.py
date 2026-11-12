"""Intent: Wesker #27 — nothing in the engine still treats a wall-clock timeout as a kill.

The residue the issue lists after the producers were removed: dead `killed_by == "timeout"` arms
that bumped `CategoryResult.timed_out` (always 0, but summed into `value_survived`), and
`isolation.mutant_verdict`, which mapped the outcome `timeout` to `killed`. A fired wall is
`undetermined`: not a kill, not a survivor.

What was kept on purpose: `mutant_disposition` and the in-process loops still read a legacy
`killed_by == "timeout"` as the unscored `undetermined`. Nothing produces that spelling any more, but
removing the guard would send a stray one down the default branch, which is a KILL. A guard for the
one direction that must never happen is not dead code.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

import Wesker
from Wesker.engine import (
    SCORED_DISPOSITIONS,
    CategoryResult,
    MutationCategory,
    mutant_disposition,
)
from Wesker.isolation import isolated_test_outcome, mutant_verdict


def test_a_hung_node_is_not_a_kill():
    assert mutant_verdict("timeout") == "undetermined"
    assert mutant_verdict("failed") == "killed"
    assert mutant_verdict("passed") == "survived"
    for harness in ("no_tests", "error", "", "unheard_of"):
        assert mutant_verdict(harness) == "harness"


def test_a_timed_out_run_is_named_timeout_before_it_is_judged():
    assert isolated_test_outcome(1, True) == "timeout"
    assert mutant_verdict(isolated_test_outcome(1, True)) != "killed"


def test_the_dead_counter_is_gone_and_unspecified_value_is_survivors_plus_crashes():
    fields = {f.name for f in dataclasses.fields(CategoryResult)}
    assert "timed_out" not in fields
    with pytest.raises(TypeError):
        CategoryResult(category=MutationCategory.VALUE, timed_out=3)  # type: ignore[call-arg]
    cr = CategoryResult(
        category=MutationCategory.VALUE,
        survived=2,
        killed_by_crash=1,
        killed_by_assertion=4,
    )
    assert cr.value_survived == 3
    assert cr.value_killed == 4


def test_a_stray_legacy_timeout_is_still_never_scored_as_a_kill():
    disposition = mutant_disposition(True, True, True, True, False, killed_by="timeout")
    assert disposition == "undetermined"
    assert disposition not in SCORED_DISPOSITIONS


def test_no_source_in_the_package_names_timeout_as_a_kill_reason():
    # The search that proves nothing produces it: no keyword `killed_by="timeout"` and no assignment
    # of the string to a `killed_by` attribute anywhere in the package.
    offenders = []
    for path in pathlib.Path(Wesker.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "killed_by"
                and isinstance(node.value, ast.Constant)
                and node.value.value == "timeout"
            ):
                offenders.append(f"{path.name}:{node.value.lineno}")
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and node.value.value == "timeout"
                and any(
                    isinstance(t, ast.Attribute) and t.attr == "killed_by"
                    for t in node.targets
                )
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
