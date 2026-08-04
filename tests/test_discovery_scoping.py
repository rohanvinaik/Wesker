"""Discovery must hand back the test files that can REACH the target — not the tree.

The regression these guard: `discover_test_callables` accepted `source_file` and
`func_names` and, on the pytest backend, used neither. Profiling one function in a
consumer repo collected 549 of 637 callables, and the ~12x that cannot reach the target
were paid for three times — in collection, in the traced baseline, and again per mutant.
Nothing failed; it was only slow, which is why it survived a green suite.

Every test here is hermetic (its own tmp tree) so a change to THIS repo's own test layout
cannot make them pass or fail for the wrong reason.
"""

from __future__ import annotations

import os

import pytest

from Wesker.ci import callable_origin, discover_test_callables, relevant_test_files

_TARGET = """\
def shipping_cost(weight_kg, distance_km):
    if weight_kg <= 0:
        raise ValueError("weight must be positive")
    return 4.5 + _surcharge(weight_kg) + distance_km * 0.05


def _surcharge(weight_kg):
    return (weight_kg - 5) * 0.6 if weight_kg > 5 else 0.0
"""

_ORPHAN = """\
def lonely(units):
    return units * 2
"""


@pytest.fixture
def project(tmp_path):
    """A tree with one conventionally-named test, one impact-only test, and one that
    mentions the target nowhere."""
    (tmp_path / "shipping.py").write_text(_TARGET)
    (tmp_path / "orphan.py").write_text(_ORPHAN)
    tests = tmp_path / "tests"
    tests.mkdir()
    # Layer 1: named for the source file.
    (tests / "test_shipping.py").write_text(
        "from shipping import shipping_cost\n\n\ndef test_base():\n    assert shipping_cost(1, 0) == 4.5\n"
    )
    # Layer 2: named for nothing, but references a function of the target file.
    (tests / "test_pricing_rules.py").write_text(
        "from shipping import _surcharge\n\n\ndef test_surcharge():\n    assert _surcharge(1) == 0.0\n"
    )
    # Neither: must never be selected for `shipping.py`.
    (tests / "test_unrelated.py").write_text(
        "def test_arith():\n    assert 2 + 2 == 4\n"
    )
    return tmp_path


def _names(paths) -> set[str]:
    return {os.path.basename(p) for p in paths}


def test_convention_and_impact_are_selected(project):
    got = _names(
        relevant_test_files(
            str(project), str(project / "shipping.py"), ["shipping_cost", "_surcharge"]
        )
    )
    assert "test_shipping.py" in got, (
        "layer 1 (convention) must select the same-named test file"
    )
    assert "test_pricing_rules.py" in got, (
        "layer 2 (static impact) must select a file naming the target"
    )


def test_unrelated_file_is_excluded(project):
    """THE regression. `discover_tests` returns this file too — its layer 3 appends every
    remaining test unconditionally, which is what made the three layers a ranking rather
    than a selection."""
    got = _names(
        relevant_test_files(
            str(project), str(project / "shipping.py"), ["shipping_cost", "_surcharge"]
        )
    )
    assert "test_unrelated.py" not in got


def test_selection_never_invents_a_file(project):
    """Narrowing may only ever drop; a scope containing something that is not a test file
    on disk would mean the selector is generating paths rather than filtering them."""
    got = relevant_test_files(
        str(project), str(project / "shipping.py"), ["shipping_cost"]
    )
    assert got, "sanity: this target has relevant tests"
    for p in got:
        assert os.path.isfile(p), f"selected a path that does not exist: {p}"
        assert os.path.basename(p).startswith("test_")


def test_orphan_target_selects_nothing(project):
    """No convention match and no file naming it: the honest answer is the empty set, and
    it must NOT widen to the tree. Callers read empty as 'synthesize'."""
    assert (
        relevant_test_files(str(project), str(project / "orphan.py"), ["lonely"]) == []
    )


def test_orphan_target_discovers_no_callables(project):
    """The empty selection has to survive the backend. Widening here would put every
    unrelated test through a full mutant pass to re-learn what the selection already said."""
    assert discover_test_callables(str(project), "orphan.py", ["lonely"]) == []


def test_discovered_callables_come_only_from_scoped_files(project):
    """End of the pipe: whatever backend answered, every callable must originate in a file
    the scope selected. Origin resolves through the contract accessor — a raw `__code__`
    read attributes every wrapper to Wesker's own runner module."""
    calls = discover_test_callables(
        str(project), "shipping.py", ["shipping_cost", "_surcharge"]
    )
    assert calls, (
        "the target has tests; discovery returning nothing would be the opposite bug"
    )
    origins = {os.path.basename(callable_origin(c) or "") for c in calls}
    assert origins <= {"test_shipping.py", "test_pricing_rules.py"}, origins
    assert "test_unrelated.py" not in origins


def test_extra_dirs_still_collected_when_scope_is_empty(project, tmp_path):
    """converge writes its suite to a --write-dir that can sit outside the tree, and the
    first profiling pass runs BEFORE anything is written there. An empty in-tree scope must
    not discard that root, or the kill count reports 0% for tests that do exist."""
    outside = tmp_path.parent / "written_out_of_tree"
    outside.mkdir(exist_ok=True)
    (outside / "test_generated_orphan.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(project)!r})\n"
        "from orphan import lonely\n\n\ndef test_lonely():\n    assert lonely(2) == 4\n"
    )
    calls = discover_test_callables(
        str(project), "orphan.py", ["lonely"], extra_dirs=[str(outside)]
    )
    assert calls, (
        "an out-of-tree write dir must still be collected when the in-tree scope is empty"
    )
