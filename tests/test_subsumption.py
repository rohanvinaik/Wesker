"""Empirical redundancy is a REPORT, never a smaller universe (issue #25).

Cost is managed today by quantity — `--max-per-category`, sampling — and every one of those
buys speed by reducing coverage of the universe, weakening the completeness claim. Redundancy
is the principled lever: a mutant the current suite cannot distinguish from another adds no
information to a REPORT, and dropping it there costs the claim nothing.

The grouping is derived from one suite's kill matrix, so it is a fact about that suite and not
about the programs — a different suite splits what this one merges. It may label a report; it
may never shrink the set a completeness claim is made over. These pin the boundary, and above
all the one inversion that would make the feature actively harmful.
"""

from __future__ import annotations

from Wesker.subsumption import distinct_obligations, redundancy_groups


def test_survivors_are_never_grouped_together():
    """THE trap. Every unkilled mutant shares the empty killer set, so a naive grouping merges
    all of them into ONE — turning 40 real gaps into a single reported obligation and inverting
    the answer. An empty set means no test detected this, not these behave the same. Absence of
    evidence groups nothing."""
    matrix = {"s1": [], "s2": [], "s3": [], "killed": ["t1"]}
    groups = redundancy_groups(matrix)
    flat = [m for g in groups for m in g]
    assert "s1" not in flat and "s2" not in flat and "s3" not in flat
    assert groups == [["killed"]]


def test_mutants_with_the_same_killers_are_one_obligation():
    """The payoff. Every test detecting one detects the other, so asking a reader to consider
    both is asking about a distinction the evidence does not contain."""
    matrix = {"A": ["t1", "t2"], "B": ["t1", "t2"]}
    assert redundancy_groups(matrix) == [["A", "B"]]


def test_killer_order_and_repetition_are_not_distinctions():
    """The matrix records a killer per kill EVENT, so ordering and duplicates are artefacts of
    evaluation order. Treating them as distinctions would split groups by scheduling noise."""
    matrix = {"A": ["t2", "t1"], "B": ["t1", "t2", "t1"]}
    assert redundancy_groups(matrix) == [["A", "B"]]


def test_different_killers_stay_separate():
    """The control against over-merging. A grouping that collapsed these would claim the suite
    cannot tell apart two mutants it demonstrably can."""
    matrix = {"A": ["t1"], "B": ["t2"]}
    assert redundancy_groups(matrix) == [["A"], ["B"]]


def test_a_subset_is_not_a_group():
    """Only IDENTICAL killer sets group. `killers(A) ⊂ killers(B)` is the subsumption relation,
    which is a stronger and directional claim — deliberately not asserted here, because getting
    its direction wrong would drop a dominator and silently lose discriminating power."""
    matrix = {"A": ["t1"], "B": ["t1", "t2"]}
    assert redundancy_groups(matrix) == [["A"], ["B"]]


def test_the_ordering_is_stable():
    """Two identical measurements must not produce a diff."""
    matrix = {"z": ["t1"], "a": ["t2"], "m": ["t2"]}
    assert redundancy_groups(matrix) == redundancy_groups(
        dict(reversed(matrix.items()))
    )
    assert redundancy_groups(matrix) == [["a", "m"], ["z"]]


def test_distinct_obligations_counts_groups_plus_each_survivor():
    """The honest headline: indistinguishable kills collapse to one apiece, survivors each stay
    their own. Derived from the groups rather than recomputed beside them, so the two numbers
    cannot drift."""
    matrix = {"A": ["t1"], "B": ["t1"], "C": ["t2"], "s": []}
    assert redundancy_groups(matrix) == [["A", "B"], ["C"]]
    assert distinct_obligations(matrix, survivors=3) == 5
