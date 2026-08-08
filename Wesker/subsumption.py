"""Empirical mutant redundancy — manage cost by duplication, not by sampling (issue #25).

Cost is managed today by QUANTITY: `--max-per-category`, pass indices, sampling. Every one of
those reduces cost by reducing coverage of the universe, which weakens the completeness claim
in exchange. The literature manages it by REDUNDANCY instead, which is the principled version
of the same lever and costs the claim nothing: a mutant whose kill is implied by another's
contributes no discriminating power, so dropping it from a REPORT removes a duplicate rather
than an obligation.

> Citation recalled, not looked up (Ammann, Delamaro & Offutt on minimal mutant sets). Verify
> before quoting publicly.

WHAT THIS IS FOR, AND THE ONE THING IT MUST NEVER DO. A user staring at 40 survivors is mostly
staring at 40 spellings of a handful of distinctions. Grouping them turns "40 unpinned
behaviours" into the honest, actionable number. But the grouping is derived from the CURRENT
suite's kill matrix, so it is an empirical fact about that suite and not a property of the
programs — a different suite splits groups this one merges. It may therefore label a report and
must never shrink the universe a completeness claim is made over. `ARCHITECTURE.md`'s standing
rule applies exactly: an empirical grouping is not a proof, and promoting one to the other by
assertion is the failure mode this whole tool exists to refuse.

Static subsumption — structurally implied pairs, e.g. `if False and X` making every operand
mutation on `X` unreachable — is a genuine proof and a different mechanism. It is NOT
implemented here; this module holds only the empirical half.
"""

from __future__ import annotations


def redundancy_groups(kill_matrix: dict[str, list[str]]) -> list[list[str]]:
    """Mutants the CURRENT suite cannot tell apart, grouped; each group is one obligation.

    Two mutants killed by exactly the same set of tests are indistinguishable BY THIS SUITE:
    every test that detects one detects the other, so reporting both asks the reader to consider
    a distinction the evidence does not contain. `len(result)` is the number of distinct
    obligations the matrix actually witnesses.

    AN EMPTY KILLER SET IS EXCLUDED, and this is the load-bearing detail. Every unkilled mutant
    shares the empty set, so a naive grouping merges all 40 survivors into ONE — precisely
    inverting the answer, since an empty set means "no test detected this" and not "these behave
    the same". Absence of evidence groups nothing. Survivors are exactly the mutants a reader
    most needs enumerated, so they are omitted here rather than silently collapsed; the caller
    still holds them and reports them as the individual gaps they are.

    Killer lists are normalised to sets: the matrix records a killer per kill event, so order and
    repetition are artefacts of evaluation order, not distinctions. Groups and their members are
    sorted so a report is stable across runs — an unstable ordering makes two identical
    measurements produce a diff.
    """
    by_killers: dict[frozenset[str], list[str]] = {}
    for mutant, killers in kill_matrix.items():
        key = frozenset(killers)
        if not key:
            continue
        by_killers.setdefault(key, []).append(mutant)
    return sorted((sorted(group) for group in by_killers.values()), key=lambda g: g[0])


def distinct_obligations(kill_matrix: dict[str, list[str]], survivors: int) -> int:
    """The honest count of distinct things a reader must act on.

    Groups of indistinguishable killed mutants collapse to one apiece; survivors each stay their
    own, for the reason :func:`redundancy_groups` excludes them. Kept separate from the grouping
    itself so the two numbers cannot drift: the count is derived from the groups rather than
    recomputed beside them.
    """
    return len(redundancy_groups(kill_matrix)) + survivors
