"""Argument-order questions at one call site: which pairs to ask, under a hard budget.

A call ``g(a, b, c)`` poses one question per PAIR of positional arguments — "are these two
positions distinguished?" — answered by transposing that pair. Through policy 6 only NEIGHBOURING
pairs were asked, so the first-and-third transposition ``g(c, b, a)`` was never a question at all:
a suite whose inputs happened to repeat a value in positions 0 and 2 read complete while that
rewrite passed it.

Every pair is its own behavioural dimension (a singleton cover), so the greedy max-coverage rule —
take the question that covers the most still-uncovered distinctions — ties across every unasked
pair, and nearest-first is the deterministic tie-break. Nothing here claims one pair discriminates
more than another; the order only matters once the budget binds.

Enumeration and selection are kept apart on purpose. Every pair has a canonical position and label
whether or not the budget lets it run; the pairs past the budget are WITHHELD — counted, carried to
the census and the certificate, never silently dropped, and never read as verified. Spending the
budget changes the recorded evidence, not what "verified" means.
"""

from __future__ import annotations

# The hard per-call-site budget on argument-order questions: every pair of a call with up to five
# positional arguments (C(5, 2) = 10). A wider call has its nearest pairs asked and the rest withheld.
SWAP_PAIR_BUDGET = 10


def swap_pairs(n_positional: int) -> list[tuple[int, int]]:
    """Every pair of positional-argument indices at a call site, in canonical order (pure — pinned).

    Nearest first: distance 1 — the neighbour transpositions policy 6 asked, in the order it asked
    them — then distance 2, 3, and so on; within one distance, by left index. A call with fewer than
    two positional arguments has no pair.
    """
    return [(i, i + d) for d in range(1, n_positional) for i in range(n_positional - d)]


def swap_plan(n_positional: int, budget: int) -> tuple[list[tuple[int, int]], int]:
    """(pairs ASKED, count WITHHELD) at one call site under a hard budget (pure — pinned).

    The asked pairs are the first ``budget`` of :func:`swap_pairs`'s canonical order; every pair past
    them is withheld and counted, so asked + withheld always equals the pair total. The budget is a
    COUNT, never a sentinel: 0 asks nothing and a negative budget is read as 0 — unlike
    ``max_per_category``, where 0 means unlimited, there is no "no cap" value here.
    """
    pairs = swap_pairs(n_positional)
    asked = pairs[: max(0, budget)]
    return asked, len(pairs) - len(asked)


def swap_label(callee: str, i: int, j: int) -> str:
    """The dimension label for transposing positional arguments ``i`` and ``j`` (pure — pinned).

    Neighbour pairs keep the labels policy 6 emitted, byte for byte — ``SWAP:<callee>`` for (0, 1)
    and ``SWAP:<callee>~p<i>`` for (i, i + 1) — so their fingerprint rows and any flag keyed to them
    are untouched. A farther pair is ``SWAP:<callee>~p<i>,<j>``, a spelling no earlier label used.
    """
    if j == i + 1:
        return f"SWAP:{callee}" if i == 0 else f"SWAP:{callee}~p{i}"
    return f"SWAP:{callee}~p{i},{j}"
