"""Intent pins for asking every argument-order question under a hard budget.

THE GAP, measured 2026-09-11 through the real CLI. For ``f(a, b, c) = g(a, b, c)`` with the
existing test ``f(1, 2, 1) == 8``, Detective reported the suite complete over f's universe of 4,
and ``verify-rewrite`` returned PRESERVED for the rewrite ``g(c, b, a)`` -- which returns 10 where
the original returns 14 at (1, 2, 3). SWAP only ever asked about NEIGHBOURING argument pairs, so the
first-and-third transposition was never a question: a suite that happens to repeat a value in
positions 0 and 2 read complete while that rewrite passed it. (A suite Detective builds itself picks
distinct values and caught the same rewrite; the hole is specific to suites that repeat a value.)

THE RULE: every pair of positional arguments is a question -- one behavioural dimension each --
asked nearest-first up to a hard per-call-site budget of 10, which covers every pair of a call with
up to five positional arguments. What the budget leaves unasked is WITHHELD and counted: never
dropped, never read as verified. Neighbour pairs keep their policy-6 labels byte for byte, so nothing
already measured moves.

These tests are written from what the plan must do, not from what it currently returns; the
generated suites beside them are characterizations and would pin a wrong answer wrong.
"""

from __future__ import annotations

from math import comb

import pytest

from Wesker.swap_plan import SWAP_PAIR_BUDGET, swap_label, swap_pairs, swap_plan


def test_the_budget_asks_every_pair_of_a_five_argument_call():
    assert SWAP_PAIR_BUDGET == comb(5, 2) == 10


@pytest.mark.parametrize("n", range(13))
def test_every_pair_is_enumerated_exactly_once(n):
    pairs = swap_pairs(n)
    assert len(pairs) == comb(n, 2)
    assert len(set(pairs)) == len(pairs)
    assert all(0 <= i < j < n for i, j in pairs)


@pytest.mark.parametrize("n", range(13))
def test_neighbour_pairs_are_a_prefix_in_policy_6_order(n):
    """The transpositions policy 6 asked come first, in the order it asked them."""
    neighbours = [(i, i + 1) for i in range(n - 1)]
    assert swap_pairs(n)[: len(neighbours)] == neighbours


def test_order_is_nearest_first_then_left_index():
    assert swap_pairs(3) == [(0, 1), (1, 2), (0, 2)]
    assert swap_pairs(4) == [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]


def test_a_three_argument_call_gains_exactly_the_first_third_question():
    """The audit's rewrite ``g(c, b, a)`` is the (0, 2) transposition -- now asked."""
    assert swap_plan(3, SWAP_PAIR_BUDGET) == ([(0, 1), (1, 2), (0, 2)], 0)


def test_five_arguments_fit_the_budget_exactly():
    asked, withheld = swap_plan(5, SWAP_PAIR_BUDGET)
    assert len(asked) == 10
    assert withheld == 0


def test_six_arguments_ask_the_ten_nearest_and_withhold_five():
    asked, withheld = swap_plan(6, SWAP_PAIR_BUDGET)
    assert asked == [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (0, 2), (1, 3), (2, 4), (3, 5),
        (0, 3),
    ]  # fmt: skip
    assert withheld == 5


def test_the_cap_can_defer_a_question_policy_6_asked():
    """Compatibility, stated rather than hidden. A 12-argument call had 11 neighbour
    transpositions under policy 6. The hard cap of 10 asks the first ten and WITHHOLDS the
    eleventh: it keeps its identity and its label, and the budget defers its execution. The cap
    is not silently raised to preserve the old schedule."""
    asked, withheld = swap_plan(12, SWAP_PAIR_BUDGET)
    assert asked == [(i, i + 1) for i in range(10)]
    assert (10, 11) not in asked
    assert withheld == comb(12, 2) - 10
    assert swap_label("g", 10, 11) == "SWAP:g~p10"


@pytest.mark.parametrize("n", range(13))
@pytest.mark.parametrize("budget", [0, 1, 3, 10, 100])
def test_asked_plus_withheld_accounts_for_every_pair(n, budget):
    asked, withheld = swap_plan(n, budget)
    assert len(asked) + withheld == comb(n, 2)
    assert asked == swap_pairs(n)[: len(asked)]


def test_zero_or_negative_budget_asks_nothing_never_everything():
    """0 is a count here -- not the "unlimited" sentinel ``max_per_category`` uses."""
    assert swap_plan(4, 0) == ([], 6)
    assert swap_plan(4, -3) == ([], 6)


def test_neighbour_labels_are_the_policy_6_labels_byte_for_byte():
    assert swap_label("g", 0, 1) == "SWAP:g"
    assert swap_label("g", 1, 2) == "SWAP:g~p1"
    assert swap_label("g", 4, 5) == "SWAP:g~p4"


def test_farther_pairs_use_a_spelling_no_other_pair_shares():
    assert swap_label("g", 0, 2) == "SWAP:g~p0,2"
    labels = [swap_label("g", i, j) for i, j in swap_pairs(12)]
    assert len(set(labels)) == len(labels)
