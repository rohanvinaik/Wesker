"""Tests for Wesker's Monty Hall categorical-exclusion layer (filter.py).

A mutation-testing tool that ships an untested exclusion filter is a
contradiction in terms — these pin every structural-signal branch, the
Layer-1 category selection, and the Layer-2 predictive priors. Oracles are
chosen to kill the surviving categories LintGate's profile flagged
(BOUNDARY, LOGICAL, SWAP, TYPE, VALUE, ARITHMETIC): the per-construct
matrix catches `isinstance→True` and `"self"→""` mutants because a
construct that should *not* set a flag asserts it stays False.
"""

from __future__ import annotations

import ast
import textwrap

from Wesker.engine import MutationCategory
from Wesker.filter import (
    CategoryPrior,
    _classify_signal_node,
    _collect_signals,
    filter_categories,
    prioritize_categories,
)


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


# ── Structural signals: one construct sets exactly one flag ──────
# Each case asserts the target flag True AND leaves the others False, so an
# isinstance→True mutant (which would flip an unrelated flag) is killed.


def test_signals_comparison_only():
    s = _collect_signals(_fn("def f(a, b):\n    return a < b\n"))
    assert s.has_comparisons is True
    assert (s.has_arithmetic, s.has_logical, s.has_isinstance) == (False, False, False)
    assert s.has_self_assigns is False


def test_signals_self_assign_requires_store_context():
    # self.x = 1 is a Store; reading self.x (Load) must NOT count as a
    # side-effecting assignment — kills the `and`→`or` mutant on the ctx check.
    store = _collect_signals(_fn("def f(self):\n    self.x = 1\n"))
    load = _collect_signals(_fn("def f(self):\n    return self.x\n"))
    assert store.has_self_assigns is True
    assert load.has_self_assigns is False


def test_signals_self_literal_matters():
    # `other.x = 1` is an attribute Store but not on self — kills the
    # VALUE mutant that blanks the "self" literal (it would then match nothing,
    # or with "self"→"" match empty ids).
    other = _collect_signals(_fn("def f(other):\n    other.x = 1\n"))
    assert other.has_self_assigns is False


def test_signals_global_and_nonlocal():
    assert (
        _collect_signals(_fn("def f():\n    global g\n    g = 1\n")).has_global_nonlocal
        is True
    )


def test_signals_isinstance():
    s = _collect_signals(_fn("def f(a):\n    return isinstance(a, int)\n"))
    assert s.has_isinstance is True
    # A non-isinstance call must not trip it (kills isinstance-name VALUE mutant).
    assert (
        _collect_signals(_fn("def f(a):\n    return len(a)\n")).has_isinstance is False
    )


def test_signals_arithmetic_binop_and_unary():
    assert (
        _collect_signals(_fn("def f(a, b):\n    return a + b\n")).has_arithmetic is True
    )
    assert _collect_signals(_fn("def f(a):\n    return -a\n")).has_arithmetic is True
    # Bitwise op is not in the arithmetic set — must stay False.
    assert (
        _collect_signals(_fn("def f(a, b):\n    return a & b\n")).has_arithmetic
        is False
    )


def test_signals_logical_boolop_and_not():
    assert (
        _collect_signals(_fn("def f(a, b):\n    return a and b\n")).has_logical is True
    )
    assert _collect_signals(_fn("def f(a):\n    return not a\n")).has_logical is True


def test_signals_constant_only_sets_nothing():
    s = _collect_signals(_fn("def f():\n    return 1\n"))
    assert not any(
        [
            s.has_comparisons,
            s.has_self_assigns,
            s.has_global_nonlocal,
            s.has_isinstance,
            s.has_arithmetic,
            s.has_logical,
        ]
    )


def test_collect_signals_counts_params():
    assert _collect_signals(_fn("def f(a, b, c):\n    return 1\n")).param_count == 3
    assert _collect_signals(_fn("def f():\n    return 1\n")).param_count == 0


def test_classify_signal_node_mutates_in_place():
    from Wesker.filter import _FunctionSignals

    sig = _FunctionSignals()
    cmp_node = ast.parse("a < b").body[0].value
    _classify_signal_node(cmp_node, sig)
    assert sig.has_comparisons is True


# ── Layer 1: filter_categories ───────────────────────────────────


def test_filter_value_always_present():
    assert MutationCategory.VALUE in filter_categories(_fn("def f():\n    return 1\n"))


def test_filter_swap_follows_call_sites_not_formal_parameters():
    """`_SwapMutator` mutates CALL SITES — adjacent argument transpositions, used-call
    unwrapping, curated callee duals. It never touches formal parameters, so `param_count >= 2`
    answered a different question and got both directions wrong.

    Under-enumeration is the dangerous one: `square` below has ONE formal parameter and a live
    `pow(x, 2) -> pow(2, x)` target. With the category filtered out, that mutant never entered
    the universe, and a suite testing only `x == 2` (where both spellings give 4) reported
    `✓ COMPLETE (operator universe) · 3/3 killed` — unqualified — while failing to distinguish a
    mutant that differs at every other x. A false SC = 1."""
    one_param_with_a_call = filter_categories(_fn("def f(x):\n    return pow(x, 2)\n"))
    assert MutationCategory.SWAP in one_param_with_a_call

    # And the other direction: two parameters but nothing to transpose is not a SWAP target.
    two_params_no_calls = filter_categories(_fn("def f(a, b):\n    return a\n"))
    assert MutationCategory.SWAP not in two_params_no_calls

    no_params_no_calls = filter_categories(_fn("def f():\n    return 1\n"))
    assert MutationCategory.SWAP not in no_params_no_calls


def test_filter_boundary_from_comparison():
    assert MutationCategory.BOUNDARY in filter_categories(
        _fn("def f(a, b):\n    return a < b\n")
    )
    assert MutationCategory.BOUNDARY not in filter_categories(
        _fn("def f():\n    return 1\n")
    )


def test_filter_state_gated_by_purity():
    src = "def f(self):\n    self.x = 1\n"
    assert MutationCategory.STATE in filter_categories(_fn(src), is_pure=False)
    # Pure functions cannot have observable state mutation — kills the
    # `not is_pure and (...)` LOGICAL mutant.
    assert MutationCategory.STATE not in filter_categories(_fn(src), is_pure=True)


def test_filter_type_arithmetic_logical():
    assert MutationCategory.TYPE in filter_categories(
        _fn("def f(a):\n    return isinstance(a, int)\n")
    )
    assert MutationCategory.ARITHMETIC in filter_categories(
        _fn("def f(a, b):\n    return a + b\n")
    )
    assert MutationCategory.LOGICAL in filter_categories(
        _fn("def f(a, b):\n    return a and b\n")
    )


# ── Layer 2: prioritize_categories ───────────────────────────────


def test_priors_uniform_without_cache():
    priors = prioritize_categories({MutationCategory.VALUE, MutationCategory.BOUNDARY})
    assert all(p.prior == 0.5 for p in priors)
    assert {p.category for p in priors} == {
        MutationCategory.VALUE,
        MutationCategory.BOUNDARY,
    }


def test_priors_computed_from_list_cache():
    cache = {
        "per_category": [
            {"category": "VALUE", "total": 10, "survived": 3},
            {"category": "BOUNDARY", "total": 4, "survived": 3},
        ]
    }
    priors = prioritize_categories(
        {MutationCategory.VALUE, MutationCategory.BOUNDARY}, cache
    )
    by_cat = {p.category: p.prior for p in priors}
    # survived/total — kills the ARITHMETIC mutant (/ → *) and VALUE rounding.
    assert by_cat[MutationCategory.VALUE] == 0.3
    assert by_cat[MutationCategory.BOUNDARY] == 0.75


def test_priors_sorted_descending():
    cache = {
        "per_category": [
            {"category": "VALUE", "total": 10, "survived": 1},
            {"category": "BOUNDARY", "total": 10, "survived": 9},
        ]
    }
    priors = prioritize_categories(
        {MutationCategory.VALUE, MutationCategory.BOUNDARY}, cache
    )
    assert [p.prior for p in priors] == sorted([p.prior for p in priors], reverse=True)
    assert priors[0].category == MutationCategory.BOUNDARY  # highest survival first


def test_priors_zero_total_uses_default():
    # total>0 guard: a zero-total entry must fall back to 0.5, not divide by zero.
    cache = {"per_category": [{"category": "VALUE", "total": 0, "survived": 0}]}
    priors = prioritize_categories({MutationCategory.VALUE}, cache)
    assert priors[0].prior == 0.5


def test_priors_rounded_to_three_places():
    cache = {"per_category": [{"category": "VALUE", "total": 3, "survived": 1}]}
    assert prioritize_categories({MutationCategory.VALUE}, cache)[0].prior == 0.333


def test_priors_accept_dict_cache_format():
    # Defensive dict branch.
    cache = {"per_category": {"VALUE": {"total": 2, "survived": 1}}}
    assert prioritize_categories({MutationCategory.VALUE}, cache)[0].prior == 0.5


def test_category_prior_field_order():
    # Kills the SWAP mutant on CategoryPrior(category=, prior=): category must
    # be a MutationCategory and prior a float, not transposed.
    p = prioritize_categories({MutationCategory.VALUE})[0]
    assert isinstance(p.category, MutationCategory)
    assert isinstance(p.prior, float)
    assert p == CategoryPrior(category=MutationCategory.VALUE, prior=0.5)
