"""Tests for Wesker's Monty Hall categorical-exclusion layer (filter.py).

A mutation-testing tool that ships an untested exclusion filter is a
contradiction in terms — these pin the Layer-1 category selection and the
Layer-2 predictive priors. Layer 1's whole contract is now a single sentence:
a category is relevant exactly when the engine counts at least one target for
it, so exclusion is lossless by construction. The structural-signal matrix
that used to live here died with ``_collect_signals``: the signals were a
second copy of each mutator's eligibility predicate, and issue #9 showed what
a second copy costs (a false ``✓ COMPLETE`` over a dropped SWAP universe).
"""

from __future__ import annotations

import ast
import textwrap

from Wesker.engine import MutationCategory, estimate_universe_size, generate_mutants
from Wesker.filter import (
    CategoryPrior,
    filter_categories,
    prioritize_categories,
)


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


# ── Layer 1: filter_categories ───────────────────────────────────


def test_filter_eligibility_is_the_target_count():
    """The one contract: cat ∈ filter_categories(f) ⇔ the engine counts ≥ 1
    target for cat. Checked over EVERY category on a function that exercises
    several (VALUE, BOUNDARY, ARITHMETIC, STATE) and misses the rest, so a
    filter branch that drifted from the engine's count fails by name.

    OUTPUT (μ⁻) is the one policy-gated exception, exactly as remove_assign is under the
    ``is_pure`` overlay: it has live targets but the DEFAULT one-sign policy withholds it, so
    its count↔eligibility contract is checked under the two-sign policy that enables it.
    """
    fn = _fn(
        """
        def f(a, b):
            if a < b:
                return a + b
            return 0
        """
    )
    cats = filter_categories(fn)
    two_sign = filter_categories(fn, two_sign=True)
    for cat in MutationCategory:
        if cat is MutationCategory.OUTPUT:
            assert (
                cat not in cats
            )  # withheld under the one-sign default despite live targets
            assert (cat in two_sign) == (estimate_universe_size(fn, {cat}) > 0), (
                cat.value
            )
        else:
            assert (cat in cats) == (estimate_universe_size(fn, {cat}) > 0), cat.value


def test_filter_value_follows_constants():
    # `return 1` carries a VALUE target; a constant-free body carries none.
    # VALUE used to be unconditionally relevant — harmless (zero targets
    # generate zero mutants) but a lie about the universe.
    assert MutationCategory.VALUE in filter_categories(_fn("def f():\n    return 1\n"))
    assert MutationCategory.VALUE not in filter_categories(
        _fn("def f(a):\n    return a\n")
    )


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
    # A caller asserting purity asserts self.x writes are unobservable — the
    # remove_assign sub-mode's targets stop counting, and nothing else in this
    # function counts toward STATE.
    assert MutationCategory.STATE not in filter_categories(_fn(src), is_pure=True)


def test_filter_purity_never_hides_return_none():
    # Purity suppresses ONLY remove_assign. A pure function that returns a
    # value still owes its callers the return_none question.
    src = "def f(a):\n    return a\n"
    assert MutationCategory.STATE in filter_categories(_fn(src), is_pure=True)


def test_filter_loop_flow_without_return_or_state_write():
    """The silent universe shrink this rewrite closes: `break` is a live
    loop_flow target, but the old signal gate (has_return_value or
    self-assign/global) excluded STATE for a procedure with neither — so
    `break` ↔ `continue` never entered its universe, the same #9 shape in a
    different category."""
    fn = _fn(
        """
        def drain(q, sink):
            while True:
                item = q.get()
                if item is None:
                    break
                sink.append(item)
        """
    )
    assert MutationCategory.STATE in filter_categories(fn)
    state_mutants = generate_mutants(fn, {MutationCategory.STATE}, max_per_category=0)
    assert any("swap break/continue" in m.description for m in state_mutants)


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


def test_filter_exception_counts_real_targets():
    assert MutationCategory.EXCEPTION in filter_categories(
        _fn("def f(a):\n    raise ValueError(a)\n")
    )
    assert MutationCategory.EXCEPTION not in filter_categories(
        _fn("def f(a):\n    return a\n")
    )


def test_filter_empty_universe_is_an_empty_set():
    # A body with no targets in ANY category filters to nothing at all —
    # under the old unconditional-VALUE rule this returned {VALUE} and callers
    # profiled a zero-mutant function.
    assert filter_categories(_fn("def f():\n    pass\n")) == set()


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
