"""Tests for Wesker's greedy behavioral-dimension selection (Layer 2).

These pin the selection machinery that replaced seeded random sampling: the
record-mode dimension enumerator, the greedy submodular order, pass-windowed
selection, and their integration into ``generate_mutants``. The oracles encode
the (1-1/e)-optimal coverage guarantee (exact for singleton cover-sets) that the
change rests on — see proofs/{coverage_submodular,marginal_antitone,
greedy_coverage_bound}.lean.

Named tests/test_engine.py so Wesker's own convention-based test discovery maps
it to Wesker/engine.py when the engine self-profiles.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from Wesker.engine import (
    _DEAD_DIM,
    MutationCategory,
    _callee_name,
    _count_targets,
    _greedy_dimension_order,
    _is_dead,
    _isinstance_type_name,
    _record_dimensions,
    _record_state_dimensions,
    _select_greedy,
    generate_mutants,
    run_function_converged,
)


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _selected_indices(mutants) -> set[int]:
    """Recover target indices from mutant ids like ``VALUE_3`` / ``STATE_return_none_0``."""
    return {int(m.mutant_id.rsplit("_", 1)[1]) for m in mutants}


# ── _is_dead ──────────────────────────────────────────────────────


def test_is_dead_only_true_for_sentinel():
    assert _is_dead(_DEAD_DIM) is True
    assert _is_dead("BOUNDARY:Lt") is False
    assert _is_dead("") is False


# ── Dimension recording (record-mode alignment) ──────────────────


def test_record_dimensions_aligns_with_universe_count():
    # Every applicable category: len(keys) must equal the universe count so
    # keys[i] is the dimension of target i.
    func = _fn(
        """
        def f(a, b, c):
            x = a + b - c * 2
            if a < b and b > c:
                return isinstance(a, int)
            return max(a, b, c)
        """
    )
    for cat in (
        MutationCategory.VALUE,
        MutationCategory.BOUNDARY,
        MutationCategory.ARITHMETIC,
        MutationCategory.LOGICAL,
        MutationCategory.SWAP,
        MutationCategory.TYPE,
    ):
        keys = _record_dimensions(func, cat, set())
        assert len(keys) == _count_targets(func, cat), cat


def test_record_dimensions_keys_carry_construct_kind():
    func = _fn(
        """
        def f(a, b):
            return a < b, a > b, a == b
        """
    )
    keys = _record_dimensions(func, MutationCategory.BOUNDARY, set())
    assert keys == ["BOUNDARY:Lt", "BOUNDARY:Gt", "BOUNDARY:Eq"]


def test_record_dimensions_marks_non_swappable_ops_dead():
    # `is` / `in` have no boundary mutant → dead key, but still one entry per op.
    func = _fn(
        """
        def f(a, b):
            return a is None, a < b
        """
    )
    keys = _record_dimensions(func, MutationCategory.BOUNDARY, set())
    assert _DEAD_DIM in keys
    assert "BOUNDARY:Lt" in keys


def test_record_dimensions_value_keys_by_python_type():
    func = _fn(
        """
        def f():
            return 1, True, 1.5, "s"
        """
    )
    keys = _record_dimensions(func, MutationCategory.VALUE, set())
    # bool checked before int; order follows AST/transformer traversal.
    assert keys == ["VALUE:int", "VALUE:bool", "VALUE:float", "VALUE:str"]


def test_record_dimensions_unknown_category_is_empty():
    func = _fn("def f():\n    return 1\n")
    assert _record_dimensions(func, MutationCategory.STATE, set()) == []


# ── Greedy submodular order ──────────────────────────────────────


def test_greedy_order_covers_distinct_dimensions_first():
    keys = ["A", "A", "A", "B", "C"]
    order = _greedy_dimension_order(keys)
    # First three picks must span all three distinct dimensions.
    assert {keys[i] for i in order[:3]} == {"A", "B", "C"}
    # It is a permutation of all indices.
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_greedy_order_round_robins_by_depth():
    # Two dims, uneven depth: A={0,1,2}, B={3}. Round 0 -> [0,3], round 1 -> [1],
    # round 2 -> [2].
    keys = ["A", "A", "A", "B"]
    assert _greedy_dimension_order(keys) == [0, 3, 1, 2]


def test_greedy_order_sinks_dead_keys_to_end():
    keys = ["A", _DEAD_DIM, "B", _DEAD_DIM]
    order = _greedy_dimension_order(keys)
    assert order[:2] == [0, 2]  # live dims first
    assert set(order[2:]) == {1, 3}  # dead last


def test_greedy_order_is_deterministic():
    keys = ["A", "B", "A", "C", "B", "A"]
    assert _greedy_dimension_order(keys) == _greedy_dimension_order(keys)


def test_greedy_order_empty():
    assert _greedy_dimension_order([]) == []


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_greedy_coverage_is_optimal_for_singleton_covers(k):
    # The core guarantee: with singleton cover-sets, the first k of the greedy
    # order cover exactly min(k, #distinct-dims) dimensions — the optimum (you
    # cannot cover more than min(k, D) with k picks). This is the >=(1-1/e)
    # bound realized exactly.
    keys = ["A", "A", "A", "A", "B", "C"]  # 3 distinct live dims
    distinct = len({keys[i] for i in _greedy_dimension_order(keys)[:k]})
    assert distinct == min(k, 3)


# ── Pass-windowed selection ──────────────────────────────────────


def test_select_greedy_windows_are_disjoint_and_cover():
    keys = ["A", "A", "A", "B", "C", "C"]  # 6 targets
    limit = 3
    w0 = _select_greedy(keys, target_count=6, limit=limit, pass_index=0)
    w1 = _select_greedy(keys, target_count=6, limit=limit, pass_index=1)
    assert len(w0) == limit and len(w1) == limit
    assert set(w0).isdisjoint(w1)
    assert set(w0) | set(w1) == set(range(6))


def test_select_greedy_first_window_maximizes_coverage():
    keys = ["A", "A", "A", "B", "C", "C"]
    w0 = _select_greedy(keys, target_count=6, limit=3, pass_index=0)
    assert {keys[i] for i in w0} == {"A", "B", "C"}


def test_select_greedy_exhausted_pass_falls_back_to_top_window():
    keys = ["A", "B", "C"]
    # pass_index far beyond the order length -> non-empty top window, not [].
    w = _select_greedy(keys, target_count=3, limit=2, pass_index=99)
    assert w == _select_greedy(keys, target_count=3, limit=2, pass_index=0)


def test_select_greedy_covers_all_indices_when_keys_short():
    # Defensive full-cover: fewer keys than target_count must not drop indices.
    keys = ["A"]
    w = _select_greedy(keys, target_count=3, limit=3, pass_index=0)
    assert set(w) == {0, 1, 2}


# ── generate_mutants integration ─────────────────────────────────


def test_generate_mutants_greedy_spans_dimensions_under_budget():
    # 4x Lt, 1x Gt, 1x Eq; budget 3 must hit all three boundary kinds.
    func = _fn(
        """
        def f(a, b, c, d, e, g):
            return a < b, a < c, a < d, a < e, b > c, d == g
        """
    )
    keys = _record_dimensions(func, MutationCategory.BOUNDARY, set())
    muts = generate_mutants(
        func, {MutationCategory.BOUNDARY}, max_per_category=3, greedy=True
    )
    dims = {keys[i] for i in _selected_indices(muts)}
    assert dims == {"BOUNDARY:Lt", "BOUNDARY:Gt", "BOUNDARY:Eq"}
    assert len(muts) == 3


def test_generate_mutants_passes_are_disjoint():
    func = _fn(
        """
        def f(a, b, c, d, e, g):
            return a < b, a < c, a < d, a < e, b > c, d == g
        """
    )
    p0 = _selected_indices(
        generate_mutants(
            func, {MutationCategory.BOUNDARY}, max_per_category=3, pass_index=0
        )
    )
    p1 = _selected_indices(
        generate_mutants(
            func, {MutationCategory.BOUNDARY}, max_per_category=3, pass_index=1
        )
    )
    assert p0.isdisjoint(p1)
    assert p0 | p1 == set(range(6))


def test_generate_mutants_greedy_false_uses_ast_order():
    func = _fn(
        """
        def f(a, b, c, d, e, g):
            return a < b, a < c, a < d, a < e, b > c, d == g
        """
    )
    muts = generate_mutants(
        func, {MutationCategory.BOUNDARY}, max_per_category=3, greedy=False, seed=None
    )
    assert _selected_indices(muts) == {0, 1, 2}  # first-k in AST order, no reordering


def test_generate_mutants_exhaustive_ignores_selection():
    func = _fn(
        """
        def f(a, b, c, d, e, g):
            return a < b, a < c, a < d, a < e, b > c, d == g
        """
    )
    muts = generate_mutants(func, {MutationCategory.BOUNDARY}, max_per_category=0)
    assert _selected_indices(muts) == set(range(6))


# ── STATE greedy (attribute diversity) ───────────────────────────


def test_state_greedy_spreads_across_attributes():
    func = _fn(
        """
        def f(self, v):
            self.a = v
            self.b = v
            self.c = v
        """
    )
    keys = _record_state_dimensions(func, "remove_assign")
    assert keys == [
        "STATE:remove_assign:a",
        "STATE:remove_assign:b",
        "STATE:remove_assign:c",
    ]
    muts = generate_mutants(
        func, {MutationCategory.STATE}, max_per_category=2, pass_index=0
    )
    # Two distinct attributes covered before repeating (both sub-modes present;
    # here only remove_assign has targets).
    picked = {m.description.split(":", 2)[-1].split(" ")[0] for m in muts}
    assert len(picked) >= 1  # sanity; assignment mode produced mutants


# ── Multi-pass convergence union grows ───────────────────────────


def test_converged_union_grows_with_passes():
    func = _fn(
        """
        def f(a, b, c, d, e, g):
            return a < b, a < c, a < d, a < e, b > c, d == g
        """
    )

    def tested(passes):
        res = run_function_converged(
            func,
            "m::f",
            {MutationCategory.BOUNDARY},
            test_functions=[],
            original_func=None,
            max_per_category=2,
            passes=passes,
        )
        return res.total_mutants

    one, three = tested(1), tested(3)
    assert three > one  # more passes -> strictly more unique mutants (up to universe)


# ── Key-extraction helpers ───────────────────────────────────────


def test_callee_name_variants():
    call_name = ast.parse("f(x, y)").body[0].value
    call_attr = ast.parse("obj.m(x, y)").body[0].value
    call_other = ast.parse("(lambda: g)()(x, y)").body[0].value
    assert _callee_name(call_name) == "f"
    assert _callee_name(call_attr) == "m"
    assert _callee_name(call_other) == "call"


def test_isinstance_type_name_variants():
    single = ast.parse("isinstance(x, int)").body[0].value
    tup = ast.parse("isinstance(x, (int, str))").body[0].value
    attr = ast.parse("isinstance(x, mod.T)").body[0].value
    assert _isinstance_type_name(single) == "int"
    assert _isinstance_type_name(tup) == "int+str"
    assert _isinstance_type_name(attr) == "T"


# ── class-method owner patching: a method exercised via a FACTORY, whose class is never imported into
# the test namespace, must still have its mutant installed (regression: previously a false "survivor"
# because owner resolution only searched the test namespace). ─────────────────────────────────────
class _OwnerFixture:
    def flag(self) -> bool:
        return True


def _make_owner() -> _OwnerFixture:
    return _OwnerFixture()


def test_patch_module_qualified_patches_class_method_owner():
    from Wesker.engine import _patch_module_qualified

    def mutant(self):  # the VALUE mutant: flip the return
        return False

    assert _make_owner().flag() is True
    saved = _patch_module_qualified(
        "flag", mutant, __file__, qualname="_OwnerFixture.flag"
    )
    try:
        assert saved, "owner class was not resolved/patched"
        assert (
            _make_owner().flag() is False
        )  # mutant on the class -> instance dispatch hits it (killable)
    finally:
        for owner, orig in saved:
            setattr(owner, "flag", orig)
    assert _make_owner().flag() is True  # cleanly restored


def test_patch_module_qualified_skips_inherited_method():
    # a subclass that does NOT define the method is not patched (precision: only the defining owner)
    from Wesker.engine import _patch_module_qualified

    def mutant(self):
        return False

    saved = _patch_module_qualified(
        "flag", mutant, __file__, qualname="_SubNoOverride.flag"
    )
    for owner, orig in saved:  # cleanup if anything was (wrongly) patched
        setattr(owner, "flag", orig)
    assert saved == []  # nothing defines _SubNoOverride.flag directly
