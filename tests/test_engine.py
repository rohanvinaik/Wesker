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
    _ValueMutator,
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
    """The positional target indices selected — now a first-class ``target_index`` field
    (``mutant_id`` is content-addressed, so the ordinal no longer lives in the id string)."""
    return {m.target_index for m in mutants}


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
    # Full ROR. An ordering carries FIVE dimensions: boundary shift, direction
    # reversal, equality collapse (does the suite pin a RANGE or only a point?), and
    # the two predicate constants (does the branch matter at all?). Equality has no
    # ordering to reverse and no range to collapse, so it carries the flip + constants.
    # Each is a distinct question, so each must be its own greedy dimension: collapsing
    # them would let one kill mark the operator settled while another question goes
    # unasked.
    assert keys == [
        "BOUNDARY:Lt",
        "BOUNDARY:Lt~dir",
        "BOUNDARY:Lt~eq",
        "BOUNDARY:Lt~true",
        "BOUNDARY:Lt~false",
        "BOUNDARY:Gt",
        "BOUNDARY:Gt~dir",
        "BOUNDARY:Gt~eq",
        "BOUNDARY:Gt~true",
        "BOUNDARY:Gt~false",
        "BOUNDARY:Eq",
        "BOUNDARY:Eq~true",
        "BOUNDARY:Eq~false",
    ]


def test_record_dimensions_identity_membership_are_live_boundary_dims():
    # `is` / `in` are now mutable comparison ops (predicate flip) → live BOUNDARY
    # dims, one per op. Previously these produced a dead key; that blind spot is closed.
    func = _fn(
        """
        def f(a, b, xs):
            return a is None, a in xs, a < b
        """
    )
    keys = _record_dimensions(func, MutationCategory.BOUNDARY, set())
    assert "BOUNDARY:Is" in keys
    assert "BOUNDARY:In" in keys
    assert "BOUNDARY:Lt" in keys
    assert _DEAD_DIM not in keys  # every comparison op is now swappable


def test_record_dimensions_value_keys_by_python_type():
    func = _fn(
        """
        def f():
            return 1, True, 1.5, "s"
        """
    )
    keys = _record_dimensions(func, MutationCategory.VALUE, set())
    # bool checked before int; order follows AST/transformer traversal. Ints
    # carry the collapse and the off-by-one as adjacent dimensions; floats the
    # collapse and the two hail-mary perturbations.
    assert keys == [
        "VALUE:int",
        "VALUE:int~off1",
        "VALUE:bool",
        "VALUE:float",
        "VALUE:float~pert+1",
        "VALUE:float~pert-1",
        "VALUE:float~pert+01",
        "VALUE:float~pert-01",
        "VALUE:str",
    ]


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
    """Pins the ORDER, not merely that two calls agree.

    This asserted ``f(keys) == f(keys)``, which can only fail if the function reaches for
    randomness or global state — it cannot see a changed order, because both sides change
    together. The claim being made is stronger than that ("Fully deterministic" means a
    SPECIFIC order, stable across runs and processes), so the expected value is written
    down.

    And the value is the round-robin itself: one index per distinct dimension (A B C)
    before any dimension repeats, then the second of each, then the tail. That ordering is
    what makes ``m = D`` cover every dimension exactly once — the 1.00-mutants-per-dimension
    result rests on this list being exactly this.
    """
    keys = ["A", "B", "A", "C", "B", "A"]
    order = _greedy_dimension_order(keys)
    assert order == [0, 1, 3, 2, 4, 5]
    assert [keys[i] for i in order] == ["A", "B", "C", "A", "B", "A"]


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
    # 4x Lt, 1x Gt, 1x Eq → dimension KINDS {Lt, Lt~dir, Gt, Gt~dir, Eq}.
    # Greedy submodular coverage ((1-1/e)-optimal) must spend a budget of 3 on 3
    # DISTINCT dimension kinds — never clustering redundant instances of one kind.
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
    assert len(muts) == 3
    assert len(dims) == 3  # 3 distinct kinds — maximal coverage under the budget


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
    # Two consecutive budget-windows of the greedy order: disjoint, each of size 3.
    assert len(p0) == 3 and len(p1) == 3
    assert len(p0 | p1) == 6


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
    # Exhaustive covers every target dimension, whatever the count.
    n = _count_targets(func, MutationCategory.BOUNDARY)
    assert _selected_indices(muts) == set(range(n))


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


def test_converged_stops_immediately_on_an_uncontained_worker(monkeypatch):
    """#14 (reopened): an uncontained worker (abandon could not stop it) is STILL ALIVE. The converged
    path used to check containment only AFTER finishing all passes, so a first-mutant runaway kept
    spawning workers alongside it. It must halt on the first uncontained result — keeping that result,
    but evaluating no more — and report the run non-gateable / depth="cut"."""
    from Wesker import engine as E

    func = _fn(
        """
        def f(a, b, c, d, e, g):
            return a < b, a < c, a < d, a < e, b > c, d == g
        """
    )
    calls = {"n": 0}

    def fake_evaluate(mutant, *a, **k):
        calls["n"] += 1
        return E.MutantResult(
            mutant=mutant, killed=False, contained=False, elapsed_ms=1.0
        )

    monkeypatch.setattr(E, "evaluate_mutant", fake_evaluate)
    monkeypatch.setattr(E, "check_equivalent", lambda *a, **k: False)

    res = run_function_converged(
        func,
        "m::f",
        {MutationCategory.BOUNDARY},
        test_functions=[],
        original_func=None,
        max_per_category=4,
        passes=3,
    )
    assert (
        calls["n"] == 1
    )  # halted after the first uncontained result — did not keep spawning
    assert res.coverage_depth == "cut"  # the measurement is invalidated
    assert res.is_gateable is False  # and honestly non-gateable


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


def test_profiling_derives_source_path_and_kills_factory_method():
    # End-to-end regression: run_function_profiling passes no source_path, so evaluate_mutant must
    # derive it from original_func — else the module-qualified/class-owner patch is inert and a method
    # reached only via a factory (owner class not in the test namespace) is a false survivor.
    from Wesker.engine import run_function_profiling

    with open(__file__) as f:
        tree = ast.parse(f.read())
    node = next(
        m
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "_OwnerFixture"
        for m in cls.body
        if isinstance(m, ast.FunctionDef) and m.name == "flag"
    )

    def _test_via_factory():
        assert _make_owner().flag() is True

    res = run_function_profiling(
        node,
        f"{__file__}::_OwnerFixture.flag",
        {MutationCategory.VALUE},
        [_test_via_factory],
        _OwnerFixture.flag,
    )
    assert res.total_mutants >= 1
    assert (
        res.total_survived == 0
    )  # source_path derived -> class owner patched -> mutant killed


def test_co_filename_matches_absolute_and_relative():
    from Wesker.engine import _co_filename_matches

    assert _co_filename_matches(
        "/a/b/regen/roleframes.py", "/a/b/regen/roleframes.py"
    )  # abspath equal
    assert _co_filename_matches(
        "/a/b/regen/roleframes.py", "regen/roleframes.py"
    )  # relative suffix
    assert not _co_filename_matches(
        "/a/b/xregen/roleframes.py", "regen/roleframes.py"
    )  # segment boundary
    assert not _co_filename_matches("/a/b/other.py", "regen/roleframes.py")  # unrelated
    assert not _co_filename_matches(None, "x") and not _co_filename_matches("x", None)


def test_profiling_uses_func_key_source_path_with_stubbed_original():
    # The real LintGate scenario: original_func is a STUB (its co_filename is useless) and func_key
    # carries a project-RELATIVE source path. source_path must come from func_key + a relative
    # co_filename match, so the class-owner patch still fires and the mutant is killed.
    import os

    from Wesker.engine import run_function_profiling

    with open(__file__) as f:
        tree = ast.parse(f.read())
    node = next(
        m
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "_OwnerFixture"
        for m in cls.body
        if isinstance(m, ast.FunctionDef) and m.name == "flag"
    )

    def _test_via_factory():
        assert _make_owner().flag() is True

    rel = os.path.relpath(__file__)
    res = run_function_profiling(
        node,
        f"{rel}::_OwnerFixture.flag",
        {MutationCategory.VALUE},
        [_test_via_factory],
        lambda *_a: None,  # STUB original_func
    )
    assert res.total_mutants >= 1
    assert res.total_survived == 0


# ── VALUE off-by-one and SWAP unwrap/adjacent-pair dimensions ────


def _exec_mutant(mutant) -> dict:
    """Compile and exec a mutant's function AST; returns the namespace."""
    mod = ast.Module(body=[mutant.mutated_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {}
    exec(compile(mod, "<mutant>", "exec"), ns)  # noqa: S102 — test-local exec
    return ns


def test_value_off_by_one_dodges_collapse():
    """The two int dimensions never share a mutant, whatever the value."""
    assert _ValueMutator._alternatives(2) == [(0, "VALUE:int"), (3, "VALUE:int~off1")]
    assert _ValueMutator._alternatives(0) == [(1, "VALUE:int"), (-1, "VALUE:int~off1")]
    assert _ValueMutator._alternatives(-1) == [(0, "VALUE:int"), (-2, "VALUE:int~off1")]


def test_value_off_by_one_reaches_round_ndigits():
    """round(x, 2) -> round(x, 3): the near-miss the 0/1 collapse cannot express.

    A 1-decimal golden value kills the collapse (ndigits 0) while never
    distinguishing 2 from 3 — this dimension is what forces that witness.
    """
    func = _fn("def f(x):\n    return round(x, 2)")
    muts = generate_mutants(func, {MutationCategory.VALUE}, max_per_category=0)
    off1 = next(m for m in muts if m.dimension == "VALUE:int~off1")
    f = _exec_mutant(off1)["f"]
    assert f(1.9375) == 1.938  # original returns 1.94


def test_swap_unwrap_replaces_call_with_first_arg():
    func = _fn("def f(x):\n    return round(x, 2)")
    muts = generate_mutants(func, {MutationCategory.SWAP}, max_per_category=0)
    unwrap = next(m for m in muts if m.dimension == "SWAP:round~unwrap")
    f = _exec_mutant(unwrap)["f"]
    assert f(1.2345) == 1.2345  # the call is gone; nothing rounds


def test_swap_unwrap_skips_statement_position_calls():
    """A discarded call is SDL's question; unwrap there is guaranteed-equivalent."""
    func = _fn("def f(items, y):\n    items.append(y)\n    return len(items)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert not any("append" in k and k.endswith("~unwrap") for k in keys)
    assert "SWAP:len~unwrap" in keys  # value-position call keeps the dimension


def test_swap_unwrap_skips_starred_first_arg():
    func = _fn("def f(xs):\n    return max(*xs)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert not any(k.endswith("~unwrap") for k in keys)


def test_swap_adjacent_pairs_one_dimension_each():
    func = _fn("def f(a, b, c):\n    return g(a, b, c)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert keys == ["SWAP:g", "SWAP:g~p1", "SWAP:g~unwrap"]
    assert _count_targets(func, MutationCategory.SWAP) == 3


def test_swap_second_pair_transposes_later_neighbors():
    func = _fn('def f(a, b, c):\n    return "{}{}{}".format(a, b, c)')
    muts = generate_mutants(func, {MutationCategory.SWAP}, max_per_category=0)
    p1 = next(m for m in muts if m.dimension.endswith("~p1"))
    f = _exec_mutant(p1)["f"]
    assert f(1, 2, 3) == "132"  # (b, c) transposed; (a, b) untouched


# ── STATE loop_flow: break ↔ continue ────────────────────────────


def test_state_loop_flow_swaps_break_to_continue():
    func = _fn(
        """
        def f(xs):
            total = 0
            for x in xs:
                if x < 0:
                    break
                total += x
            return total
        """
    )
    muts = generate_mutants(func, {MutationCategory.STATE}, max_per_category=0)
    flow = next(m for m in muts if m.dimension == "STATE:loop_flow:break")
    g = _exec_mutant(flow)["f"]
    # Original stops at the first negative (returns 3); the mutant skips
    # negatives and keeps summing.
    assert g([1, 2, -1, 4]) == 7


def test_state_loop_flow_swaps_continue_to_break():
    func = _fn(
        """
        def f(xs):
            total = 0
            for x in xs:
                if x < 0:
                    continue
                total += x
            return total
        """
    )
    muts = generate_mutants(func, {MutationCategory.STATE}, max_per_category=0)
    flow = next(m for m in muts if m.dimension == "STATE:loop_flow:continue")
    g = _exec_mutant(flow)["f"]
    # Original skips the negative (returns 7); the mutant stops there.
    assert g([1, 2, -1, 4]) == 3


def test_state_loop_flow_count_aligns_with_record():
    func = _fn(
        """
        def f(xs):
            for x in xs:
                if x:
                    break
                continue
        """
    )
    keys = _record_state_dimensions(func, "loop_flow")
    assert keys == ["STATE:loop_flow:break", "STATE:loop_flow:continue"]
    assert _count_targets(func, MutationCategory.STATE) == 2  # no assigns/returns


# ── SWAP callee-dual dimension (issue #5): min↔max, any↔all, floor↔ceil ────


def test_swap_dual_min_becomes_max():
    func = _fn("def f(a, b):\n    return min(a, b)")
    muts = generate_mutants(func, {MutationCategory.SWAP}, max_per_category=0)
    dual = next(m for m in muts if m.dimension == "SWAP:min~dual")
    f = _exec_mutant(dual)["f"]
    assert f(1, 9) == 9  # the fold direction flipped; original returns 1


def test_swap_dual_any_becomes_all():
    """The wrong-fold-direction class no transposition can express: any(xs) and
    all(xs) take identical arguments in every order."""
    func = _fn("def f(xs):\n    return any(xs)")
    muts = generate_mutants(func, {MutationCategory.SWAP}, max_per_category=0)
    dual = next(m for m in muts if m.dimension == "SWAP:any~dual")
    f = _exec_mutant(dual)["f"]
    assert (
        f([True, False]) is False
    )  # some-but-not-all: the one input class that distinguishes


def test_swap_dual_attribute_callee_keeps_its_module():
    func = _fn("def f(x):\n    import math\n    return math.floor(x)")
    muts = generate_mutants(func, {MutationCategory.SWAP}, max_per_category=0)
    dual = next(m for m in muts if m.dimension == "SWAP:floor~dual")
    f = _exec_mutant(dual)["f"]
    assert f(1.2) == 2  # math.ceil — the module survived, only the attr flipped


def test_swap_dual_absent_for_uncurated_callees():
    """Every entry inflates every universe that calls it — the table is closed."""
    func = _fn("def f(xs):\n    return sorted(xs)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert not any(k.endswith("~dual") for k in keys)


def test_swap_dual_counts_align_with_generation():
    """The single-source-of-truth guarantee: estimate and generation cannot drift."""
    func = _fn("def f(a, b):\n    return min(a, b) + max(a, 0)")
    from Wesker.engine import estimate_universe_size

    est = estimate_universe_size(func, {MutationCategory.SWAP})
    muts = generate_mutants(func, {MutationCategory.SWAP}, max_per_category=0)
    assert len(muts) == est
    assert sum(1 for m in muts if m.dimension.endswith("~dual")) == 2


def test_swap_dual_skips_arbitrary_attribute_callees():
    """`obj.floor(...)` is any object's method — a dual there asserts a duality the
    code never promised. Only the literal `math.floor`/`math.ceil` qualifies."""
    func = _fn("def f(obj, x):\n    return obj.floor(x)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert not any(k.endswith("~dual") for k in keys)


def test_swap_dual_skips_shadowed_builtins():
    """A parameter or local named `min` is the user's object, not the builtin."""
    param = _fn("def f(min, a, b):\n    return min(a, b)")
    assert not any(
        k.endswith("~dual")
        for k in _record_dimensions(param, MutationCategory.SWAP, set())
    )
    local = _fn("def f(a, b):\n    max = a\n    return max(a, b)")
    assert not any(
        k.endswith("~dual")
        for k in _record_dimensions(local, MutationCategory.SWAP, set())
    )


def test_swap_dual_local_import_of_math_still_qualifies():
    """`import math` binds the RESOLVED module — an import is not shadowing."""
    func = _fn("def f(x):\n    import math\n    return math.ceil(x)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert "SWAP:ceil~dual" in keys


def test_swap_dual_foreign_import_is_not_the_builtin():
    """`from custom import min` has no established relationship to builtin max."""
    func = _fn("def f(a, b):\n    from custom import min\n    return min(a, b)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert not any(k.endswith("~dual") for k in keys)


def test_swap_dual_numpy_as_math_is_not_math():
    """The literal spelling `math.floor` resolves to NumPy here — no dual."""
    func = _fn("def f(x):\n    import numpy as math\n    return math.floor(x)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert not any(k.endswith("~dual") for k in keys)


def test_swap_dual_nested_scope_param_does_not_shadow_outer():
    """g's parameter `min` binds g's frame; the outer builtin keeps its dual."""
    func = _fn(
        "def f(a, b):\n    def g(min):\n        return min\n    return min(a, b)"
    )
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert "SWAP:min~dual" in keys


def test_swap_dual_math_alias_is_resolved():
    """`import math as m; m.floor` IS math.floor — provenance, not spelling."""
    func = _fn("def f(x):\n    import math as m\n    return m.floor(x)")
    muts = generate_mutants(func, {MutationCategory.SWAP}, max_per_category=0)
    dual = next(m for m in muts if m.dimension == "SWAP:floor~dual")
    f = _exec_mutant(dual)["f"]
    assert f(1.2) == 2  # m.ceil


def test_swap_dual_from_math_import_floor_is_resolved():
    func = _fn("def f(x):\n    from math import floor\n    return floor(x)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert "SWAP:floor~dual" in keys


def test_swap_dual_bare_floor_without_import_abstains():
    """`floor` is not a builtin; with no visible import it is unresolvable."""
    func = _fn("def f(x):\n    return floor(x)")
    keys = _record_dimensions(func, MutationCategory.SWAP, set())
    assert not any(k.endswith("~dual") for k in keys)


def test_value_float_perturbations_join_the_collapse():
    """Floats carry the collapse plus all four hail-mary perturbations."""
    assert _ValueMutator._alternatives(150.0) == [
        (0.0, "VALUE:float"),
        (151.0, "VALUE:float~pert+1"),
        (149.0, "VALUE:float~pert-1"),
        (150.1, "VALUE:float~pert+01"),
        (149.9, "VALUE:float~pert-01"),
    ]
    # v=0.0: +1.0 lands on the collapse (1.0) and drops out; the rest survive
    assert _ValueMutator._alternatives(0.0) == [
        (1.0, "VALUE:float"),
        (-1.0, "VALUE:float~pert-1"),
        (0.1, "VALUE:float~pert+01"),
        (-0.1, "VALUE:float~pert-01"),
    ]
    # v=-1.0: +1.0 lands on the collapse (0.0) and drops out
    assert _ValueMutator._alternatives(-1.0) == [
        (0.0, "VALUE:float"),
        (-2.0, "VALUE:float~pert-1"),
        (-0.9, "VALUE:float~pert+01"),
        (-1.1, "VALUE:float~pert-01"),
    ]


def test_value_float_perturbation_drops_out_at_magnitude_and_nan():
    """A delta swallowed by float magnitude (both directions) is skipped, never
    emitted as a mutant equal to the original; NaN never perturbs at all."""
    # at 1e20 the float spacing exceeds every delta in every direction:
    # each perturbation would equal the original, so only the collapse remains
    big = _ValueMutator._alternatives(1e20)
    assert [lab for _, lab in big] == ["VALUE:float"]
    nan = _ValueMutator._alternatives(float("nan"))
    assert [lab for _, lab in nan] == ["VALUE:float"]


def test_value_float_perturbation_shifts_a_threshold():
    """150.0 -> 151.0 on a comparison: any input in the shifted interval kills,
    where BOUNDARY's endpoint shift would need the exact edge."""
    func = _fn("def free(v):\n    return v >= 150.0")
    muts = generate_mutants(func, {MutationCategory.VALUE}, max_per_category=0)
    up = next(m for m in muts if m.dimension == "VALUE:float~pert+1")
    f_up = _exec_mutant(up)["free"]
    assert f_up(150.5) is False  # original returns True — in-interval input kills
    # the down-mutant is the one a 150.0 -> 149.0 hand-bug hides behind
    down = next(m for m in muts if m.dimension == "VALUE:float~pert-1")
    f_down = _exec_mutant(down)["free"]
    assert f_down(149.5) is True  # original returns False
