"""DATAFLOW (issue #10): reference substitution enters the universe.

The anchor is the issue's own minimal witness: ``pick(x, y): return x`` with a
suite that only ever tests ``x == y`` kills every pre-DATAFLOW mutant while
never distinguishing ``return x`` from ``return y`` — a wrong-reference fault,
the signature fault family of extraction refactors, invisible to a universe
without reference identity. These tests pin the witness and every conservatism
rule the candidate analysis promises, each in the direction that WITHHOLDS a
dimension rather than fabricating one.
"""

from __future__ import annotations

import ast
import textwrap

from Wesker.engine import (
    MutationCategory,
    estimate_universe_size,
    generate_mutants,
)
from Wesker.filter import filter_categories


def _fn(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    node = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node


def _dataflow_mutants(src: str):
    return generate_mutants(_fn(src), {MutationCategory.DATAFLOW}, max_per_category=0)


def test_the_pick_witness_is_in_the_universe():
    """Issue #10's minimal witness, end to end: the return_sub mutant exists,
    agrees with the original exactly where the under-testing suite looked
    (x == y), and differs where it did not — so a suite at that blind spot can
    no longer be called complete."""
    src = "def pick(x, y):\n    return x\n"
    fn = _fn(src)
    assert estimate_universe_size(fn, {MutationCategory.DATAFLOW}) == 1
    assert MutationCategory.DATAFLOW in filter_categories(fn)

    (mutant,) = _dataflow_mutants(src)
    assert mutant.dimension == "DATAFLOW:return:x→y"

    ns_orig: dict = {}
    ns_mut: dict = {}
    exec(compile(ast.Module([mutant.original_node], []), "<o>", "exec"), ns_orig)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([mutant.mutated_node], [])),
            "<m>",
            "exec",
        ),
        ns_mut,
    )
    assert ns_orig["pick"](7, 7) == ns_mut["pick"](7, 7)  # the suite's blind spot
    assert ns_orig["pick"](1, 2) != ns_mut["pick"](1, 2)  # the behavior it missed


def test_return_sub_owns_the_returned_name():
    # `return total` is one return_sub site per candidate — never doubled as
    # a name_sub site.
    mutants = _dataflow_mutants(
        """
        def f(a, b):
            total = a + b
            return total
        """
    )
    dims = {m.dimension for m in mutants}
    assert "DATAFLOW:return:total→a" in dims
    assert "DATAFLOW:return:total→b" in dims
    assert "DATAFLOW:total→a" not in dims
    assert "DATAFLOW:total→b" not in dims


def test_candidates_are_bound_before_under_the_narrow_rule():
    # `b` binds inside an inner block; the narrow rule (parameter, or earlier
    # statement in the SAME block) keeps it out of every candidate set, so no
    # substitution can manufacture an UnboundLocalError.
    mutants = _dataflow_mutants(
        """
        def f(a):
            if a:
                b = 1
            c = 2
            return c
        """
    )
    assert {m.dimension for m in mutants} == {"DATAFLOW:return:c→a"}


def test_nested_frames_are_skipped_whole():
    # Neither inner's parameter nor its body loads leak into the outer
    # function's sites or candidates.
    mutants = _dataflow_mutants(
        """
        def f(a, b):
            def inner(z):
                return z
            return a
        """
    )
    assert {m.dimension for m in mutants} == {"DATAFLOW:return:a→b"}


def test_name_sub_is_implemented_but_not_enrolled():
    """The un-enrolled slice stays record-countable (the next enrollment is a
    candidate restraint plus one tuple entry), and its rules hold: the callee
    position belongs to SWAP, the argument load is a site."""
    from Wesker.engine import _DATAFLOW_SUB_MODES, _record_dataflow_dimensions

    assert [mode for mode, _desc in _DATAFLOW_SUB_MODES] == ["return_sub"]
    fn = _fn(
        """
        def f(g, x):
            return g(x)
        """
    )
    assert _record_dataflow_dimensions(fn, "name_sub") == ["DATAFLOW:x→g"]
    # And because name_sub is not enrolled, the call expression contributes
    # nothing to the generated universe (the return value is a Call, not a
    # Name, so return_sub abstains too).
    assert _dataflow_mutants("def f(g, x):\n    return g(x)\n") == []


def test_receiver_names_never_flow():
    # A method's self is neither a site nor a candidate; with only one other
    # pool name there is nothing to substitute at all.
    src = """
        class C:
            def m(self, x):
                self.slot = x
                return x
        """
    node = ast.parse(textwrap.dedent(src)).body[0].body[0]
    assert isinstance(node, ast.FunctionDef)
    assert estimate_universe_size(node, {MutationCategory.DATAFLOW}) == 0


def test_dimension_collapses_across_sites_like_swap():
    # Two `return a` statements ask ONE behavioral question ("is a
    # distinguished from b?") — exhaustive mode visits both sites, DOF mode
    # spends exactly one mutant on the dimension, the same collapse SWAP
    # applies per callee.
    src = """
        def f(a, b):
            if b:
                return a
            return a
        """
    fn = _fn(src)
    exhaustive = generate_mutants(fn, {MutationCategory.DATAFLOW}, max_per_category=0)
    a_to_b = [m for m in exhaustive if m.dimension == "DATAFLOW:return:a→b"]
    assert len(a_to_b) == 2
    dof = generate_mutants(fn, {MutationCategory.DATAFLOW}, max_per_category=None)
    assert sum(1 for m in dof if m.dimension == "DATAFLOW:return:a→b") == 1


def test_candidate_analysis_pins_every_binding_and_block_shape():
    """One rich probe, exact record output — pins the branches the simpler
    tests cannot reach (found as killable witnesses by converge on
    `_dataflow_candidates` itself): AnnAssign binds only WITH a value,
    AugAssign targets bind, tuple/starred targets never join the pool,
    keyword arguments flow, and all four Try blocks are walked. The exact
    lists also pin candidate ORDER (signature, then binding order)."""
    from Wesker.engine import _record_dataflow_dimensions

    fn = _fn(
        """
        def probe(a, b=0, *args, kw=None, **kwargs):
            c: int = a
            d: int
            c += b
            e, f = a, b
            g = h(a, key=b)
            try:
                i = c + 1
            except ValueError:
                j = c
            else:
                k = c
            finally:
                m = c
            return g
        """
    )
    # *args/**kwargs never in the pool; d (valueless AnnAssign) binds nothing;
    # e, f (tuple target) bind nothing; candidates are bound-before per site.
    assert _record_dataflow_dimensions(fn, "return_sub") == [
        "DATAFLOW:return:g→a",
        "DATAFLOW:return:g→b",
        "DATAFLOW:return:g→kw",
        "DATAFLOW:return:g→c",
    ]
    assert _record_dataflow_dimensions(fn, "name_sub") == [
        # c: int = a — c not yet bound
        "DATAFLOW:a→b",
        "DATAFLOW:a→kw",
        # c += b — c now visible
        "DATAFLOW:b→a",
        "DATAFLOW:b→kw",
        "DATAFLOW:b→c",
        # e, f = a, b
        "DATAFLOW:a→b",
        "DATAFLOW:a→kw",
        "DATAFLOW:a→c",
        "DATAFLOW:b→a",
        "DATAFLOW:b→kw",
        "DATAFLOW:b→c",
        # g = h(a, key=b) — the keyword value flows, the callee does not
        "DATAFLOW:a→b",
        "DATAFLOW:a→kw",
        "DATAFLOW:a→c",
        "DATAFLOW:b→a",
        "DATAFLOW:b→kw",
        "DATAFLOW:b→c",
        # try body / handler / orelse / finalbody each load c, g visible
        "DATAFLOW:c→a",
        "DATAFLOW:c→b",
        "DATAFLOW:c→kw",
        "DATAFLOW:c→g",
        "DATAFLOW:c→a",
        "DATAFLOW:c→b",
        "DATAFLOW:c→kw",
        "DATAFLOW:c→g",
        "DATAFLOW:c→a",
        "DATAFLOW:c→b",
        "DATAFLOW:c→kw",
        "DATAFLOW:c→g",
        "DATAFLOW:c→a",
        "DATAFLOW:c→b",
        "DATAFLOW:c→kw",
        "DATAFLOW:c→g",
    ]


def test_candidate_map_shape_direct():
    """Direct-call value pins for the last converge witnesses: the analysis
    returns THE map (not the tree, not None); a valueless AnnAssign binds
    nothing; and a scope-boundary node is SKIPPED, never allowed to stop the
    scan of its whole expression (continue, not break — the trailing lambda
    would otherwise eat the loads popped after it)."""
    from Wesker.engine import _dataflow_candidates

    fn = _fn(
        """
        def probe2(x, y):
            d: int
            g = h(x, lambda: 0)
            if g:
                return
            return g + x
        """
    )
    sites = _dataflow_candidates(fn)
    assert isinstance(sites, dict)
    assert sorted(sites.values()) == [
        ("name_sub", "g", ("x", "y")),  # the `if g:` test
        ("name_sub", "g", ("x", "y")),  # the `return g + x` load
        ("name_sub", "x", ("y",)),  # the call argument, before g binds
        ("name_sub", "x", ("y", "g")),  # the return-expression load
    ]


def test_non_function_root_returns_the_empty_map():
    # The guard's contract: anything that is not a function yields exactly the
    # empty map — never the node handed in, never None.
    from Wesker.engine import _dataflow_candidates

    stmt = ast.parse("x = 1").body[0]
    assert _dataflow_candidates(stmt) == {}


def test_augassign_binds_and_non_name_targets_bind_nothing():
    """An AugAssign can be a name's first SAME-BLOCK pool entry (its inner-if
    binding never escaped the narrow rule) — t must be a candidate afterwards.
    And AnnAssign/AugAssign with non-Name targets (self.x, xs[0]) bind
    nothing: the branch that would read ``.id`` off an Attribute or Subscript
    must not be reachable for them."""
    from Wesker.engine import _record_dataflow_dimensions

    fn = _fn(
        """
        def probe3(self, a, xs):
            if a:
                t = 1
            t += a
            self.x: int = a
            xs[0] += a
            return a
        """
    )
    assert _record_dataflow_dimensions(fn, "return_sub") == [
        "DATAFLOW:return:a→xs",
        "DATAFLOW:return:a→t",
    ]
    # And the candidate container is a TUPLE by contract — hashable, reusable,
    # equality-comparable — never a one-shot generator.
    from Wesker.engine import _dataflow_candidates

    assert ("return_sub", "a", ("xs", "t")) in _dataflow_candidates(fn).values()


def test_match_case_bodies_are_declared_unwalked():
    # match/case blocks are not DATAFLOW sites yet — a withheld dimension,
    # declared here and in the policy exclusions, never a silently-registered
    # one (the walker's block fields are body/orelse/finalbody/handlers).
    from Wesker.engine import _record_dataflow_dimensions

    fn = _fn(
        """
        def probe4(a, b):
            match a:
                case 0:
                    u = a + b
            return a
        """
    )
    # The match SUBJECT is an ordinary load in the statement and registers;
    # the case body's `u = a + b` loads do not.
    assert _record_dataflow_dimensions(fn, "name_sub") == ["DATAFLOW:a→b"]
    assert _record_dataflow_dimensions(fn, "return_sub") == ["DATAFLOW:return:a→b"]


def test_generation_is_deterministic():
    src = """
        def f(a, b, c):
            total = a + b
            total = total + c
            return total
        """
    first = [(m.mutant_id, m.dimension) for m in _dataflow_mutants(src)]
    second = [(m.mutant_id, m.dimension) for m in _dataflow_mutants(src)]
    assert first == second
    assert len(first) == len(set(first))
