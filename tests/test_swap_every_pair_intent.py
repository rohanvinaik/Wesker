"""Policy 7 at the call site: every positional pair is a question, and the budget is disclosed.

THE GAP, measured 2026-09-11 through the real CLI. For ``f(a, b, c) = g(a, b, c)`` with the
hand-written test ``f(1, 2, 1) == 8``, Detective reported the suite complete and ``verify-rewrite``
returned PRESERVED for the rewrite ``g(c, b, a)`` — which returns 10 where the original returns 14
at (1, 2, 3). SWAP only ever asked about NEIGHBOURING pairs, so the first-and-third transposition
was never a question, and a suite that happened to repeat a value across positions 0 and 2 read
complete while that rewrite passed it.

These are INTENT tests for the surfaces the generated suites cannot reach on their own: argument
expressions that are not plain names (starred expansions, side-effecting calls), and the one case
where a "symmetric" builtin is not symmetric at all. A characterization would pin whatever the
engine does today; these say what it must do.
"""

from __future__ import annotations

import ast

from Wesker.engine import (
    MutationCategory,
    _count_targets,
    _record_dimensions,
    _SwapMutator,
    generate_mutants,
)
from Wesker.swap_plan import SWAP_PAIR_BUDGET


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _labels(src: str) -> list[str]:
    return _record_dimensions(_fn(src), MutationCategory.SWAP, set())


def _pairs_only(labels: list[str]) -> list[str]:
    return [lab for lab in labels if not lab.endswith(("~unwrap", "~dual"))]


def _exec_mutant(mutant, **globals_) -> dict:
    """Compile and exec a mutant's function AST, with the callees it references injected."""
    mod = ast.Module(body=[mutant.mutated_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = dict(globals_)
    # S102: test-local exec
    exec(compile(mod, "<mutant>", "exec"), ns)  # noqa: S102
    return ns


# ── starred expansions: the pairs are over AST entries, not runtime arguments ──


def test_a_starred_block_is_one_argument_expression():
    """``f(a, *rest)`` has TWO positional entries whatever `rest` holds at runtime, so it poses
    exactly one pair. The budget counts AST entries because that is what a transposition can move;
    the order WITHIN an expansion is a separate surface this policy does not claim."""
    assert _pairs_only(_labels("def f(a, rest):\n    return g(a, *rest)")) == ["SWAP:g"]


def test_multiple_starred_blocks_still_pair_by_position():
    """Two expansions are two entries — one pair — and three entries are three pairs, exactly as
    for plain names. Nothing about a star changes the arity of the question."""
    assert _pairs_only(_labels("def f(xs, ys):\n    return g(*xs, *ys)")) == ["SWAP:g"]
    assert _pairs_only(_labels("def f(a, xs, b):\n    return g(a, *xs, b)")) == [
        "SWAP:g",
        "SWAP:g~p1",
        "SWAP:g~p0,2",
    ]


def test_a_transposition_across_a_star_stays_valid_python():
    """The mutant must COMPILE — a swap that moved a star into a position the grammar forbids
    would read as a crash-only kill and pad the bucket rather than pin a behaviour."""
    muts = generate_mutants(
        _fn("def f(a, xs, b):\n    return g(a, *xs, b)"),
        {MutationCategory.SWAP},
        max_per_category=0,
    )
    rendered = [ast.unparse(m.mutated_node) for m in muts]
    assert any("g(b, *xs, a)" in r for r in rendered)
    for m in muts:  # every one compiles
        mod = ast.Module(body=[m.mutated_node], type_ignores=[])
        ast.fix_missing_locations(mod)
        compile(mod, "<mutant>", "exec")


def test_an_empty_expansion_is_still_a_question_at_the_ast():
    """`rest == []` at runtime does not make the pair vanish: eligibility is structural, decided
    before any value exists. An engine that asked only when the expansion was non-empty would be
    deciding eligibility from data it cannot see."""
    mut = next(
        m
        for m in generate_mutants(
            _fn("def f(a, rest):\n    return g(a, *rest)"),
            {MutationCategory.SWAP},
            max_per_category=0,
        )
        if m.dimension == "SWAP:g"
    )
    ns = _exec_mutant(mut, g=lambda *args: args)
    assert ns["f"](1, []) == (1,)  # g(*rest, a) with rest empty -> (1,)


# ── argument evaluation ORDER is a behaviour a transposition changes ──


def test_transposing_side_effecting_arguments_changes_evaluation_order():
    """Arguments are evaluated left to right, so transposing two CALLS reorders their effects even
    when the callee ignores its arguments. A suite that only checks the return value does not pin
    this — which is why the dimension has to exist for the suite to be asked about it."""
    mut = next(
        m
        for m in generate_mutants(
            _fn("def f(log):\n    return g(p(log), q(log))"),
            {MutationCategory.SWAP},
            max_per_category=0,
        )
        if m.dimension == "SWAP:g"
    )
    order: list[str] = []
    ns = _exec_mutant(
        mut,
        g=lambda *a: None,
        p=lambda log: log.append("p"),
        q=lambda log: log.append("q"),
    )
    ns["f"](order)
    assert order == ["q", "p"], (
        "the mutant must evaluate the transposed arguments in the new order"
    )


# ── min/max are NOT symmetric: ties break by POSITION, so the test must not use == ──


def test_min_and_max_break_ties_by_position_so_equality_cannot_see_it():
    """The reason there is no blanket skip for 'symmetric' builtins. ``max(1, 1.0)`` is ``1`` and
    ``max(1.0, 1)`` is ``1.0``: equal under ``==``, different objects of different types. A
    regression test written with ``==`` would pass against the transposed call and certify a
    behaviour change as preserved — so any such test checks TYPE or IDENTITY."""
    assert max(1, 1.0) == max(1.0, 1)  # the trap: equality sees nothing
    assert type(max(1, 1.0)) is int
    assert type(max(1.0, 1)) is float
    assert type(max(1, 1.0)) is not type(max(1.0, 1))


def test_a_transposed_max_is_distinguishable_by_type():
    """The same fact as a live mutant: the SWAP alternative on ``max(a, b)`` is killable, so it
    must not be excluded from the universe as 'obviously equivalent'."""
    mut = next(
        m
        for m in generate_mutants(
            _fn("def f(a, b):\n    return max(a, b)"),
            {MutationCategory.SWAP},
            max_per_category=0,
        )
        if m.dimension == "SWAP:max"
    )
    ns = _exec_mutant(mut)
    assert type(ns["f"](1, 1.0)) is float  # the original returns int at this input
    assert type(max(1, 1.0)) is int


# ── the budget binds what is ASKED, and only that ──


def test_the_budget_bounds_the_questions_asked_at_one_call_site():
    """Ten is a cap on the pairs ASKED, not a per-pass window — a per-pass limit would let a later
    pass spend more than the intended total."""
    wide = "def f(a, b, c, d, e, h):\n    return g(a, b, c, d, e, h)"
    assert len(_pairs_only(_labels(wide))) == SWAP_PAIR_BUDGET
    # `~unwrap` is a dimension but not a pair, so the target count is the budget plus it.
    assert _count_targets(_fn(wide), MutationCategory.SWAP) == SWAP_PAIR_BUDGET + 1


def test_the_count_and_the_generator_cannot_disagree():
    """`_count_targets` reads `_alternatives`, the same function generation walks. Issue #9 shipped
    a false COMPLETE when SWAP's counter answered a different question than its mutator."""
    for src in (
        "def f(a, b):\n    return g(a, b)",
        "def f(a, b, c):\n    return g(a, b, c)",
        "def f(a, b, c, d, e, h):\n    return g(a, b, c, d, e, h)",
        "def f(a, xs, b):\n    return g(a, *xs, b)",
    ):
        node = _fn(src)
        generated = generate_mutants(node, {MutationCategory.SWAP}, max_per_category=0)
        assert len(generated) == _count_targets(node, MutationCategory.SWAP), src


def test_every_emitted_pair_label_is_unique_at_a_call_site():
    """Two questions sharing a signifier would be one question in the report."""
    labels = _labels("def f(a, b, c, d, e, h):\n    return g(a, b, c, d, e, h)")
    assert len(set(labels)) == len(labels)


def test_neighbour_dimensions_keep_their_policy_6_position():
    """Compatibility, asserted rather than assumed: the labels policy 6 emitted come first and in
    its order, so a per-site prefix does not shift under the new questions."""
    labels = _labels("def f(a, b, c, d):\n    return g(a, b, c, d)")
    assert labels[:4] == ["SWAP:g", "SWAP:g~p1", "SWAP:g~p2", "SWAP:g~unwrap"]


def test_the_withheld_pairs_are_not_generated():
    """The budget must actually bind generation, not merely be reported — otherwise the census
    would describe a narrowing that never happened."""
    node = _fn("def f(a, b, c, d, e, h):\n    return g(a, b, c, d, e, h)")
    dims = {
        m.dimension
        for m in generate_mutants(node, {MutationCategory.SWAP}, max_per_category=0)
    }
    assert "SWAP:g~p0,3" in dims  # the tenth pair, inside the budget
    assert "SWAP:g~p0,5" not in dims  # the widest pair, withheld
    assert "SWAP:g~p1,5" not in dims


def test_alternatives_and_withheld_come_from_one_plan():
    """Generation and the census read the SAME `swap_plan` call, so a budget change moves both or
    neither — a second copy of the selection rule is the defect class this project names."""
    node = next(
        n
        for n in ast.walk(
            _fn("def f(a, b, c, d, e, h):\n    return g(a, b, c, d, e, h)")
        )
        if isinstance(n, ast.Call)
    )
    asked = [
        s
        for s, _lab in _SwapMutator._alternatives(node, set(), {})
        if isinstance(s, tuple)
    ]
    assert len(asked) + _SwapMutator._withheld(node) == 15  # C(6, 2)
