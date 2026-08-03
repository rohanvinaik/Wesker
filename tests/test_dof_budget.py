"""DOF-derived budget: dimension_budget / dof_universe.

Hand-written, deliberately. `detective converge Wesker/engine.py::dimension_budget`
names these degrees of freedom correctly but cannot synthesize them: its `--input`
is literal-only (`ast.literal_eval`) and this function's arguments are an
`ast.FunctionDef` and a `MutationCategory` enum. Converge therefore stops at 1/5
killed and asks for an input it has no way to express, so the STATE-branch DOF are
pinned here instead.
"""

import ast

import pytest

from Wesker.engine import (
    MutationCategory,
    dimension_budget,
    dof_universe,
)


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


# ── the STATE branch: two sub-modes, summed ──────────────────────────────


def test_state_budget_sums_both_sub_modes():
    """STATE DOF = distinct assigned fields + the return_none dimension.

    Pins the sub-mode tuple: dropping either ``remove_assign`` or ``return_none``
    changes the count, and so does swapping (func_node, mode).
    """
    node = _fn("def f(self):\n    self.a = 1\n    self.b = 2\n    return self.a")
    # 2 distinct assign dims (a, b) + 1 return_none dim
    assert dimension_budget(node, MutationCategory.STATE) == 3


def test_state_budget_counts_distinct_fields_not_assignments():
    """Two writes to the SAME field are one dimension, not two."""
    node = _fn("def f(self):\n    self.a = 1\n    self.a = 2")
    assert dimension_budget(node, MutationCategory.STATE) == 1


def test_state_budget_return_only():
    """A return with no state writes has exactly the return_none dimension."""
    node = _fn("def f(self):\n    return 1")
    assert dimension_budget(node, MutationCategory.STATE) == 1


def test_state_budget_assign_only():
    """State writes with no return contribute only assign dimensions."""
    node = _fn("def f(self):\n    self.a = 1\n    self.b = 2")
    assert dimension_budget(node, MutationCategory.STATE) == 2


# ── the non-STATE branch: argument order into _record_dimensions ──────────


def test_value_budget_counts_distinct_constant_types():
    """VALUE DOF is the distinct dimension KEYS, not the constant count.

    Three constants of two types = 3 dimensions: ints carry a collapse AND an
    off-by-one key (VALUE:int, VALUE:int~off1), strings one. Also pins the
    argument order of the _record_dimensions call: swapping (func_node, category)
    resolves no mutator factory and collapses the count to 0.
    """
    node = _fn("def g():\n    x = 1\n    y = 'a'\n    z = 2\n    return x, y, z")
    assert dimension_budget(node, MutationCategory.VALUE) == 3


def test_arithmetic_budget_counts_distinct_operators():
    node = _fn("def h(a, b):\n    return a + b - a")
    assert dimension_budget(node, MutationCategory.ARITHMETIC) == 2


def test_budget_is_zero_when_category_has_no_target():
    """A category with no syntactic target has no degrees of freedom."""
    node = _fn("def g():\n    return 1")
    assert dimension_budget(node, MutationCategory.ARITHMETIC) == 0


# ── the DOF universe ─────────────────────────────────────────────────────


def test_dof_universe_sums_categories():
    node = _fn("def g():\n    x = 1\n    y = 'a'\n    return x + y")
    cats = {MutationCategory.VALUE, MutationCategory.ARITHMETIC}
    assert dof_universe(node, cats) == (
        dimension_budget(node, MutationCategory.VALUE)
        + dimension_budget(node, MutationCategory.ARITHMETIC)
    )


def test_dof_universe_empty_categories_is_zero():
    assert dof_universe(_fn("def g():\n    return 1"), set()) == 0


# ── the property that makes the budget the RIGHT budget ──────────────────


@pytest.mark.parametrize(
    "src,category",
    [
        (
            "def g():\n    x = 1\n    y = 'a'\n    z = 2\n    return x, y, z",
            MutationCategory.VALUE,
        ),
        ("def h(a, b):\n    return a + b - a * b", MutationCategory.ARITHMETIC),
        ("def k(a, b):\n    return a < b or a > b", MutationCategory.BOUNDARY),
    ],
)
def test_budget_selects_one_mutant_per_dimension(src, category):
    """The DOF budget generates exactly one mutant per behavioral dimension.

    This is the property the DOF-coverage claim rests on: at budget = D the greedy
    round-robin covers every dimension exactly once and repeats none.
    """
    from Wesker.engine import generate_mutants

    node = _fn(src)
    d = dimension_budget(node, category)
    mutants = generate_mutants(node, {category}, max_per_category=None)
    dims = [m.dimension for m in mutants]
    assert len(set(dims)) == d, (
        f"expected {d} distinct dimensions, got {sorted(set(dims))}"
    )
    assert len(dims) == len(set(dims)), f"a dimension was covered twice: {dims}"


def test_dof_mode_never_exceeds_the_universe():
    """A DOF budget larger than the target count cannot over-generate."""
    from Wesker.engine import estimate_universe_size, generate_mutants

    node = _fn("def g():\n    return 1")
    cats = {MutationCategory.VALUE}
    mutants = generate_mutants(node, cats, max_per_category=None)
    assert len(mutants) <= estimate_universe_size(node, cats)
