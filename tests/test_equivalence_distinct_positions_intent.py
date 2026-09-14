"""Intent: the equivalence probe can tell argument positions apart.

The defect, found in a cold read and confirmed by running it: for a function of three or more
parameters `_generate_boundary_inputs` produced `(0, 0, 0), (1, 1, 1), (-1, -1, -1), (0.0, 0.0, 0.0),
(1.0, 1.0, 1.0)` — one value repeated in every position. `check_equivalent` runs a survivor and its
original on those rows and calls the mutant "likely equivalent" when every row agrees, so a SWAP mutant
such as `pow(a, c)` → `pow(c, a)` agreed on every row, read as equivalent, and was removed from the
effective kill rate. That is the repeated-value blind spot Wesker 1.1.0 closed on the GENERATION side
(every pair of positions is now a question); the equivalence side still could not see it.

`boundary_input_rows` now adds rotation rows in which every position holds a different value, and keeps
the uniform rows, which a BOUNDARY mutant needs (`a < c` → `a <= c` differs only where operands are
equal). On the old code the position and end-to-end tests below fail.
"""

import ast
import itertools

import pytest

from Wesker.engine import (
    MutationCategory,
    boundary_input_rows,
    check_equivalent,
    generate_mutants,
)


@pytest.mark.parametrize("n_params", [3, 4, 5])
def test_every_pair_of_positions_differs_in_some_row(n_params) -> None:
    rows = boundary_input_rows(n_params)
    for i, j in itertools.combinations(range(n_params), 2):
        assert any(row[i] != row[j] for row in rows), (
            f"positions {i} and {j} never differ"
        )


@pytest.mark.parametrize("n_params", [3, 4, 5, 6])
def test_the_uniform_rows_a_boundary_mutant_needs_are_kept(n_params) -> None:
    rows = boundary_input_rows(n_params)
    assert any(len(set(row)) == 1 for row in rows)


def test_one_and_two_parameter_rows_are_unchanged() -> None:
    assert boundary_input_rows(1) == [
        (0,),
        (1,),
        (-1,),
        (2,),
        (-2,),
        (0.0,),
        (1.0,),
        (-1.0,),
        (0.5,),
        (True,),
        (False,),
    ]
    base = [0, 1, -1, 0.0, 1.0]
    assert boundary_input_rows(2) == [(a, b) for a in base for b in base]
    assert boundary_input_rows(0) == [()]


def test_a_swap_mutant_of_a_three_argument_function_is_not_called_equivalent() -> None:
    """End to end through the real generator and the real equivalence check."""
    node = ast.parse("def f(a, b, c):\n    return pow(a, c) - b\n").body[0]
    mutants = generate_mutants(node, {MutationCategory.SWAP}, max_per_category=0)
    swaps = [m for m in mutants if m.category is MutationCategory.SWAP]
    assert swaps, "the SWAP operator should produce pow(c, a)"
    for mutant in swaps:
        assert check_equivalent(node, mutant) is False, mutant.description
