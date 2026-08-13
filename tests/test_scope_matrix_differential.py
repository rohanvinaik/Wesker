"""The differential oracle: scoped and unscoped profiling must agree on the FULL disposition
matrix, not merely on total-killed counts.

`test_scoped_and_unscoped_verdicts_agree` (test_scope_tests_verdict_exact.py) compares only
`total_mutants`/`total_killed` on three zero-fixture functions. That is too weak to be the gate for
Fix B's target-first routing: a routing change could kill mutant A instead of B (same total, wrong
mutant) or re-attribute a kill to a non-distinguishing test, and the count check would stay green.

This oracle is the standing gate every Fix-B slice must keep green. For each target it asserts,
between `scope_tests=True` and `scope_tests=False`:

  * identical per-mutant DISPOSITION (which mutant ids are killed vs survived vs equivalent);
  * identical `kill_matrix` — the exact `mutant -> {killer test-ids}` map. This holds because the
    attribution filter bars a baseline-failing test from BOTH paths, and a baseline-passing test
    that does not execute the mutated line behaves identically under the mutation, so it is credited
    with the kill in NEITHER. Scoping therefore changes only WHICH tests are RUN, never the matrix.

The suites deliberately include a non-covering-but-passing test and a baseline-FAILING test, the two
attribution hazards, so the oracle proves the matrix is stable in their presence — not just on a
suite where every test covers everything.
"""

import ast

from Wesker.engine import MutationCategory, run_function_profiling

_CATS = {
    MutationCategory.VALUE,
    MutationCategory.ARITHMETIC,
    MutationCategory.SWAP,
    MutationCategory.BOUNDARY,
    MutationCategory.LOGICAL,
}


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _profile(node, key, tests, original, *, scope_tests):
    return run_function_profiling(
        node, key, _CATS, tests, original, max_per_category=0, scope_tests=scope_tests
    )


def _matrix(result):
    """The comparable disposition of a run: normalized kill_matrix + the survivor/equivalent id sets."""
    kill = {m: sorted(killers) for m, killers in result.kill_matrix.items()}
    survivors = sorted(r.get("mutant_id") for r in result.survivor_records)
    return {
        "total_mutants": result.total_mutants,
        "total_killed": result.total_killed,
        "total_equivalent": result.total_equivalent,
        "kill_matrix": kill,
        "survivors": survivors,
    }


def _assert_scoping_is_matrix_exact(node, key, tests, original):
    unscoped = _profile(node, key, tests, original, scope_tests=False)
    scoped = _profile(node, key, tests, original, scope_tests=True)
    mu, ms = _matrix(unscoped), _matrix(scoped)
    assert ms["total_mutants"] == mu["total_mutants"], key
    assert ms["survivors"] == mu["survivors"], (
        f"{key}: scoping changed WHICH mutants survive — "
        f"unscoped {mu['survivors']} vs scoped {ms['survivors']}"
    )
    assert ms["kill_matrix"] == mu["kill_matrix"], (
        f"{key}: scoping changed the kill matrix (killer attribution diverged), not verdict-exact"
    )
    return scoped, unscoped


_SCORE_SRC = 'def scoreit(a, b, flag):\n    """doc"""\n    if flag:\n        return a * 2 + b\n    return a - b\n'


def _bind(tests, original, name):
    for t in tests:
        t.__globals__[name] = original


def test_matrix_exact_with_a_non_covering_passing_test():
    node = _fn(_SCORE_SRC)
    ns: dict = {}
    exec(compile(ast.parse(_SCORE_SRC), "<m>", "exec"), ns)  # noqa: S102 — test fixture source
    original = ns["scoreit"]

    def test_true():
        assert scoreit(1, 2, True) == 4  # noqa: F821

    def test_false():
        assert scoreit(5, 3, False) == 2  # noqa: F821

    def test_true_other():
        assert scoreit(3, 4, True) == 10  # noqa: F821

    def test_unrelated_but_passing():
        # Touches nothing in scoreit — the case unscoped must NOT credit with a kill.
        assert 1 + 1 == 2

    tests = [test_true, test_false, test_true_other, test_unrelated_but_passing]
    _bind(tests, original, "scoreit")
    _assert_scoping_is_matrix_exact(node, "<m>::scoreit", tests, original)


def test_matrix_exact_with_a_baseline_failing_test_present():
    node = _fn(_SCORE_SRC)
    ns: dict = {}
    exec(compile(ast.parse(_SCORE_SRC), "<m2>", "exec"), ns)  # noqa: S102 — test fixture source
    original = ns["scoreit"]

    def test_true():
        assert scoreit(1, 2, True) == 4  # noqa: F821

    def test_false():
        assert scoreit(5, 3, False) == 2  # noqa: F821

    def test_broken_on_original():
        # FAILS on the unmutated function: must be barred from attribution in BOTH paths, so it
        # can never inflate one side's kill matrix over the other's.
        assert scoreit(1, 2, True) == 999  # noqa: F821

    tests = [test_true, test_false, test_broken_on_original]
    _bind(tests, original, "scoreit")
    scoped, _ = _assert_scoping_is_matrix_exact(node, "<m2>::scoreit", tests, original)
    # And the broken test never appears as a killer of anything.
    assert not any(
        "test_broken_on_original" in k
        for killers in scoped.kill_matrix.values()
        for k in killers
    )


_BRANCHY_SRC = (
    "def classify(x):\n"
    '    """doc"""\n'
    "    if x <= 0:\n"
    "        return -1\n"
    "    if x < 10:\n"
    "        return 0\n"
    "    return 1\n"
)


def test_matrix_exact_on_a_multi_branch_function():
    node = _fn(_BRANCHY_SRC)
    ns: dict = {}
    exec(compile(ast.parse(_BRANCHY_SRC), "<m3>", "exec"), ns)  # noqa: S102 — test fixture source
    original = ns["classify"]

    def test_neg():
        assert classify(-5) == -1  # noqa: F821

    def test_mid():
        assert classify(5) == 0  # noqa: F821

    def test_high():
        assert classify(50) == 1  # noqa: F821

    def test_boundary():
        assert classify(10) == 1  # noqa: F821

    tests = [test_neg, test_mid, test_high, test_boundary]
    _bind(tests, original, "classify")
    _assert_scoping_is_matrix_exact(node, "<m3>::classify", tests, original)
