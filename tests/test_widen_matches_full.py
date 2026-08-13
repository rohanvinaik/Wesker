"""The Fix B driver oracle: seed + lazy-widen produces the SAME disposition as a full baseline.

`test_seed_then_expand_equals_a_full_rebuild` pins that the BASELINE is identical; this pins that
`run_function_converged`'s widen pass, driven off a seeded session baseline, ends at the same
`kill_matrix` / survivor set as a run over the full baseline. The target is chosen so different
tests kill different mutants: `test_true` covers only the `flag=True` branch, `test_false` only the
`flag=False` branch. Seeding `[test_true]` and widening with `[test_false]` means the `a - b`
mutants SURVIVE the seed and are killed only in the widen pass — exactly the case the driver exists
to handle, and the case a broken widen would report as a false survivor.
"""

import ast

from Wesker.engine import (
    _SESSION_BASELINE,
    LazySessionBaseline,
    MutationCategory,
    build_session_baseline,
    run_function_converged,
)

_CATS = {
    MutationCategory.VALUE,
    MutationCategory.ARITHMETIC,
    MutationCategory.SWAP,
    MutationCategory.BOUNDARY,
}

_SRC = "def scoreit(a, b, flag):\n    if flag:\n        return a * 2 + b\n    return a - b\n"


def _fn(src):
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _matrix(result):
    return {
        "total_mutants": result.total_mutants,
        "total_killed": result.total_killed,
        "total_equivalent": result.total_equivalent,
        "kill_matrix": {m: sorted(k) for m, k in result.kill_matrix.items()},
        "survivors": sorted(r.get("mutant_id") for r in result.survivor_records),
    }


def _run(node, tests, original, holder, **kw):
    tok = _SESSION_BASELINE.set(holder)
    try:
        return run_function_converged(
            node,
            f"{original.__code__.co_filename}::scoreit",
            _CATS,
            tests,
            original,
            max_per_category=0,
            **kw,
        )
    finally:
        _SESSION_BASELINE.reset(tok)


def test_seed_widen_matches_full_baseline():
    node = _fn(_SRC)
    ns: dict = {}
    exec(compile(ast.parse(_SRC), "<sw>", "exec"), ns)  # noqa: S102 — test fixture source
    original = ns["scoreit"]

    def test_true():
        assert scoreit(1, 2, True) == 4  # noqa: F821 — flag=True branch only

    def test_false():
        assert scoreit(5, 3, False) == 2  # noqa: F821 — flag=False branch only

    tests = [test_true, test_false]
    for t in tests:
        t.__globals__["scoreit"] = original

    target_files = {original.__code__.co_filename}

    def build(subset=None):
        return build_session_baseline(
            tests if subset is None else list(subset), target_files
        )

    # FULL baseline (get() builds over both tests).
    full = _run(node, tests, original, LazySessionBaseline(build))

    # SEED [test_true] only, WIDEN with [test_false]: the flag=False mutants survive the seed and
    # must be killed by the widen pass to match `full`.
    holder = LazySessionBaseline(build)
    holder.seed([test_true])
    seeded = _run(node, tests, original, holder, widen_tests=[test_false])

    assert _matrix(seeded) == _matrix(full), (
        "seed+widen diverged from the full baseline — the widen pass is not disposition-exact"
    )
    # And the widen genuinely mattered: some mutant is killed ONLY by the widened test.
    assert any(
        "test_false" in k for killers in full.kill_matrix.values() for k in killers
    )


def test_widen_with_no_survivors_is_a_noop():
    # When the seed already kills everything, the widen pass must not run the unknowns at all — and
    # must still match the full baseline.
    node = _fn(_SRC)
    ns: dict = {}
    exec(compile(ast.parse(_SRC), "<sw2>", "exec"), ns)  # noqa: S102 — test fixture source
    original = ns["scoreit"]

    def test_true():
        assert scoreit(1, 2, True) == 4  # noqa: F821

    def test_false():
        assert scoreit(5, 3, False) == 2  # noqa: F821

    tests = [test_true, test_false]
    for t in tests:
        t.__globals__["scoreit"] = original
    target_files = {original.__code__.co_filename}

    def build(subset=None):
        return build_session_baseline(
            tests if subset is None else list(subset), target_files
        )

    full = _run(node, tests, original, LazySessionBaseline(build))
    # Seed BOTH (nothing left to widen); widen list is irrelevant.
    holder = LazySessionBaseline(build)
    holder.seed(tests)
    seeded = _run(node, tests, original, holder, widen_tests=[])
    assert _matrix(seeded) == _matrix(full)
