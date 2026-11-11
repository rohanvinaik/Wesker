"""Intent: the stale wrapper's memory ends with the test that made it (Wesker #49).

THE DEFECT, measured 2026-10-25 on Wesker 6ba533d by evaluating OUTPUT `return_stale` mutants directly
over hand-written tests. `_stale_wrapper` held the previous call's return for the whole evaluation,
across every test it ran:

    (a)   one call per test: survives                       (b)  one test, two calls, two results: killed
    (c)   two tests, one call each, order [a, b]: killed, credited to b
    (c')  the same two tests, order [b, a]: killed, credited to a
    (c'') test b alone: survives                            (d)  two calls, the same result: survives

In (c) neither test detects stale. The pair does, through state the wrapper leaks across a test
boundary, and the kill matrix credits only the second. The verdict depends on test order and on which
subset ran, and a consumer minimising the suite could drop the predecessor and silently un-kill stale.

THE RULING: Def. 11.8 (i) of docs/theory/NEGATIVE_SPECIFICATION.md (Detective): stale stands for "the
output is a pure function of *this* call, carrying no leaked state or accidental memoization". A TEST
detects it by calling twice and seeing two results. So the wrapper's memory must not outlive a test:
reset at every test boundary, in both execution modes. Then (c) and (c') SURVIVE, (b) is killed by
assertion and credited to the test that made both calls, and (a), (c''), (d) are unchanged.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import sys
import textwrap

import pytest

from Wesker.engine import (
    MutationCategory,
    evaluate_mutant,
    generate_mutants,
)
from Wesker.isolation import IsolatedMutantWorker, mutant_verdict
from Wesker.mu_minus import begin_test, form_b, stale_memory_standing

# ── the pure decision ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("last", "epoch", "standing"),
    [
        (None, 0, "first"),
        (None, 7, "first"),
        (3, 3, "carry"),
        (0, 0, "carry"),
        (3, 4, "reset"),
        (0, 1, "reset"),
        # An epoch that moved backwards is still not the same test: nothing is carried across it.
        (4, 3, "reset"),
    ],
)
def test_the_memory_is_carried_only_inside_the_test_that_made_it(last, epoch, standing):
    assert stale_memory_standing(last, epoch) == standing


# ── the wrapper itself ──────────────────────────────────────────────────────────────────────────


def test_inside_one_test_each_call_answers_with_the_previous_return():
    stale = form_b("return_stale")(lambda x: x * 2)
    begin_test()
    assert [stale(1), stale(2), stale(3)] == [2, 2, 4]


def test_the_first_call_of_every_test_answers_with_its_own_value():
    stale = form_b("return_stale")(lambda x: x * 2)
    begin_test()
    assert stale(1) == 2
    assert stale(2) == 2  # carried inside the test
    begin_test()
    assert stale(5) == 10, (
        "the previous test's return must not be answered in a new test"
    )
    begin_test()
    assert stale(7) == 14


def test_two_wrappers_do_not_share_a_memory_and_a_boundary_resets_both():
    left = form_b("return_stale")(lambda x: ("l", x))
    right = form_b("return_stale")(lambda x: ("r", x))
    begin_test()
    left(1)
    right(1)
    assert left(2) == ("l", 1) and right(2) == ("r", 1)
    begin_test()
    assert left(3) == ("l", 3) and right(3) == ("r", 3)


def test_a_caller_that_never_marks_a_boundary_sees_the_memory_it_always_saw():
    """An unmanaged use of the wrapper (an equivalence probe, a library caller) crosses no test
    boundary, so nothing resets: the behaviour before #49."""
    stale = form_b("return_stale")(lambda x: x * 2)
    begin_test()  # one boundary, then none
    assert [stale(1), stale(2), stale(3), stale(4)] == [2, 2, 4, 6]


def test_the_async_wrapper_resets_too():
    async def double(x):
        return x * 2

    stale = form_b("return_stale")(double)

    async def calls(*xs):
        return [await stale(x) for x in xs]

    begin_test()
    assert asyncio.run(calls(1, 2)) == [2, 2]
    begin_test()
    assert asyncio.run(calls(3)) == [6]


# ── the evaluator: shapes (a) to (d), in-process ────────────────────────────────────────────────

_SRC = "def f(x):\n    return x * 2\n"


def _stale_mutant():
    node = ast.parse(_SRC).body[0]
    mutants = generate_mutants(node, {MutationCategory.OUTPUT}, max_per_category=0)
    return next(m for m in mutants if m.dimension.split(":")[1] == "return_stale")


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "stale_target.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("stale_target", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stale_target"] = mod
    spec.loader.exec_module(mod)
    yield mod, str(path)
    sys.modules.pop("stale_target", None)


def _run(target, *tests, record_all_killers=False):
    mod, path = target
    return evaluate_mutant(
        _stale_mutant(),
        list(tests),
        mod.f,
        qualname="f",
        source_path=path,
        record_all_killers=record_all_killers,
    )


def _tests(mod):
    def a():
        assert mod.f(1) == 2

    def b():
        assert mod.f(2) == 4

    def both():  # shape (b): two calls, two results, one test
        assert mod.f(1) == 2
        assert mod.f(2) == 4

    def same():  # shape (d): two calls, the same result
        assert mod.f(1) == 2
        assert mod.f(1) == 2

    return a, b, both, same


def test_shape_a_one_call_per_test_survives(target):
    a, _, _, _ = _tests(target[0])
    result = _run(target, a)
    assert result.constructed and result.installed and not result.killed


def test_shape_b_two_calls_in_one_test_is_killed_by_assertion_and_credited_to_it(
    target,
):
    _, _, both, _ = _tests(target[0])
    result = _run(target, both)
    assert result.killed and result.killed_by == "assertion"
    assert result.test_name.endswith("both")


@pytest.mark.parametrize("order", ["ab", "ba"])
def test_shapes_c_and_c_prime_two_tests_one_call_each_survive_in_either_order(
    target, order
):
    a, b, _, _ = _tests(target[0])
    tests = {"a": a, "b": b}
    result = _run(target, *(tests[k] for k in order))
    assert result.constructed and result.installed
    assert not result.killed, "state leaked across the test boundary credited a kill"
    assert result.killed_by is None


def test_shape_c_double_prime_a_test_alone_survives(target):
    _, b, _, _ = _tests(target[0])
    assert not _run(target, b).killed


def test_shape_d_two_calls_with_the_same_result_survive(target):
    _, _, _, same = _tests(target[0])
    assert not _run(target, same).killed


def test_the_verdict_does_not_depend_on_which_subset_ran(target):
    """The same test gives the same verdict whatever ran before it, and a test that detects stale does
    so alone, so a minimiser that drops its neighbours cannot un-kill it."""
    a, b, both, same = _tests(target[0])
    alone = _run(target, both)
    for suite in ([a, both], [both, a], [a, b, both], [b, a, both, same], [same, both]):
        result = _run(target, *suite)
        assert result.killed and result.killed_by == "assertion", suite
        assert result.test_name == alone.test_name, suite
    # And the ones that do not detect it do not, whatever they are run among.
    for suite in ([a, b], [b, a], [a, b, same], [same, a, b]):
        assert not _run(target, *suite).killed, suite


def test_full_matrix_credits_only_the_test_that_made_both_calls(target):
    a, b, both, _ = _tests(target[0])
    result = _run(target, a, b, both, a, record_all_killers=True)
    assert result.killed
    assert len(result.killed_by_tests) == 1
    assert result.killed_by_tests[0].endswith("both")


# ── the evaluator: shapes, isolated worker ──────────────────────────────────────────────────────

_ISOLATED_TESTS = textwrap.dedent(
    """
    import pytest
    from t import f


    def test_a():
        assert f(1) == 2


    def test_b():
        assert f(2) == 4


    def test_both():
        assert f(1) == 2
        assert f(2) == 4


    def test_same():
        assert f(1) == 2
        assert f(1) == 2

    """
)


@pytest.fixture
def worker(tmp_path):
    (tmp_path / "t.py").write_text(_SRC)
    (tmp_path / "test_t.py").write_text(_ISOLATED_TESTS)
    w = IsolatedMutantWorker(str(tmp_path), [], "t.py", "f")
    yield w
    w.close()


def _isolated(worker, *names):
    source = ast.unparse(_stale_mutant().mutated_node)
    return worker.evaluate(
        source, 60.0, node_ids=[f"test_t.py::{name}" for name in names]
    )


@pytest.mark.parametrize("names", [("test_a", "test_b"), ("test_b", "test_a")])
def test_isolated_two_tests_one_call_each_survive_in_either_order(worker, names):
    result = _isolated(worker, *names)
    assert mutant_verdict(result.outcome) == "survived", names
    assert result.killed_by is None


def test_isolated_one_test_two_calls_is_killed_by_assertion_and_credited_to_it(worker):
    result = _isolated(worker, "test_a", "test_both", "test_b")
    assert mutant_verdict(result.outcome) == "killed"
    assert result.killed_by == "assertion"
    assert result.test_name.endswith("test_both")


def test_isolated_same_result_survives_and_a_test_alone_survives(worker):
    assert mutant_verdict(_isolated(worker, "test_same").outcome) == "survived"
    assert mutant_verdict(_isolated(worker, "test_b").outcome) == "survived"


def test_isolated_the_same_worker_gives_the_same_verdicts_after_other_mutants(worker):
    """The epoch is process-wide and the worker is reused: an earlier evaluation must leave nothing
    that changes a later one."""
    first = _isolated(worker, "test_a", "test_b")
    killed = _isolated(worker, "test_both")
    again = _isolated(worker, "test_a", "test_b")
    assert mutant_verdict(first.outcome) == mutant_verdict(again.outcome) == "survived"
    assert mutant_verdict(killed.outcome) == "killed"
