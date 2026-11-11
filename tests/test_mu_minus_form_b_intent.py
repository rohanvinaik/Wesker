"""μ⁻ Form B (#44): the stale return, the swallowed raise, and yields that reach the isolated worker.

Intent tests authored from docs/theory/NEGATIVE_SPECIFICATION.md, Def. 11.6 and Def. 11.8 (i), (viii).

THE DEFECTS.
- Def. 11.8 (i) `stale` (the output is a pure function of THIS call) and (viii) `raise` (the
  raise-versus-return contract) were not generated.
- Measured on a three-test generator: the four yield mutants were killed in process and SURVIVED
  under `--isolated`. The isolated worker compiles source alone. A yield mutant's node was a marker
  string that defines nothing, so the worker installed the ORIGINAL function as the "mutant", and
  every isolated verdict on a yield mutant was a false survivor. Converge's reproducibility check
  re-observes the function in isolated workers before it certifies.
- The in-process yield factory called a bound classmethod with the `cls` the installed wrapper is
  handed: `cls` bound twice, a TypeError, a crash kill. It also could not wrap an async generator.

WHAT MUST HOLD.
- Which Form B modes a function gets is read off its own body, the same way for counting and
  generation.
- A Form B mutant is source that every evaluator builds alike, with the perturbation innermost
  under the target's own decorators.
- stale is killed only by a second call with a different result. swallow is killed by a test that
  pins the raise, as a value kill.
- Neither mode eats the engine's BaseException interrupts, nor changes how the function must be
  called (sync, async, method, classmethod, staticmethod).
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import sys
import textwrap

import pytest

from Wesker.engine import (
    MutationCategory,
    _form_b_node,
    _has_own_raise,
    _make_generator_wrapper,
    estimate_universe_size,
    evaluate_mutant,
    form_b_output_modes,
    generate_mutants,
    isolated_mutant_node,
)
from Wesker.isolation import IsolatedMutantWorker, mutant_verdict
from Wesker.mu_minus import CALL_MODES, YIELD_MODES, form_b
from Wesker.tce import nodes_equivalent


def _node(src: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    return next(
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _form_b(src: str) -> dict:
    muts = generate_mutants(_node(src), {MutationCategory.OUTPUT}, max_per_category=0)
    out = {}
    for m in muts:
        mode = m.dimension.split(":")[1]
        if mode in YIELD_MODES + CALL_MODES:
            out[mode] = m
    return out


def _load(tmp_path, name, src):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(src))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, str(path)


# ── which modes, read off the function's own body ──────────────────────────────────────────


def test_form_b_modes_follow_how_the_output_is_delivered():
    # The pure decision, every row. A generator's output is its yielded sequence; anything else
    # delivers a return value across calls, and a raise only where its own body raises.
    assert form_b_output_modes(True, False) == YIELD_MODES
    assert form_b_output_modes(True, True) == YIELD_MODES
    assert form_b_output_modes(False, False) == ("return_stale",)
    assert form_b_output_modes(False, True) == ("return_stale", "raise_swallow")


def test_an_own_raise_is_one_this_function_executes():
    assert _has_own_raise(_node("def f(x):\n    if x:\n        raise ValueError(x)\n"))
    assert _has_own_raise(
        _node(
            "def f(x):\n    try:\n        return g(x)\n    except KeyError:\n        raise\n"
        )
    )
    # A nested def's or nested class's raise is that scope's contract, not this function's.
    assert not _has_own_raise(
        _node("def f(x):\n    def g():\n        raise ValueError\n    return g\n")
    )
    assert not _has_own_raise(
        _node(
            "def f(x):\n    class C:\n        def m(self):\n            raise E\n    return C\n"
        )
    )
    assert not _has_own_raise(_node("def f(x):\n    return int(x)\n"))


@pytest.mark.parametrize(
    "src",
    [
        "def g(n):\n    yield from range(n)\n",
        "async def g(n):\n    for i in range(n):\n        yield i\n",
        "def f(x):\n    return x * 2\n",
        "def f(x):\n    if x < 0:\n        raise ValueError(x)\n    return x\n",
        "async def f(x):\n    if x:\n        raise KeyError(x)\n    return x\n",
        "def f(self, x):\n    self.n = x\n",
    ],
)
def test_count_equals_generation_with_form_b(src):
    # The census counts the Form B modes from the same two facts generation reads (#9).
    node = _node(src)
    muts = generate_mutants(node, {MutationCategory.OUTPUT}, max_per_category=0)
    assert len(muts) == estimate_universe_size(node, {MutationCategory.OUTPUT})


# ── the mutant is source, decorated innermost ─────────────────────────────────────────────


def test_a_call_mode_mutant_is_the_definition_with_the_decorator_innermost():
    src = "@functools.lru_cache\ndef f(x):\n    return x * 2\n"
    m = _form_b(src)["return_stale"]
    assert m.wrapper_factory is None  # compiled like any other mutant, everywhere
    decorators = [ast.unparse(d) for d in m.mutated_node.decorator_list]
    assert (
        decorators[0] == "functools.lru_cache"
    )  # the target's own, outermost, untouched
    assert decorators[-1] == (
        "__import__('Wesker.mu_minus', fromlist=['form_b']).form_b('return_stale')"
    )
    assert m.mutated_line == m.original_node.lineno


def test_a_form_b_mutant_is_never_bytecode_equivalent_to_the_original():
    # TCE runs on every survivor before the boundary probes. The decorator lives in module-level
    # bytecode beside the function's (identical) code object; were it invisible to the key, every
    # Form B survivor would be marked equivalent without running anything.
    node = _node(
        "def f(x):\n    if x < 0:\n        raise ValueError(x)\n    return x\n"
    )
    for mode in ("return_stale", "raise_swallow"):
        assert not nodes_equivalent(node, _form_b_node(node, mode))


def test_an_unknown_mode_refuses_to_build_rather_than_wrap_nothing():
    # A silent no-op would install the original as the mutant: a false survivor.
    with pytest.raises(ValueError, match="not a μ⁻ Form B mode"):
        form_b("return_bogus")


# ── stale ─────────────────────────────────────────────────────────────────────────────────


def test_stale_answers_each_call_with_the_previous_calls_return():
    perturbed = form_b("return_stale")(lambda x: x * 2)
    assert [perturbed(1), perturbed(2), perturbed(3)] == [2, 2, 4]


def test_stale_is_killed_by_a_second_call_and_survives_a_single_one(tmp_path):
    mod, path = _load(tmp_path, "mu_stale", "def f(x):\n    return x * 2\n")
    try:
        m = _form_b("def f(x):\n    return x * 2\n")["return_stale"]

        def first():
            assert mod.f(2) == 4

        def second():
            assert mod.f(3) == 6

        def twice():
            assert mod.f(2) == 4
            assert mod.f(3) == 6

        # One call cannot see a previous call: the suite has not pinned that f carries no state.
        alone = evaluate_mutant(m, [first], mod.f, qualname="f", source_path=path)
        assert alone.constructed and alone.installed and not alone.killed
        # Two TESTS that call once each do not either (W#49): the wrapper's memory ends with the test
        # that made it, so neither test, run alone, detects stale, and the pair does not by leaking.
        for order in ([first, second], [second, first]):
            pair = evaluate_mutant(m, order, mod.f, qualname="f", source_path=path)
            assert pair.constructed and pair.installed and not pair.killed, order
        # A second call with a different result, in ONE test, can, by ASSERTION, credited to it.
        both = evaluate_mutant(m, [twice], mod.f, qualname="f", source_path=path)
        assert both.killed and both.killed_by == "assertion"
        assert both.test_name is not None and both.test_name.endswith("twice")
    finally:
        sys.modules.pop("mu_stale", None)


def test_stale_on_a_constant_output_cannot_be_told_apart():
    # Where every call returns the same value, the previous return IS this call's: no test can
    # distinguish them, so the perturbation is a true no-op there.
    perturbed = form_b("return_stale")(lambda x: 7)
    assert [perturbed(1), perturbed(2)] == [7, 7]


# ── swallow ───────────────────────────────────────────────────────────────────────────────


def test_swallow_returns_none_for_an_ordinary_raise_only():
    class Interrupt(BaseException):
        pass

    def boom(kind):
        if kind == "value":
            raise ValueError(kind)
        raise Interrupt(kind)

    perturbed = form_b("raise_swallow")(boom)
    assert perturbed("value") is None
    # The engine's own interrupts (Abandoned, MeasurementRefused, StepBudgetExceeded) are
    # BaseException precisely so that code like this cannot eat them.
    with pytest.raises(Interrupt):
        perturbed("interrupt")


def test_swallow_is_killed_by_a_test_that_pins_the_raise(tmp_path):
    src = "def f(x):\n    if x < 0:\n        raise ValueError(x)\n    return x\n"
    mod, path = _load(tmp_path, "mu_swallow", src)
    try:
        m = _form_b(src)["raise_swallow"]

        def pins_value():
            assert mod.f(3) == 3

        def pins_raise():
            with pytest.raises(ValueError):
                mod.f(-1)

        # A suite that never exercises the raise has not pinned the raise-versus-return contract.
        assert not evaluate_mutant(
            m, [pins_value], mod.f, qualname="f", source_path=path
        ).killed
        r = evaluate_mutant(m, [pins_raise], mod.f, qualname="f", source_path=path)
        # A VALUE kill: the suite saw a return where its raises contract says raise, which Wesker
        # names `exception` and counts beside `assertion` as value-pinned. Never a crash, which
        # would credit the run, not the specification.
        assert r.killed and r.killed_by == "exception", r.killed_by
    finally:
        sys.modules.pop("mu_swallow", None)


# ── the call shape survives: async, methods, descriptors ──────────────────────────────────


def test_async_functions_stay_awaitable():
    async def double(x):
        return x * 2

    async def boom():
        raise ValueError

    stale = form_b("return_stale")(double)
    assert inspect.iscoroutinefunction(stale)

    async def calls():
        return [await stale(1), await stale(2)]

    assert asyncio.run(calls()) == [2, 2]
    swallow = form_b("raise_swallow")(boom)
    assert inspect.iscoroutinefunction(swallow)
    assert asyncio.run(swallow()) is None


def test_methods_and_descriptors_bind_once_under_the_innermost_decorator():
    # Built from the real generated node shape: each method gets the decorator innermost, so
    # classmethod/staticmethod/property wrap the perturbation as they wrap the function.
    src = textwrap.dedent(
        """
        class C:
            def inst(self, x):
                return x * 2

            @classmethod
            def cls_m(cls, x):
                return x * 3

            @staticmethod
            def static_m(x):
                return x * 4

            @property
            def prop(self):
                return self.v
        """
    )
    tree = ast.parse(src)
    cls_node = tree.body[0]
    cls_node.body = [_form_b_node(fn, "return_stale") for fn in cls_node.body]
    ns: dict = {}
    exec(compile(ast.fix_missing_locations(tree), "<c>", "exec"), ns)  # noqa: S102
    c = ns["C"]
    assert [c().inst(1), c().inst(2)] == [2, 2]
    assert [c.cls_m(1), c.cls_m(2)] == [3, 3]
    assert [c.static_m(1), c.static_m(2)] == [4, 4]
    a, b = c(), c()
    a.v, b.v = "a", "b"
    assert [a.prop, b.prop] == ["a", "a"]  # state leaked across instances: the question


def test_the_wrapper_reads_as_the_function_and_never_evaluates_annotations():
    # A deferred annotation that would raise must not be evaluated by building a mutant (W#47).
    ns: dict = {}
    exec(  # noqa: S102
        "def f(x: NameNotDefinedAnywhere) -> NameNotDefinedAnywhere:\n"
        '    """Doc."""\n'
        "    return x\n",
        ns,
    )
    perturbed = form_b("return_stale")(ns["f"])
    assert perturbed.__name__ == "f" and perturbed.__doc__ == "Doc."
    assert perturbed.__wrapped__ is ns["f"]


# ── yields: the async generator, the bound classmethod, and the isolated worker ──────────


def test_an_async_generator_stays_an_async_generator():
    async def agen(n):
        for i in range(n):
            yield i

    perturbed = form_b("yield_truncate")(agen)
    assert inspect.isasyncgenfunction(perturbed)

    async def collect():
        return [i async for i in perturbed(3)]

    assert asyncio.run(collect()) == [0, 1]


def test_the_yield_factory_binds_a_classmethods_cls_once():
    class C:
        @classmethod
        def gen(cls, n):
            yield from range(n)

    # `C.gen` reaches the factory BOUND. The installed mutant is classmethod-wrapped and handed
    # `cls` again; calling the bound method with it bound `cls` twice (a crash kill).
    wrapped = _make_generator_wrapper("yield_truncate")(C.gen)
    C.gen = classmethod(wrapped)
    assert list(C.gen(3)) == [0, 1]


def test_an_isolated_worker_measures_the_yield_mutant_not_the_original(tmp_path):
    # The grounded defect, end to end. The marker defines nothing, so the worker installed the
    # original and the mutant "survived"; the isolated twin is the decorated definition.
    (tmp_path / "g.py").write_text(
        "def g(n):\n    for i in range(n):\n        yield i\n"
    )
    (tmp_path / "test_g.py").write_text(
        "from g import g\n\n\ndef test_g():\n    assert list(g(3)) == [0, 1, 2]\n"
    )
    m = _form_b("def g(n):\n    for i in range(n):\n        yield i\n")[
        "yield_truncate"
    ]
    worker = IsolatedMutantWorker(str(tmp_path), ["test_g.py::test_g"], "g.py", "g")
    try:
        legacy = worker.evaluate(ast.unparse(m.mutated_node), 30.0)
        assert mutant_verdict(legacy.outcome) == "survived"  # the false survivor
        twin = worker.evaluate(ast.unparse(isolated_mutant_node(m)), 30.0)
        assert mutant_verdict(twin.outcome) == "killed"
        assert twin.killed_by == "assertion"
    finally:
        worker.close()


def test_an_isolated_worker_measures_stale_by_two_calls_in_one_test(tmp_path):
    (tmp_path / "t.py").write_text("def f(x):\n    return x * 2\n")
    (tmp_path / "test_t.py").write_text(
        "from t import f\n\n\n"
        "def test_a():\n    assert f(2) == 4\n\n\n"
        "def test_b():\n    assert f(3) == 6\n\n\n"
        "def test_twice():\n    assert f(2) == 4\n    assert f(3) == 6\n"
    )
    m = _form_b("def f(x):\n    return x * 2\n")["return_stale"]
    assert isolated_mutant_node(m) is m.mutated_node  # already source
    source = ast.unparse(m.mutated_node)
    worker = IsolatedMutantWorker(str(tmp_path), [], "t.py", "f")
    try:
        # Two tests, one call each, in either order: survived (W#49).
        for nodes in (
            ["test_t.py::test_a", "test_t.py::test_b"],
            ["test_t.py::test_b", "test_t.py::test_a"],
        ):
            r = worker.evaluate(source, 30.0, node_ids=nodes)
            assert mutant_verdict(r.outcome) == "survived", nodes
        # Two calls in one test: killed by assertion, by that test.
        r = worker.evaluate(source, 30.0, node_ids=["test_t.py::test_twice"])
        assert mutant_verdict(r.outcome) == "killed"
        assert r.killed_by == "assertion"
        assert r.test_name and r.test_name.endswith("test_twice")
    finally:
        worker.close()
