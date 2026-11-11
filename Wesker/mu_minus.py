"""μ⁻ Form B at runtime, and the catalog of Def. 11.8's output operators.

Form B (docs/theory/NEGATIVE_SPECIFICATION.md, Def. 11.6) perturbs an output the return-site rewrite
cannot reach:
- a generator's yielded sequence;
- the previous call's return (stale);
- a raise, swallowed.

A Form B mutant is ordinary SOURCE. It is a copy of the target's own definition with
``@__import__("Wesker.mu_minus", fromlist=["form_b"]).form_b("<mode>")`` applied innermost. Every site
that builds a mutant therefore builds this one the same way: the in-process evaluator, the isolated
worker (which receives nothing but source), ``check_equivalent``, TCE and Detective's mutant compiler.

The yield modes were first a runtime closure that never reached source. The isolated worker then
compiled a bare marker string, found no definition, and installed the ORIGINAL function as the
mutant. Measured on a three-test generator: all four yield mutants were killed in process and
survived under ``--isolated``, so every isolated verdict on them was false.

Applied innermost, the wrapper wraps the raw function, and the target's own decorators wrap it in
turn. A classmethod therefore binds ``cls`` once, and ``property`` and ``lru_cache`` see the
perturbed function, the way they see the real one.

Mutant code imports this module at definition time, so it imports only the standard library.
"""

from __future__ import annotations

import functools
import inspect
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Bounds an INFINITE generator so a wrapper can never hang the evaluator; high enough not to bite a
# real finite sequence. A wrapper materializes at most this many items before perturbing.
GENERATOR_MATERIALIZE_CAP = 10_000

YIELD_MODES = (
    "yield_truncate",
    "yield_drop_first",
    "yield_duplicate_last",
    "yield_empty",
)
CALL_MODES = ("return_stale", "raise_swallow")

# What the wrapper copies from the function it wraps. `__annotations__` is deliberately absent:
# reading it evaluates deferred annotations, which can raise and must never run as a side effect of
# building a mutant (W#47).
_WRAPPER_ASSIGNED = ("__module__", "__name__", "__qualname__", "__doc__")


def perturb_yields(items: list, mode: str) -> list:
    """The yielded-sequence perturbation for one yield mode. A pure list transform. On some inputs
    it is a no-op (truncate or drop on an empty sequence): a runtime candidate-equivalent there."""
    if mode == "yield_truncate":
        return items[:-1]
    if mode == "yield_drop_first":
        return items[1:]
    if mode == "yield_duplicate_last":
        return items + items[-1:]
    if mode == "yield_empty":
        return []
    return items


def _like(wrapper: Callable[..., Any], fn: Callable[..., Any]) -> Callable[..., Any]:
    """``wrapper`` carrying ``fn``'s name, qualname, module, doc and ``__wrapped__``, so a test that
    reads them, or ``inspect.signature``, sees the function it called rather than the perturbation."""
    return functools.update_wrapper(wrapper, fn, assigned=_WRAPPER_ASSIGNED, updated=())


def _yield_wrapper(fn: Callable[..., Any], mode: str) -> Callable[..., Any]:
    if inspect.isasyncgenfunction(fn):

        async def agen(*args: Any, **kwargs: Any) -> Any:
            source = fn(*args, **kwargs)
            items: list = []
            try:
                async for item in source:
                    items.append(item)
                    if len(items) >= GENERATOR_MATERIALIZE_CAP:
                        break
            finally:
                await source.aclose()
            for item in perturb_yields(items, mode):
                yield item

        return _like(agen, fn)

    def gen(*args: Any, **kwargs: Any) -> Any:
        items = list(itertools.islice(fn(*args, **kwargs), GENERATOR_MATERIALIZE_CAP))
        yield from perturb_yields(items, mode)

    return _like(gen, fn)


#: The number of test boundaries crossed so far in this process (#49). Whoever RUNS tests against a
#: mutant (the in-process evaluator, the isolated worker) calls :func:`begin_test` as each test starts;
#: a stale wrapper compares it with the boundary it last answered in. A counter rather than a flag so
#: that nothing needs to remember to clear it, and an unmanaged caller (the wrapper used directly, an
#: equivalence probe) sees one unchanging epoch, which is the wrapper's previous behaviour.
_TEST_EPOCH = [0]


def begin_test() -> None:
    """Mark a test boundary: no stale wrapper carries a previous return across it (#49)."""
    _TEST_EPOCH[0] += 1


def stale_memory_standing(last_epoch: int | None, epoch: int) -> str:
    """Whether a stale wrapper may answer with the previous call's return (#49, pure — pinned).

    Def. 11.8(i): ``p_stale`` rules out "the output is a pure function of *this* call, carrying no leaked
    state or accidental memoization". A TEST detects it by calling the function twice and seeing two
    results. The wrapper used to hold its previous return for the whole evaluation, across every test
    the evaluation ran, so two tests that each call once and each pass alone killed it together, the
    kill was credited to whichever ran second, and the verdict changed with test order and with which
    subset ran. That is state the HARNESS leaks across a test boundary, not a property of the function.

      first    the wrapper has answered no call yet: nothing to carry. The first call of a mutant has no
               previous one, and answering it with None would repeat return_none's question.
      reset    a new test has begun since the last call: the memory ends with the test that made it.
      carry    the same test is still running: the previous return stands, and a second call in it that
               returns a different result is the detection.
    """
    if last_epoch is None:
        return "first"
    return "carry" if last_epoch == epoch else "reset"


def _stale_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    # The previous call's return, held from one call to the next WITHIN ONE TEST (#49): it is dropped
    # at every test boundary (`begin_test`), so a stale kill needs two calls in one test, is credited to
    # that test, and does not depend on test order. Empty until a call returns, so the FIRST call of a
    # test returns its own value.
    previous: list = []
    last_epoch: list = [None]

    def swap(value: Any) -> Any:
        epoch = _TEST_EPOCH[0]
        if stale_memory_standing(last_epoch[0], epoch) == "reset":
            previous.clear()
        last_epoch[0] = epoch
        out = previous[0] if previous else value
        previous[:] = [value]
        return out

    if inspect.iscoroutinefunction(fn):

        async def acall(*args: Any, **kwargs: Any) -> Any:
            return swap(await fn(*args, **kwargs))

        return _like(acall, fn)

    def call(*args: Any, **kwargs: Any) -> Any:
        return swap(fn(*args, **kwargs))

    return _like(call, fn)


def _swallow_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    # `Exception`, never `BaseException`: the engine's own interrupts (Abandoned, MeasurementRefused,
    # StepBudgetExceeded) derive from BaseException precisely so that code like this cannot eat them.
    if inspect.iscoroutinefunction(fn):

        async def acall(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            # BLE001: swallowing every ordinary raise IS the perturbation
            except Exception:  # noqa: BLE001
                return None

        return _like(acall, fn)

    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        # BLE001: swallowing every ordinary raise IS the perturbation
        except Exception:  # noqa: BLE001
            return None

    return _like(call, fn)


def form_b(mode: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """The Form B perturbation ``mode`` as a decorator ``fn -> perturbed fn``.

    The shape is preserved: a generator stays a generator, an async generator an async generator,
    and a coroutine function a coroutine function, so the call site cannot tell the perturbation
    from the function by how it must be called. An unknown mode raises at decoration time. That
    makes the mutant un-buildable (harness_error), never a silent no-op that would read as a
    survivor."""
    if mode in YIELD_MODES:
        return lambda fn: _yield_wrapper(fn, mode)
    if mode == "return_stale":
        return _stale_wrapper
    if mode == "raise_swallow":
        return _swallow_wrapper
    raise ValueError(f"not a μ⁻ Form B mode: {mode!r}")


# ── The catalog: every operator of Def. 11.8, and what realizes it ─────────────────────────────


@dataclass(frozen=True)
class MuMinusOperator:
    """One operator of Def. 11.8's perturbation family Π, and its standing in this engine.

    ``status`` is one of four named states. They are not collapsible into "supported or not",
    because each tells a reader of a two-sign certificate something different:

    * ``built``       — generated: ``modes`` are the sub-modes that realize it;
    * ``partial``     — some of its variants generated (``modes``); ``note`` names the rest and why;
    * ``realized_by`` — an existing operator already asks exactly this question (``modes`` names
      it), so a separate mutant would duplicate it;
    * ``unsupported`` — not generated; ``note`` says why it cannot be built soundly.

    ``modes`` entries are category-qualified (``OUTPUT:return_drop``, ``STATE:remove_assign``)."""

    key: str
    family: str
    form: str
    status: str
    modes: tuple[str, ...]
    note: str = ""


_I, _II, _III, _IV = "(i) universal", "(ii) boolean", "(iii) numeric", "(iv) string"
_V, _VI, _VII, _VIII = (
    "(v) container",
    "(vi) structured",
    "(vii) ordered",
    "(viii) effect",
)

MU_MINUS_CATALOG: tuple[MuMinusOperator, ...] = (
    MuMinusOperator("none", _I, "A", "built", ("OUTPUT:return_none",)),
    MuMinusOperator("const", _I, "A", "built", ("OUTPUT:return_const",)),
    MuMinusOperator("id", _I, "A", "built", ("OUTPUT:return_identity",)),
    MuMinusOperator(
        "zero-of",
        _I,
        "A",
        "built",
        ("OUTPUT:return_zero_of",),
        "float, complex, Decimal, Fraction; an int's zero is return_const's 0 and a sized "
        "type's is return_empty",
    ),
    MuMinusOperator(
        "type",
        _I,
        "A",
        "built",
        ("OUTPUT:return_type",),
        "the same value as another type: str -> bytes, bytes -> str, bool -> int, "
        "int/float -> Decimal",
    ),
    MuMinusOperator(
        "stale",
        _I,
        "B",
        "built",
        ("OUTPUT:return_stale",),
        "the previous call's return within one evaluation; the first call returns its own",
    ),
    MuMinusOperator("not", _II, "A", "built", ("OUTPUT:return_not",)),
    MuMinusOperator("true", _II, "A", "built", ("OUTPUT:return_true",)),
    MuMinusOperator("false", _II, "A", "built", ("OUTPUT:return_false",)),
    MuMinusOperator("neg", _III, "A", "built", ("OUTPUT:return_negate",)),
    MuMinusOperator("abs", _III, "A", "built", ("OUTPUT:return_abs",)),
    MuMinusOperator(
        "±1", _III, "A", "built", ("OUTPUT:return_plus_one", "OUTPUT:return_minus_one")
    ),
    MuMinusOperator(
        "±ε",
        _III,
        "A",
        "built",
        ("OUTPUT:return_next_up", "OUTPUT:return_next_down"),
        "the adjacent float (math.nextafter)",
    ),
    MuMinusOperator("scale", _III, "A", "built", ("OUTPUT:return_scale",), "k = 2"),
    MuMinusOperator(
        "round", _III, "A", "built", ("OUTPUT:return_round",), "to a whole number"
    ),
    MuMinusOperator("NaN", _III, "A", "built", ("OUTPUT:return_nan",)),
    MuMinusOperator(
        "inf",
        _III,
        "A",
        "built",
        ("OUTPUT:return_inf",),
        "+inf; -inf asks the same finiteness question",
    ),
    MuMinusOperator(
        "0",
        _III,
        "A",
        "realized_by",
        ("OUTPUT:return_const",),
        "return_const IS v -> 0",
    ),
    MuMinusOperator(
        "empty",
        _IV,
        "A",
        "built",
        ("OUTPUT:return_empty",),
        "also (v)'s container empty: one operator over str, bytes and every sized container",
    ),
    MuMinusOperator(
        "trunc",
        _IV,
        "A",
        "built",
        ("OUTPUT:return_truncate",),
        "s[:-1]; str and bytes",
    ),
    MuMinusOperator(
        "case",
        _IV,
        "A",
        "built",
        ("OUTPUT:return_swapcase",),
        "swapcase; str and bytes",
    ),
    MuMinusOperator(
        "ws",
        _IV,
        "A",
        "built",
        ("OUTPUT:return_pad",),
        "pad with a trailing space; strip is a no-op on most values, so pad is the total member",
    ),
    MuMinusOperator(
        "rev", _IV, "A", "built", ("OUTPUT:return_reverse",), "str and bytes"
    ),
    MuMinusOperator(
        "perm",
        _V,
        "A",
        "built",
        ("OUTPUT:return_reorder",),
        "reversal of a list or tuple; a no-op under set and dict-key semantics",
    ),
    MuMinusOperator(
        "drop",
        _V,
        "A",
        "built",
        ("OUTPUT:return_drop",),
        "the last element (list, tuple), the last key (dict), the least-repr element (set)",
    ),
    MuMinusOperator(
        "dup",
        _V,
        "A",
        "built",
        ("OUTPUT:return_duplicate",),
        "the last element repeated",
    ),
    MuMinusOperator(
        "singleton",
        _V,
        "A",
        "built",
        ("OUTPUT:return_singleton",),
        "the first element (list, tuple), the least-repr element (set)",
    ),
    MuMinusOperator(
        "ctype",
        _V,
        "A",
        "built",
        ("OUTPUT:return_ctype",),
        "list <-> tuple and set <-> frozenset; list -> set is not total (unhashable "
        "elements) and set -> list has no deterministic order",
    ),
    MuMinusOperator(
        "dict",
        _V,
        "A",
        "partial",
        ("OUTPUT:return_drop", "OUTPUT:return_field_default"),
        "drop-field (the last key) and default-field (the last key's value -> None); the "
        "key/value swap raises on an unhashable value, so it is not total on dict",
    ),
    MuMinusOperator(
        "field",
        _VI,
        "A",
        "unsupported",
        (),
        "the observed codomain is a bare type name: it cannot show that a return is a "
        "record, nor its fields' types, so neither applicability nor the field's own "
        "family is known before generation",
    ),
    MuMinusOperator(
        "class",
        _VI,
        "A/B",
        "unsupported",
        (),
        "a super- or sub-type instance has to be constructed, and no total construction "
        "follows from a type name",
    ),
    MuMinusOperator(
        "adj",
        _VII,
        "A",
        "built",
        ("OUTPUT:return_adjacent",),
        "date, datetime, timedelta; int and float adjacency is ±1 and ±ε",
    ),
    MuMinusOperator(
        "clamp",
        _VII,
        "A",
        "unsupported",
        (),
        "clip(v, l, u) needs the range the invariant names; no premise for it is observed",
    ),
    MuMinusOperator(
        "sortbreak",
        _VII,
        "A",
        "built",
        ("OUTPUT:return_sortbreak",),
        "transposes the first two elements of a list or tuple",
    ),
    MuMinusOperator(
        "yield",
        _VIII,
        "B",
        "built",
        tuple(f"OUTPUT:{mode}" for mode in YIELD_MODES),
        "generators and async generators",
    ),
    MuMinusOperator(
        "raise",
        _VIII,
        "B",
        "partial",
        ("OUTPUT:raise_swallow",),
        "swallow, where the function's own body raises; an injected raise fails every "
        "test that reaches the call, so its kill measures reach, not specification",
    ),
    MuMinusOperator(
        "state-noop",
        _VIII,
        "B",
        "realized_by",
        ("STATE:remove_assign", "STMT"),
        "omitting a side effect is the positive STATE remove_assign and STMT deletion",
    ),
    MuMinusOperator(
        "state-extra",
        _VIII,
        "B",
        "unsupported",
        (),
        "an extra mutation must choose which object and attribute to change; the function "
        "determines no such target",
    ),
)
