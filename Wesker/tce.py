"""Trivial Compiler Equivalence — sound, execution-free detection of equivalent mutants.

A share of the `candidate-equivalent — UNPROVEN` pile is decidable for free: if two programs
compile to identical code objects they cannot behave differently, because the interpreter has
nothing left to tell them apart. Equivalence is undecidable in general (Budd & Angluin, 1982),
so the UNPROVEN posture stays for everything else — this does not decide the hard cases, it
removes the trivial ones before they cost a user attention.

> Citations recalled, not looked up. Verify before quoting publicly.

SOUND IN THE ONLY DIRECTION THAT MATTERS. Identical bytecode cannot behave differently, so this
never false-claims equivalence and cannot corrupt a certificate. The converse is NOT claimed:
different bytecode says nothing at all, and a `False` here is not evidence of inequivalence.
That asymmetry is the whole reason this is safe to admit to a proof tool.

WHY IT EARNS ITS PLACE HERE: today a bytecode-identical mutant reaches the user as "supply an
input that distinguishes this" — a request that provably cannot be satisfied. Every one of
those removed is a demand for work that does not exist.

THE COMPARISON IS OVER THE RESOLVED INSTRUCTION STREAM, not the raw code-object fields, and
that is forced rather than stylistic. Measured on this interpreter:

    compile("x = 1 + 1")  ->  co_consts == (1, None)
    compile("x = 2")      ->  co_consts == (2, None)

with BYTE-IDENTICAL ``co_code`` and identical disassembly — a small int is an inline
``LOAD_SMALL_INT`` operand and never reaches the constant table, so the folded ``1`` is dead
residue nothing references. Comparing ``co_consts`` therefore REJECTS the most canonical
equivalence this exists to catch; comparing ``co_code`` alone would ACCEPT ``x = 1000`` versus
``x = 1001``, which differ only in the table. Neither field decides it. Resolving each operand
through ``dis.get_instructions`` compares what the interpreter will actually do.

Position (``co_firstlineno``, ``co_filename``) is ignored: a mutant is compiled from a rewritten
AST, so line numbers routinely differ while the emitted instructions do not. Nested code objects
recurse rather than compare by identity, since a comprehension or nested def compiles to a fresh
object every time. The calling convention — argument counts, flags, var/free/cell names — is
compared verbatim. Operand TYPE is part of the key alongside value, because ``0 == False`` and
``1 == 1.0`` are true in Python while the loaded objects are distinguishable.
"""

from __future__ import annotations

import ast
import dis
import struct
import types
from typing import Any

#: The warrant a proven equivalence carries. A boundary-probe agreement is a HEURISTIC — it
#: says no input Wesker tried distinguished the two — while this is a proof. They must never
#: share a label: promoting `candidate-equivalent — UNPROVEN` to `equivalent` by assertion is
#: the one discipline the whole tool's credibility rests on.
WARRANT_BYTECODE = "bytecode_identity"


def _op_key(ins: Any) -> tuple:
    """One instruction reduced to what the interpreter will DO with it.

    Keyed on the RESOLVED operand (`argval`) rather than the raw index, because an index is a
    position in a table and two identical programs may lay their tables out differently. This
    is not a refinement — comparing `co_consts` directly is WRONG on current CPython:

        compile("x = 1 + 1")  ->  co_consts == (1, None)
        compile("x = 2")      ->  co_consts == (2, None)

    with BYTE-IDENTICAL `co_code`, because a small int is an inline `LOAD_SMALL_INT` operand
    and never enters the table at all — so the folded `1` is dead residue nothing references.
    Strict const equality therefore rejects the single most canonical equivalence TCE exists to
    catch, while `co_code` equality alone would ACCEPT `x = 1000` vs `x = 1001`, which differ
    only in the table. Neither field decides it; the resolved instruction stream does.

    Type is part of the key alongside value. `0 == False` and `1 == 1.0` are true in Python
    while the loaded objects are distinguishable, and a bare equality test would call those
    mutations equivalent when they are not — the one way this could produce an unsound True.
    """
    return (ins.opname, _const_key(ins.argval))


def _const_key(v: Any) -> tuple:
    """A constant reduced to a stable, typed, STRUCTURAL identity — never ``repr()`` (#24).

    ``repr()`` was two hazards in a proof tool. It is UNSTABLE: a code object (or a container of
    one) reprs with a memory address, so the same program keys differently run to run. And it can
    COLLIDE: for exotic or lossy values two distinguishable constants can share a repr, which is the
    one way this could produce an UNSOUND ``True``. Instead recurse structurally with the type in the
    key at every level:

    * a code object recurses through :func:`_code_key`;
    * a tuple recurses element-wise; a frozenset is order-canonicalised first (it is unordered);
    * a float / complex uses its exact IEEE-754 bit pattern, so ``0.0`` vs ``-0.0`` and distinct
      NaNs are distinguished (``==`` calls them equal, the bytes do not);
    * ``bool`` is keyed before ``int`` (it is a subclass) so ``0`` and ``False`` never merge, and
      ``int`` / ``str`` / ``bytes`` / ``None`` carry their own value beside their type name, so
      ``1`` and ``1.0`` stay distinct.

    An argval that is none of these (an exotic operand of some opcode) falls back to ``(type, repr)``
    — a last resort that cannot be less precise than the old universal repr, only more."""
    if isinstance(v, types.CodeType):
        return ("code", _code_key(v))
    if isinstance(
        v, bool
    ):  # before int: bool is an int subclass, and 0/False must not merge
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        return ("float", struct.pack(">d", v))
    if isinstance(v, complex):
        return ("complex", struct.pack(">d", v.real), struct.pack(">d", v.imag))
    if isinstance(v, (str, bytes)):
        return (type(v).__name__, v)
    if v is None:
        return ("none",)
    if isinstance(v, tuple):
        return ("tuple", tuple(_const_key(e) for e in v))
    if isinstance(v, frozenset):
        return ("frozenset", tuple(sorted((_const_key(e) for e in v), key=repr)))
    return (type(v).__name__, repr(v))


def _code_key(code: types.CodeType) -> tuple:
    """A code object reduced to its behaviour: calling convention plus resolved instructions.

    Nested code objects recurse through :func:`_op_key`, since a comprehension or nested def
    compiles to a fresh object every time and comparing by identity would make every function
    containing one trivially 'different'.
    """
    return (
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_posonlyargcount,
        code.co_flags,
        # Stack shape and the EXCEPTION TABLE (#24). `dis.get_instructions` surfaces the instruction
        # stream but NOT the zero-cost exception table (3.11+), so two functions with an identical
        # stream but different try/finally/except* regions — behaviourally different when something
        # raises — would key identically without this. Adding it can only make the key STRICTER
        # (fewer equivalences claimed), never less sound. `getattr` tolerates pre-3.11 interpreters.
        code.co_stacksize,
        getattr(code, "co_exceptiontable", b""),
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(_op_key(i) for i in dis.get_instructions(code)),
    )


def code_identical(a: types.CodeType, b: types.CodeType) -> bool:
    """Whether two code objects are indistinguishable to the interpreter.

    Position is ignored (see the module docstring); everything the evaluation loop reads is
    compared, through the resolved instruction stream rather than the raw tables.
    """
    return _code_key(a) == _code_key(b)


def bytecode_equivalent(original_src: str, mutant_src: str) -> bool:
    """True when two sources compile to indistinguishable code — a PROVEN equivalence.

    The entry point, and deliberately typed over SOURCE rather than code objects so the whole
    contract sits inside a literal grammar and can be pinned.

    A compile failure returns False rather than raising. This runs over generated mutants, and a
    mutant that does not compile is already handled as a harness error upstream (#18); making
    equivalence detection the thing that crashes a run would trade a cheap optimisation for a
    liability. False is also the safe answer: it claims nothing.

    Expected catches are mutations the compiler folds away — constant arithmetic it evaluates at
    compile time, reorderings the peephole optimiser normalises, and dead-branch rewrites such as
    `if False and ...`. The hit rate is a fact about the operator universe worth measuring on a
    real corpus, but the value does not depend on it: every catch is a question removed that the
    user could not have answered.
    """
    try:
        a = compile(original_src, "<tce>", "exec")
        b = compile(mutant_src, "<tce>", "exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return code_identical(a, b)


def nodes_equivalent(original_node: Any, mutant_node: Any) -> bool:
    """:func:`bytecode_equivalent` over two AST nodes — the engine-facing adapter.

    Thin on purpose: the decision lives in `bytecode_equivalent`, typed over source so it sits
    inside a literal grammar and can be pinned. This only unparses.

    Any failure returns False. `ast.unparse` can raise on a malformed node, and a mutant built
    from a rewritten AST is exactly where malformed nodes appear — but an equivalence check is
    an optimisation, and one that can fail a run is a liability. False claims nothing.
    """
    try:
        return bytecode_equivalent(ast.unparse(original_node), ast.unparse(mutant_node))
    except Exception:  # noqa: BLE001 — an optimisation must never fail the run
        return False
