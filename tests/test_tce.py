"""Trivial Compiler Equivalence: sound one way, silent the other (issue #24).

Identical bytecode cannot behave differently, so a True here is a PROOF and may promote a
mutant out of `candidate-equivalent — UNPROVEN`. Different bytecode says nothing at all, so a
False is never evidence of inequivalence. Everything below defends that asymmetry: the
positive cases must be real catches, and the negative cases must never become True.

The most dangerous failure mode is not missing a catch — it is an unsound True, because that
would let a mutant the tests genuinely fail to detect be written off as equivalent, silently
inflating a completeness claim. `0 == False` and `1 == 1.0` are true in Python while the
compiled constants are distinguishable, which is why constants are compared by type AND value.
"""

from __future__ import annotations

from Wesker.tce import bytecode_equivalent


# ── proven equivalences: mutations the compiler folds away ──────────────────────


def test_constant_folding_is_proven_equivalent():
    """The canonical catch. The peephole optimiser evaluates both to the same constant, so no
    input can distinguish them and asking a user for one is asking for the impossible."""
    assert bytecode_equivalent("x = 1 + 1", "x = 2") is True


def test_a_source_is_equivalent_to_itself():
    """The floor. If this ever fails, every other answer is suspect."""
    src = "def f(n):\n    return n * 2\n"
    assert bytecode_equivalent(src, src) is True


def test_nested_code_objects_do_not_defeat_the_comparison():
    """A comprehension compiles to a fresh nested code object every time, so comparing consts
    by identity would call every function containing one 'different' and the check would be
    dead weight on exactly the code most worth checking."""
    src = "def f(xs):\n    return [x + 1 for x in xs]\n"
    assert bytecode_equivalent(src, src) is True


# ── real differences: must never be claimed equivalent ──────────────────────────


def test_a_different_return_value_is_not_equivalent():
    assert (
        bytecode_equivalent("def f():\n    return 1", "def f():\n    return 2") is False
    )


def test_bool_and_int_constants_are_not_conflated():
    """`0 == False` is true in Python. Comparing constants by equality alone would call this
    mutation equivalent, and it is not — the interpreter loads a different object."""
    assert bytecode_equivalent("x = 0", "x = False") is False


def test_table_resident_constants_are_compared():
    """The trap for a `co_code`-only check. Small ints are inline operands, but larger ones live
    in `co_consts` — so these two emit the SAME instruction bytes and differ only in the table.
    An implementation that compared `co_code` alone would call them equivalent, which is the
    unsound direction: a mutant the tests genuinely fail to detect written off as impossible."""
    assert bytecode_equivalent("x = 1000", "x = 1001") is False


def test_int_and_float_constants_are_not_conflated():
    """`1 == 1.0` is true in Python; the constants are not the same object."""
    assert bytecode_equivalent("x = 1", "x = 1.0") is False


def test_a_nested_difference_is_found_through_the_recursion():
    """The recursion has to be a real comparison, not a rubber stamp: a difference INSIDE a
    nested code object is still a difference."""
    a = "def f(xs):\n    return [x + 1 for x in xs]\n"
    b = "def f(xs):\n    return [x + 2 for x in xs]\n"
    assert bytecode_equivalent(a, b) is False


def test_a_renamed_global_is_not_equivalent():
    """`co_names` is compared: the instructions are byte-identical here and only the name
    table differs, which is exactly the case a `co_code`-only check would get wrong."""
    assert (
        bytecode_equivalent("def f():\n    return a", "def f():\n    return b") is False
    )


# ── the refusal path ───────────────────────────────────────────────────────────


def test_uncompilable_input_claims_nothing():
    """A mutant that does not compile is a harness error upstream (#18), not an equivalence.
    Returning False rather than raising keeps a cheap optimisation from becoming the thing
    that crashes a run — and False claims nothing, which is the honest answer."""
    assert bytecode_equivalent("def f(): pass", "def f(: pass") is False
    assert bytecode_equivalent("def f(: pass", "def f(): pass") is False


def test_constant_identity_is_typed_and_structural_not_repr():
    """#24 hardening: constants key by a recursive TYPED structure, never repr(). A True must be a
    proof, so no two distinguishable constants may collide, and the key must be stable."""
    # 0.0 == -0.0 in Python, but the loaded constants differ — the bit pattern must separate them.
    assert bytecode_equivalent("x = 0.0", "x = -0.0") is False
    # a genuinely identical constant (including nested container types) stays a catch, stably.
    assert (
        bytecode_equivalent("x = (1.0, 'a', b'z', None)", "x = (1.0, 'a', b'z', None)")
        is True
    )
    # the canonical constant-fold catch survives the finer key.
    assert bytecode_equivalent("x = 1 + 1", "x = 2") is True


def test_exception_regions_are_part_of_the_key():
    """#24: `dis.get_instructions` does not surface the zero-cost exception table (3.11+), so two
    functions with an identical stream but different try/finally/except regions — behaviourally
    different when the body raises — must not key identically. `co_exceptiontable` closes that."""
    fin = "def g(f):\n    try:\n        x = f()\n    finally:\n        y = 2\n    return x\n"
    lin = "def g(f):\n    x = f()\n    y = 2\n    return x\n"
    assert bytecode_equivalent(fin, lin) is False
