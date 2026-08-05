"""Type-impossible arithmetic exclusion (issue #12, policy 5).

The anchor is the field witness: a string-builder's `'if ' + ' and
'.join(...)` drew Add→Sub mutants that can only TypeError, and eight of them
padded the certificate's unproven-equivalent bucket ("COMPLETE modulo 12"
where the honest number was 4). A crash-only-BY-TYPE mutant measures
reachability, not specification — the same principle that keeps first-binding
deletions and non-mutable constants out of the universe. The exclusion trusts
only PROOF: a wrongly-excluded live mutant is under-enumeration, the one
failure the completeness claim cannot survive, so every boundary case here
pins the conservative direction.
"""

from __future__ import annotations

import ast
import textwrap

from Wesker.engine import (
    MutationCategory,
    _statically_str,
    _type_impossible_swap,
    estimate_universe_size,
    generate_mutants,
)


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _arith_dims(src: str) -> list[str]:
    return [
        m.dimension
        for m in generate_mutants(
            _fn(src), {MutationCategory.ARITHMETIC}, max_per_category=0
        )
    ]


def test_the_field_witness_is_out_of_the_universe():
    # `'if ' - ' and '.join(...)` can only raise; it is not a dimension.
    fn = _fn(
        """
        def serialize(parts):
            return 'if ' + ' and '.join(parts) + ' then done'
        """
    )
    assert estimate_universe_size(fn, {MutationCategory.ARITHMETIC}) == 0


def test_one_sided_str_add_stays_in():
    # `'a' + x` can meet a custom __radd__ — the swap is behavioral until
    # BOTH sides are proven str. The conservative direction is IN.
    assert _arith_dims("def f(x):\n    return 'a' + x\n") == ["ARITHMETIC:Add"]
    assert _arith_dims("def f(x, y):\n    return x + y\n") == ["ARITHMETIC:Add"]


def test_str_repetition_mult_is_out_but_bool_literal_stays():
    assert _arith_dims("def f():\n    return '-' * 40\n") == []
    assert _arith_dims("def f():\n    return 40 * '-'\n") == []
    # bool is an int, but a bool literal is not the proven-int case.
    assert _arith_dims("def f():\n    return '-' * True\n") == ["ARITHMETIC:Mult"]


def test_statically_str_proof_surface():
    def expr(src: str) -> ast.AST:
        return ast.parse(src, mode="eval").body

    assert _statically_str(expr("'a'"))
    assert _statically_str(expr("f'{x}!'"))
    assert _statically_str(expr("' and '.join(xs)"))
    assert _statically_str(expr("'a'.upper().strip()"))
    assert _statically_str(expr("'a' + 'b'"))
    # Unprovable shapes — names, annotations don't exist here, non-str
    # receivers, one-sided adds — must all answer False.
    assert not _statically_str(expr("s"))
    assert not _statically_str(expr("xs.join(s)"))
    assert not _statically_str(expr("'a' + x"))
    assert not _statically_str(expr("str(x)"))


def test_type_impossible_is_add_and_mult_only():
    # `'%s' % x` -> `'%s' * x` is VALID when x is an int — Mod swaps stay in,
    # even with a str-literal left and an int-literal right.
    assert _arith_dims("def f(x):\n    return '%s' % x\n") == ["ARITHMETIC:Mod"]
    assert _arith_dims("def f():\n    return 'a%d' % 3\n") == ["ARITHMETIC:Mod"]
    tmpl = ast.parse("'a' % 'b'", mode="eval").body
    assert isinstance(tmpl, ast.BinOp)
    assert not _type_impossible_swap(tmpl)


def test_half_proven_operands_stay_in():
    # Every one-sided proof keeps the dimension: int-literal × name, name ×
    # str-literal — the other operand may carry the dunder that makes the
    # swap behavioral.
    assert _arith_dims("def f(x):\n    return 3 * x\n") == ["ARITHMETIC:Mult"]
    assert _arith_dims("def f(x):\n    return x * 40\n") == ["ARITHMETIC:Mult"]
    assert _arith_dims("def f(x):\n    return x * '-'\n") == ["ARITHMETIC:Mult"]
    assert _arith_dims("def f(x):\n    return x + 'b'\n") == ["ARITHMETIC:Add"]


def test_augassign_never_excluded_in_v1():
    # A target NAME's str-ness needs flow analysis; v1 keeps every AugAssign.
    assert _arith_dims("def f(s):\n    s += 'x'\n    return s\n") == ["ARITHMETIC:Add"]
