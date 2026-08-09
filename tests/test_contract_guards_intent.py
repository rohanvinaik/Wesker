"""Contracts that were true, unwritten, and enforced by nothing.

Four sites in this repo were correct only because of an invariant established somewhere else:
a different function, a different branch, or a hundred-odd lines away. `ty` found each of them
by asking a question no test could — "can this value be what the signature says it is, on EVERY
path" — and in all four the previous answer was a `# type: ignore` comment, which is not an
answer but a decision to stop asking.

None of these were live bugs. That is exactly the point: each was one refactor from becoming
one, and the reason it was safe existed nowhere a reader or a test could find it. The guards
below fail LOUDLY and by name, so a future edit that breaks the invariant says which invariant
it broke instead of raising `AttributeError: 'Return' object has no attribute 'target'` from a
line that never mentions the analysis that actually drifted.
"""

from __future__ import annotations

import ast

import pytest

from Wesker.engine import (
    LazySessionBaseline,
    _deletable_stmt_ids,
    _mutant_module,
    _stmt_label,
)
from Wesker.filter import _census_row


# ── _stmt_label: the labeller and the analysis must admit the same statement kinds ──


def test_the_four_deletable_kinds_all_get_a_label():
    """`_deletable_stmt_ids` admits exactly Expr, AugAssign, Assign and AnnAssign. Every one of
    them must label, or the operator drops a dimension it claimed to cover."""
    src = "def f(a, cfg):\n    a = 1\n    b: int = 2\n    a += 1\n    cfg[a] = 2\n    print(a)\n"
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)
    for stmt in fn.body:
        assert _stmt_label(stmt)  # non-empty label, no exception


def test_a_statement_kind_the_analysis_never_admits_is_named_not_dereferenced():
    """THE guard. `else: [node.target]` was safe only because `_deletable_stmt_ids` — 127 lines
    away — admits four kinds and no others. A `Return` has no `.target`, so drift surfaced as an
    AttributeError naming the attribute rather than the analysis."""
    stmt = ast.parse("return 1").body[0]
    with pytest.raises(TypeError, match="not a deletable statement kind"):
        _stmt_label(stmt)


def test_the_analysis_still_admits_only_labellable_kinds():
    """The two must not drift apart in EITHER direction. This asserts the coupling itself, so
    widening `_deletable_stmt_ids` without widening `_stmt_label` fails here rather than in a
    profiling run."""
    src = (
        "def f(a, cfg):\n"
        "    a = 1\n"
        "    b: int = 2\n"
        "    a += 1\n"
        "    cfg[a] = 2\n"
        "    print(a)\n"
        "    if a:\n"
        "        return a\n"
    )
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)
    deletable = _deletable_stmt_ids(fn)
    for stmt in ast.walk(fn):
        if not isinstance(stmt, ast.stmt):
            continue
        pos = (getattr(stmt, "lineno", -1), getattr(stmt, "col_offset", -1))
        if pos in deletable:
            assert _stmt_label(stmt), (
                f"admitted but unlabellable: {type(stmt).__name__}"
            )


# ── _mutant_module: one owner for a wrap that was written twice ──


def test_a_mutant_body_becomes_a_compilable_module():
    fn = ast.parse("def f():\n    return 1\n").body[0]
    mod = _mutant_module(fn)
    assert isinstance(mod, ast.Module)
    compile(ast.fix_missing_locations(mod), "<t>", "exec")


def test_a_mutant_body_that_is_not_a_statement_is_refused_by_name():
    """`Mutant.mutated_node` is typed `ast.AST`; `ast.Module(body=...)` needs statements. Both
    construction sites carried the same `type: ignore[list-item]` — the same silenced question
    written twice, which is how two call sites drift into two behaviours."""
    with pytest.raises(TypeError, match="must be a statement"):
        _mutant_module(ast.Name(id="x", ctx=ast.Load()))


# ── LazySessionBaseline: two fields encoding one state ──


def test_the_memo_builds_once_and_returns_the_value():
    calls = []

    def build():
        calls.append(1)
        return "baseline"

    lazy = LazySessionBaseline(build)
    assert lazy.built is False
    assert lazy.get() == "baseline"
    assert lazy.get() == "baseline"
    assert lazy.built is True
    assert len(calls) == 1, "the pass must run at most once"


def test_a_built_but_empty_memo_raises_instead_of_returning_none():
    """`_built` and `_value` encode ONE state in two fields, and only their agreement made
    `get() -> SessionBaseline` honest. A build closure returning None used to hand that None
    straight to a caller annotated otherwise — the failure would surface wherever the baseline
    was eventually read, not here."""
    lazy = LazySessionBaseline(lambda: None)
    with pytest.raises(RuntimeError, match="inconsistent"):
        lazy.get()


# ── _census_row: one shape, two producers ──


def test_the_disposition_comes_from_the_arguments_not_a_re_read():
    """Both counts positive: `generated` wins, and the withheld count still travels."""
    row = _census_row(3, 2)
    assert row["disposition"] == "generated"
    assert row["withheld"] == 2


def test_no_candidate_site_is_not_the_same_as_every_site_suppressed():
    """The distinction the census exists for: nothing to decide vs a judgement a reader must be
    able to disagree with."""
    assert _census_row(0, 0)["disposition"] == "not_applicable"
    assert _census_row(0, 4)["disposition"] == "withheld"


def test_both_producers_agree_on_shape_and_key_order():
    """The census is serialised, so key ORDER is observable. The STATE arm carries `sub_modes`
    and the other does not — that is the only difference permitted between them."""
    plain = _census_row(1, 0)
    stateful = _census_row(1, 0, {"remove_assign": {"targets": 0, "withheld_by": ""}})
    assert list(plain) == ["generated", "withheld", "disposition"]
    assert list(stateful) == ["generated", "withheld", "sub_modes", "disposition"]
    assert plain["disposition"] == stateful["disposition"]
