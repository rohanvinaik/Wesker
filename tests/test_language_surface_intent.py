"""Intent tests for the language-surface census — and its CI enforcement.

Authored from INTENT. `Wesker/language_surface.py` is a set of CLAIMS about what the fault model
covers, and a table of claims that nothing checks is worse than no table: it reads as coverage
while being whatever somebody last typed. Two of these tests are therefore adversarial against the
census itself rather than against user code.

WHAT THE CENSUS IS FOR. `mutation_policy()` already declared `categories` and `exclusions`. That
is a NUMERATOR — what somebody thought to name. It had no denominator, so the question "is there a
semantic surface of Python no category mentions at all?" had no mechanical answer. This census is
derived from CPython's own `ast` grammar, so the denominator tracks the language rather than a
hand-kept idea of it, and `test_the_running_interpreters_surface_is_fully_declared` is the gate:
Python adds a node, the build fails until somebody decides what that node means.

IT PAID FOR ITSELF ON THE FIRST RUN, twice.

  * Six binary operators (MatMult, LShift, RShift, BitAnd, BitOr, BitXor) and two unary ones
    (Invert, UAdd) have no operator family at all. They were never withheld and never declared —
    simply absent, which is indistinguishable from coverage until something counts.
  * The first version of the census was exhaustive on 3.14 and FAILED on 3.11/3.12/3.13, which
    still carry the pre-3.8 constant aliases (`Num`, `Str`, `Bytes`, `NameConstant`, `Ellipsis`).
    A census that holds only on the author's interpreter is the exact subspace error this project
    exists to refuse, and the matrix check caught it before it could be believed.
"""

from __future__ import annotations

import ast

import pytest

from Wesker.language_surface import (
    DISPOSITIONS,
    REASONS,
    SURFACE_CENSUS,
    census_gaps,
    interpreter_surface,
    surface_disposition,
    surface_id,
    unsupported_slots,
)
from Wesker.policy import mutation_policy


# ── the gate ────────────────────────────────────────────────────────────────


def test_the_running_interpreters_surface_is_fully_declared() -> None:
    """THE CI ENFORCEMENT. Every semantic slot this Python exposes carries a disposition.

    Runs against the interpreter executing the tests, so Wesker's CI matrix (3.11/3.12/3.13) is
    what makes the claim hold across supported versions rather than on one machine.

    If this fails, the message is the whole point: a language construct exists that the fault
    model has never been asked about. The fix is a decision, not a suppression.
    """
    gaps = census_gaps(list(interpreter_surface()))
    assert not gaps, (
        f"{len(gaps)} semantic slot(s) in this Python carry no disposition: {list(gaps)}. "
        "Declare each in SURFACE_CENSUS — covered / withheld / not_applicable / unsupported."
    )


def test_declaring_ahead_of_this_interpreter_is_allowed() -> None:
    """The check is ONE-DIRECTIONAL on purpose.

    The census declares nodes that only exist on other versions (3.14's TemplateStr, 3.11's Num).
    Present-but-undeclared is the failure; declared-but-absent is how one table serves the whole
    support matrix. A two-directional check would make the census un-authorable.
    """
    absent = set(SURFACE_CENSUS) - set(interpreter_surface())
    assert census_gaps(list(interpreter_surface())) == ()
    assert absent, "expected some declarations for versions other than this one"


# ── adversarial against the census itself ───────────────────────────────────


def test_covered_operators_are_really_in_the_engines_tables() -> None:
    """The anti-fiction test, and the most important one here.

    A census is a claim about the ENGINE. Nothing stops someone typing `covered` next to a node
    the engine has never heard of, and the table would then read as coverage forever. So this
    checks the claim against `mutation_policy()`'s own derived tables — which are themselves read
    off `_ArithmeticMutator._BIN_SWAP` and friends, not prose.
    """
    cats = mutation_policy().categories
    binop_swaps = set(cats["ARITHMETIC"]["binop_swaps"])
    boundary = set(cats["BOUNDARY"]["alternatives"])
    boolops = set(cats["LOGICAL"]["boolop_swaps"])

    for node in _concrete_subclasses(ast.operator):
        claimed = surface_disposition(node) == "covered"
        assert claimed == (node in binop_swaps), (
            f"census says {node} is {surface_disposition(node)}, engine table says "
            f"{'present' if node in binop_swaps else 'absent'}"
        )
    for node in _concrete_subclasses(ast.cmpop):
        assert (surface_disposition(node) == "covered") == (node in boundary), node
    for node in _concrete_subclasses(ast.boolop):
        assert (surface_disposition(node) == "covered") == (node in boolops), node


def test_the_uncovered_operator_families_are_named_not_absent() -> None:
    """The census's first finding, pinned so it cannot silently regress in either direction.

    If somebody adds `@`, shifts or bitwise ops to `_BIN_SWAP`, the test above fails and this one
    must be updated deliberately — which is correct: gaining coverage is a policy change worth
    noticing, and so is losing it.
    """
    for node in (
        "MatMult",
        "LShift",
        "RShift",
        "BitAnd",
        "BitOr",
        "BitXor",
        "Invert",
        "UAdd",
    ):
        assert surface_disposition(node) == "unsupported", node
        assert REASONS.get(node), f"{node} is unsupported and must say why"


def test_comparisons_are_completely_covered() -> None:
    """The other half of the finding, and the reason the census is not merely a list of gaps:
    BOUNDARY turns out to be complete over every cmpop the grammar has."""
    for node in _concrete_subclasses(ast.cmpop):
        assert surface_disposition(node) == "covered", node


# ── the vocabulary ──────────────────────────────────────────────────────────


def test_every_disposition_is_one_of_the_four() -> None:
    """No fifth state, and never a bare bool: each disposition has a distinct consequence."""
    bad = {k: v for k, v in SURFACE_CENSUS.items() if v not in DISPOSITIONS}
    assert not bad, f"undeclared disposition values: {bad}"


def test_undeclared_is_not_the_same_as_unsupported() -> None:
    """The distinction that keeps the bookkeeping honest.

    `unsupported` is an admitted limit of the FAULT MODEL. `undeclared` is a limit of this MODULE.
    Collapsing them would hide every future bookkeeping gap behind an existing, respectable-looking
    admission — absence dressed as a negative result, which is the failure mode the whole project
    is organised against.
    """
    assert surface_disposition("Invert") == "unsupported"
    assert surface_disposition("SomeNodeFromPython4.field") == "undeclared"
    assert surface_disposition("") == "undeclared"
    assert "undeclared" not in DISPOSITIONS


def test_every_non_covered_slot_explains_itself() -> None:
    """A withheld or unsupported slot without a reason is an assertion, not a declaration."""
    missing = [
        k
        for k, v in SURFACE_CENSUS.items()
        if v in ("withheld", "unsupported", "not_applicable") and not REASONS.get(k)
    ]
    assert not missing, f"non-covered slots with no recorded reason: {missing[:12]}"


def test_the_unsupported_set_is_enumerable_by_name() -> None:
    """A consumer deciding whether a target sits inside the model needs the slots BY NAME. A count
    would be a summary of exactly the thing that must not be summarised."""
    slots = unsupported_slots()
    assert slots, (
        "the fault model does not cover everything; the list must not be empty"
    )
    assert all(surface_disposition(s) == "unsupported" for s in slots)
    assert slots == tuple(sorted(slots)), "stable order, so a diff is readable"


# ── identity, and what it must NOT disturb ──────────────────────────────────


def test_the_census_does_not_change_the_policy_id() -> None:
    """REGRESSION GUARD, and the reason the census sits outside the manifest digest.

    Wesker mutates exactly what it mutated before this census existed — the fault model did not
    change, only the bookkeeping about it. Folding the census into `policy_id` would invalidate
    every receipt and cached verdict in existence to record a change in documentation. These two
    ids were measured before the census landed and must not move because of it.
    """
    assert mutation_policy().policy_id == "5.751e8e9f4f11"
    assert mutation_policy(True).policy_id == "5+neg.ae3eae86a312"


def test_the_census_carries_its_own_identity() -> None:
    """A claim that depends on the coverage declaration keys on THIS, not on policy_id."""
    first = surface_id()
    assert surface_id() == first
    assert first.startswith("1.")
    assert mutation_policy().language_surface["surface_id"] == first


def test_the_policy_exposes_the_surface_for_consumers() -> None:
    """Detective reads this to decide whether a target sits inside the model."""
    ls = mutation_policy().language_surface
    assert ls["slots"] == len(SURFACE_CENSUS)
    assert set(ls["dispositions"]) <= set(DISPOSITIONS)
    assert ls["unsupported"] == list(unsupported_slots())


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("Add", "covered"),
        ("Call.args", "covered"),
        ("Constant.value", "covered"),
        ("Name.ctx", "not_applicable"),
        ("Attribute.attr", "withheld"),
        ("Await.value", "unsupported"),
    ],
)
def test_representative_dispositions(key: str, expected: str) -> None:
    """A spot-check across all four states, so a wholesale table edit cannot pass unnoticed."""
    assert surface_disposition(key) == expected


def _concrete_subclasses(base: type) -> list[str]:
    return [
        c.__name__
        for c in vars(ast).values()
        if isinstance(c, type)
        and issubclass(c, base)
        and c is not base
        and not c.__subclasses__()
    ]
