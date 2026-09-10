"""The Python language surface, and what the fault model says about every part of it.

WHY THIS EXISTS. Wesker's ``mutation_policy()`` already declares its categories and, in
``exclusions``, several deliberate withholdings. That list is honest and it is a NUMERATOR: it
records what somebody thought to exclude. It has never had a DENOMINATOR. Nothing answered "are
there semantic surfaces of Python that no category mentions at all", because nothing enumerated
the surfaces.

This module is the denominator. It is derived from CPython's own ``ast`` module — which is
generated from the ASDL grammar — so it tracks the language rather than a hand-kept idea of it.
For every concrete AST node and every semantic field, the census records exactly one disposition,
and a test asserts that the RUNNING interpreter's surface is fully declared. Python 3.15 adds a
node tomorrow, the build fails until somebody decides what that node means.

That converts the open-ended question

    "are there mutation categories we forgot?"

into the closed one

    "there are zero undeclared semantic fields in this Python under policy X."

WHAT IT FOUND IMMEDIATELY, which is the argument for having built it. Six binary operators
(``MatMult``, ``LShift``, ``RShift``, ``BitAnd``, ``BitOr``, ``BitXor``) and two unary operators
(``Invert``, ``UAdd``) have no operator family at all: ``_ArithmeticMutator._BIN_SWAP`` covers
seven of thirteen. They were not withheld and not declared — they were simply absent, and absence
is indistinguishable from coverage until something counts. Comparisons, by contrast, turn out to
be complete: all ten ``cmpop`` nodes are in BOUNDARY's table.

WHAT A DISPOSITION MEANS. The question a field answers is not "is this syntax important" but
"if THIS slot's own content were altered, does the fault model see it?" — where a slot holding a
child node or a list of them is usually answered by the walk reaching those children instead.

    covered         an operator family generates alternatives for this slot
    withheld        behaviour exists here and the policy deliberately excludes it, with a reason
    not_applicable  altering this slot cannot change the observation model (a container the walk
                    descends through, an annotation never executed, a context determined by
                    position, a deprecated alias the parser never emits)
    unsupported     behaviour exists here and no family models it — the honest gap. A certificate
                    that depends on this slot being pinned is not available.

`unsupported` is the load-bearing state, and it is deliberately not comfortable. The point is not
that Wesker covers everything; it is that nothing is UNCLASSIFIED. Below 100% is allowed. Unknown
territory is not.

SCOPE. Wesker mutates ONE FUNCTION, so module-level and class-level surfaces are declared
``not_applicable`` with that reason rather than omitted — the boundary of the tool is part of what
the census is for, and leaving them out would restore exactly the silence this closes.
"""

from __future__ import annotations

import ast
import hashlib

DISPOSITIONS = ("covered", "withheld", "not_applicable", "unsupported")

SURFACE_VERSION = 1
"""Bump on any change to a disposition or to the set of declared keys."""

# ── shared reasons, so the table states a rationale without restating prose ──

_RECURSES = (
    "a slot holding child nodes; the walk descends and the fault model applies there"
)
_ANNOTATION = "an annotation, never evaluated as behaviour by the runtime"
_TYPE_COMMENT = "a type comment; no runtime effect"
_CTX = "expression context is fixed by position; altering it yields an invalid tree"
_DEPRECATED = "a deprecated alias the parser never emits on this Python"
_NOT_A_FUNCTION_BODY = "outside the unit Wesker mutates: one function"
_TARGET_IDENTITY = "the target's own identity, fixed by the harness that selected it"
_NO_SHIFT_DUAL = "no ARITHMETIC dual; shifts are absent from _BIN_SWAP"
_NO_BITWISE_DUAL = "no ARITHMETIC dual; bitwise ops are absent from _BIN_SWAP"
_NO_IMPORT_REDIRECT = "no family redirects an import"
_MATCH_UNWALKED = "match/case bodies are declared unwalked"
_COMPREHENSION_BODY = "comprehension bodies are out of DATAFLOW's slice"
_COMPREHENSION_SHAPE = "comprehension iteration structure is unmodelled"
_NO_SIGNATURE_PERTURBATION = "signature perturbation is unmodelled"
_NO_LOOP_ELSE = "no family adds or removes a loop else-clause"
_CM_LIFETIME = "context-manager lifetime is unmodelled"
_NO_ASSIGN_TARGET_SUB = "DATAFLOW does not substitute assignment targets"
_NO_SLICE_BOUNDS = "no family perturbs slice bounds"
_NO_DECORATOR_MODEL = "a decorator can replace dispatch entirely; unmodelled"
_LOOP_TARGET = "loop targets are declared out of DATAFLOW's candidate set"
_KEYWORD_BINDING = "keyword binding is not perturbed"

REASONS: dict[str, str] = {}
"""Populated below alongside the census so every non-``covered`` entry can explain itself."""


def _d(key: str, disposition: str, reason: str = "") -> tuple[str, str]:
    if reason:
        REASONS[key] = reason
    return key, disposition


# ── the census ───────────────────────────────────────────────────────────────
#
# Keys are "Node.field", or a bare "Node" for a field-less node whose IDENTITY is the semantic
# choice (`Add` versus `Sub` is the whole mutation; there is no field to point at).

# Built from an explicit list so every key appears literally: adding a node to the emitter
# without declaring it here fails the exhaustiveness test rather than defaulting silently.
_ENTRIES: list[tuple[str, str]] = [
    # ── operator identities: ARITHMETIC's _BIN_SWAP ─────────────────────
    _d("Add", "covered"),
    _d("Sub", "covered"),
    _d("Mult", "covered"),
    _d("Div", "covered"),
    _d("FloorDiv", "covered"),
    _d("Mod", "covered"),
    _d("Pow", "covered"),
    # ── operator identities with NO family. The census's first finding. ──
    _d("MatMult", "unsupported", "no ARITHMETIC dual; @ is absent from _BIN_SWAP"),
    _d(
        "LShift",
        "unsupported",
        _NO_SHIFT_DUAL,
    ),
    _d(
        "RShift",
        "unsupported",
        _NO_SHIFT_DUAL,
    ),
    _d(
        "BitAnd",
        "unsupported",
        _NO_BITWISE_DUAL,
    ),
    _d(
        "BitOr",
        "unsupported",
        _NO_BITWISE_DUAL,
    ),
    _d(
        "BitXor",
        "unsupported",
        _NO_BITWISE_DUAL,
    ),
    # ── boolean and comparison identities ───────────────────────────────
    _d("And", "covered"),
    _d("Or", "covered"),
    _d("Eq", "covered"),
    _d("NotEq", "covered"),
    _d("Lt", "covered"),
    _d("LtE", "covered"),
    _d("Gt", "covered"),
    _d("GtE", "covered"),
    _d("Is", "covered"),
    _d("IsNot", "covered"),
    _d("In", "covered"),
    _d("NotIn", "covered"),
    # ── unary identities ────────────────────────────────────────────────
    _d("Not", "covered"),
    _d("USub", "covered"),
    _d("Invert", "unsupported", "no family removes or dualises ~"),
    _d("UAdd", "unsupported", "no family removes unary +"),
    # ── expression contexts and deprecated aliases ──────────────────────
    _d("Load", "not_applicable", _CTX),
    _d("Store", "not_applicable", _CTX),
    _d("Del", "not_applicable", _CTX),
    _d("AugLoad", "not_applicable", _DEPRECATED),
    _d("AugStore", "not_applicable", _DEPRECATED),
    _d("Param", "not_applicable", _DEPRECATED),
    _d("Suite", "not_applicable", _DEPRECATED),
    _d("Index", "not_applicable", _DEPRECATED),
    _d("ExtSlice", "not_applicable", _DEPRECATED),
    # Pre-3.8 constant aliases. Present on 3.11 (all six) and 3.12/3.13 (`Ellipsis` only),
    # gone by 3.14. Declared rather than filtered: the census is checked against the RUNNING
    # interpreter, and the first matrix run failed on exactly these — which is the mechanism
    # working, not a nuisance. Keeping them costs six lines and keeps 3.11 exhaustive.
    _d("Num.n", "not_applicable", _DEPRECATED),
    _d("Str.s", "not_applicable", _DEPRECATED),
    _d("Bytes.s", "not_applicable", _DEPRECATED),
    _d("NameConstant.value", "not_applicable", _DEPRECATED),
    _d("NameConstant.kind", "not_applicable", _DEPRECATED),
    _d("Ellipsis", "not_applicable", _DEPRECATED),
    # ── simple statements whose identity is the whole node ──────────────
    _d("Pass", "covered"),
    _d("Break", "covered"),
    _d("Continue", "covered"),
    # ── mod: outside the unit ───────────────────────────────────────────
    _d("Module.body", "not_applicable", _NOT_A_FUNCTION_BODY),
    _d("Module.type_ignores", "not_applicable", _TYPE_COMMENT),
    _d("Interactive.body", "not_applicable", _NOT_A_FUNCTION_BODY),
    _d("Expression.body", "not_applicable", _NOT_A_FUNCTION_BODY),
    _d("FunctionType.argtypes", "not_applicable", _ANNOTATION),
    _d("FunctionType.returns", "not_applicable", _ANNOTATION),
    # ── the function definition itself ──────────────────────────────────
    _d("FunctionDef.name", "not_applicable", _TARGET_IDENTITY),
    _d(
        "FunctionDef.args",
        "unsupported",
        "no family perturbs the signature (arity, defaults, order)",
    ),
    _d("FunctionDef.body", "not_applicable", _RECURSES),
    _d(
        "FunctionDef.decorator_list",
        "unsupported",
        _NO_DECORATOR_MODEL,
    ),
    _d("FunctionDef.returns", "not_applicable", _ANNOTATION),
    _d("FunctionDef.type_comment", "not_applicable", _TYPE_COMMENT),
    _d("FunctionDef.type_params", "not_applicable", _ANNOTATION),
    _d("AsyncFunctionDef.name", "not_applicable", _TARGET_IDENTITY),
    _d("AsyncFunctionDef.args", "unsupported", "no family perturbs the signature"),
    _d("AsyncFunctionDef.body", "not_applicable", _RECURSES),
    _d(
        "AsyncFunctionDef.decorator_list",
        "unsupported",
        _NO_DECORATOR_MODEL,
    ),
    _d("AsyncFunctionDef.returns", "not_applicable", _ANNOTATION),
    _d("AsyncFunctionDef.type_comment", "not_applicable", _TYPE_COMMENT),
    _d("AsyncFunctionDef.type_params", "not_applicable", _ANNOTATION),
    _d("ClassDef.name", "not_applicable", _NOT_A_FUNCTION_BODY),
    _d("ClassDef.bases", "not_applicable", _NOT_A_FUNCTION_BODY),
    _d(
        "ClassDef.keywords",
        "unsupported",
        "metaclass keywords can synthesise members; unmodelled",
    ),
    _d("ClassDef.body", "not_applicable", _NOT_A_FUNCTION_BODY),
    _d("ClassDef.decorator_list", "unsupported", _NO_DECORATOR_MODEL),
    _d("ClassDef.type_params", "not_applicable", _ANNOTATION),
    # ── statements Wesker's families reach ──────────────────────────────
    _d("Return.value", "covered"),
    _d(
        "Assign.targets",
        "withheld",
        "DATAFLOW's enrolled slice is return-name substitution only",
    ),
    _d("Assign.value", "not_applicable", _RECURSES),
    _d("Assign.type_comment", "not_applicable", _TYPE_COMMENT),
    _d(
        "AugAssign.target",
        "withheld",
        _NO_ASSIGN_TARGET_SUB,
    ),
    _d("AugAssign.op", "covered"),
    _d("AugAssign.value", "not_applicable", _RECURSES),
    _d(
        "AnnAssign.target",
        "withheld",
        _NO_ASSIGN_TARGET_SUB,
    ),
    _d("AnnAssign.annotation", "not_applicable", _ANNOTATION),
    _d("AnnAssign.value", "not_applicable", _RECURSES),
    _d("AnnAssign.simple", "not_applicable", "a parser flag, not a behaviour"),
    _d("Expr.value", "covered"),
    _d("Delete.targets", "withheld", "no family removes or redirects a del target"),
    _d("If.test", "not_applicable", _RECURSES),
    _d("If.body", "not_applicable", _RECURSES),
    _d("If.orelse", "not_applicable", _RECURSES),
    _d("While.test", "not_applicable", _RECURSES),
    _d("While.body", "not_applicable", _RECURSES),
    _d(
        "While.orelse",
        "unsupported",
        _NO_LOOP_ELSE,
    ),
    _d(
        "For.target",
        "withheld",
        _LOOP_TARGET,
    ),
    _d("For.iter", "not_applicable", _RECURSES),
    _d("For.body", "not_applicable", _RECURSES),
    _d("For.orelse", "unsupported", _NO_LOOP_ELSE),
    _d("For.type_comment", "not_applicable", _TYPE_COMMENT),
    _d(
        "AsyncFor.target",
        "withheld",
        _LOOP_TARGET,
    ),
    _d("AsyncFor.iter", "not_applicable", _RECURSES),
    _d("AsyncFor.body", "not_applicable", _RECURSES),
    _d(
        "AsyncFor.orelse",
        "unsupported",
        _NO_LOOP_ELSE,
    ),
    _d("AsyncFor.type_comment", "not_applicable", _TYPE_COMMENT),
    _d(
        "With.items",
        "unsupported",
        "context-manager lifetime and ordering are unmodelled",
    ),
    _d("With.body", "not_applicable", _RECURSES),
    _d("With.type_comment", "not_applicable", _TYPE_COMMENT),
    _d(
        "AsyncWith.items",
        "unsupported",
        "async context-manager lifetime is unmodelled",
    ),
    _d("AsyncWith.body", "not_applicable", _RECURSES),
    _d("AsyncWith.type_comment", "not_applicable", _TYPE_COMMENT),
    _d("Raise.exc", "covered"),
    _d(
        "Raise.cause",
        "unsupported",
        "no family alters an exception cause (`raise X from Y`)",
    ),
    _d("Try.body", "not_applicable", _RECURSES),
    _d("Try.handlers", "covered"),
    _d("Try.orelse", "unsupported", "no family adds or removes a try else-clause"),
    _d(
        "Try.finalbody",
        "unsupported",
        "no family alters a finally block's presence or content",
    ),
    _d("TryStar.body", "not_applicable", _RECURSES),
    _d(
        "TryStar.handlers",
        "unsupported",
        "exception GROUPS are not modelled by the EXCEPTION family",
    ),
    _d("TryStar.orelse", "unsupported", "no family alters a try* else-clause"),
    _d("TryStar.finalbody", "unsupported", "no family alters a try* finally block"),
    _d("Assert.test", "not_applicable", _RECURSES),
    _d(
        "Assert.msg",
        "not_applicable",
        "the message is observed only when the assert already fails",
    ),
    _d("Import.names", "unsupported", _NO_IMPORT_REDIRECT),
    _d("ImportFrom.module", "unsupported", _NO_IMPORT_REDIRECT),
    _d("ImportFrom.names", "unsupported", _NO_IMPORT_REDIRECT),
    _d("ImportFrom.level", "unsupported", "no family alters relative-import depth"),
    _d("Global.names", "unsupported", "global binding relocation is unmodelled"),
    _d("Nonlocal.names", "unsupported", "closure-cell relocation is unmodelled"),
    _d("Match.subject", "withheld", _MATCH_UNWALKED),
    _d("Match.cases", "withheld", _MATCH_UNWALKED),
    _d("TypeAlias.name", "not_applicable", _ANNOTATION),
    _d("TypeAlias.type_params", "not_applicable", _ANNOTATION),
    _d("TypeAlias.value", "not_applicable", _ANNOTATION),
    # ── expressions ─────────────────────────────────────────────────────
    _d("BinOp.left", "not_applicable", _RECURSES),
    _d("BinOp.op", "covered"),
    _d("BinOp.right", "not_applicable", _RECURSES),
    _d("BoolOp.op", "covered"),
    _d("BoolOp.values", "not_applicable", _RECURSES),
    _d("UnaryOp.op", "covered"),
    _d("UnaryOp.operand", "not_applicable", _RECURSES),
    _d("Compare.left", "not_applicable", _RECURSES),
    _d("Compare.ops", "covered"),
    _d("Compare.comparators", "not_applicable", _RECURSES),
    _d("Call.func", "covered"),
    _d("Call.args", "covered"),
    _d(
        "Call.keywords",
        "withheld",
        "keyword binding is not perturbed; SWAP transposes positionals only",
    ),
    _d("Constant.value", "covered"),
    _d(
        "Constant.kind",
        "not_applicable",
        "the u-prefix marker has no runtime effect",
    ),
    _d("Name.id", "covered"),
    _d("Name.ctx", "not_applicable", _CTX),
    _d("Attribute.value", "not_applicable", _RECURSES),
    _d(
        "Attribute.attr",
        "withheld",
        "attribute selectors are declared out of DATAFLOW's slice",
    ),
    _d("Attribute.ctx", "not_applicable", _CTX),
    _d("Subscript.value", "not_applicable", _RECURSES),
    _d(
        "Subscript.slice",
        "withheld",
        "subscript keys are declared out of DATAFLOW's slice",
    ),
    _d("Subscript.ctx", "not_applicable", _CTX),
    _d("Slice.lower", "unsupported", _NO_SLICE_BOUNDS),
    _d("Slice.upper", "unsupported", _NO_SLICE_BOUNDS),
    _d("Slice.step", "unsupported", "no family perturbs a slice step"),
    _d("Starred.value", "not_applicable", _RECURSES),
    _d("Starred.ctx", "not_applicable", _CTX),
    _d("Tuple.elts", "unsupported", "no family reorders or repacks a tuple display"),
    _d("Tuple.ctx", "not_applicable", _CTX),
    _d("List.elts", "unsupported", "no family reorders a list display"),
    _d("List.ctx", "not_applicable", _CTX),
    _d("Set.elts", "unsupported", "no family reorders a set display"),
    _d("Dict.keys", "unsupported", "no family perturbs dict key/value pairing"),
    _d("Dict.values", "unsupported", "no family perturbs dict key/value pairing"),
    _d("IfExp.test", "not_applicable", _RECURSES),
    _d(
        "IfExp.body",
        "unsupported",
        "no family swaps a conditional expression's arms",
    ),
    _d(
        "IfExp.orelse",
        "unsupported",
        "no family swaps a conditional expression's arms",
    ),
    _d("Lambda.args", "unsupported", "no family perturbs a lambda signature"),
    _d(
        "Lambda.body",
        "withheld",
        "loads inside lambdas are declared out of DATAFLOW's slice",
    ),
    _d("NamedExpr.target", "unsupported", "walrus binding is unmodelled"),
    _d("NamedExpr.value", "not_applicable", _RECURSES),
    _d(
        "Await.value",
        "unsupported",
        "await scheduling and cancellation are unmodelled",
    ),
    _d("Yield.value", "unsupported", "generator suspension is unmodelled"),
    _d(
        "YieldFrom.value",
        "unsupported",
        "delegated generator suspension is unmodelled",
    ),
    _d("JoinedStr.values", "unsupported", "f-string assembly is unmodelled"),
    _d("FormattedValue.value", "not_applicable", _RECURSES),
    _d(
        "FormattedValue.conversion",
        "unsupported",
        "no family alters an !r/!s conversion",
    ),
    _d(
        "FormattedValue.format_spec",
        "unsupported",
        "no family alters a format spec",
    ),
    _d(
        "TemplateStr.values",
        "unsupported",
        "PEP 750 template strings are unmodelled",
    ),
    _d("Interpolation.value", "not_applicable", _RECURSES),
    _d(
        "Interpolation.str",
        "unsupported",
        "template interpolation text is unmodelled",
    ),
    _d(
        "Interpolation.conversion",
        "unsupported",
        "no family alters an interpolation conversion",
    ),
    _d(
        "Interpolation.format_spec",
        "unsupported",
        "no family alters an interpolation format spec",
    ),
    _d(
        "ListComp.elt",
        "withheld",
        _COMPREHENSION_BODY,
    ),
    _d(
        "ListComp.generators",
        "unsupported",
        _COMPREHENSION_SHAPE,
    ),
    _d(
        "SetComp.elt",
        "withheld",
        _COMPREHENSION_BODY,
    ),
    _d(
        "SetComp.generators",
        "unsupported",
        _COMPREHENSION_SHAPE,
    ),
    _d(
        "DictComp.key",
        "withheld",
        _COMPREHENSION_BODY,
    ),
    _d(
        "DictComp.value",
        "withheld",
        _COMPREHENSION_BODY,
    ),
    _d(
        "DictComp.generators",
        "unsupported",
        _COMPREHENSION_SHAPE,
    ),
    _d(
        "GeneratorExp.elt",
        "withheld",
        _COMPREHENSION_BODY,
    ),
    _d(
        "GeneratorExp.generators",
        "unsupported",
        _COMPREHENSION_SHAPE,
    ),
    # ── helper nodes ────────────────────────────────────────────────────
    _d("ExceptHandler.type", "covered"),
    _d(
        "ExceptHandler.name",
        "unsupported",
        "no family rebinds the caught-exception name",
    ),
    _d("ExceptHandler.body", "covered"),
    _d(
        "arguments.posonlyargs",
        "unsupported",
        _NO_SIGNATURE_PERTURBATION,
    ),
    _d("arguments.args", "unsupported", _NO_SIGNATURE_PERTURBATION),
    _d("arguments.vararg", "unsupported", _NO_SIGNATURE_PERTURBATION),
    _d(
        "arguments.kwonlyargs",
        "unsupported",
        _NO_SIGNATURE_PERTURBATION,
    ),
    _d("arguments.kw_defaults", "unsupported", "default values are unmodelled"),
    _d("arguments.kwarg", "unsupported", _NO_SIGNATURE_PERTURBATION),
    _d("arguments.defaults", "unsupported", "default values are unmodelled"),
    _d("arg.arg", "unsupported", "parameter names are unmodelled"),
    _d("arg.annotation", "not_applicable", _ANNOTATION),
    _d("arg.type_comment", "not_applicable", _TYPE_COMMENT),
    _d("keyword.arg", "withheld", _KEYWORD_BINDING),
    _d("keyword.value", "not_applicable", _RECURSES),
    _d("alias.name", "unsupported", _NO_IMPORT_REDIRECT),
    _d("alias.asname", "unsupported", "no family redirects an import alias"),
    _d(
        "withitem.context_expr",
        "unsupported",
        _CM_LIFETIME,
    ),
    _d(
        "withitem.optional_vars",
        "unsupported",
        "context-manager binding is unmodelled",
    ),
    _d(
        "comprehension.target",
        "withheld",
        "comprehension targets are out of DATAFLOW's candidates",
    ),
    _d("comprehension.iter", "not_applicable", _RECURSES),
    _d("comprehension.ifs", "unsupported", "comprehension filters are unmodelled"),
    _d("comprehension.is_async", "unsupported", "async comprehension is unmodelled"),
    _d("match_case.pattern", "withheld", _MATCH_UNWALKED),
    _d("match_case.guard", "withheld", _MATCH_UNWALKED),
    _d("match_case.body", "withheld", _MATCH_UNWALKED),
    # ── patterns (all inside match, which is declared unwalked) ─────────
    _d("MatchValue.value", "withheld", _MATCH_UNWALKED),
    _d(
        "MatchSingleton.value",
        "withheld",
        _MATCH_UNWALKED,
    ),
    _d(
        "MatchSequence.patterns",
        "withheld",
        _MATCH_UNWALKED,
    ),
    _d("MatchMapping.keys", "withheld", _MATCH_UNWALKED),
    _d(
        "MatchMapping.patterns",
        "withheld",
        _MATCH_UNWALKED,
    ),
    _d("MatchMapping.rest", "withheld", _MATCH_UNWALKED),
    _d("MatchClass.cls", "withheld", _MATCH_UNWALKED),
    _d("MatchClass.patterns", "withheld", _MATCH_UNWALKED),
    _d(
        "MatchClass.kwd_attrs",
        "withheld",
        _MATCH_UNWALKED,
    ),
    _d(
        "MatchClass.kwd_patterns",
        "withheld",
        _MATCH_UNWALKED,
    ),
    _d("MatchStar.name", "withheld", _MATCH_UNWALKED),
    _d("MatchAs.pattern", "withheld", _MATCH_UNWALKED),
    _d("MatchAs.name", "withheld", _MATCH_UNWALKED),
    _d("MatchOr.patterns", "withheld", _MATCH_UNWALKED),
    # ── type parameters and type-ignore records ─────────────────────────
    _d("TypeVar.name", "not_applicable", _ANNOTATION),
    _d("TypeVar.bound", "not_applicable", _ANNOTATION),
    _d("TypeVar.default_value", "not_applicable", _ANNOTATION),
    _d("ParamSpec.name", "not_applicable", _ANNOTATION),
    _d("ParamSpec.default_value", "not_applicable", _ANNOTATION),
    _d("TypeVarTuple.name", "not_applicable", _ANNOTATION),
    _d("TypeVarTuple.default_value", "not_applicable", _ANNOTATION),
    _d("TypeIgnore.lineno", "not_applicable", _TYPE_COMMENT),
    _d("TypeIgnore.tag", "not_applicable", _TYPE_COMMENT),
]

SURFACE_CENSUS: dict[str, str] = dict(_ENTRIES)


def surface_id() -> str:
    """A stable identity for the census contents, as ``<version>.<digest>``."""
    body = ";".join(f"{k}={v}" for k, v in sorted(SURFACE_CENSUS.items()))
    return f"{SURFACE_VERSION}.{hashlib.sha256(body.encode()).hexdigest()[:12]}"


def surface_disposition(key: str) -> str:
    """What the policy says about one ``Node.field`` (or bare ``Node``) slot (pure -- pinned).

    Returns ``undeclared`` for a key the census never named, which is the one answer that means
    "this module has a gap", not "your code has a problem". It is kept distinct from
    ``unsupported`` for exactly that reason: one is an admitted limit of the fault model, the
    other is a limit of the bookkeeping, and collapsing them would hide the second behind the
    first forever.
    """
    if not key:
        return "undeclared"
    return SURFACE_CENSUS.get(key, "undeclared")


def census_gaps(present: list[str]) -> tuple[str, ...]:
    """Which of the interpreter's actual surface keys carry no disposition (pure -- pinned).

    ``present`` is what THIS Python exposes; the census may legitimately declare keys absent here
    (a node from a newer or older version), so the check is one-directional on purpose. Only
    present-but-undeclared is a failure — declaring ahead is not.
    """
    return tuple(sorted(k for k in present if k not in SURFACE_CENSUS))


def interpreter_surface() -> tuple[str, ...]:
    """Every semantic slot the RUNNING interpreter's ``ast`` grammar exposes.

    Abstract bases are skipped (they are never instantiated); a concrete node with no fields
    contributes its bare name, because for those the node's identity IS the semantic choice.
    """
    keys: list[str] = []
    for obj in vars(ast).values():
        if not (
            isinstance(obj, type) and issubclass(obj, ast.AST) and obj is not ast.AST
        ):
            continue
        if obj.__subclasses__():
            continue
        fields = tuple(getattr(obj, "_fields", ()))
        if not fields:
            keys.append(obj.__name__)
        else:
            keys.extend(f"{obj.__name__}.{f}" for f in fields)
    return tuple(sorted(keys))


def unsupported_slots() -> tuple[str, ...]:
    """Every slot the fault model admits it cannot see. The honest gap, enumerable on demand."""
    return tuple(sorted(k for k, v in SURFACE_CENSUS.items() if v == "unsupported"))
