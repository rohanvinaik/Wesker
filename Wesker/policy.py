"""The versioned mutation policy — what μ_Wesker IS, as one inspectable object.

Specification completeness is rigorous only relative to a fixed mutation
policy: SC = 1 means "every non-equivalent alternative in THIS universe is
distinguished", and that claim is only as auditable as the universe's
definition (issue #8). This module makes the definition a first-class,
versioned, machine-readable artifact instead of a property smeared across the
engine. Downstream consumers (Detective's proof receipts, verdict caches) key
on ``policy_id``, so a policy change invalidates exactly the claims it
undermines — and nothing else.

Two mechanisms keep the id honest, neither of which is discipline:

* the manifest is DERIVED from the same tables the engine dispatches on
  (``_STATE_SUB_MODES``, ``_EXCEPTION_SUB_MODES``, ``_SwapMutator._DUALS``,
  the operator swap tables, ``_ValueMutator._alternatives``), so it cannot
  drift from them; and
* ``policy_id`` hashes the manifest TOGETHER WITH the engine's target counts
  over the embedded fingerprint corpus below, so an eligibility change
  anywhere in the record/count path changes the id by construction, even when
  no declared table moved.

``POLICY_VERSION`` is the human-owned counter on top: bump it when the
universe changes meaning (a new category or sub-mode, a changed eligibility
rule), so a receipt reads "policy 2.<hash>" and a human can say "since
policy 2". The golden test in ``tests/test_policy.py`` fails on any id drift
and its failure message says which of the two moves to make.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

from .engine import (
    _DATAFLOW_SUB_MODES,
    _EXCEPTION_SUB_MODES,
    _STATE_SUB_MODES,
    MutationCategory,
    _ArithmeticMutator,
    _BoundaryMutator,
    _LogicalMutator,
    _SwapMutator,
    _ValueMutator,
    estimate_universe_size,
)

# 2: DATAFLOW landed (issue #10) — reference substitution enters the universe,
#    and the fingerprint corpus gained fp_dataflow. Every completeness verdict
#    under policy 1 is a claim about a universe without reference identity.
# 3: STATE.remove_assign covers all three self-write spellings — plain,
#    annotated (self.x: T = v), augmented (self.x += v) — closing the gap the
#    counter migration exposed; the corpus gained fp_state_spellings.
# 4: remove_assign un-binds PER TARGET — `self.a = self.b = x` now yields two
#    DISTINCT mutants (drop a, keep b; drop b, keep a) instead of two
#    dimensions sharing one byte-identical statement->pass mutant. Counts are
#    unchanged; the universe's meaning is not. Single-target mutants are
#    byte-identical to policy 3.
# 5: type-impossible ARITHMETIC swaps leave the universe (issue #12): Add on
#    two provably-str operands and Mult on str × int-literal can only raise
#    TypeError, so they measured reachability and padded the
#    unproven-equivalent bucket ("COMPLETE modulo 12" where the honest number
#    was 4). Proof-only inference — literals, f-strings, str-literal-receiver
#    methods — never annotations; the corpus gained fp_str_arith.
POLICY_VERSION = 5

# The behavioral fingerprint corpus: small functions that together reach every
# category, every sub-mode, dead dimensions, the docstring skip, int
# double-dimensions, duals, unwrap, and the deletable-statement shapes. The
# policy id hashes the engine's target counts over these, so the corpus is
# part of the policy definition — extend it when a new category lands (that is
# a policy change by definition), never rearrange it casually.
_FINGERPRINT_CORPUS: tuple[str, ...] = (
    # VALUE: docstring skipped; int (two dims), str, bool, float constants.
    'def fp_value():\n    """Doc."""\n    n = 1\n    n = n\n    return (n, "s", True, 1.5)\n',
    # BOUNDARY: orderings (boundary shift, direction, equality collapse, two
    # predicate constants), equality, and identity/membership (flip-only).
    "def fp_boundary(a, b):\n    return (a < b) == (a in b) is (a is b)\n",
    # ARITHMETIC + LOGICAL: BinOp, AugAssign, unary minus; and/or, not.
    "def fp_arith_logic(a, b):\n    a += 1\n    return -(a * b) if a and not b else a - b\n",
    # SWAP: adjacent transposition, used-call unwrap, builtin dual (min),
    # provenance-resolved math dual (floor).
    "def fp_swap(xs):\n    import math\n    return math.floor(min(len(xs), 2)) + pow(len(xs), 2)\n",
    # STATE (all three sub-modes) + TYPE + EXCEPTION (raise_type,
    # handler_swallow, handler_broaden).
    "def fp_state_type_exc(self, xs):\n"
    "    self.total = 0\n"
    "    for x in xs:\n"
    "        if isinstance(x, bool):\n"
    "            continue\n"
    "        if x is None:\n"
    "            break\n"
    "        try:\n"
    "            self.total = int(x)\n"
    "        except ValueError:\n"
    "            self.total = -1\n"
    "    if not xs:\n"
    "        raise LookupError(xs)\n"
    "    return self.total\n",
    # STMT: discarded-value call, subscript/attribute aliased writes, a
    # rebinding (deletable) after a first binding (not deletable).
    "def fp_stmt(items, cfg):\n"
    "    items.append(1)\n"
    '    cfg["k"] = 2\n'
    "    total = 3\n"
    "    total = abs(total)\n"
    "    return total\n",
    # Async carries the same universe as sync.
    "async def fp_async(x):\n    return x + 1\n",
    # DATAFLOW (enrolled slice): return_sub over both visible candidates.
    "def fp_dataflow(x, y):\n    total = x + y\n    return total\n",
    # Type-impossible arithmetic (policy 5): the two-str Adds and the
    # str × int-literal Mult carry NO arithmetic dimension; the one-sided
    # f-string Add and the name-name Adds still do (a name may carry
    # __radd__ — provable cases only).
    "def fp_str_arith(parts, n):\n"
    "    label = 'if ' + ' and '.join(parts) + ' then '\n"
    "    bar = '-' * 3\n"
    "    grown = f'{label}!' + label\n"
    "    total = n + 1\n"
    "    return label + bar\n",
    # STATE remove_assign across all three write spellings (policy 3); the
    # valueless declaration is NOT a write and must stay out. The two-target
    # write (policy 4) carries one dimension PER attribute, each with its own
    # per-target mutant.
    "def fp_state_spellings(self, v):\n"
    "    self.plain = v\n"
    "    self.annotated: int = v\n"
    "    self.augmented += v\n"
    "    self.left = self.right = v\n"
    "    self.declared: int\n",
)


@dataclass(frozen=True)
class MutationPolicy:
    """The declared mutation policy: identity, semantics, and per-category
    surface. ``policy_id`` is the one field consumers should key on."""

    policy_version: int
    policy_id: str
    order: int
    generation: str
    eligibility: str
    observation: tuple[str, ...]
    purity_overlay: str
    categories: dict[str, dict[str, Any]]
    exclusions: tuple[str, ...]
    fingerprint_counts: dict[str, dict[str, int]]

    def to_json(self) -> str:
        """Canonical serialization — stable key order, no whitespace variance."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _value_alternative_labels() -> dict[str, list[str]]:
    """Dimension labels per constant type, derived from the live table."""
    return {
        type(v).__name__: [label for _repl, label in _ValueMutator._alternatives(v)]
        for v in (True, 1, 1.5, "s")
    }


def _boundary_alternative_labels() -> dict[str, list[str]]:
    """Dimension labels per comparison operator, derived from the live table."""
    return {
        op_type.__name__: [
            label for _repl, label in _BoundaryMutator._alternatives(op_type())
        ]
        for op_type in _BoundaryMutator._SWAP
    }


def _categories() -> dict[str, dict[str, Any]]:
    """Per-category declared surface, derived from engine tables wherever a
    table exists; prose only where the rule is an analysis, with a pointer to
    the function that owns it."""
    return {
        MutationCategory.VALUE.value: {
            "question": "is this constant's exact value pinned?",
            "mutable_types": [t.__name__ for t in _ValueMutator._MUTABLE_TYPES],
            "alternatives": _value_alternative_labels(),
            "skips": ["docstring constants", "None/bytes/complex/Ellipsis constants"],
        },
        MutationCategory.BOUNDARY.value: {
            "question": "are this comparison's endpoint, direction, and range pinned?",
            "alternatives": _boundary_alternative_labels(),
            "dead_dimensions": "an unrecognised comparison op counts one dead dimension",
        },
        MutationCategory.ARITHMETIC.value: {
            "question": "is this computation's operator pinned?",
            "binop_swaps": {
                k.__name__: v.__name__ for k, v in _ArithmeticMutator._BIN_SWAP.items()
            },
            "also": ["AugAssign under the same table", "unary minus removal (USub)"],
        },
        MutationCategory.LOGICAL.value: {
            "question": "is this conditional composition pinned?",
            "boolop_swaps": {
                k.__name__: v.__name__ for k, v in _LogicalMutator._BOOL_SWAP.items()
            },
            "also": ["`not` removal"],
        },
        MutationCategory.SWAP.value: {
            "question": "are argument positions, call effect, and fold direction pinned?",
            "alternatives": [
                "adjacent positional transposition per pair",
                "unwrap (f(x, ...) -> x) when the call's value is used",
                "curated callee dual, resolved by import provenance, not spelling",
            ],
            "duals": dict(_SwapMutator._DUALS),
        },
        MutationCategory.STATE.value: {
            "question": "are side effects, return values, and loop control pinned?",
            "sub_modes": {mode: desc for mode, desc in _STATE_SUB_MODES},
        },
        MutationCategory.TYPE.value: {
            "question": "is this type guard exercised?",
            "alternatives": ["isinstance(x, T) -> True"],
        },
        MutationCategory.STMT.value: {
            "question": "does this statement do anything the suite can see?",
            "rule": (
                "delete statements that bind nothing new — discarded-value "
                "expressions, subscript/attribute writes, rebindings of "
                "already-bound names; first bindings excluded (see "
                "_deletable_stmt_ids)"
            ),
        },
        MutationCategory.EXCEPTION.value: {
            "question": "are the raised type, handler effect, and caught type pinned?",
            "sub_modes": {mode: desc for mode, desc in _EXCEPTION_SUB_MODES},
        },
        MutationCategory.DATAFLOW.value: {
            "question": "does this expression use the correct available value?",
            "sub_modes": {mode: desc for mode, desc in _DATAFLOW_SUB_MODES},
            "candidates": (
                "parameters (never self/cls, *args, **kwargs) and plain "
                "single-name assignment targets, bound before the load's "
                "enclosing statement (parameter, or earlier statement in the "
                "same block — no flow analysis, so no substitution can be "
                "unbound)"
            ),
            "skips": [
                "callee positions (SWAP owns callable identity)",
                "loads inside nested functions, lambdas, comprehensions",
                "loop and comprehension targets as candidates",
                "match/case bodies (unwalked — a declared withholding)",
            ],
            "dimension_collapse": (
                "DATAFLOW:x→y is one dimension however many sites load x — "
                "the question is whether x and y are distinguished, the same "
                "collapse SWAP applies per callee"
            ),
        },
    }


def _fingerprint_counts() -> dict[str, dict[str, int]]:
    """The engine's target counts over the fingerprint corpus — the behavioral
    half of the policy id. Runs the same record-mode counting every consumer
    runs; no generation, no execution."""
    counts: dict[str, dict[str, int]] = {}
    for src in _FINGERPRINT_CORPUS:
        node = ast.parse(src).body[0]
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        counts[node.name] = {
            cat.value: estimate_universe_size(node, {cat}) for cat in MutationCategory
        }
    return counts


@cache
def mutation_policy() -> MutationPolicy:
    """The current mutation policy, id included. Cached — the policy is a
    constant of the installed engine."""
    fingerprint = _fingerprint_counts()
    manifest: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "order": 1,
        "generation": "original-ast",
        "eligibility": (
            "a category is eligible exactly when its target count is nonzero; "
            "every count runs the category's mutator in record mode or its "
            "shared _alternatives static — never a second predicate"
        ),
        "observation": [
            "a mutant is killed when a covering test fails under it",
            "assertion kills (value-specified) are distinguished from "
            "crash/timeout kills (run-only), and an assertion kill takes "
            "precedence over a crash kill for the same mutant",
            "survivors probe synthesized boundary inputs for likely "
            "equivalence; equivalence is reported, never silently claimed",
            "covering-test scoping and categorical exclusion are "
            "verdict-preserving reductions",
        ],
        "purity_overlay": (
            "filter_categories(is_pure=True) suppresses the remove_assign "
            "sub-mode's contribution to STATE eligibility; nothing else is "
            "purity-gated"
        ),
        "categories": _categories(),
        "exclusions": [
            "first-order only: mutants are generated one at a time from the "
            "original AST; kills do not certify compositions of operators "
            "(see the composite-blind-spot witness in issue #11)",
            "DATAFLOW's enrolled slice is returned-name substitution only: "
            "general name-load substitution is implemented but not enrolled "
            "(measured at +217% dimensions on Wesker itself — awaiting a "
            "candidate restraint), and attribute selectors, subscript keys, "
            "callable references, and receiver (self/cls) substitutions are "
            "declared out of the supported slice, not silently covered "
            "(issue #10's remaining subfamilies)",
            "sorted(reverse=) has no curated dual until a measured case wants it",
            "bare re-raise carries no EXCEPTION target; an already-pass "
            "handler is not a handler_swallow target; an untyped except: is "
            "not a handler_broaden target",
            "type-impossible ARITHMETIC swaps are not targets: Add on two "
            "provably-str operands and Mult on str x int-literal can only "
            "raise TypeError (reachability, not specification); provable "
            "inference only — annotations are never trusted, and a one-sided "
            "str Add stays in the universe",
        ],
        "fingerprint_corpus": list(_FINGERPRINT_CORPUS),
        "fingerprint_counts": fingerprint,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return MutationPolicy(
        policy_version=POLICY_VERSION,
        policy_id=f"{POLICY_VERSION}.{digest}",
        order=1,
        generation="original-ast",
        eligibility=manifest["eligibility"],
        observation=tuple(manifest["observation"]),
        purity_overlay=manifest["purity_overlay"],
        categories=manifest["categories"],
        exclusions=tuple(manifest["exclusions"]),
        fingerprint_counts=fingerprint,
    )
