"""AST mutation engine — in-process mutant generation and evaluation.

Implements §6.4 dispatch table: category→AST-transform mapping.
Generates mutants by AST rewriting (no subprocess spawning), evaluates
them by running targeted tests in the same process against a sandboxed
namespace. Respects per-function time budgets.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import math
import time
import types
from dataclasses import dataclass, field
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Any

from .interrupt import abandon as _abandon
from .line_coverage import coverage_from_trace as _coverage_from_trace
from .line_coverage import executable_lines as _executable_lines
from .line_coverage import failing_on_baseline as _failing_on_baseline
from .line_coverage import trace_line_coverage as _trace_line_coverage
from .line_coverage import trace_suite as _trace_suite
from .memory_guard import over_budget as _over_budget
from .memory_guard import reclaim as _reclaim
from .memory_guard import resolve_budget as _resolve_budget

if TYPE_CHECKING:
    from collections.abc import Callable


# The default per-test TRACE budget, in seconds. A backstop, NOT a target: it is generous enough
# that no honest test should ever meet it (the untraced per-test timeout beside it is 5s, and
# tracing costs a callback per executed line), so meeting it is evidence of the pathological case
# — a test whose traced cost is effectively unbounded. Bounded by DEFAULT because the failure it
# replaces is a SILENT HANG with no output and no diagnosis, which reads as a broken tool rather
# than a slow test; a cut is always reported by name, so the loud-and-wrong outcome is preferred
# to the quiet-and-wrong one. Pass None to opt out and restore the historical unbounded pass.
DEFAULT_TRACE_BUDGET_S = 50.0

# The default budget for the WHOLE traced baseline pass. Independent of the per-test cap above,
# because they bound different things and neither implies the other: a per-test cap × N tests is
# still N× unbounded, and on a 2000-test suite the 50s cap alone permits a day of tracing. The
# tests it did not reach are reported by name, so a partial baseline is never mistaken for a
# complete one. Pass None for the historical unbounded pass.
#
# WAS 300s, on the reasoning that five minutes is "this project's own stated intolerable case (the
# whole-file audit that ran 5 min with zero output)". That premise has since been retired by the
# thing that fixed it: `trace_progress` now reports this pass from its first test (see
# `ci.run_with_live_suite`), so outrunning five minutes is a VISIBLE slow measurement, not a
# dead-looking tool. The budget was sized to a silence that no longer exists.
#
# What the old number cost, measured on Regenesis (240 scoped tests, 3.06s/test, 734.8s to trace
# in full): the 300s cap cut 152 tests, whose line coverage is then under-counted, so no test could
# be credited with the kills it actually makes — `greedy_coverage` reported 0 of 45 behaviours
# pinned where the truth is 22. Not a slower answer: a confidently wrong one, in the direction that
# invites `converge` to write tests for behaviour the suite already pins. Undersizing this is
# silent and wrong; oversizing it is loud and slow. Prefer loud.
#
# 1800s is read off that measurement (~2.4x), not picked: the order of magnitude a genuine suite
# needs, while still bounding the pathological case this exists for. The per-test cap above is
# deliberately NOT raised with it — on the same measurement, (50,300) and (∞,300) cut an identical
# 152 tests, so the per-test bound was never the binding constraint and there is no evidence to
# change it. Raise a default the evidence indicts; leave the rest alone.
#
# NOTE THE UNITS, because they decide how to read that 734.8s: these budgets are WALL-CLOCK
# (`time.monotonic`, see `line_coverage.trace_suite`) while the work is CPU-bound and single-core.
# So the wall-clock cost of the SAME suite rises with whatever else is running, and truncation is a
# property of the machine at that moment, not of the suite. That cuts one way for sizing: 734.8s
# was measured on a CONTENDED box (test suites running alongside it), which makes it the
# conservative sample and the right one to size against — a number taken on an idle machine would
# under-size the budget for precisely the loaded runs that need the headroom. It also means no
# budget can be "correct": a busy enough box cuts at any finite value. The bound is here to stop a
# pathological hang, not to certify a measurement; when the answer must be exact, pass 0.
DEFAULT_TRACE_SESSION_BUDGET_S = 1800.0

# How long to let an ABANDONED test thread unwind, while its stdout/stderr are still redirected.
# An abandoned frame runs its `finally`/`__exit__` blocks on the way out, so a test (or any
# library it called) that entered its own `redirect_stdout` reinstalls what IT captured — our
# StringIO. Landing after we restore, that replaces the process's real `sys.stdout` with a dead
# buffer and every later write is discarded in silence. Waiting here keeps the unwind inside our
# own redirect, so our restoration is the last writer. Deliberately short: a courtesy window on a
# thread already declared a timeout, not a second timeout — the verdict is decided either way, and
# a thread outliving it is no worse off than before this wait existed. See `_run_test_with_timeout`.
_ABANDON_UNWIND_S = 0.25


class MutationCategory(str, Enum):
    """Semantic mutation category (§6.4 dispatch table)."""

    VALUE = "VALUE"
    SWAP = "SWAP"
    STATE = "STATE"
    BOUNDARY = "BOUNDARY"
    TYPE = "TYPE"
    ARITHMETIC = "ARITHMETIC"
    LOGICAL = "LOGICAL"
    STMT = "STMT"
    # Exception behavior: raised type, handler swallowing, handler widening. Carries
    # three orthogonal sub-modes (see _ExceptionMutator) counted against their own
    # target sets, the same shape STATE uses.
    EXCEPTION = "EXCEPTION"


@dataclass
class Mutant:
    """A single AST-level mutation."""

    category: MutationCategory
    original_node: ast.AST
    mutated_node: ast.AST
    description: str
    location: int = 0
    mutant_id: str = ""
    # The positional target index within its (category, sub-mode) — the engine's internal
    # selection ordinal. ``mutant_id`` is content-addressed (invocation-stable); this stays
    # positional for greedy-selection bookkeeping and order/coverage assertions.
    target_index: int = -1
    # Absolute source line the mutation changed (from the mutator's fire site). The exact
    # line a test must EXECUTE to observe this mutant — the key to test-impact scoping.
    # None when the mutator could not report it (falls back to running the full suite).
    mutated_line: int | None = None
    # The behavioral dimension this mutant pins (``VALUE:int``, ``ARITHMETIC:Add``, …) —
    # the cover set of the greedy selection, which is a SINGLETON. Carried so a run can
    # report DOF coverage exactly (distinct dimensions reached / this function's DOF)
    # instead of inferring it from the selection. "" when unrecorded (greedy=False).
    dimension: str = ""


@dataclass
class MutantResult:
    """Result of evaluating a single mutant against tests."""

    mutant: Mutant
    killed: bool = False
    killed_by: str | None = None  # "assertion" | "exception" | "crash" | "timeout"
    test_name: str | None = None  # first killing test (first-killer mode)
    elapsed_ms: float = 0.0
    equivalent: bool = False
    killed_by_tests: list[str] = field(
        default_factory=list
    )  # all killers (full-matrix mode)


@dataclass
class CategoryResult:
    """Aggregated results for one mutation category."""

    category: MutationCategory
    total: int = 0
    killed: int = 0
    survived: int = 0
    killed_by_assertion: int = 0
    killed_by_exception: int = 0
    killed_by_crash: int = 0
    timed_out: int = 0
    equivalent: int = 0

    @property
    def survival_rate(self) -> float:
        return self.survived / self.total if self.total > 0 else 0.0

    @property
    def value_killed(self) -> int:
        """Mutants whose VALUE behavior is pinned. An assertion kill qualifies, and so does
        an EXCEPTION kill: a test that says ``pytest.raises(ValueError)`` has stated what the
        function does on that input as precisely as ``== 3`` states it elsewhere. Raising IS
        the return behaviour of an error path.

        Only crash/timeout are excluded, and for the original reason: they prove the code RAN,
        not WHAT it did. Counting a declared failure among them read a pin as a gap — and since
        an error path can be pinned ONLY this way, it made every input-validating function
        permanently unspecifiable rather than merely unspecified."""
        return self.killed_by_assertion + self.killed_by_exception

    @property
    def value_survived(self) -> int:
        """Value-unspecified DOF: survivors PLUS crash/timeout kills. For specification
        these are equivalent — none pins the return value."""
        return self.survived + self.killed_by_crash + self.timed_out


@dataclass
class SamplingResult:
    """Result of inline mutation sampling for a function."""

    function_key: str = ""
    categories_tested: int = 0
    total_mutants: int = 0
    total_killed: int = 0
    total_survived: int = 0
    survival_rate: float = 0.0
    coverage_depth: str = "sampled"
    per_category: list[CategoryResult] = field(default_factory=list)
    budget_exhausted: bool = False
    elapsed_ms: float = 0.0
    total_equivalent: int = 0
    universe_size: int = 0

    def to_dict(self) -> dict:
        effective_total = self.total_mutants - self.total_equivalent
        effective_kill_pct = (
            round(100 * self.total_killed / effective_total)
            if effective_total > 0
            else 100
        )
        return {
            "function_key": self.function_key,
            "categories_tested": self.categories_tested,
            "total_mutants": self.total_mutants,
            "total_killed": self.total_killed,
            "total_survived": self.total_survived,
            "total_equivalent": self.total_equivalent,
            "universe_size": self.universe_size,
            "survival_rate": round(self.survival_rate, 3),
            "effective_kill_pct": effective_kill_pct,
            "coverage_depth": self.coverage_depth,
            "budget_exhausted": self.budget_exhausted,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "per_category": [
                {
                    "category": cr.category.value,
                    "total": cr.total,
                    "killed": cr.killed,
                    "survived": cr.survived,
                    "equivalent": cr.equivalent,
                    "survival_rate": round(cr.survival_rate, 3),
                }
                for cr in self.per_category
            ],
        }


@dataclass
class ProfilingResult:
    """Result of exhaustive mutation profiling for a function."""

    function_key: str = ""
    categories_tested: int = 0
    total_mutants: int = 0
    total_killed: int = 0
    total_survived: int = 0
    survival_rate: float = 0.0
    coverage_depth: str = "profiled"
    is_gateable: bool = True
    per_category: list[CategoryResult] = field(default_factory=list)
    kill_matrix: dict[str, list[str]] = field(default_factory=dict)
    survivor_records: list[dict] = field(default_factory=list)
    killed_records: list[dict] = field(default_factory=list)
    budget_exhausted: bool = False
    elapsed_ms: float = 0.0
    total_equivalent: int = 0
    universe_size: int = 0
    # Tests whose TRACED baseline pass hit `trace_budget_s` and was CUT. Their line coverage is
    # under-counted, so this travels WITH the result: an unreported cut is indistinguishable from
    # "no test reaches this line", which turns a timing accident into a false completeness verdict
    # — the one thing a completeness tool must never do quietly.
    trace_truncated: list[str] = field(default_factory=list)
    # --- DOF coverage: the claim a bounded run can actually make ------------------
    # ``universe_size`` counts mutation TARGETS; these count the distinct behavioral
    # DIMENSIONS those targets pin. Because each target's cover set is a singleton,
    # the greedy round-robin covers min(picks, D) of D exactly — so dof_covered /
    # dof_total is a measured, exact coverage fraction, not an estimate or a bound.
    # It states which DIMENSIONS were reached, NOT that untested mutants would die:
    # two sites sharing a dimension are still distinct behaviors.
    dof_total: int = 0
    dof_covered: int = 0
    # THE SPECIFICATION METRIC — distinct dimensions whose mutant a test actually KILLED.
    #
    # ``dof_covered`` is a property of the SELECTION (did greedy reach every dimension), and
    # under the DOF budget it is ~always dof_total, because that is exactly the theorem. It
    # says nothing about the tests, and reporting it as "specification completeness" would
    # publish a number that is 100% on a repo with no working tests at all.
    #
    # ``dof_pinned`` is the property of the SUITE: of this function's behavioral dimensions,
    # how many does some test distinguish? Its denominator (``dof_total``) is derived from the
    # AST — a property of the code, identical on any machine at any budget — which is what
    # makes the ratio comparable ACROSS repos, unlike a kill rate whose denominator is however
    # many mutants a given run happened to sample.
    dof_pinned: int = 0
    # Second completeness axis, from a traced baseline pass over the unmutated
    # function: which target lines each test covers, and the executable-line
    # denominator. Empty when no baseline pass ran (backward-compatible).
    line_coverage: dict[str, list[int]] = field(default_factory=dict)
    executable_lines: list[int] = field(default_factory=list)
    # Tests whose assertion fails on the UNMUTATED function — broken/stale, surfaced
    # for a human (a wrong assertion or a real regression), never auto-removed.
    failing_tests: list[str] = field(default_factory=list)
    # How many test callables were discovered for this function. 0 means the kill
    # rate is 0% because there is NOTHING to kill with — a discovery/"write a test"
    # signal, not weak tests. -1 = not populated (older callers), so consumers can
    # tell "no tests" apart from "unknown". Prevents a silent, misleading 0%.
    tests_discovered: int = -1

    # --- Value-specification view -------------------------------------------------
    # An assertion kill pins WHAT the function returns; a crash/timeout kill only proves
    # it RUNS. For SPECIFICATION only assertion kills count, so crash/timeout kills are
    # unspecified value-DOF. Derived (not stored) so they can never drift from the record
    # of record — any ProfilingResult, however constructed, reports the split correctly.

    @property
    def value_killed(self) -> int:
        """Mutants whose return value is pinned — assertion kills only."""
        return sum(cr.value_killed for cr in self.per_category)

    @property
    def value_survived(self) -> int:
        """Value-unspecified DOF: true survivors PLUS crash/timeout kills."""
        return sum(cr.value_survived for cr in self.per_category)

    @property
    def value_survivor_records(self) -> list[dict]:
        """Survivor-shaped record for every value-unspecified mutant — the true survivors
        plus each crash/timeout kill (reshaped from ``killed_records``, carrying its diff)
        so a value-distinguishing witness can be sought for behavior the tests only ran.

        Carries ``mutated_line``, ``dimension`` and ``change`` through from the kill record.
        They are what let a consumer report the gap AT the line it lives on and name the
        behavior nobody pinned — a SARIF result or an editor annotation rather than a count.
        Without them a crash-kill survivor arrives as `line None` with a blank dimension: a
        warning that cannot be located, attached to a file, which is worse than silence.
        """
        crash_survivors = [
            {
                "mutant_id": r.get("mutant_id"),
                "mutant": r.get("mutant"),
                "category": r.get("category"),
                "mutated_line": r.get("mutated_line"),
                "dimension": r.get("dimension"),
                "change": r.get("change", ""),
                "diff_summary": r.get("diff_summary", ""),
                "killed_by": r.get("killed_by"),
                "elapsed_ms": r.get("elapsed_ms", 0.0),
            }
            for r in self.killed_records
            if r.get("killed_by") not in ("assertion", "exception")
        ]
        return list(self.survivor_records) + crash_survivors

    def to_dict(self) -> dict:
        effective_total = self.total_mutants - self.total_equivalent
        effective_kill_pct = (
            round(100 * self.total_killed / effective_total)
            if effective_total > 0
            else 100
        )
        d = {
            "function_key": self.function_key,
            "categories_tested": self.categories_tested,
            "total_mutants": self.total_mutants,
            "total_killed": self.total_killed,
            "total_survived": self.total_survived,
            "total_equivalent": self.total_equivalent,
            "universe_size": self.universe_size,
            "dof_total": self.dof_total,
            "dof_covered": self.dof_covered,
            "dof_pct": (
                round(100 * self.dof_covered / self.dof_total)
                if self.dof_total > 0
                else 100
            ),
            "dof_pinned": self.dof_pinned,
            # Specification completeness: the fraction of this function's behavioral
            # dimensions that some test pins. 0 dimensions = nothing to specify = complete,
            # which is why the empty case is 100 rather than a division error.
            "spec_pct": (
                round(100 * self.dof_pinned / self.dof_total)
                if self.dof_total > 0
                else 100
            ),
            "survival_rate": round(self.survival_rate, 3),
            "effective_kill_pct": effective_kill_pct,
            "coverage_depth": self.coverage_depth,
            "is_gateable": self.is_gateable,
            "budget_exhausted": self.budget_exhausted,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "per_category": [
                {
                    "category": cr.category.value,
                    "total": cr.total,
                    "killed": cr.killed,
                    "survived": cr.survived,
                    "equivalent": cr.equivalent,
                    "killed_by_assertion": cr.killed_by_assertion,
                    "killed_by_crash": cr.killed_by_crash,
                    "survival_rate": round(cr.survival_rate, 3),
                }
                for cr in self.per_category
            ],
        }
        if self.kill_matrix:
            d["kill_matrix"] = self.kill_matrix
        if self.survivor_records:
            d["survivor_records"] = self.survivor_records
        # The value-unspecified set: true survivors PLUS crash/timeout kills. This is what a
        # SPECIFICATION consumer wants — `spec_pct` counts assertion kills alone, so a report
        # that listed only `survivor_records` would claim a gap and then name none of it, and
        # the SARIF/annotations built from it would show a clean diff under a red badge.
        # `survivor_records` stays exactly as it was: Detective reads that name.
        if self.value_survivor_records:
            d["value_survivor_records"] = self.value_survivor_records
        if self.killed_records:
            d["killed_records"] = self.killed_records
        if self.line_coverage:
            d["line_coverage"] = self.line_coverage
        if self.executable_lines:
            d["executable_lines"] = self.executable_lines
        if self.failing_tests:
            d["failing_tests"] = self.failing_tests
        return d


# ── §6.4 Dispatch Table: Category → AST Transform ────────────────


class _BaseMutator(ast.NodeTransformer):
    """Base class for all category mutators — tracks ``applied`` state.

    Doubles as a *dimension recorder*. When ``keys`` is a list (record mode,
    entered by constructing with ``target_index=-1``), each mutator calls
    ``_note(dim_key)`` exactly once per candidate site — at the same point it
    increments ``self.current`` — so ``keys[i]`` is the behavioral dimension of
    target index ``i`` in the *identical* traversal order the transformer uses
    to consume that index. Alignment is therefore by construction, not by a
    re-implementation of the walk. When ``keys`` is ``None`` (normal mutation),
    ``_note`` is a no-op.
    """

    def __init__(self, target_index: int = 0):
        self.current = 0
        self.target = target_index
        self.applied = False
        self.keys: list[str] | None = None
        # The absolute source line the mutation changed — the exact line a test
        # must EXECUTE to observe this mutant. Captured at the fire site so
        # test-impact scoping can run only the covering tests (verdict-preserving:
        # a test that never runs the mutated line cannot kill the mutant).
        self.mutated_lineno: int | None = None

    def _mark_applied(self, node: ast.AST) -> None:
        """Record that the mutation fired at ``node`` — sets the applied flag and
        the source line changed. Mutators call this in place of a bare applied-flag
        set so every category reports WHERE it mutated with no per-category drift."""
        self.applied, self.mutated_lineno = True, getattr(node, "lineno", None)

    def _note(self, dim_key: str) -> None:
        """Record the behavioral dimension of the current candidate site."""
        if self.keys is not None:
            self.keys.append(dim_key)


class _ValueMutator(_BaseMutator):
    """Replace constants with boundary values.

    Ints carry TWO dimensions: the 0/1 COLLAPSE (is the constant read at all?)
    and an OFF-BY-ONE (is its exact value pinned?). The collapse alone lets a
    near-miss hide: ``round(cost, 2) -> round(cost, 0)`` is killed by any
    1-decimal golden value, while ``2 -> 3`` changes nothing a coarse input can
    see — only the off-by-one forces the witness that pins the exact value.
    Ints only: a float/str collapse already forces an exact-value assertion
    wherever a value test exists, and a threshold used in a comparison is
    BOUNDARY's job (its endpoint shift generates the same witness ±1 would).
    """

    # Types we can actually mutate — others (None, bytes, complex, Ellipsis)
    # are left unchanged by _mutate_constant, so we must not count them as
    # targets or mark ``applied`` when we encounter them.
    _MUTABLE_TYPES = (bool, int, float, str)

    def __init__(
        self,
        target_index: int = 0,
        docstring_positions: set[tuple[int, int]] | None = None,
    ):
        super().__init__(target_index)
        self._ds_pos = docstring_positions or set()

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.applied:
            return node
        if not isinstance(node.value, self._MUTABLE_TYPES):
            return node
        # Skip docstring constants — they produce equivalent mutants.
        if (
            self._ds_pos
            and isinstance(node.value, str)
            and (node.lineno, node.col_offset) in self._ds_pos
        ):
            return node
        # One dimension per alternative (mirrors _BoundaryMutator): the counter
        # derives from _alternatives, so count and visit order cannot drift.
        for repl, label in self._alternatives(node.value):
            selected = not self.applied and self.current == self.target
            self._note(label)
            self.current += 1
            if not selected:
                continue
            self._mark_applied(node)
            return ast.Constant(value=repl)
        return node

    @staticmethod
    def _alternatives(v: Any) -> list[tuple[Any, str]]:
        """Ordered (replacement value, dimension label) for one constant.

        Single source of truth: the mutator and ``_count_value_target`` both
        read it. bool is checked before int — it IS an int, and ``True + 1``
        is not a boolean dimension.
        """
        if isinstance(v, bool):
            return [(not v, "VALUE:bool")]
        if isinstance(v, int):
            collapse = 0 if v != 0 else 1
            # ±1 dodges the collapse value so the two dimensions never share a
            # mutant (v=0: collapse 1, off-by-one -1; v=-1: collapse 0, off-by-one -2).
            off1 = v + 1 if v + 1 != collapse else v - 1
            return [(collapse, "VALUE:int"), (off1, "VALUE:int~off1")]
        if isinstance(v, float):
            return [(0.0 if v else 1.0, "VALUE:float")]
        if isinstance(v, str):
            return [("" if v else "mutated", "VALUE:str")]
        return []


class _BoundaryMutator(_BaseMutator):
    """Relational-operator mutation (ROR), complete for a comparison.

    Four independent questions per operator, each its own behavioral dimension:

      * BOUNDARY shift (``<`` -> ``<=``) — is the endpoint pinned?
      * DIRECTION reversal (``<`` -> ``>``) — is the ordering pinned?
      * EQUALITY collapse (``<`` -> ``==``) — is the RANGE pinned, or only the point?
        A suite testing one value either side of a threshold kills the shift and the
        reversal while never distinguishing "less than" from "exactly equal".
      * PREDICATE constant (``x < y`` -> ``True`` / ``False``) — does the branch matter
        at all? This is the classic ROR pair, and it is the one that catches a condition
        no test ever drives both ways: dead branches, defensive guards nothing exercises.

    Identity/membership operators (``is``, ``in``) take the flip only — there is no
    ordering to reverse and no meaningful equality collapse.

    ``_alternatives`` is the single source of truth: the mutator and
    ``_count_boundary_target`` both read it, so the target count and the visit order
    cannot drift.
    """

    # Boundary / predicate flip — the always-present alternative for every
    # comparison operator.
    _SWAP = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        # Identity / membership predicate flips — whole operator classes that
        # previously produced no mutant, leaving a real behavioral DOF unpinned.
        ast.Is: ast.IsNot,
        ast.IsNot: ast.Is,
        ast.In: ast.NotIn,
        ast.NotIn: ast.In,
    }

    # Direction reversal — a SECOND alternative on ordering comparisons only.
    # Distinct behavioral DOF from the boundary shift (`<` vs `>` vs `<=`).
    _DIRECTION = {
        ast.Lt: ast.Gt,
        ast.Gt: ast.Lt,
        ast.LtE: ast.GtE,
        ast.GtE: ast.LtE,
    }

    # Equality collapse — a THIRD alternative on orderings: does the suite pin a RANGE,
    # or merely a point? Absent, an ordering whose tests only probe equality looks pinned.
    _EQUALITY = {
        ast.Lt: ast.Eq,
        ast.Gt: ast.Eq,
        ast.LtE: ast.Eq,
        ast.GtE: ast.Eq,
    }

    @staticmethod
    def _alternatives(op: ast.cmpop) -> list[tuple[Any, str]]:
        """Ordered (replacement, dimension label) for one comparison operator.

        A replacement is either a ``cmpop`` CLASS (swap the operator) or a ``bool``
        (replace the whole comparison with that constant). Single source of truth for
        the mutation dimensions, so the mutator and ``_count_boundary_target`` cannot
        drift.
        """
        alts: list[tuple[Any, str]] = []
        name = type(op).__name__
        boundary = _BoundaryMutator._SWAP.get(type(op))
        if boundary is not None:
            alts.append((boundary, f"BOUNDARY:{name}"))
        direction = _BoundaryMutator._DIRECTION.get(type(op))
        if direction is not None:
            alts.append((direction, f"BOUNDARY:{name}~dir"))
        equality = _BoundaryMutator._EQUALITY.get(type(op))
        if equality is not None:
            alts.append((equality, f"BOUNDARY:{name}~eq"))
        if boundary is not None:
            # Predicate constants — only where an operator is recognised at all, so an
            # exotic comparison stays a single dead dimension rather than sprouting two.
            alts.append((True, f"BOUNDARY:{name}~true"))
            alts.append((False, f"BOUNDARY:{name}~false"))
        return alts

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        new_ops = list(node.ops)
        for pos, op in enumerate(node.ops):
            alts = self._alternatives(op)
            if not alts:
                # No swap for this op → dead dimension (still one entry per op),
                # sinks to the end of the greedy order.
                self._note(_DEAD_DIM)
                self.current += 1
                continue
            # One dimension per alternative; apply the one the target selects.
            for repl, label in alts:
                selected = not self.applied and self.current == self.target
                self._note(label)
                self.current += 1
                if not selected:
                    continue
                if isinstance(repl, bool):
                    # Replace the ENTIRE comparison — a chained compare collapses too,
                    # which is correct: the predicate's value is what the branch reads.
                    self._mark_applied(node)
                    return ast.Constant(value=repl)
                new_ops[pos] = repl()
                self._mark_applied(node)
        node.ops = new_ops
        return self.generic_visit(node)


def _stmt_call_ids(root: ast.AST) -> set[int]:
    """ids of Call nodes in expression-STATEMENT position (``log.info(x)``,
    ``items.append(y)``). Their value is discarded, so unwrap (call -> first
    arg) is a guaranteed-equivalent no-op there — and SDL already owns the
    "does this discarded call do anything?" question by deleting the statement.
    ids are safe here: the set is built and consumed within one traversal of
    one live tree, never persisted (contrast ``_deletable_stmt_ids``)."""
    return {
        id(stmt.value)
        for stmt in ast.walk(root)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
    }


class _SwapMutator(_BaseMutator):
    """Call-shape mutations: transpose adjacent arguments, or unwrap the call.

    Per call site, one dimension per alternative (mirrors _BoundaryMutator):

      * a TRANSPOSITION per adjacent argument pair — ``f(a, b, c)`` carries
        (a,b) and (b,c). The first pair keeps the bare ``SWAP:<callee>`` label
        the universe has always had; later pairs get ``~p<i>``.
      * an UNWRAP (``f(x, ...) -> x``) when the call's value is USED — the
        "does this call do anything the suite can see?" question. It forces a
        witness where the call is observable: ``round(cost, 2) -> cost``
        demands an input with 3+ decimals, which also pins the ndigits
        constant no VALUE collapse can reach. Skipped in expression-statement
        position (guaranteed-equivalent there) and for a starred first arg.

    ``_alternatives`` is the single source of truth: the mutator and
    ``_count_targets``'s SWAP case both read it, so the target count and the
    visit order cannot drift.
    """

    def __init__(self, target_index: int = 0):
        super().__init__(target_index)
        self._stmt_ids: set[int] | None = None

    def visit(self, node: ast.AST) -> ast.AST:
        # The first visit sees the root: capture statement-position calls once,
        # so per-Call eligibility below needs no parent pointers.
        if self._stmt_ids is None:
            self._stmt_ids = _stmt_call_ids(node)
        return super().visit(node)

    @staticmethod
    def _alternatives(node: ast.Call, stmt_ids: set[int]) -> list[tuple[Any, str]]:
        """Ordered (spec, dimension label) for one call site — a spec is the
        left index of the adjacent pair to transpose, or ``"unwrap"``."""
        name = _callee_name(node)
        alts: list[tuple[Any, str]] = []
        for i in range(len(node.args) - 1):
            alts.append((i, f"SWAP:{name}" if i == 0 else f"SWAP:{name}~p{i}"))
        if (
            node.args
            and not isinstance(node.args[0], ast.Starred)
            and id(node) not in stmt_ids
        ):
            alts.append(("unwrap", f"SWAP:{name}~unwrap"))
        return alts

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        for spec, label in self._alternatives(node, self._stmt_ids or set()):
            selected = not self.applied and self.current == self.target
            self._note(label)
            self.current += 1
            if not selected:
                continue
            self._mark_applied(node)
            if spec == "unwrap":
                # Returned as-is, unvisited: applied is set, so no site after
                # this one is consumed anywhere in the tree this pass — the
                # skipped subtree costs nothing (record mode never fires this
                # branch, so key alignment is untouched).
                return node.args[0]
            node.args = list(node.args)
            node.args[spec], node.args[spec + 1] = node.args[spec + 1], node.args[spec]
        return self.generic_visit(node)


class _StateMutator(_BaseMutator):
    """Remove self.x = ... assignments or replace return with return None."""

    def __init__(self, target_index: int = 0, mode: str = "remove_assign"):
        super().__init__(target_index)
        self.mode = mode

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if self.applied or self.mode != "remove_assign":
            return node
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                if self.current == self.target:
                    self._mark_applied(node)
                    return ast.Pass()
                self._note(f"STATE:remove_assign:{target.attr}")
                self.current += 1
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if self.applied or self.mode != "return_none":
            return node
        if node.value is not None:
            if self.current == self.target:
                self._mark_applied(node)
                return ast.Return(value=ast.Constant(value=None))
            self._note("STATE:return_none")
            self.current += 1
        return node


class _TypeMutator(_BaseMutator):
    """Replace isinstance(x, T) with True."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            if self.current == self.target:
                self._mark_applied(node)
                return ast.Constant(value=True)
            self._note(f"TYPE:{_isinstance_type_name(node)}")
            self.current += 1
        return self.generic_visit(node)


def _exc_type_name(node: ast.AST | None) -> str:
    """Readable name for a raised/caught exception expression."""
    if node is None:
        return "bare"
    if isinstance(node, ast.Call):
        return _exc_type_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Tuple):
        return ",".join(_exc_type_name(e) for e in node.elts)
    return type(node).__name__


def _swapped_exc(name: str) -> str:
    """A DIFFERENT builtin exception to raise instead of ``name``.

    A builtin, because the mutant's namespace is seeded from the source module's globals:
    a sentinel class of our own would not resolve there and the mutant would die of
    NameError — a crash kill that measures our plumbing, not the suite.
    """
    return "TypeError" if name == "ValueError" else "ValueError"


class _ExceptionMutator(_BaseMutator):
    """Exception-behavior mutation — the operator class Wesker had none of.

    Nothing else in the taxonomy touches exceptions: no operator changes a raised type,
    swallows a handler, or widens what is caught. That is the gap that bites REFACTORING
    hardest, because moving code across a ``try`` boundary changes exactly this and
    nothing in the universe pins it — the extracted block that used to raise inside the
    ``try`` now raises inside a helper called from somewhere else. A suite can be at
    100% and not notice.

    Three independent questions, each its own greedy dimension:

      * ``raise X(...)`` -> ``raise <other builtin>(...)`` — does any test pin the TYPE?
        A suite asserting ``pytest.raises(ValueError)`` kills it; one asserting
        ``pytest.raises(Exception)`` does not, and should not — it genuinely did not
        pin the type.
      * ``except X: <body>`` -> ``except X: pass`` — does any test notice the handler
        stopped doing its work? This is exception SWALLOWING, the failure mode where an
        error is silently discarded.
      * ``except X:`` -> ``except BaseException:`` — does any test notice the handler
        now catches strictly more? A refactor that widens a catch swallows errors that
        used to propagate.

    A handler whose body is already ``pass`` is skipped for the swallow mode: replacing
    ``pass`` with ``pass`` is an equivalent mutant by construction, and generating it
    would inflate the universe with a guaranteed survivor.
    """

    def __init__(self, target: int, mode: str = "raise_type", *a, **k) -> None:  # type: ignore[no-untyped-def]
        super().__init__(target, *a, **k)
        self.mode = mode

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        if self.mode != "raise_type" or self.applied or node.exc is None:
            return self.generic_visit(node)
        name = _exc_type_name(node.exc)
        if self.current == self.target:
            repl = ast.Name(id=_swapped_exc(name), ctx=ast.Load())
            if isinstance(node.exc, ast.Call):
                node.exc = ast.Call(
                    func=repl, args=node.exc.args, keywords=node.exc.keywords
                )
            else:
                node.exc = ast.Call(func=repl, args=[], keywords=[])
            self._mark_applied(node)
            return node
        self._note(f"EXCEPTION:raise:{name}")
        self.current += 1
        return self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        name = _exc_type_name(node.type)
        if self.mode == "handler_swallow":
            if not _handler_is_noop(node):
                if self.current == self.target:
                    node.body = [ast.Pass()]
                    self._mark_applied(node)
                    return node
                self._note(f"EXCEPTION:swallow:{name}")
                self.current += 1
        elif self.mode == "handler_broaden" and node.type is not None:
            if self.current == self.target:
                node.type = ast.Name(id="BaseException", ctx=ast.Load())
                self._mark_applied(node)
                return node
            self._note(f"EXCEPTION:broaden:{name}")
            self.current += 1
        return self.generic_visit(node)


def _handler_is_noop(node: ast.ExceptHandler) -> bool:
    """True when a handler's body is already a no-op, so swallowing it changes nothing."""
    return len(node.body) == 1 and isinstance(node.body[0], ast.Pass)


class _ArithmeticMutator(_BaseMutator):
    """Replace arithmetic operators: + ↔ -, * ↔ /, // → /, % → *, ** → *.

    Also removes unary negation (-x → x). Covers AOR and UOI from the
    DeMillo/Lipton/Sayward operator set.
    """

    _BIN_SWAP: dict[type, type] = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
        ast.FloorDiv: ast.Div,
        ast.Mod: ast.Mult,
        ast.Pow: ast.Mult,
    }

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        swapped = self._BIN_SWAP.get(type(node.op))
        if swapped:
            if self.current == self.target:
                self._mark_applied(node)
                node.op = swapped()
            self._note(f"ARITHMETIC:{type(node.op).__name__}")
            self.current += 1
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        if isinstance(node.op, ast.USub):
            if self.current == self.target:
                self._mark_applied(node)
                return self.generic_visit(node.operand)
            self._note("ARITHMETIC:USub")
            self.current += 1
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        # Augmented assignment (x += 1) carries the same operator DOF as a
        # BinOp — swap it under the same table so `+=`/`*=`/… get pinned.
        if self.applied:
            return self.generic_visit(node)
        swapped = self._BIN_SWAP.get(type(node.op))
        if swapped:
            if self.current == self.target:
                self._mark_applied(node)
                node.op = swapped()
            self._note(f"ARITHMETIC:{type(node.op).__name__}")
            self.current += 1
        return self.generic_visit(node)


class _LogicalMutator(_BaseMutator):
    """Replace logical operators: and ↔ or; remove not.

    Covers COR (Conditional Operator Replacement) from the standard
    mutation operator set.
    """

    _BOOL_SWAP: dict[type, type] = {
        ast.And: ast.Or,
        ast.Or: ast.And,
    }

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        swapped = self._BOOL_SWAP.get(type(node.op))
        if swapped:
            if self.current == self.target:
                self._mark_applied(node)
                node.op = swapped()
            self._note(f"LOGICAL:{type(node.op).__name__}")
            self.current += 1
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if self.applied:
            return self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            if self.current == self.target:
                self._mark_applied(node)
                return self.generic_visit(node.operand)
            self._note("LOGICAL:Not")
            self.current += 1
        return self.generic_visit(node)


def _deletable_stmt_ids(func_node: ast.AST) -> set[tuple[int, int]]:
    """The SOURCE POSITION ``(lineno, col_offset)`` of every statement in ``func_node``
    whose deletion cannot raise NameError.

    POSITIONS, NOT ``id()``. An ``id()`` is a memory address: it is unique per run and
    meaningless across runs, so this function's return value could never be asserted
    against. Detective made that concrete — it could prove 29 of its mutants killable and
    still write only ``assert _deletable_stmt_ids(...) == set()``, because the empty set
    is the one answer stable enough to pin; every input with a deletable statement was
    correctly dropped as non-deterministic. A function whose output cannot be written down
    cannot be specified, and no amount of test generation fixes that from the outside.
    A source position is stable across runs and identical in a ``deepcopy``, which is
    exactly the property the caller needs (see :class:`_StmtMutator`) — so nothing is
    lost and the contract becomes observable.

    STATEMENT DELETION (SDL) is the highest-value operator per the deletion-operator
    literature (Delamaro & Offutt): it is cheap, and it catches what operator-REPLACEMENT
    structurally cannot. Replacing an operator asks "is this operator right?"; deleting a
    statement asks "does this statement do anything the suite can see?" — the question a
    refactor most often gets wrong.

    Three shapes qualify, and the single rule behind them is "binds nothing new":

      * ``ast.Expr`` (a discarded-value call: ``log.info(x)``, ``items.append(y)``) —
        binds nothing, always deletable.
      * ``x[k] = v`` / ``x.attr = v`` — a Subscript/Attribute target binds NO name, so
        deletion is always safe. This is the ALIASING case: ``def f(cfg): cfg[k] = v``
        mutates a caller's object, and no other operator generates "drop that write" —
        STATE only ever targeted ``self.x``. A refactor that copies instead of aliasing
        passes every return-value assertion a suite has.
      * ``x = expr`` / ``x += expr`` where ``x`` is ALREADY bound — rebinding, not
        binding, so the name survives deletion with its earlier value. This is the
        ``total = abs(total)`` case.

    Excluded: a FIRST binding (``x = ...`` where ``x`` is new). Deleting it makes every
    later use a NameError — a mutant that always crashes is killed by any test that runs
    the line, so it measures reachability, not specification, and it inflates the
    universe with trivial kills. The prior implementation excluded ALL bound names for
    this reason; that is right for a first binding and wrong for a rebinding, which is
    exactly the case worth testing.

    Conservative on conditional binding: a name counts as bound only if it is a parameter
    or was bound by an EARLIER statement in the SAME block. So::

        if flag:
            x = 1
        x = 2      # NOT deletable — x is unbound when flag is False

    stays out of the universe rather than becoming a spurious crash-kill. The same rule
    means a rebinding inside a nested block is only seen when its first binding is in
    that block too — deliberately narrow, since proving otherwise needs flow analysis.
    """
    out: set[tuple[int, int]] = set()

    def _pos(stmt: ast.stmt) -> tuple[int, int]:
        return (getattr(stmt, "lineno", -1), getattr(stmt, "col_offset", -1))

    def _names(target: ast.AST) -> tuple[list[str], bool]:
        """(bound names, binds_nothing) for one assignment target."""
        if isinstance(target, ast.Name):
            return [target.id], False
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            return [], True  # mutates an existing object; binds no name
        if isinstance(target, (ast.Tuple, ast.List)):
            names: list[str] = []
            nothing = True
            for el in target.elts:
                n, only_mut = _names(el)
                names.extend(n)
                nothing = nothing and only_mut
            return names, nothing and not names
        return [], False  # Starred and friends: don't reason, don't delete

    def _blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
        found: list[list[ast.stmt]] = []
        for attr in ("body", "orelse", "finalbody"):
            b = getattr(stmt, attr, None)
            if isinstance(b, list) and b:
                found.append(b)
        for handler in getattr(stmt, "handlers", []) or []:
            if getattr(handler, "body", None):
                found.append(handler.body)
        return found

    def _walk_block(stmts: list[ast.stmt], bound: set[str]) -> None:
        local = set(bound)
        for st in stmts:
            if isinstance(st, ast.Expr):
                if not isinstance(st.value, ast.Constant):
                    out.add(_pos(st))  # docstring / bare literal has no side effect
            elif isinstance(st, ast.AugAssign):
                # ``x += 1`` REQUIRES x to already exist, or the original itself raises.
                # So deletion is always safe regardless of what we can prove here.
                out.add(_pos(st))
            elif isinstance(st, (ast.Assign, ast.AnnAssign)):
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                names: list[str] = []
                binds_nothing = True
                for t in targets:
                    n, only_mut = _names(t)
                    names.extend(n)
                    binds_nothing = binds_nothing and only_mut
                if getattr(st, "value", None) is None:
                    pass  # bare annotation (``x: int``) — no runtime effect to delete
                elif binds_nothing:
                    out.add(_pos(st))  # x[k]=v / x.attr=v — the aliasing case
                elif names and all(n in local for n in names):
                    out.add(_pos(st))  # rebinding — the SDL case
                local.update(names)
            # Nested blocks see the bindings established BEFORE them at this level.
            for block in _blocks(st):
                _walk_block(block, local)

    params: set[str] = set()
    args = getattr(func_node, "args", None)
    if args is not None:
        for a in (
            list(getattr(args, "posonlyargs", []))
            + list(args.args)
            + list(getattr(args, "kwonlyargs", []))
        ):
            params.add(a.arg)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                params.add(extra.arg)
    _walk_block(list(getattr(func_node, "body", [])), params)
    return out


def _stmt_label(node: ast.stmt) -> str:
    """Dimension label for a deletable statement.

    Distinct side effects must be distinct greedy dimensions, so the label names WHAT is
    being dropped, not merely that something was: the callee for a call
    (``STMT:append``), the mutated container/attribute for a write (``STMT:cfg[]``,
    ``STMT:obj.attr``), the rebound name for a rebinding (``STMT:=total``). Collapsing
    these to one label would let greedy cover ``log.info(x)`` and call the dimension
    settled while ``items.append(y)`` on the next line goes untested.
    """
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            return _callee_name(node.value)
        return type(node.value).__name__
    if isinstance(node, ast.AugAssign):
        return f"aug:{_assign_target_label(node.target)}"
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]  # type: ignore[attr-defined]
    return "=" + ",".join(_assign_target_label(t) for t in targets)


def _assign_target_label(target: ast.AST) -> str:
    """Stable name for an assignment target, used as its dimension key."""
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return f"{_assign_target_label(target.value)}.{target.attr}"
    if isinstance(target, ast.Subscript):
        return f"{_assign_target_label(target.value)}[]"
    if isinstance(target, (ast.Tuple, ast.List)):
        return "(" + ",".join(_assign_target_label(e) for e in target.elts) + ")"
    return type(target).__name__


class _StmtMutator(_BaseMutator):
    """Statement deletion (SDL) — replace a statement with ``pass`` and ask whether any
    test notices the side effect is gone.

    Targets exactly the statements :func:`_deletable_stmt_ids` proves cannot raise
    NameError when removed: discarded-value calls (``items.append(y)``), writes through
    an existing object (``cfg[k] = v``, ``obj.attr = v``), and rebindings
    (``total = abs(total)``, ``total += x``). See that function for why a FIRST binding
    is excluded and why conditional binding is treated conservatively.

    Deletability is derived from the tree this mutator is handed, not passed in: the
    engine visits a deepcopy of the function, so precomputed node ``id()``s from the
    original would not match. ``_count_stmt_target`` runs the SAME analysis, so the
    counter and the mutator cannot drift.
    """

    def __init__(self, target: int, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(target, *args, **kwargs)
        self._deletable: set[tuple[int, int]] | None = None

    def visit(self, node: ast.AST) -> ast.AST:
        # The first node handed to a run IS the function; analyse it once, here, so both
        # record mode and mutate mode see an identical target set in identical order.
        if self._deletable is None and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            self._deletable = _deletable_stmt_ids(node)
        return super().visit(node)

    def _consider(self, node: ast.stmt) -> ast.AST:
        # Keyed by SOURCE POSITION, not id(): this mutator visits a deepcopy, whose nodes
        # are different objects from the ones analysed but carry identical positions.
        # (An id() set happened to work here — the analysis ran on this same copy — but it
        # made the analysis's RETURN VALUE un-assertable, so the operator could not be
        # specified. Positions cost nothing and are observable.)
        pos = (getattr(node, "lineno", -1), getattr(node, "col_offset", -1))
        if self.applied or self._deletable is None or pos not in self._deletable:
            return node
        if self.current == self.target:
            self._mark_applied(node)
            return ast.Pass()
        self._note(f"STMT:{_stmt_label(node)}")
        self.current += 1
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        return self._consider(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        return self._consider(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        return self._consider(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        return self._consider(node)


# ── Mutant Generation ─────────────────────────────────────────────


def _docstring_positions(func_node: ast.FunctionDef) -> set[tuple[int, int]]:
    """Return (lineno, col_offset) of docstring Constant nodes in a function.

    A docstring is the first statement if it's ``Expr(Constant(str))``.
    We collect positions so that both counting and mutation can skip them
    using position-based identity (survives ``copy.deepcopy``).
    """
    positions: set[tuple[int, int]] = set()
    if (
        func_node.body
        and isinstance(func_node.body[0], ast.Expr)
        and isinstance(func_node.body[0].value, ast.Constant)
        and isinstance(func_node.body[0].value.value, str)
    ):
        ds = func_node.body[0].value
        positions.add((ds.lineno, ds.col_offset))
    return positions


def _count_targets(func_node: ast.FunctionDef, category: MutationCategory) -> int:
    """Count how many mutation targets exist for a category in a function."""
    counter = _TARGET_COUNTERS.get(category)
    if counter is None:
        return 0
    # VALUE needs docstring exclusion — pass positions through.
    if category == MutationCategory.VALUE:
        ds_pos = _docstring_positions(func_node)
        return sum(_count_value_target(node, ds_pos) for node in ast.walk(func_node))
    # STMT deletability depends on what is bound BEFORE a statement, which no per-node
    # counter can see. Same analysis the mutator runs, so the two cannot drift.
    if category == MutationCategory.STMT:
        return len(_deletable_stmt_ids(func_node))
    # SWAP eligibility (unwrap) depends on statement position, which no per-node
    # counter can see — same shape as STMT. Derive from _SwapMutator._alternatives
    # over the same tree, so the count cannot drift from the mutator's _note calls.
    if category == MutationCategory.SWAP:
        stmt_ids = _stmt_call_ids(func_node)
        return sum(
            len(_SwapMutator._alternatives(node, stmt_ids))
            for node in ast.walk(func_node)
            if isinstance(node, ast.Call)
        )
    # EXCEPTION's sub-modes each have their own target space, and their skip rules
    # (bare re-raise, already-``pass`` handler, untyped ``except:``) live in the
    # mutator — so count by running it, not by re-encoding the rules here.
    if category == MutationCategory.EXCEPTION:
        return sum(
            _count_exception_targets(func_node, mode)
            for mode, _desc in _EXCEPTION_SUB_MODES
        )
    return sum(counter(node) for node in ast.walk(func_node))


def _count_value_target(
    node: ast.AST,
    docstring_positions: set[tuple[int, int]] | None = None,
) -> int:
    # Only count constants whose types _ValueMutator can actually mutate.
    # None, bytes, complex, and Ellipsis are left unchanged by _mutate_constant,
    # so counting them produces phantom mutants that always survive.
    # Skip docstring constants — they produce equivalent mutants that waste budget.
    if isinstance(node, ast.Constant) and isinstance(
        node.value, _ValueMutator._MUTABLE_TYPES
    ):
        if (
            docstring_positions
            and isinstance(node.value, str)
            and (node.lineno, node.col_offset) in docstring_positions
        ):
            return 0
        # Derived from _alternatives so the count can never drift from the
        # mutator's _note calls (ints carry two dimensions, everything else one).
        return len(_ValueMutator._alternatives(node.value))
    return 0


def _count_boundary_target(node: ast.AST) -> int:
    if not isinstance(node, ast.Compare):
        return 0
    # One dimension per alternative per op (a dead op still notes once) — derive
    # from _alternatives so this never drifts from _BoundaryMutator's _note count.
    return sum(len(_BoundaryMutator._alternatives(op)) or 1 for op in node.ops)


def _count_swap_target(node: ast.AST) -> int:
    """Per-node SWAP counter — NOT used for SWAP.

    Unwrap eligibility is a statement-position property no per-node counter can
    see, so ``_count_targets`` special-cases SWAP over the whole function via
    ``_SwapMutator._alternatives`` (same shape as STMT). Retained only so the
    ``_TARGET_COUNTERS`` dispatch stays total.
    """
    return 0


def _is_self_assign(target: ast.AST) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    )


def _count_state_assign_target(node: ast.AST) -> int:
    """Count self.x = ... assignments (remove_assign mode)."""
    if isinstance(node, ast.Assign):
        return sum(1 for t in node.targets if _is_self_assign(t))
    return 0


def _count_state_return_target(node: ast.AST) -> int:
    """Count return-with-value nodes (return_none mode)."""
    if isinstance(node, ast.Return) and node.value is not None:
        return 1
    return 0


def _count_state_target(node: ast.AST) -> int:
    return _count_state_assign_target(node) + _count_state_return_target(node)


def _count_type_target(node: ast.AST) -> int:
    return (
        1
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        else 0
    )


def _count_arithmetic_target(node: ast.AST) -> int:
    """Count arithmetic mutation targets (BinOp + AugAssign + unary negation).

    Must stay in lockstep with ``_ArithmeticMutator``'s ``visit_*`` methods —
    the count defines the index range generation iterates, and it has to equal
    the number of ``_note`` calls the mutator makes over the same constructs.
    """
    if isinstance(node, ast.BinOp) and type(node.op) in _ArithmeticMutator._BIN_SWAP:
        return 1
    if (
        isinstance(node, ast.AugAssign)
        and type(node.op) in _ArithmeticMutator._BIN_SWAP
    ):
        return 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return 1
    return 0


def _count_logical_target(node: ast.AST) -> int:
    """Count logical mutation targets (BoolOp + not removal)."""
    if isinstance(node, ast.BoolOp) and type(node.op) in _LogicalMutator._BOOL_SWAP:
        return 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return 1
    return 0


def _count_stmt_target(node: ast.AST) -> int:
    """Per-node STMT counter — NOT used for STMT.

    STMT deletability is a FUNCTION-level property (is this name already bound?), so it
    cannot be decided from a node in isolation the way every other category's counter can.
    ``_count_targets`` therefore special-cases STMT and calls :func:`_deletable_stmt_ids`
    over the whole function, exactly as ``_StmtMutator`` does — one analysis, so the
    counter and the mutator cannot disagree about how many targets exist.

    Retained only so the ``_TARGET_COUNTERS`` dispatch stays total.
    """
    return 0


_TARGET_COUNTERS: dict[MutationCategory, Callable[[ast.AST], int]] = {
    MutationCategory.VALUE: _count_value_target,
    MutationCategory.BOUNDARY: _count_boundary_target,
    MutationCategory.SWAP: _count_swap_target,
    MutationCategory.STATE: _count_state_target,
    MutationCategory.TYPE: _count_type_target,
    MutationCategory.ARITHMETIC: _count_arithmetic_target,
    MutationCategory.LOGICAL: _count_logical_target,
    MutationCategory.STMT: _count_stmt_target,
    # Like STMT, a stub: EXCEPTION is counted per sub-mode in _count_targets, which
    # cannot be expressed as a per-node function. Present so the dispatch stays total.
    MutationCategory.EXCEPTION: lambda _node: 0,
}


def _content_mutant_id(category: MutationCategory, mutated_node: ast.AST) -> str:
    """A content-addressed, invocation-stable mutant id.

    A short hash of the mutation's CONTENT — the mutated function's AST including source
    locations — so the SAME mutation gets the SAME id in every mode, pass, and process.
    Positional ``CATEGORY_i`` ids drift because greedy/fast passes emit different index
    subsets, and the index carries no meaning; the content id does not drift, which is what
    makes a cross-invocation reference to "this mutant" (the audit→flag handoff) resolvable.
    """
    content = ast.dump(mutated_node, include_attributes=True)
    digest = hashlib.sha1(content.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :8
    ]
    return f"{category.value}_{digest}"


def _generate_state_mutants(
    func_node: ast.FunctionDef,
    max_per_category: int | None,
    greedy: bool = True,
    pass_index: int = 0,
) -> list[Mutant]:
    """Generate STATE mutants across both sub-modes (assign + return).

    STATE has two independent sub-modes with separate target indices:
    - remove_assign: replaces ``self.x = expr`` with ``pass``
    - return_none: replaces ``return expr`` with ``return None``

    Each sub-mode gets its own target count and transformer pass so that
    target indices align correctly with what the transformer visits. Under
    ``greedy`` the assign sub-mode is ordered by distinct attribute (its
    behavioral dimension) so a budget spreads across state fields before
    repeating one; both sub-modes are always represented (they are two
    distinct dimensions the greedy never collapses).
    """
    mutants: list[Mutant] = []
    cat = MutationCategory.STATE

    sub_modes = [
        ("remove_assign", "remove state assignment", _count_state_assign_target),
        ("return_none", "replace return with None", _count_state_return_target),
    ]

    for mode, desc, counter in sub_modes:
        target_count = sum(counter(node) for node in ast.walk(func_node))
        # Each sub-mode is selected against its own budget; in DOF mode that is
        # the sub-mode's own degrees of freedom (distinct state fields, or the
        # single return_none dimension).
        keys = _record_state_dimensions(func_node, mode) if greedy else []
        budget = (
            _live_dimension_count(keys)
            if max_per_category is None
            else max_per_category
        )
        limit = min(target_count, budget) if budget > 0 else target_count

        if greedy and budget > 0 and target_count > limit:
            selected = _select_greedy(keys, target_count, limit, pass_index)
        else:
            selected = list(range(limit))

        for i in selected:
            mutated_tree = copy.deepcopy(func_node)
            transformer = _StateMutator(i, mode)
            mutated_node = transformer.visit(mutated_tree)
            ast.fix_missing_locations(mutated_node)

            if transformer.applied:
                mid = _content_mutant_id(cat, mutated_node)
                mutants.append(
                    Mutant(
                        category=cat,
                        original_node=func_node,
                        mutated_node=mutated_node,
                        description=f"{mid}: {desc}",
                        location=getattr(func_node, "lineno", 0),
                        mutant_id=mid,
                        target_index=i,
                        mutated_line=transformer.mutated_lineno,
                        dimension=keys[i] if i < len(keys) else "",
                    )
                )

    return mutants


# ── Behavioral-Dimension Coverage (Layer 2, greedy submodular) ───
#
# A mutant is an unconstrained *behavioral degree of freedom* (§2.1 of the SSL
# homology: surviving mutant ↔ candidate reading). The behavioral *dimension* it
# probes is (category, construct-kind) — e.g. ``BOUNDARY:Lt``, ``ARITHMETIC:Add``.
# The set-cover value f(S) = |dimensions covered by S| is monotone submodular
# (proofs/coverage_submodular.lean), so greedily selecting mutants by marginal
# coverage κ = |cover(v) \ cover(S)| — which is antitone (marginal_antitone.lean)
# — reaches ≥(1−1/e) of the optimally-coverable dimension set within any budget
# k (greedy_coverage_bound.lean). This replaces seeded random sampling: instead
# of "((n−k)/n)^K probability we missed a survivor," we *select* the provably
# near-optimal covering set. Multi-pass slicing (pass p takes window
# [p·k, (p+1)·k) of the greedy order) makes cross-pass accrual the gap-contraction
# the bound is stated over.

_DEAD_DIM = "\x00dead"  # sentinel: a candidate site that yields no mutant


def _is_dead(dim_key: str) -> bool:
    return dim_key == _DEAD_DIM


def _callee_name(node: ast.Call) -> str:
    """Best-effort callable name for SWAP dimension keys."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return "call"


def _isinstance_type_name(node: ast.Call) -> str:
    """Type-argument name of an ``isinstance(x, T)`` call for TYPE dimension keys."""
    if len(node.args) >= 2:
        t = node.args[1]
        if isinstance(t, ast.Name):
            return t.id
        if isinstance(t, ast.Attribute):
            return t.attr
        if isinstance(t, ast.Tuple):
            names = [e.id for e in t.elts if isinstance(e, ast.Name)]
            return "+".join(names) if names else "tuple"
    return "type"


# Record-mode mutator factory per category (STATE is handled separately by
# _generate_state_mutants / _record_state_dimensions). Dispatch table keeps
# _record_dimensions a flat regime-A function rather than a 6-way branch.
_RECORD_MUTATOR_FACTORIES: dict[
    MutationCategory, Callable[[set[tuple[int, int]] | None], _BaseMutator]
] = {
    MutationCategory.VALUE: lambda ds: _ValueMutator(-1, ds),
    MutationCategory.BOUNDARY: lambda ds: _BoundaryMutator(-1),
    MutationCategory.ARITHMETIC: lambda ds: _ArithmeticMutator(-1),
    MutationCategory.LOGICAL: lambda ds: _LogicalMutator(-1),
    MutationCategory.SWAP: lambda ds: _SwapMutator(-1),
    MutationCategory.TYPE: lambda ds: _TypeMutator(-1),
    MutationCategory.STMT: lambda ds: _StmtMutator(-1),
}


def _record_dimensions(
    func_node: ast.FunctionDef,
    category: MutationCategory,
    docstring_positions: set[tuple[int, int]] | None = None,
) -> list[str]:
    """Behavioral dimension of each target index, in transformer-visit order.

    Runs the *actual* category mutator in record mode (``target_index=-1`` so
    nothing applies) over a copy of the tree; each mutator notes one key per
    candidate site at the same point it would increment its index counter, so
    ``keys[i]`` is guaranteed to be the dimension of target ``i``. STATE is
    generated by ``_generate_state_mutants`` and recorded via
    ``_record_state_dimensions`` instead.
    """
    factory = _RECORD_MUTATOR_FACTORIES.get(category)
    if factory is None:
        return []
    mutator = factory(docstring_positions)
    mutator.keys = []
    mutator.visit(copy.deepcopy(func_node))
    return mutator.keys


def _record_state_dimensions(func_node: ast.FunctionDef, mode: str) -> list[str]:
    """Dimension keys for one STATE sub-mode, in transformer-visit order."""
    tree = copy.deepcopy(func_node)
    mutator = _StateMutator(-1, mode)
    mutator.keys = []
    mutator.visit(tree)
    return mutator.keys


def _record_exception_dimensions(func_node: ast.FunctionDef, mode: str) -> list[str]:
    """Dimension keys for one EXCEPTION sub-mode, in transformer-visit order."""
    tree = copy.deepcopy(func_node)
    mutator = _ExceptionMutator(-1, mode)
    mutator.keys = []
    mutator.visit(tree)
    return mutator.keys


def _count_exception_targets(func_node: ast.FunctionDef, mode: str) -> int:
    """Targets for one EXCEPTION sub-mode. Counted by RUNNING the mutator in record
    mode, so the counter and the transformer cannot drift — the skip rules (a bare
    ``raise``, an already-``pass`` handler, an untyped ``except:``) live in one place."""
    return len(_record_exception_dimensions(func_node, mode))


_EXCEPTION_SUB_MODES = (
    ("raise_type", "replace raised exception type"),
    ("handler_swallow", "swallow exception handler body"),
    ("handler_broaden", "widen caught exception type"),
)


def _generate_exception_mutants(
    func_node: ast.FunctionDef,
    max_per_category: int | None,
    greedy: bool = True,
    pass_index: int = 0,
) -> list[Mutant]:
    """Generate EXCEPTION mutants across all three sub-modes.

    Same shape as :func:`_generate_state_mutants`: each sub-mode has its OWN target
    index space and its own budget, because a shared index would misalign against the
    transformer (mode ``handler_swallow`` visiting target 3 is a different statement
    from mode ``raise_type`` visiting target 3). In DOF mode each sub-mode's budget is
    its own degrees of freedom, so a function that raises but catches nothing spends
    nothing on handler modes: they count zero and contribute nothing.
    """
    mutants: list[Mutant] = []
    cat = MutationCategory.EXCEPTION

    for mode, desc in _EXCEPTION_SUB_MODES:
        keys = _record_exception_dimensions(func_node, mode) if greedy else []
        target_count = (
            len(keys) if greedy else _count_exception_targets(func_node, mode)
        )
        budget = (
            _live_dimension_count(keys)
            if max_per_category is None
            else max_per_category
        )
        limit = min(target_count, budget) if budget > 0 else target_count

        if greedy and budget > 0 and target_count > limit:
            selected = _select_greedy(keys, target_count, limit, pass_index)
        else:
            selected = list(range(limit))

        for i in selected:
            mutated_tree = copy.deepcopy(func_node)
            transformer = _ExceptionMutator(i, mode)
            mutated_node = transformer.visit(mutated_tree)
            ast.fix_missing_locations(mutated_node)

            if transformer.applied:
                mid = _content_mutant_id(cat, mutated_node)
                mutants.append(
                    Mutant(
                        category=cat,
                        original_node=func_node,
                        mutated_node=mutated_node,
                        description=f"{mid}: {desc}",
                        location=getattr(func_node, "lineno", 0),
                        mutant_id=mid,
                        target_index=i,
                        mutated_line=transformer.mutated_lineno,
                        dimension=keys[i] if i < len(keys) else "",
                    )
                )

    return mutants


def _live_dimension_count(keys: list[str]) -> int:
    """Distinct behavioral dimensions among candidate sites (dead sites excluded)."""
    return len({k for k in keys if not _is_dead(k)})


class SessionBaseline:
    """The suite-global half of the baseline, computed ONCE per session.

    Three passes ran per FUNCTION and two of them provably cannot vary by function:

      * ``failing_on_baseline`` calls a bare ``test_fn()``; ``original_func`` only gates
        a ``__code__`` check. Its answer is a property of the SUITE.
      * ``_baseline_failures`` patches the original OVER ITSELF — a no-op — so it also
        just asks "does this test pass?".
      * ``trace_line_coverage`` traces the whole suite and then keeps one function's
        lines. The TRACE is global; only the intersection is per-function.

    So the cost was ``O(3 × suite × functions)`` before any mutant ran. Hoisted here it
    is ``O(3 × suite)`` once, plus a set-intersection per function. Measured on Prism
    (445 tests): 28.6s of baseline PER FUNCTION, 89% of wall clock, and enough to eat a
    10s per-file budget whole — that function then reported 0 mutants, i.e. the budget
    was spent entirely on recomputing a constant.

    ONLY VALID FOR ZERO-ARG CALLABLES, which is why this is opt-in and set exclusively
    by the live-session path. ``_baseline_failures`` is suite-global only because the
    patch is a no-op AND the call convention does not vary; under the LEGACY runner the
    unpatched path calls ``test_fn(original)``, whose convention depends on ``qualname``
    per function, so the answer is not a constant there. Live-session callables are
    keyword-only zero-arg wrappers, so both hold.

    ``truncated`` names the tests a ``trace_budget_s`` CUT mid-trace. Their coverage is
    under-counted by construction, so it must be reported rather than folded in silently: a
    budget-shortened trace is indistinguishable downstream from "no test reaches this line",
    which would turn a timing accident into a false completeness verdict.
    """

    __slots__ = ("traced", "failing", "inert", "n_tests", "truncated")

    def __init__(
        self,
        traced: dict[str, dict[str, set[int]]],
        failing: list[str],
        inert: set[int],
        n_tests: int,
        truncated: set[str] | None = None,
    ) -> None:
        self.traced = traced
        self.failing = failing
        self.inert = inert
        self.n_tests = n_tests
        self.truncated = truncated or set()

    def replaced(
        self,
        affected: set[str],
        removed_ids: set[int],
        partial: "SessionBaseline",
        n_tests: int,
    ) -> "SessionBaseline":
        """This baseline with every test named in ``affected`` re-measured from ``partial``.

        Writing ONE test file invalidated the whole baseline, so the next read re-traced the
        entire suite to learn what one file changed — and a consumer that writes tests in a loop
        (converge) paid that per pass. Measured on Detective's own 312-test suite: 4 full traces,
        8.6s, 60% of wall clock, to write 3 tests. Nothing about the other 311 tests changed, and
        re-measuring a constant is the same refusal-of-provably-irrelevant-work the engine already
        applies to mutants and `trace_suite` applies to the per-function trace.

        ``affected`` is keyed by test NAME because ``traced`` is, and that keying is why this
        takes a name set rather than a file: ``trace_suite`` UNIONS duplicate ``__name__``s
        across files, so a name the written file merely SHARES with another module cannot be
        dropped — its other owner's coverage would vanish with it, and a test whose coverage is
        absent is one `_build_test_scope` never runs, which reports a mutant it kills as a
        surviving behavioral gap. The caller therefore re-traces every CURRENT owner of an
        affected name and hands the result here; this method only splices.

        ``removed_ids`` are the ``id()``s of callables that left the suite. They must be dropped
        explicitly: ``inert`` is keyed by object identity, and an id whose object has been freed
        can be reused by a later allocation — a stale entry would bar an unrelated test from kill
        attribution. Every id kept here belongs to a callable the live suite still references, so
        none can dangle.
        """
        traced = {k: v for k, v in self.traced.items() if k not in affected}
        traced.update(partial.traced)
        return SessionBaseline(
            traced,
            [n for n in self.failing if n not in affected] + partial.failing,
            {i for i in self.inert if i not in removed_ids} | partial.inert,
            n_tests,
            {n for n in self.truncated if n not in affected} | partial.truncated,
        )


class LazySessionBaseline:
    """A :class:`SessionBaseline` that is not built until something actually reads it.

    The hoist made the baseline suite-global; this makes it DEMAND-driven, and the two together
    are what let a cache save anything. Built eagerly, the baseline is the whole cost of a run —
    it traces the entire suite before the consumer's first line of work — and a consumer whose
    own cache answers the question then never touches it. Measured on Regenesis: a warm-cache
    `detective diagnose` still paid 191s to trace 240 tests, then served the profile from disk
    and dropped the trace unread. The cost sat OUTSIDE the region the cache protects, so the
    cache could not amortise the only thing worth amortising.

    Deferring costs nothing: a run that needs the baseline builds it on first read and every
    later function reuses it exactly as before, while a run answered from cache never builds it
    at all. The eager/lazy difference is invisible to a caller — same object, same values, same
    once-per-session guarantee.

    The build closure must carry its OWN stream guard. Eagerly, the caller could wrap the build;
    lazily it fires deep inside the consumer's call stack, where nothing is wrapping it, and the
    baseline RUNS the target's suite — arbitrary code that can leave `sys.stdout` replaced.
    Whatever wraps this must therefore live in the closure, not around the site that stores it.
    """

    __slots__ = ("_build", "_value", "_built", "_budgets")

    def __init__(
        self,
        build: Callable[..., SessionBaseline],
        budgets: tuple[float | None, float | None] | None = None,
    ) -> None:
        # Takes an optional subset of callables: the same closure builds the whole baseline
        # (subset=None) and the partial one `refresh` splices in, so both are measured under
        # identical target files and budgets by construction rather than by discipline.
        self._build = build
        self._value: SessionBaseline | None = None
        self._built = False
        self._budgets = budgets

    def get(self) -> SessionBaseline:
        """The baseline, building it on first call. Memoised — the pass runs at most once."""
        if not self._built:
            self._value = self._build()
            self._built = True
        return self._value  # type: ignore[return-value]

    @property
    def built(self) -> bool:
        """Whether the pass has actually run — for callers that must not force it."""
        return self._built

    @property
    def budgets(self) -> tuple[float | None, float | None] | None:
        """The ``(per_test, session)`` trace budgets this baseline is built with, or None.

        Readable WITHOUT forcing the build: the budgets are fixed when the closure is stored, not
        when it runs, so a consumer can ask what a verdict WOULD be measured under before deciding
        whether it needs the measurement at all. That ordering is the whole point — the one caller
        that needs this reads it to build a cache key, and a key that forced the trace it is trying
        to avoid would defeat the laziness above.
        """
        return self._budgets

    def invalidate(self) -> None:
        """Discard the built baseline so the next read rebuilds it from the CURRENT suite.

        The baseline is a measurement OF a suite: which test covers which line, which tests fail
        on the unmutated original. Add a test to the suite and every one of those answers is out
        of date — the new test is simply absent from `traced`, so `_build_test_scope` finds no
        covering tests for it and the mutation loop never runs it. It is not "a bit stale": the
        test cannot kill anything, so a consumer that WRITES tests scores the suite it had before
        it wrote them, and reports its own work as unspecified behaviour (measured: 18 kills on
        disk, 2 reported).

        Cheap precisely because the baseline is lazy: this drops the value, and the next reader
        pays for a rebuild only if there IS a next reader. A run that writes tests and stops
        never re-traces at all.
        """
        self._value = None
        self._built = False

    def refresh(
        self,
        affected: set[str],
        removed_ids: set[int],
        retrace: list[Callable[..., None]],
        n_tests: int,
    ) -> bool:
        """Re-measure only the tests ``affected`` names, keeping the rest. True if spliced.

        `invalidate` is correct but total: it answers "one file changed" by re-measuring every
        file. The other tests' coverage is a CONSTANT across a write that did not touch them —
        so the next read paid the whole suite to rediscover what it already knew. Measured on
        Detective's own suite: converge did 4 full 312-test traces (8.6s, 60% of wall clock) to
        write 3 tests; only the written file's tests were ever new.

        Nothing is built if nothing was built: the lazy build already reads the CURRENT live
        suite, so a not-yet-forced baseline is not stale, and forcing a trace here to service a
        write is precisely the eager cost `LazySessionBaseline` exists to defer.

        DEGRADES TO `invalidate`, never to a wrong answer. A partial build runs the consumer's
        own test code, so it can fail in ways this module cannot enumerate; a half-spliced
        baseline would under-report coverage, and an under-covered test is one the mutation loop
        never runs — a false survivor, the exact lie the live session exists to prevent. So any
        failure drops the whole value and the next reader re-traces, which is what today does
        unconditionally: the fast path can only ever be skipped, never be wrong.
        """
        if not self._built or self._value is None:
            return False
        try:
            partial = self._build(retrace)
            self._value = self._value.replaced(affected, removed_ids, partial, n_tests)
        except Exception:  # noqa: BLE001 — see DEGRADES above; correctness over speed
            self.invalidate()
            return False
        return True


# Set only by the live-session path; None everywhere else, so every existing caller
# (Detective included) keeps the exact per-function behaviour it has today.
_SESSION_BASELINE: ContextVar[LazySessionBaseline | None] = ContextVar(
    "wesker_session_baseline", default=None
)


def session_baseline() -> SessionBaseline | None:
    """The live session's baseline, built on demand, or None outside a live session.

    The single read point. Both consumers go through here so neither can forget to resolve the
    holder, and so "is there a session?" stays separable from "build it".
    """
    holder = _SESSION_BASELINE.get()
    return holder.get() if holder is not None else None


def session_budgets() -> tuple[float | None, float | None] | None:
    """The ``(per_test, session)`` trace budgets the live session's baseline is built under, or
    None outside a live session. Does NOT force the build.

    The read point for "what actually produced this verdict". Inside a live session the baseline —
    and therefore ``truncated`` and every absent ``line_coverage`` — comes from THESE budgets;
    the per-function ``trace_budget_s`` arguments are not consulted at all (see
    ``_build_test_scope``'s precedence). A consumer that caches a verdict must key it on these,
    or it keys on numbers that had no bearing on the answer and serves one budget's measurement
    to another's question.
    """
    holder = _SESSION_BASELINE.get()
    return holder.budgets if holder is not None else None


def build_session_baseline(
    test_functions: list[Callable[..., None]],
    target_files: set[str],
    timeout_ms: float = 5000,
    trace_budget_s: float | None = DEFAULT_TRACE_BUDGET_S,
    trace_progress: Callable[[int, int, float], None] | None = None,
    trace_session_budget_s: float | None = DEFAULT_TRACE_SESSION_BUDGET_S,
    project_root: str | None = None,
) -> SessionBaseline:
    """Run the suite-global baseline passes ONCE. See :class:`SessionBaseline`.

    ``project_root`` turns on the PERSISTENT cache (`Wesker.trace_cache`) and is the difference
    between once-per-session and once-per-suite-state. Both passes here measure constants, and a
    ContextVar forgets them at process exit, so a CLI — one target per invocation — re-measured
    the whole suite for every function. `None` keeps the pre-cache behaviour exactly.

    ``inert`` is computed by running each test AS-IS (no patch): with the original code
    in place that is precisely "does this test fail regardless of any mutation?", which
    is what bars it from kill attribution.

    ``trace_budget_s`` caps EACH test's traced pass; the names it cuts land on
    ``SessionBaseline.truncated`` for the caller to report. The run passes below already have
    ``timeout_ms``; the TRACE pass had no bound at all, and it is the slower of the two by far
    (a callback per line). Because this baseline is computed once and reused by every function,
    one heavy test stalls the whole session before a single mutant runs — the failure mode is a
    silent hang, not a slow answer. ``None`` = unbounded = the historical behavior.
    """
    from Wesker import trace_cache  # local: trace_cache imports nothing from engine

    # Both passes below measure a CONSTANT: the trace is function-independent (see `trace_suite`),
    # and so is "does this test pass on the unmutated original". They were re-run per invocation
    # because the result lived in a ContextVar. Persisted, the first target in a repo pays and
    # every later one does not. Off (project_root=None) it behaves exactly as before.
    budgets = (trace_budget_s, trace_session_budget_s)
    targets_fp = trace_cache.targets_fingerprint(target_files) if project_root else ""
    cache = (
        trace_cache.load(project_root, targets_fp, budgets) if project_root else None
    )
    before = len(cache) if cache is not None else 0

    truncated: set[str] = set()
    traced = _trace_suite(
        test_functions,
        target_files,
        trace_budget_s,
        truncated,
        trace_progress,
        trace_session_budget_s,
        cache,
    )
    failing: list[str] = []
    inert: set[int] = set()
    # `inert` is keyed by id() — a fact about THIS heap — so it can never be read from disk.
    # The NAMES are what persist; the ids are rebuilt here against the live callables. Same
    # information, addressed by something that survives a process boundary.
    cached_failing, cached_inert = (
        trace_cache.load_outcomes(project_root)
        if (project_root and cache)
        else ([], [])
    )
    reuse_outcomes = bool(cache) and before > 0 and len(cache) == before
    if reuse_outcomes:
        inert_names = set(cached_inert)
        failing = list(cached_failing)
        inert = {
            id(t) for t in test_functions if getattr(t, "__name__", "") in inert_names
        }
    else:
        inert_names_out: list[str] = []
        for test_fn in test_functions:
            outcome = _run_test_with_timeout(test_fn, None, True, timeout_ms)
            if outcome is not None:
                inert.add(id(test_fn))
                inert_names_out.append(getattr(test_fn, "__name__", "unknown"))
                if outcome == "assertion":
                    # An assertion that fails on correct code is a WRONG EXPECTATION — the
                    # narrower thing failing_on_baseline reports to a human. Other outcomes
                    # are ambiguous and are barred from attribution without accusation.
                    failing.append(getattr(test_fn, "__name__", "unknown"))
        cached_inert = inert_names_out
    if project_root and cache is not None and not truncated:
        # Never persist a truncated pass: what a budget cut is absent, not zero, and a cache
        # that remembers the cut serves a false gap forever.
        trace_cache.save(
            project_root, targets_fp, budgets, cache, failing, list(cached_inert)
        )
    return SessionBaseline(traced, failing, inert, len(test_functions), truncated)


def _baseline_failures(
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any] | None,
    qualname: str | None,
    timeout_ms: float = 5000,
) -> set[int]:
    """``id()`` of every test that FAILS against the UNMUTATED function, under
    ``evaluate_mutant``'s own call convention.

    Such a test fails no matter what the mutation does, so crediting it with a kill
    measures the harness, not the suite. It cannot distinguish correct code from a
    mutant and must be barred from attribution entirely.

    Keyed by identity, not name: parametrized cases share a ``__name__``, and only
    some of them may be runnable.

    WHY this exists next to ``failing_on_baseline`` rather than inside it: the two
    answer different questions and must not be merged. ``failing_on_baseline`` asks
    "is this test's EXPECTATION wrong?" and counts only ``AssertionError`` on a bare
    ``test_fn()``, staying deliberately conservative because its answer is shown to a
    human as "your test may be broken". This asks "can this test distinguish anything
    AT ALL?" — any outcome other than pass disqualifies it, because a test that
    cannot run cannot detect. Conflating them either accuses innocent tests or
    credits inert ones.

    Measured on prism/economics.py::analyze (131 mutants): TestAnalyze.test_basic_output
    was credited with 123 crash "kills" while calling ``analyze`` exactly ZERO times —
    it is a bound method needing ``(tmp_path, monkeypatch)`` fixtures, so it raised
    TypeError before reaching the function under test, identically on the original.
    """
    if original_func is None or not qualname:
        return set()
    func_name = qualname.split(".")[-1]
    # A baseline is only meaningful against the GENUINE original. Several callers
    # deliberately STUB original_func (e.g. ``lambda *_a: None``) when they only want
    # the mutation loop; running the suite against a stub makes every real test "fail
    # on baseline", drops the entire suite, and reports the resulting survivors as an
    # honest result. Identity check, matching the convention the tracer already uses:
    # the callable must actually be the function we are mutating.
    probe = _unwrap_descriptor(original_func)
    if getattr(probe, "__name__", None) != func_name:
        return set()
    inert: set[int] = set()
    for test_fn in test_functions:
        patched, saved, patch_target = _patch_mutant_into_test(
            test_fn, qualname, original_func
        )
        try:
            if _run_test_with_timeout(test_fn, probe, patched, timeout_ms) is not None:
                inert.add(id(test_fn))
        except Exception:  # noqa: BLE001 — an unrunnable baseline is itself inert
            inert.add(id(test_fn))
        finally:
            _unpatch_mutant(patched, saved, patch_target, func_name)
    return inert


def _build_test_scope(
    func_node: ast.FunctionDef,
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any] | None,
    scope_tests: bool,
    precomputed_line_data: tuple[dict[str, list[int]], list[str]] | None = None,
    qualname: str | None = None,
    trace_budget_s: float | None = None,
    truncated: set[str] | None = None,
    trace_progress: Callable[[int, int, float], None] | None = None,
    trace_session_budget_s: float | None = None,
) -> tuple[
    Callable[[Mutant], list[Callable[..., None]]],
    dict[str, list[int]],
    list[int],
    list[str],
]:
    """Build the mutant -> covering-tests resolver shared by both profiling entry points.

    Two independent filters, in order:

    1. ATTRIBUTION. A test that fails against the UNMUTATED original fails regardless
       of any mutation, so it cannot distinguish a mutant from correct code. Such a
       test is dropped from the kill loop entirely (see ``_baseline_failures``). This
       is the honesty guard: without it, one unrunnable test manufactures a 100% kill
       rate. It applies to the scoped AND unscoped paths — they share this resolver,
       so a defect here cannot hide on one side.

    2. SCOPING. A test can only kill a mutant if it EXECUTES the mutated line, so
       evaluating each mutant against just the tests covering that line yields
       identical verdicts at a fraction of the cost. Verdict-EXACTNESS rests on:

         * an empty covering set is only meaningful for a line the coverage data COULD
           have described. A line outside the traced denominator means "no data", not
           "no test", and must fall back to the full set.

    Filter 1 is what makes filter 2 sound without a compensation hack. A fails-on-
    baseline test used to be force-joined to EVERY scoped set, so that scoped matched
    unscoped — the two agreed, but on an inflated number. Barring it from attribution
    makes both honest, and the two still agree.

    Baseline data comes from three places, in precedence order: an explicit
    ``precomputed_line_data``; a live-session :class:`SessionBaseline` (computed once
    for the whole suite — see that class for why the per-function passes were redundant);
    or, failing both, the per-function passes themselves.

    Returns ``(_tests_for, line_cov, exec_lines, failing)`` so callers can also report
    the line-coverage axis. Lives here, used by both ``run_function_profiling`` and
    ``run_function_converged``, so the two can never drift apart on soundness.
    """
    exec_lines = sorted(_executable_lines(func_node))
    # Resolving the holder is what BUILDS the baseline (see `LazySessionBaseline`). Reaching
    # here means a real profiling pass is under way and the coverage is about to be used, which
    # is exactly the demand the laziness waits for.
    session = session_baseline()
    inert: set[int] = set()
    if precomputed_line_data is not None:
        # An adaptive-probe caller already ran this baseline over the same tests+function;
        # reuse it so a probe + follow-up run don't trace twice. Deterministic, so the
        # reused map is identical to what a fresh trace would produce here.
        line_cov, failing = precomputed_line_data
        inert = _baseline_failures(test_functions, original_func, qualname)
    elif session is not None:
        # Suite-global baseline, already paid for once. Only the per-function
        # intersection is left, and it is a set operation over data in hand.
        target_file = getattr(
            getattr(original_func, "__code__", None), "co_filename", None
        )
        line_cov = _coverage_from_trace(
            session.traced, target_file or "", set(exec_lines)
        )
        failing = session.failing
        inert = session.inert
        if truncated is not None:
            truncated |= (
                session.truncated
            )  # the suite-level cut is this function's cut too
    elif original_func is not None:
        line_cov = _trace_line_coverage(
            test_functions,
            original_func,
            set(exec_lines),
            trace_budget_s,
            truncated,
            trace_progress,
            trace_session_budget_s,
        )
        failing = _failing_on_baseline(test_functions, original_func)
        inert = _baseline_failures(test_functions, original_func, qualname)
    else:
        line_cov, failing = {}, []

    # Filter 1 — bar tests that cannot distinguish anything from the kill loop.
    usable = (
        [t for t in test_functions if id(t) not in inert] if inert else test_functions
    )

    # Parametrized cases share a __name__, so one name maps to many callables.
    tests_by_name: dict[str, list[Callable[..., None]]] = {}
    for _tf in usable:
        tests_by_name.setdefault(getattr(_tf, "__name__", "unknown"), []).append(_tf)
    covering_by_line: dict[int, list[Callable[..., None]]] = {}
    if scope_tests and line_cov:
        for tname, lines in line_cov.items():
            fns = tests_by_name.get(tname, [])
            for ln in lines:
                covering_by_line.setdefault(ln, []).extend(fns)

    exec_line_set = set(exec_lines)

    def _tests_for(mutant: Mutant) -> list[Callable[..., None]]:
        if not scope_tests or not line_cov or mutant.mutated_line is None:
            return usable  # cannot scope safely — run the full usable set
        if mutant.mutated_line not in exec_line_set:
            return usable  # no data for this line — cannot scope safely
        return covering_by_line.get(mutant.mutated_line, [])

    return _tests_for, line_cov, exec_lines, failing


def dimension_budget(
    func_node: ast.FunctionDef,
    category: MutationCategory,
    docstring_positions: set[tuple[int, int]] | None = None,
) -> int:
    """The DOF-derived per-category budget: this function's degrees of freedom.

    A category's cover sets are SINGLETONS — each target site pins exactly one
    behavioral dimension (``VALUE:int``, ``ARITHMETIC:Add``, …), recorded by the
    category's own mutator in record mode. Under singleton covers the greedy
    round-robin of :func:`_greedy_dimension_order` covers ``min(m, D)`` of ``D``
    dimensions after ``m`` picks, so ``m = D`` covers every dimension EXACTLY —
    greedy is optimal here, not merely within ``(1−1/e)``.

    ``D`` is therefore the budget at which one pass reaches full DOF coverage,
    and any larger budget buys no additional dimension. It is the natural budget
    the theory names; a hardcoded constant is either short of it (partial DOF) or
    past it (redundant mutants within an already-covered dimension).

    STATE and EXCEPTION are generated as independent sub-modes with separate target
    indices, each selected against its own budget, so their DOF is the sum over
    sub-modes. A category listed here MUST match its generator's sub-mode list, or the
    budget disagrees with what is generated: too low silently truncates the category's
    coverage, too high spends budget on mutants that do not exist.
    Static: AST walk only, no compilation, no execution.
    """
    if category is MutationCategory.STATE:
        return sum(
            _live_dimension_count(_record_state_dimensions(func_node, mode))
            for mode in ("remove_assign", "return_none")
        )
    if category is MutationCategory.EXCEPTION:
        return sum(
            _live_dimension_count(_record_exception_dimensions(func_node, mode))
            for mode, _desc in _EXCEPTION_SUB_MODES
        )
    return _live_dimension_count(
        _record_dimensions(func_node, category, docstring_positions)
    )


def dof_universe(
    func_node: ast.FunctionDef,
    categories: set[MutationCategory],
) -> int:
    """Total degrees of freedom of a function — the DOF-coverage denominator.

    The behavioral-dimension analogue of :func:`estimate_universe_size`: that
    counts mutation *targets*, this counts the distinct *dimensions* those targets
    pin. Reported alongside the mutant universe so a run states what fraction of
    the DOF space it covered.
    """
    return sum(dimension_budget(func_node, cat) for cat in categories)


def _greedy_dimension_order(keys: list[str]) -> list[int]:
    """Greedy submodular order over target indices by their dimension keys.

    Round-robins across distinct (live) dimensions in first-appearance order:
    round 0 takes one index per dimension (each a marginal-coverage-1 pick),
    round 1 a second per dimension, and so on. Prefixes therefore maximize the
    number of distinct behavioral dimensions covered — the greedy max-coverage
    schedule whose gap contracts by (1−1/k) per pick. Dead sites (no mutant)
    sink to the end. Deterministic; no seed.
    """
    groups: dict[str, list[int]] = {}
    key_order: list[str] = []
    for i, k in enumerate(keys):
        if k not in groups:
            groups[k] = []
            key_order.append(k)
        groups[k].append(i)

    live = [k for k in key_order if not _is_dead(k)]
    dead = [k for k in key_order if _is_dead(k)]

    result: list[int] = []
    depth = 0
    while True:
        progressed = False
        for k in live:
            g = groups[k]
            if depth < len(g):
                result.append(g[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    for k in dead:
        result.extend(groups[k])
    return result


def _select_greedy(
    keys: list[str],
    target_count: int,
    limit: int,
    pass_index: int,
) -> list[int]:
    """Select ``limit`` target indices for pass ``pass_index`` by greedy coverage.

    Pass p takes the window [p·limit, (p+1)·limit) of the greedy order, so the
    union across passes is a growing prefix of the (1−1/e)-optimal schedule.
    Falls back to the top window once the order is exhausted (converged).
    """
    order = [i for i in _greedy_dimension_order(keys) if i < target_count]
    seen = set(order)
    order += [i for i in range(target_count) if i not in seen]  # defensive: full cover
    lo = pass_index * limit
    window = order[lo : lo + limit]
    return window if window else order[:limit]


def generate_mutants(
    func_node: ast.FunctionDef,
    categories: set[MutationCategory],
    max_per_category: int | None = 0,
    seed: int | None = None,
    category_order: list[MutationCategory] | None = None,
    greedy: bool = True,
    pass_index: int = 0,
) -> list[Mutant]:
    """Generate mutants for a function across specified categories.

    Args:
        func_node: The function AST node to mutate.
        max_per_category: Max mutants per category. ``None`` (DOF mode) derives
              the budget per category from the function itself —
              :func:`dimension_budget`, the count of distinct behavioral
              dimensions — so one pass covers every dimension exactly once and
              no budget is spent re-covering one. ``0`` = unlimited (exhaustive);
              a positive int pins an explicit budget.
        seed: Legacy deterministic shuffle seed. Only consulted when
              ``greedy=False`` (see below); retained for backward compatibility
              and the exhaustive/random-sampling fallback. ``None`` preserves
              AST-walk order.
        category_order: Optional priority ordering of categories. When provided,
              mutants are generated in this order (high-priority first). Categories
              in this list but not in ``categories`` are skipped. When None, uses
              alphabetical order.
        greedy: When True (default) and ``max_per_category > 0``, targets are
              selected by greedy behavioral-dimension coverage (Layer 2), which
              reaches ≥(1−1/e) of the optimally-coverable dimensions per budget
              (greedy_coverage_bound.lean) rather than sampling randomly.
        pass_index: Convergence pass. Pass p takes window
              [p·max_per_category, (p+1)·max_per_category) of the greedy order,
              so the union across passes grows the (1−1/e)-optimal prefix.
    """
    mutants: list[Mutant] = []
    ds_pos = _docstring_positions(func_node)

    if category_order is not None:
        order = [c for c in category_order if c in categories]
        # Append any categories not in the ordering (shouldn't happen, but defensive)
        for c in sorted(categories, key=lambda c: c.value):
            if c not in order:
                order.append(c)
    else:
        order = sorted(categories, key=lambda c: c.value)

    for cat in order:
        # STATE needs special handling: two independent sub-modes with
        # separate target counts so indices align with the transformer.
        if cat == MutationCategory.STATE:
            mutants.extend(
                _generate_state_mutants(
                    func_node, max_per_category, greedy=greedy, pass_index=pass_index
                )
            )
            continue
        # EXCEPTION, like STATE, carries independent sub-modes with separate target
        # index spaces, so it cannot go through the single-transformer path below.
        if cat == MutationCategory.EXCEPTION:
            mutants.extend(
                _generate_exception_mutants(
                    func_node, max_per_category, greedy=greedy, pass_index=pass_index
                )
            )
            continue

        target_count = _count_targets(func_node, cat)
        # DOF mode (max_per_category is None): the budget IS this category's
        # degrees of freedom, so one pass covers every dimension exactly once.
        keys = _record_dimensions(func_node, cat, ds_pos) if greedy else []
        budget = (
            _live_dimension_count(keys)
            if max_per_category is None
            else max_per_category
        )
        limit = min(target_count, budget) if budget > 0 else target_count

        if budget > 0 and target_count > limit:
            if greedy:
                # Layer 2: greedy submodular selection by behavioral dimension.
                selected = _select_greedy(keys, target_count, limit, pass_index)
            elif seed is not None:
                # Legacy fallback: deterministic pseudo-random shuffle.
                indices = _stable_target_order(
                    list(range(target_count)), seed=seed, category=cat.value
                )
                selected = indices[:limit]
            else:
                selected = list(range(limit))
        else:
            # Exhaustive for this category (budget ≥ targets): order is irrelevant.
            selected = list(range(limit))

        for i in selected:
            mutated_tree = copy.deepcopy(func_node)
            transformer, desc = _make_transformer(cat, i, ds_pos)
            mutated_node = transformer.visit(mutated_tree)
            ast.fix_missing_locations(mutated_node)

            if transformer.applied:
                mid = _content_mutant_id(cat, mutated_node)
                mutants.append(
                    Mutant(
                        category=cat,
                        original_node=func_node,
                        mutated_node=mutated_node,
                        description=f"{mid}: {desc}",
                        location=getattr(func_node, "lineno", 0),
                        mutant_id=mid,
                        target_index=i,
                        mutated_line=transformer.mutated_lineno,
                        dimension=keys[i] if i < len(keys) else "",
                    )
                )

    return mutants


def _stable_target_order(indices: list[int], *, seed: int, category: str) -> list[int]:
    """Return a deterministic pseudo-shuffled order for target indices."""
    return sorted(indices, key=lambda idx: _stable_target_key(seed, category, idx))


def _stable_target_key(seed: int, category: str, idx: int) -> bytes:
    """Build a stable hash key for deterministic mutant sampling order."""
    payload = f"{seed}:{category}:{idx}".encode()
    return hashlib.sha256(payload).digest()


def _make_transformer(
    category: MutationCategory,
    index: int,
    docstring_positions: set[tuple[int, int]] | None = None,
) -> tuple[_BaseMutator, str]:
    """Create the appropriate transformer for a category + target index."""
    if category == MutationCategory.VALUE:
        return _ValueMutator(
            index, docstring_positions
        ), "replace constant with boundary value"
    if category == MutationCategory.BOUNDARY:
        return _BoundaryMutator(index), "off-by-one comparison"
    if category == MutationCategory.SWAP:
        return _SwapMutator(index), "transpose call arguments"
    if category == MutationCategory.STATE:
        return _StateMutator(index, "remove_assign"), "remove state assignment"
    if category == MutationCategory.TYPE:
        return _TypeMutator(index), "replace isinstance with True"
    if category == MutationCategory.ARITHMETIC:
        return _ArithmeticMutator(index), "replace arithmetic operator"
    if category == MutationCategory.LOGICAL:
        return _LogicalMutator(index), "replace logical operator"
    if category == MutationCategory.STMT:
        return _StmtMutator(index), "delete statement"
    if category == MutationCategory.EXCEPTION:
        # Reached only by a caller doing single-transformer generation; the normal path
        # routes EXCEPTION through _generate_exception_mutants (independent sub-modes).
        return _ExceptionMutator(index, "raise_type"), "replace raised exception type"
    msg = f"Unknown category: {category}"
    raise ValueError(msg)


@dataclass
class BoundaryInput:
    """A synthesized boundary test input from a Compare mutation."""

    parameter: str
    boundary_value: int | float
    inputs: list[tuple[str, int | float]]  # [(param, value), ...]


def extract_boundary_inputs(mutant: Mutant) -> list[BoundaryInput]:
    """Extract boundary test inputs from a BOUNDARY mutant.

    Walks the original Compare node to find the parameter name and constant
    involved, then synthesizes inputs at boundary, boundary-1, boundary+1.
    Only works for Compare nodes comparing a Name to a numeric Constant.
    """
    if mutant.category != MutationCategory.BOUNDARY:
        return []

    results: list[BoundaryInput] = []
    orig_compares = [
        n for n in ast.walk(mutant.original_node) if isinstance(n, ast.Compare)
    ]
    mut_compares = [
        n for n in ast.walk(mutant.mutated_node) if isinstance(n, ast.Compare)
    ]

    for orig_cmp, mut_cmp in zip(orig_compares, mut_compares, strict=False):
        # Find the op that changed
        for orig_op, mut_op in zip(orig_cmp.ops, mut_cmp.ops, strict=False):
            if type(orig_op) is type(mut_op):
                continue
            # Found the mutated comparison — extract param + constant
            param, const = _extract_compare_parts(orig_cmp)
            if param and const is not None and isinstance(const, (int, float)):
                offsets = [0, -1, 1]
                inputs = [(param, const + off) for off in offsets]
                results.append(
                    BoundaryInput(
                        parameter=param,
                        boundary_value=const,
                        inputs=inputs,
                    )
                )
    return results


def _extract_compare_parts(
    cmp_node: ast.Compare,
) -> tuple[str | None, int | float | None]:
    """Extract (parameter_name, constant_value) from a Compare node.

    Handles both ``x < 10`` and ``10 < x`` orientations.
    """
    left = cmp_node.left
    comparators = cmp_node.comparators

    if isinstance(left, ast.Name) and len(comparators) == 1:
        comp = comparators[0]
        if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
            return left.id, comp.value
    if (
        isinstance(left, ast.Constant)
        and isinstance(left.value, (int, float))
        and len(comparators) == 1
        and isinstance(comparators[0], ast.Name)
    ):
        return comparators[0].id, left.value
    return None, None


# ── Mutant Evaluation ─────────────────────────────────────────────


def _is_private_copy(module_name: str, private_prefix: str) -> bool:
    """True for a module belonging to the private self-profiling copy of this package.

    THE ONE THING THIS EXISTS FOR. When Wesker profiles Wesker, the engine driving the run is
    imported a second time under ``private_prefix`` (see ``Wesker.self_profile``) so that the
    public ``Wesker.*`` modules can be mutated freely without the engine eating its own mutant
    mid-flight. Both copies are compiled from the SAME source files, so they carry the SAME
    ``co_filename`` — which means ``_co_filename_matches`` cannot tell them apart, and the
    patch loop would install the mutant into the private copy too, reintroducing exactly the
    self-mutation this design removes.

    The module NAME is the only thing that differs between the two copies, so it is the only
    thing that can discriminate them. Matching on the name rather than on the filename is
    therefore not a stylistic choice; it is the whole mechanism.

    Takes the NAME rather than the module so it stays a total function of two strings: the
    caller does the ``getattr``. A predicate that took the module object could only be
    exercised with a live module, which is not expressible as a test input — and a guard whose
    correctness cannot be pinned is not a guard worth having.

    Costs nothing when no private copy exists: no module is named ``_wesker_self`` in an
    ordinary run, so this is a failed prefix check per module and the patch behaviour for every
    other project is bit-for-bit unchanged.
    """
    return module_name.startswith(private_prefix)


def _patch_module_qualified(
    func_name: str | None,
    mutated_obj: Any,
    source_path: str | None,
    qualname: str | None = None,
) -> list[tuple[Any, Any]]:
    """Patch every module-level binding of the ORIGINAL function to the mutant.

    Real-world suites call functions through the imported module
    (``import pkg as p; p.func(...)``) rather than a bare name in the test's
    globals. Patching only the test globals leaves those call sites pointing at
    the original, so the mutant is never exercised. This patches the function in
    its defining module *and* every module that re-exports it (``pkg.func``,
    ``import as`` aliases share the same module object), so module-qualified
    call sites hit the mutant.

    When ``qualname`` is class-qualified (``Class.method``), it ALSO patches the
    method on its owner class in the defining module. Without this, a suite that
    exercises a method via a factory (``make_thing(...).method()``) WITHOUT
    importing the class leaves the owner absent from the test namespace, so both
    the test-namespace patch and the module-level patch miss it and the mutant is
    never installed — a false "survivor" (an impact-map/patch blind spot, not a
    test gap).

    Matching on the original's ``__code__.co_filename == source_path`` means
    only the specific function under test is patched — unrelated same-named
    functions in other modules are left untouched. Returns ``[(target, saved)]``
    for restoration (``target`` is a module or an owner class; the caller restores
    ``func_name`` on it). No-ops (empty list) when ``source_path`` is unavailable,
    so the caller's behaviour and output are unchanged in that case.
    """
    if not func_name or not source_path:
        return []
    import sys

    from Wesker.self_profile import PRIVATE_PREFIX

    saved: list[tuple[Any, Any]] = []
    for mod in list(sys.modules.values()):
        if mod is None or _is_private_copy(
            getattr(mod, "__name__", ""), PRIVATE_PREFIX
        ):
            continue
        try:
            obj = getattr(mod, func_name, None)
        except Exception:
            continue
        code = getattr(obj, "__code__", None)
        if code is None:
            continue
        try:
            if _co_filename_matches(code.co_filename, source_path):
                setattr(mod, func_name, mutated_obj)
                saved.append((mod, obj))
        except Exception:
            continue

    # Class-method owner patch: resolve ``Class.method`` within the defining module and patch the
    # method on the class, so instance-dispatch call sites hit the mutant even when the class was never
    # imported into the test namespace. Only a class that DEFINES the method directly (not inherited)
    # and whose original method lives in source_path is touched — same precision as the module loop.
    if qualname and "." in qualname:
        owner_parts = qualname.split(".")[:-1]
        method = qualname.split(".")[-1]
        for mod in list(sys.modules.values()):
            if mod is None or _is_private_copy(
                getattr(mod, "__name__", ""), PRIVATE_PREFIX
            ):
                continue
            owner: Any = mod
            try:
                for part in owner_parts:
                    owner = getattr(owner, part)
            except Exception:
                continue
            if not isinstance(owner, type) or method not in getattr(
                owner, "__dict__", {}
            ):
                continue
            existing = _get_raw_attr(owner, method)
            code = getattr(_unwrap_descriptor(existing), "__code__", None)
            if code is None:
                continue
            try:
                if _co_filename_matches(code.co_filename, source_path):
                    setattr(
                        owner, method, _preserve_descriptor_shape(existing, mutated_obj)
                    )
                    saved.append((owner, existing))
            except Exception:
                continue
    return saved


def _co_filename_matches(co_filename: str | None, source_path: str | None) -> bool:
    """True when a code object's file is the function-under-test's source. `source_path` may be absolute
    OR project-RELATIVE (callers derive it from func_key = 'rel/path.py::Q'), while co_filename is
    absolute — so accept exact abspath equality OR an absolute co_filename ending in the normalized
    relative source_path. The relative match is bounded to a full path-segment suffix so 'a/b.py' does
    not match '.../xa/b.py'."""
    if not co_filename or not source_path:
        return False
    import os

    try:
        a = os.path.abspath(co_filename).replace("\\", "/")
        if a == os.path.abspath(source_path).replace("\\", "/"):
            return True
    except Exception:
        return False
    rel = source_path.replace("\\", "/").lstrip("./")
    return bool(rel) and (a == rel or a.endswith("/" + rel))


def _patch_mutant_into_test(
    test_fn: Callable[..., None],
    qualname: str | None,
    mutated_obj: Any,
) -> tuple[bool, Any, Any]:
    """Patch mutated function into the test's namespace.

    Tries __globals__ first (works for dynamically imported modules),
    then falls back to inspect.getmodule.

    Returns (patched, saved_original, patch_target) where patch_target
    is either a dict (__globals__) or a module object.
    """
    if not qualname:
        return False, None, None

    func_name = qualname.split(".")[-1]

    # Primary: use __globals__ — the test function's defining module globals.
    # Works for bound methods, regular functions, and dynamically imported modules.
    test_globals = getattr(test_fn, "__globals__", None)
    # For bound methods, __globals__ is on the underlying function
    if test_globals is None:
        underlying = getattr(test_fn, "__func__", None)
        if underlying is not None:
            test_globals = getattr(underlying, "__globals__", None)

    closure_bindings = _get_closure_bindings(test_fn)

    import inspect

    test_module = inspect.getmodule(test_fn)

    owner = _resolve_qualified_owner(
        test_globals, closure_bindings, test_module, qualname
    )
    if owner is not None and hasattr(owner, func_name):
        saved = _get_raw_attr(owner, func_name)
        setattr(owner, func_name, _preserve_descriptor_shape(saved, mutated_obj))
        return True, saved, owner

    closure_cell = _find_closure_cell(closure_bindings, func_name)
    if closure_cell is not None:
        saved = closure_cell.cell_contents
        closure_cell.cell_contents = _preserve_closure_binding_shape(saved, mutated_obj)
        return True, saved, ("closure_cell", closure_cell)

    if test_globals is not None and func_name in test_globals:
        saved = test_globals[func_name]
        test_globals[func_name] = _preserve_closure_binding_shape(saved, mutated_obj)
        return True, saved, test_globals

    # Fallback: inspect.getmodule (works for regular module-level functions)
    if test_module is not None and hasattr(test_module, func_name):
        saved = getattr(test_module, func_name)
        setattr(
            test_module, func_name, _preserve_closure_binding_shape(saved, mutated_obj)
        )
        return True, saved, test_module

    return False, None, None


def _resolve_qualified_owner(
    test_globals: dict[str, Any] | None,
    closure_bindings: list[tuple[str, Any, Any]],
    test_module: Any,
    qualname: str,
) -> Any:
    """Resolve the owning object for a qualified symbol like ``Class.method``."""
    if "." not in qualname:
        return None

    import inspect

    owner_parts = qualname.split(".")[:-1]
    root_name = owner_parts[0]
    candidates: list[Any] = []
    seen: set[int] = set()

    def _add_candidate(obj: Any) -> None:
        if obj is None:
            return
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        candidates.append(obj)

    def _add_from_value(value: Any) -> None:
        if value is None:
            return
        if inspect.ismodule(value) and hasattr(value, root_name):
            _add_candidate(getattr(value, root_name))
            return
        if isinstance(value, type):
            if value.__name__ == root_name:
                _add_candidate(value)
            return
        bound_self = getattr(value, "__self__", None)
        if bound_self is not None:
            owner = bound_self if isinstance(bound_self, type) else type(bound_self)
            if getattr(owner, "__name__", "") == root_name:
                _add_candidate(owner)
            return
        owner_type = type(value)
        if getattr(owner_type, "__name__", "") == root_name:
            _add_candidate(owner_type)

    for _, value, _ in closure_bindings:
        _add_from_value(value)

    if test_globals is not None:
        _add_candidate(test_globals.get(root_name))
        for value in test_globals.values():
            _add_from_value(value)

    if test_module is not None and hasattr(test_module, root_name):
        _add_candidate(getattr(test_module, root_name))

    for candidate in candidates:
        current = candidate
        for part in owner_parts[1:]:
            if not hasattr(current, part):
                current = None
                break
            current = getattr(current, part)
        if current is not None:
            return current
    return None


def _get_closure_bindings(test_fn: Callable[..., None]) -> list[tuple[str, Any, Any]]:
    """Extract ``(freevar_name, value, cell)`` bindings from a test closure."""
    underlying = getattr(test_fn, "__func__", test_fn)
    cells = getattr(underlying, "__closure__", None) or ()
    code = getattr(underlying, "__code__", None)
    freevars = getattr(code, "co_freevars", ())

    bindings: list[tuple[str, Any, Any]] = []
    for name, cell in zip(freevars, cells, strict=False):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        bindings.append((name, value, cell))
    return bindings


def _find_closure_cell(
    closure_bindings: list[tuple[str, Any, Any]],
    func_name: str,
) -> Any:
    """Find the closure cell that directly binds a symbol name."""
    for name, _, cell in closure_bindings:
        if name == func_name:
            return cell
    return None


def _get_raw_attr(owner: Any, attr_name: str) -> Any:
    """Get the raw stored attribute to preserve descriptor identity."""
    from collections.abc import Mapping

    namespace = getattr(owner, "__dict__", None)
    if isinstance(namespace, Mapping) and attr_name in namespace:
        return namespace[attr_name]
    return getattr(owner, attr_name)


def _unwrap_descriptor(obj: Any) -> Any:
    """Extract the underlying callable from classmethod/staticmethod wrappers."""
    if isinstance(obj, (classmethod, staticmethod)):
        return obj.__func__
    return obj


def _preserve_descriptor_shape(original: Any, mutated_obj: Any) -> Any:
    """Wrap the mutant to match the original descriptor semantics."""
    if isinstance(original, classmethod):
        if isinstance(mutated_obj, classmethod):
            return mutated_obj
        return classmethod(_unwrap_descriptor(mutated_obj))
    if isinstance(original, staticmethod):
        if isinstance(mutated_obj, staticmethod):
            return mutated_obj
        return staticmethod(_unwrap_descriptor(mutated_obj))
    return _unwrap_descriptor(mutated_obj)


def _preserve_closure_binding_shape(original: Any, mutated_obj: Any) -> Any:
    """Wrap the mutant to match common closure-bound callable shapes."""
    if isinstance(original, types.MethodType):
        return types.MethodType(_unwrap_descriptor(mutated_obj), original.__self__)
    return _unwrap_descriptor(mutated_obj)


def _unpatch_mutant(
    patched: bool,
    saved: Any,
    patch_target: Any,
    func_name: str | None,
) -> None:
    """Restore the original function after mutation evaluation."""
    if not patched or saved is None or func_name is None:
        return
    if isinstance(patch_target, dict):
        patch_target[func_name] = saved
    elif (
        isinstance(patch_target, tuple)
        and len(patch_target) == 2
        and patch_target[0] == "closure_cell"
    ):
        patch_target[1].cell_contents = saved
    else:
        setattr(patch_target, func_name, saved)


def evaluate_mutant(
    mutant: Mutant,
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any],
    timeout_ms: float = 5000,
    qualname: str | None = None,
    record_all_killers: bool = False,
    source_path: str | None = None,
) -> MutantResult:
    """Evaluate a mutant against test functions.

    Compiles the mutated function, then monkey-patches it into each test's
    module namespace before invoking the test with zero args (standard pytest
    contract). The original function is restored after each test.

    Kill attribution follows VALUE-SPECIFICATION PRECEDENCE (crash-as-spec): an
    *assertion* kill pins the return value, so it is the strongest verdict and
    ends the search immediately. A *crash*/*timeout* kill only proves the mutant
    runs differently — it does not pin the value — so it is provisional: the
    search keeps going, looking for a later test that kills by assertion, and
    only settles for the crash/timeout verdict once the covering tests are
    exhausted. This makes ``killed_by`` independent of test order: a mutant that
    ANY test kills by assertion is recorded value-killed, never a crash-survivor.

    With ``record_all_killers=True`` (full-matrix mode) every test is run and
    ``killed_by_tests`` records *all* killers; ``killed_by`` is ``"assertion"``
    when any killer pinned the value, else the first reason. Full-matrix mode
    shares ``timeout_ms`` across the whole test set, so callers must budget it
    for the full suite.
    """
    start = time.monotonic()

    # The module-qualified patch (module-level bindings AND class-method owners) needs the ABSOLUTE
    # source path to match call-site objects by co_filename. Callers pass qualname but not source_path,
    # so derive it from the original function's own code object — authoritative and absolute. Without
    # this the whole module-qualified patch was inert, so a method exercised via a factory whose class
    # is not imported into the test namespace (e.g. make_role_frame(...).relationP()) was a false survivor.
    if source_path is None:
        source_path = getattr(
            getattr(original_func, "__code__", None), "co_filename", None
        )

    # Compile mutated function
    try:
        module_ast = ast.Module(body=[mutant.mutated_node], type_ignores=[])  # type: ignore[list-item]
        ast.fix_missing_locations(module_ast)
        code = compile(module_ast, "<mutant>", "exec")
        # Seed the mutant's namespace with the source module's globals so it can
        # resolve sibling helpers, module constants, and imports. Without this a
        # function that calls a module-level helper raises NameError under EVERY
        # mutant — a false all-crash 100% that hides whether the mutation's
        # behavior is actually caught. Degrades to an empty namespace (the prior
        # behavior) when the caller passes no original_func.
        namespace: dict[str, Any] = dict(
            getattr(original_func, "__globals__", None) or {}
        )
        exec(code, namespace)  # noqa: S102  # nosec B102 — intentional: compiling AST mutants
        func_name = getattr(mutant.mutated_node, "name", None)
        mutated_obj = namespace.get(func_name) if func_name else None
        if mutated_obj is None:
            return MutantResult(
                mutant=mutant,
                killed=True,
                killed_by="crash",
                elapsed_ms=_elapsed(start),
            )
    except Exception:
        return MutantResult(
            mutant=mutant,
            killed=True,
            killed_by="crash",
            elapsed_ms=_elapsed(start),
        )

    # Patch module-qualified bindings (pkg.func / mi.func) to the mutant for the
    # whole evaluation, so tests that call through the module namespace exercise
    # the mutant — not only tests that call a bare imported name. Restored in the
    # finally regardless of how the loop exits. No-op when source_path is absent,
    # so existing callers/output are unchanged.
    module_saved = _patch_module_qualified(
        func_name, mutated_obj, source_path, qualname
    )
    try:
        # Run tests against mutated function
        killers: list[str] = []
        reasons: list[str] = []
        first_reason: str | None = (
            None  # provisional crash/timeout kill (no assertion yet)
        )
        first_killer: str | None = None
        for test_fn in test_functions:
            remaining_ms = timeout_ms - _elapsed(start)
            if remaining_ms <= 0:
                if record_all_killers and killers:
                    break  # budget hit — keep the killers already collected
                if first_reason is not None:
                    break  # already have a crash/timeout kill; settle for it below
                return MutantResult(
                    mutant=mutant,
                    killed=True,
                    killed_by="timeout",
                    elapsed_ms=_elapsed(start),
                )
            # Strategy: monkey-patch the mutated function into the test's namespace
            # so the test calls the mutant instead of the original. Uses __globals__
            # (the test function's defining module globals) which works reliably for
            # both regular imports and dynamically loaded test modules. Falls back to
            # inspect.getmodule for inline test callables without __globals__.
            patch_name = qualname or func_name
            patched, saved, patch_target = _patch_mutant_into_test(
                test_fn, patch_name, mutated_obj
            )
            try:
                result = _run_test_with_timeout(
                    test_fn,
                    _unwrap_descriptor(mutated_obj),
                    patched,
                    remaining_ms,
                )
                # A failure is only a KILL if the mutation CAUSED it. When the mutant
                # could not be patched into the test's namespace, the unpatched path
                # INJECTS it as a positional argument — a contract only Wesker's own
                # inline tests observe. A discovered test with an unfilled fixture
                # parameter receives the mutant AS the fixture and fails on garbage;
                # that failure is about the fixture, not the mutation. Confirm by
                # re-running with the ORIGINAL injected identically: if it fails the
                # same way, the test cannot distinguish the two and detected nothing.
                #
                # Measured on prism/economics.py::analyze: without this,
                # test_nudge_contains_tool_count (another module; never references
                # `analyze`; needs a `tmp_state` fixture) was credited with 118 of 131
                # "assertion kills" — a 100% kill rate that was almost entirely this
                # artifact. failing_on_baseline cannot catch it: it calls test_fn()
                # with no argument, which raises TypeError, and it only counts
                # AssertionError. Only on failure, so a passing suite pays nothing.
                if result is not None and not patched and original_func is not None:
                    result = (
                        None
                        if _outcome_on_original(
                            test_fn,
                            original_func,
                            module_saved,
                            func_name,
                            max(timeout_ms - _elapsed(start), 1.0),
                        )
                        == result
                        else result
                    )
                if result is not None:
                    tname = getattr(test_fn, "__name__", "unknown")
                    if record_all_killers:
                        killers.append(tname)
                        reasons.append(result)
                        if first_reason is None:
                            first_reason = result
                    elif result in ("assertion", "exception"):
                        # Strongest verdict: the value is pinned — stop here. An exception
                        # kill is the same strength as an assertion: both are a test stating
                        # what the function does and the mutant contradicting it. Scanning on
                        # for a "better" kill would find none, and treating it as provisional
                        # would let a later crash-kill overwrite a real pin.
                        return MutantResult(
                            mutant=mutant,
                            killed=True,
                            killed_by=result,
                            test_name=tname,
                            elapsed_ms=_elapsed(start),
                        )
                    elif first_reason is None:
                        # Provisional crash/timeout kill — remember it, but keep
                        # scanning: a later test may pin the value by assertion.
                        first_reason, first_killer = result, tname
            finally:
                _unpatch_mutant(patched, saved, patch_target, func_name)

        if record_all_killers and killers:
            return MutantResult(
                mutant=mutant,
                killed=True,
                killed_by=(
                    "assertion"
                    if "assertion" in reasons
                    else "exception"
                    if "exception" in reasons
                    else first_reason
                ),
                test_name=killers[0],
                killed_by_tests=killers,
                elapsed_ms=_elapsed(start),
            )
        if first_reason is not None:
            # Killed, but only ever by crash/timeout — no test pinned the value.
            return MutantResult(
                mutant=mutant,
                killed=True,
                killed_by=first_reason,
                test_name=first_killer,
                elapsed_ms=_elapsed(start),
            )
        return MutantResult(mutant=mutant, killed=False, elapsed_ms=_elapsed(start))
    finally:
        for _mod, _orig in module_saved:
            try:
                setattr(_mod, func_name, _orig)
            except Exception:
                pass


def _outcome_on_original(
    test_fn: Callable[..., None],
    original_func: Callable[..., Any],
    module_saved: list[tuple[Any, Any]],
    func_name: str | None,
    remaining_ms: float,
) -> str | None:
    """Re-run ``test_fn`` against the ORIGINAL and return its outcome — the attribution
    control for a test the mutant could not be patched INTO.

    The module-qualified bindings must be restored for this call, not merely the injected
    argument. Injection is a contract only Wesker's own inline tests observe; a DISCOVERED
    test calls through its module and ignores the injected value, so with ``module_saved``
    still installed both runs execute the MUTANT, agree trivially, and a real kill is
    discarded. That is not hypothetical: a parametrized case is bound through a wrapper
    whose ``__globals__`` carry no target binding, so ``patched`` is False for EVERY
    parametrized test — every kill they earn would be nullified, silently, in any suite
    that uses ``@pytest.mark.parametrize``.

    The live patched objects are captured and re-installed verbatim afterwards, which keeps
    the descriptor shape ``_patch_module_qualified`` built for class-method owners.
    """
    live: list[tuple[Any, Any]] = []
    for target, saved in module_saved:
        try:
            live.append((target, _get_raw_attr(target, func_name)))
            setattr(target, func_name, saved)
        except Exception:
            continue
    try:
        return _run_test_with_timeout(
            test_fn, _unwrap_descriptor(original_func), False, remaining_ms
        )
    finally:
        for target, current in live:
            try:
                setattr(target, func_name, current)
            except Exception:
                pass


def _is_declared_failure(exc: BaseException) -> bool:
    """True for pytest's ``Failed`` — the outcome pytest raises when a test DECLARES a
    failure rather than blowing up: a violated ``pytest.raises(...)`` contract ("DID NOT
    RAISE"), or an explicit ``pytest.fail()``.

    It matters because ``Failed`` derives from ``BaseException``, NOT ``AssertionError``
    (MRO: Failed -> OutcomeException -> BaseException). So ``except AssertionError`` cannot
    see it and it lands in the BaseException fallback beside genuine crashes — a category
    error. A crash means the mutant blew up and no test said anything about its VALUE; a
    declared failure means a test stated a contract and the mutant broke it. The second
    pins behaviour; the first does not.

    Identified by its BASE, because ``Failed.__module__`` is the string ``"builtins"`` —
    pytest rewrites it so tracebacks read ``Failed`` rather than ``_pytest.outcomes.Failed``.
    Trusting that attribute silently matches nothing, which is worse than not checking:
    every raises-kill keeps its crash verdict and the classifier looks like it simply does
    not work. ``OutcomeException.__module__`` is NOT rewritten, so the base is the honest
    signal, and requiring it keeps an unrelated user class named ``Failed`` out.

    Deliberately narrow. ``Skipped`` shares the ``OutcomeException`` base and would match a
    base-only check, but a skip is not a failure and must not read as a kill — hence the
    exact name. ``Exit`` derives from ``Exception`` and never reaches here at all.

    Detected STRUCTURALLY, never by importing pytest: this engine is zero-dependency and
    pytest is an optional extra (``Wesker[pytest]``), so importing ``_pytest.outcomes`` here
    would make the engine unimportable for consumers who never opted in, in order to
    classify an exception they cannot raise. With pytest absent, no ``Failed`` exists and
    this correctly returns False for everything.
    """
    cls = type(exc)
    if cls.__name__ != "Failed":
        return False
    return any(
        base.__name__ == "OutcomeException" and base.__module__.startswith("_pytest")
        for base in cls.__mro__
    )


def _run_test_with_timeout(
    test_fn: Callable[..., None],
    mutated_func: Any,
    patched: bool,
    timeout_ms: float,
) -> str | None:
    """Run a single test function with a hard thread-based timeout.

    Returns the kill reason ("assertion", "exception", "crash", "timeout") if killed,
    or None if the test passed (mutant survived this test).

    The timeout bounds the WAIT and, via `interrupt.abandon`, the thread itself — see there for what
    that can and cannot reach.
    """
    import contextlib
    import io
    import threading

    result_box: list[str | None] = [None]  # None = survived

    def _target() -> None:
        try:
            if patched:
                test_fn()
            else:
                try:
                    test_fn(mutated_func)
                except TypeError:
                    test_fn()
        except AssertionError:
            result_box[0] = "assertion"
        except Exception:
            result_box[0] = "crash"
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # A test DECLARING failure is not a crash — see `_is_declared_failure`. Reading
            # it as one discards a real pin: `value_survivor_records` re-lists every
            # non-value kill, so a mutant killed by a `pytest.raises` contract came back as
            # an unpinned survivor, was re-classified killable off the same witness, and the
            # residual asked for an input that would write the same test and be discarded
            # again. That is not a slow path to a verdict; it is a loop with no exit.
            result_box[0] = "exception" if _is_declared_failure(exc) else "crash"

    thread = threading.Thread(target=_target, daemon=True)
    # Isolate the discovered test's own stdout/stderr (argparse usage banners,
    # prints, logging) so consumer-test side-effects never pollute the engine's
    # report. Set up in the main thread around start+join so restoration is
    # guaranteed even when the worker hangs and is abandoned as a timeout.
    timed_out = False
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        thread.start()
        thread.join(timeout=timeout_ms / 1000.0)
        timed_out = thread.is_alive()
        if timed_out:
            # Timed out. STOP the runaway rather than abandoning it: the verdict is already
            # decided (below), so what is left is the thread itself, and a daemon thread is only
            # reclaimed at PROCESS exit — i.e. never, across a run. Every timeout used to leave
            # one live thread burning a core for the rest of the session, so later mutants timed
            # out BECAUSE earlier ones were still running and the failure compounded.
            #
            # The abandon and the unwind it triggers MUST happen INSIDE the redirect above, which
            # is why they are in here rather than after it. `redirect_stdout` restores the value
            # IT captured; the abandoned test's own frames may hold their own. A test — or any
            # library it called — that entered `redirect_stdout` and is cut mid-block unwinds
            # through that `__exit__` and reinstalls what IT saved, which is this StringIO. Run
            # after the `with` exited, that write lands AFTER our restoration and therefore wins:
            # `sys.stdout` is left a dead buffer for the rest of the PROCESS and every later
            # print — the engine's own report included — is discarded in silence. Measured on
            # Regenesis: one JVM-backed test overran the 5s cap and `detective diagnose` then
            # exited 0 having printed nothing at all, which in CI is an empty artifact and a
            # green check. Unwinding in here means any such restore is itself captured, and OUR
            # `__exit__` is the last writer.
            _abandon(thread)
            thread.join(timeout=_ABANDON_UNWIND_S)

    if timed_out:
        return "timeout"

    return result_box[0]


def _elapsed(start: float) -> float:
    return (time.monotonic() - start) * 1000


def _mutant_change(mutant: Mutant) -> str:
    """The single changed line, as ``'n >= 10 → n > 10'``.

    ``_mutant_diff`` promises a minimal diff but the mutators set ``original_node`` to the
    enclosing FunctionDef, so it unparses the WHOLE function twice — fine for the oracle
    synthesis that consumes it, useless as an annotation an engineer reads in a diff. This
    recovers the actual edit by line-diffing the two unparsed forms, which works precisely
    because they are the same function differing at one place.

    Returns "" when the change is not a single line (or cannot be unparsed) rather than
    guessing: the caller then falls back to the category description, which is vaguer but true.
    """
    try:
        original = ast.unparse(mutant.original_node)
        mutated = ast.unparse(mutant.mutated_node)
    except Exception:
        return ""
    if original == mutated:
        return ""

    o_lines = original.splitlines()
    m_lines = mutated.splitlines()
    matcher = difflib.SequenceMatcher(None, o_lines, m_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
            return f"{o_lines[i1].strip()} → {m_lines[j1].strip()}"
        if tag == "delete" and i2 - i1 == 1:
            # Statement deletion (SDL): there is no replacement text to show, and saying so
            # is the whole point of the operator.
            return f"{o_lines[i1].strip()} → (statement deleted)"
        return ""
    return ""


def _mutant_diff(mutant: Mutant) -> str:
    """A minimal ``'- <original>\\n+ <mutated>'`` diff of the mutated node.

    Gives downstream oracle synthesis the specific change (e.g. ``n >= 5`` →
    ``n > 5``) rather than only the generic category description. Empty when the
    nodes can't be unparsed or don't differ textually.
    """
    try:
        original = ast.unparse(mutant.original_node).strip()
        mutated = ast.unparse(mutant.mutated_node).strip()
    except Exception:
        return ""
    return f"- {original}\n+ {mutated}" if original != mutated else ""


# ── Sampling & Profiling ──────────────────────────────────────────


def run_function_sampling(
    func_node: ast.FunctionDef,
    func_key: str,
    categories: set[MutationCategory],
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any],
    budget_ms: float = 500,
    max_per_category: int = 3,
    per_mutant_timeout_ms: float = 500,
    seed: int | None = None,
) -> SamplingResult:
    """Inline sampling mode — generate ≤max_per_category mutants per category.

    Evaluates within time budget. This is the "active hypothesis testing"
    from §6.2: each sampled mutant tests whether the test suite distinguishes
    a specific behavioral dimension.

    Args:
        budget_ms: Total wall-clock budget for the entire sampling run.
        per_mutant_timeout_ms: Timeout for evaluating a single mutant.
            Separate from budget_ms to prevent one slow mutant from
            consuming the entire budget.
        seed: Convergence pass index for greedy dimension selection. Each value
            draws the next window of the (1−1/e)-optimal coverage order, so
            successive iterations extend coverage rather than re-roll a random
            subset. (Name retained for backward compatibility.)
    """
    start = time.monotonic()
    mutants = generate_mutants(
        func_node,
        categories,
        max_per_category=max_per_category,
        pass_index=seed or 0,
    )

    results_by_cat: dict[MutationCategory, CategoryResult] = {}
    budget_exhausted = False
    all_results: list[MutantResult] = []
    qualname = (
        func_key.split("::", 1)[1]
        if "::" in func_key
        else getattr(func_node, "name", None)
    )
    # func_key = 'rel/path.py::Qualname' — the (project-relative) source path callers know but do not
    # pass to evaluate_mutant (original_func is stubbed by some callers, so its co_filename is useless).
    source_path = func_key.split("::", 1)[0] if "::" in func_key else None

    for mutant in mutants:
        if _elapsed(start) > budget_ms:
            budget_exhausted = True
            break

        result = evaluate_mutant(
            mutant,
            test_functions,
            original_func,
            timeout_ms=per_mutant_timeout_ms,
            qualname=qualname,
            source_path=source_path,
        )
        all_results.append(result)

        cr = results_by_cat.setdefault(
            mutant.category, CategoryResult(category=mutant.category)
        )
        cr.total += 1
        if result.killed:
            cr.killed += 1
            if result.killed_by == "assertion":
                cr.killed_by_assertion += 1
            elif result.killed_by == "exception":
                # Its own counter, never folded into assertion: `value_killed` needs it to
                # count, and a reader needs to still see WHICH contract pinned the mutant.
                cr.killed_by_exception += 1
            elif result.killed_by == "crash":
                cr.killed_by_crash += 1
        else:
            cr.survived += 1

    per_cat = list(results_by_cat.values())
    total = sum(cr.total for cr in per_cat)
    killed = sum(cr.killed for cr in per_cat)
    survived = total - killed

    return SamplingResult(
        function_key=func_key,
        categories_tested=len(per_cat),
        total_mutants=total,
        total_killed=killed,
        total_survived=survived,
        survival_rate=survived / total if total > 0 else 0.0,
        per_category=per_cat,
        budget_exhausted=budget_exhausted,
        elapsed_ms=_elapsed(start),
    )


def run_function_profiling(
    func_node: ast.FunctionDef,
    func_key: str,
    categories: set[MutationCategory],
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any],
    per_mutant_timeout_ms: float = 5000,
    budget_ms: float | None = None,
    mem_budget_mb: int | None = None,
    max_per_category: int = 0,
    pass_index: int = 0,
    progress: Callable[[int, int, float], None] | None = None,
    scope_tests: bool = True,
    mutant_slice: tuple[int, int] | None = None,
    precomputed_line_data: tuple[dict[str, list[int]], list[str]] | None = None,
    pregenerated: list[Mutant] | None = None,
    trace_budget_s: float | None = DEFAULT_TRACE_BUDGET_S,
    trace_progress: Callable[[int, int, float], None] | None = None,
    trace_session_budget_s: float | None = DEFAULT_TRACE_SESSION_BUDGET_S,
) -> ProfilingResult:
    """Profiling mode — generate mutants (exhaustive by default), evaluate with budget.

    ``progress(done, total, elapsed_ms)`` — optional callback invoked before each mutant
    evaluation AND once at the end (done == total). ``total`` is known up front (the
    generated mutant count), so a caller can stream ``K/N`` with a running-average ETA and
    a final completion line. Cheap: one function call per mutant; throttling is the caller's.

    Returns full survival profile with kill matrix for convergence analysis.
    Result has coverage_depth="profiled" and is_gateable=True.

    Args:
        per_mutant_timeout_ms: Timeout for evaluating a single mutant.
        budget_ms: Optional total wall-clock budget. None means unlimited.
            When exceeded, returns partial results with budget_exhausted=True.
        max_per_category: 0 (default) tests every mutant — exhaustive / comprehensive,
            identical to classical mutation testing. N > 0 tests the N greedily-selected
            (``(1−1/e)``-optimal) mutants per category — fast mode.
        pass_index: Convergence pass; pass p draws the greedy window
            [p·max_per_category, (p+1)·max_per_category), so successive passes extend
            coverage rather than re-roll the same subset.
        scope_tests: When True (default), each mutant is evaluated only against the tests
            that EXECUTE its mutated line (test-impact selection) — a verdict-preserving
            speedup, since a test that never runs the mutated line behaves identically under
            the mutation. False evaluates every mutant against the full set (the A/B baseline
            for verifying the scoping is bit-identical).
    """
    start = time.monotonic()
    # Reuse a caller's already-generated mutant list when given (an adaptive probe generates
    # once and hands the same list to both the probe and the follow-up run), else generate.
    # Deterministic, so the reused list is identical to a fresh generation here.
    mutants = (
        pregenerated
        if pregenerated is not None
        else generate_mutants(
            func_node,
            categories,
            max_per_category=max_per_category,
            pass_index=pass_index,
        )
    )
    # Shard for parallel evaluation: generation is deterministic, so mutants[a:b] here is
    # the SAME set a serial run would evaluate at those indices — a worker owns one slice
    # and the parent merges. The baseline line-coverage/failing pass below still runs over
    # the full test set (cheap, and each shard needs the same coverage map for scoping).
    if mutant_slice is not None:
        mutants = mutants[mutant_slice[0] : mutant_slice[1]]

    # Baseline line-coverage pass over the UNMUTATED function (the second completeness
    # axis) plus the test-impact scoping resolver built from it. Each test runs once
    # against the original under a tracer; the mutation loop below stays untraced (and
    # fast). Degrades to the full test set when the original/line data is unavailable.
    qualname = (
        func_key.split("::", 1)[1]
        if "::" in func_key
        else getattr(func_node, "name", None)
    )
    # func_key = 'rel/path.py::Qualname' — the (project-relative) source path callers know but do not
    # pass to evaluate_mutant (original_func is stubbed by some callers, so its co_filename is useless).
    source_path = func_key.split("::", 1)[0] if "::" in func_key else None

    _trace_truncated: set[str] = set()
    _tests_for, line_cov, exec_lines, failing = _build_test_scope(
        func_node,
        test_functions,
        original_func,
        scope_tests,
        precomputed_line_data,
        qualname,
        trace_budget_s,
        _trace_truncated,
        trace_progress,
        trace_session_budget_s,
    )

    results_by_cat: dict[MutationCategory, CategoryResult] = {}
    kill_matrix: dict[str, list[str]] = {}
    survivor_records: list[dict] = []
    killed_records: list[dict] = []
    budget_exhausted = False

    mem_budget = _resolve_budget(mem_budget_mb)
    total_m = len(mutants)
    for count, mutant in enumerate(mutants):
        if progress is not None:
            progress(count, total_m, _elapsed(start))
        if budget_ms is not None and _elapsed(start) > budget_ms:
            budget_exhausted = True
            break
        # Memory guard: if this run has crossed the (capacity-derived, user-
        # selectable) RAM budget, stop accumulating and reclaim rather than climb
        # past the ceiling — the guarantee that a profile cannot take over the box.
        if count % 16 == 0 and _over_budget(mem_budget):
            budget_exhausted = True
            _reclaim()
            break

        try:
            result = evaluate_mutant(
                mutant,
                _tests_for(mutant),
                original_func,
                timeout_ms=per_mutant_timeout_ms,
                qualname=qualname,
                source_path=source_path,
            )
        except Exception as exc:  # noqa: BLE001
            # A pathological mutant can crash the evaluation harness itself —
            # e.g. self-profiling the engine's own internals, where the mutant
            # replaces the live machinery that runs the profile. One bad mutant
            # must never abort the whole run: record it as an un-evaluable
            # survivor (conservative — never inflates the kill score) and move on.
            cr = results_by_cat.setdefault(
                mutant.category, CategoryResult(category=mutant.category)
            )
            cr.total += 1
            cr.survived += 1
            survivor_records.append(
                {
                    "mutant_id": mutant.mutant_id,
                    "mutant": mutant.description,
                    "category": mutant.category.value,
                    "mutated_line": mutant.mutated_line,
                    "dimension": mutant.dimension,
                    "change": _mutant_change(mutant),
                    "diff_summary": _mutant_diff(mutant),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": 0.0,
                }
            )
            continue

        cr = results_by_cat.setdefault(
            mutant.category, CategoryResult(category=mutant.category)
        )
        cr.total += 1
        if result.killed:
            cr.killed += 1
            if result.killed_by == "assertion":
                cr.killed_by_assertion += 1
            elif result.killed_by == "exception":
                # Its own counter, never folded into assertion: `value_killed` needs it to
                # count, and a reader needs to still see WHICH contract pinned the mutant.
                cr.killed_by_exception += 1
            elif result.killed_by == "crash":
                cr.killed_by_crash += 1
            elif result.killed_by == "timeout":
                cr.timed_out += 1
            if result.test_name:
                kill_matrix.setdefault(mutant.description, []).append(result.test_name)
            # Carry diff_summary on EVERY kill: a crash/timeout kill is a value-survivor
            # (see ProfilingResult.value_survivor_records) and needs the diff for a
            # value-distinguishing witness search downstream.
            killed_records.append(
                {
                    "mutant_id": mutant.mutant_id,
                    "mutant": mutant.description,
                    "category": mutant.category.value,
                    "mutated_line": mutant.mutated_line,
                    "dimension": mutant.dimension,
                    "change": _mutant_change(mutant),
                    "killed_by": result.killed_by,
                    "test": result.test_name,
                    "diff_summary": _mutant_diff(mutant),
                    "elapsed_ms": round(result.elapsed_ms, 1),
                }
            )
        else:
            cr.survived += 1
            survivor_records.append(
                {
                    "mutant_id": mutant.mutant_id,
                    "mutant": mutant.description,
                    "category": mutant.category.value,
                    "mutated_line": mutant.mutated_line,
                    "dimension": mutant.dimension,
                    "change": _mutant_change(mutant),
                    "diff_summary": _mutant_diff(mutant),
                    "elapsed_ms": round(result.elapsed_ms, 1),
                }
            )

    if progress is not None:
        progress(total_m, total_m, _elapsed(start))
    per_cat = list(results_by_cat.values())
    total = sum(cr.total for cr in per_cat)
    killed = sum(cr.killed for cr in per_cat)
    survived = total - killed

    return ProfilingResult(
        function_key=func_key,
        categories_tested=len(per_cat),
        total_mutants=total,
        total_killed=killed,
        total_survived=survived,
        survival_rate=survived / total if total > 0 else 0.0,
        per_category=per_cat,
        kill_matrix=kill_matrix,
        survivor_records=survivor_records,
        killed_records=killed_records,
        budget_exhausted=budget_exhausted,
        elapsed_ms=_elapsed(start),
        line_coverage=line_cov,
        executable_lines=exec_lines,
        failing_tests=failing,
        tests_discovered=len(test_functions),
        trace_truncated=sorted(_trace_truncated),
    )


# ── Universe Estimation ──────────────────────────────────────────


def estimate_universe_size(
    func_node: ast.FunctionDef,
    categories: set[MutationCategory],
) -> int:
    """Count total possible mutation targets without generating mutants.

    Cheap (AST walk only, no compilation or test execution). Used to
    report sampling coverage: tested/killed out of universe_size.
    """
    return sum(_count_targets(func_node, cat) for cat in categories)


def coverage_floor(
    target_counts: tuple[int, ...],
    max_per_category: int,
    passes: int,
) -> float:
    """Provable LOWER BOUND on behavioral-dimension coverage for a greedy run.

    Each entry of ``target_counts`` is one category's mutant universe — an
    independent maximum-coverage problem. With budget ``max_per_category`` = k
    over ``passes`` = N, greedy selection takes ``min(target, N*k)`` mutants per
    category by marginal behavioral-dimension coverage. Two regimes follow:

    * A category whose universe fits the budget (``target <= N*k``, or
      ``k == 0`` = comprehensive) is covered **exhaustively** → 1.0.
    * A larger one is covered to ``>= 1 - (1/e)**N`` of its optimally-coverable
      dimensions: the per-pick optimality-gap contraction ``g_{i+1} <= (1-1/k)
      g_i`` (greedy_coverage_bound.lean) compounds to ``<= e**-N`` after the
      ``N*k`` picks accrued across passes.

    The result is the universe-weighted mean of those per-category guarantees —
    the fraction of the DOF space the greedy run provably reaches. It is a
    *floor*: the measured kill rate meets or beats it. Deterministic, no I/O.
    """
    universe = sum(target_counts)
    if universe == 0:
        return 1.0
    exhaustive = max_per_category <= 0
    per_pass_floor = 1.0 - (1.0 / math.e) ** passes if passes > 0 else 0.0
    covered = 0.0
    for target in target_counts:
        if target <= 0:
            continue
        selected = target if exhaustive else min(target, passes * max_per_category)
        covered += target if selected >= target else target * per_pass_floor
    return covered / universe


def greedy_coverage_guarantee(
    func_node: ast.FunctionDef,
    categories: set[MutationCategory],
    max_per_category: int,
    passes: int,
) -> float:
    """Coverage floor (see :func:`coverage_floor`) over a function's categories.

    Reuses the same per-category target counts as :func:`estimate_universe_size`
    so the guarantee's denominator matches the reported DOF universe exactly.
    """
    counts = tuple(_count_targets(func_node, cat) for cat in categories)
    return coverage_floor(counts, max_per_category, passes)


# ── Equivalence Detection ────────────────────────────────────────


def _generate_boundary_inputs(
    func_node: ast.FunctionDef,
) -> list[tuple]:
    """Generate boundary test inputs based on parameter count.

    Uses a fixed set of boundary values: 0, 1, -1, 0.5, True, False, "", "x".
    For multi-param functions, generates combinations of the first few values.
    """
    n_params = len(func_node.args.args)
    # Skip 'self'/'cls' parameter — can't provide meaningful instance
    if n_params > 0 and func_node.args.args[0].arg in ("self", "cls"):
        n_params -= 1

    if n_params == 0:
        return [()]

    int_vals = [0, 1, -1, 2, -2]
    float_vals = [0.0, 1.0, -1.0, 0.5]
    bool_vals = [True, False]

    if n_params == 1:
        return [(v,) for v in int_vals + float_vals + bool_vals]

    if n_params == 2:
        inputs = []
        for a in int_vals[:3] + float_vals[:2]:
            for b in int_vals[:3] + float_vals[:2]:
                inputs.append((a, b))
        return inputs[:25]

    base = int_vals[:3] + float_vals[:2]
    return [tuple(base[i % len(base)] for _ in range(n_params)) for i in range(5)]


def check_equivalent(
    func_node: ast.FunctionDef,
    mutant: Mutant,
) -> bool:
    """Check if a surviving mutant is semantically equivalent.

    Compiles both original and mutated functions, runs them on boundary
    inputs, and compares outputs. If all outputs match, the mutant is
    likely equivalent — no test can distinguish them.

    Skips methods (self/cls parameter) since we cannot synthesize a
    meaningful instance for boundary testing.
    """
    # Methods: can't provide meaningful self — skip equivalence check
    if func_node.args.args and func_node.args.args[0].arg in ("self", "cls"):
        return False

    try:
        orig_mod = ast.Module(body=[func_node], type_ignores=[])  # type: ignore[list-item]
        ast.fix_missing_locations(orig_mod)
        orig_code = compile(orig_mod, "<original>", "exec")
        orig_ns: dict[str, Any] = {}
        exec(orig_code, orig_ns)  # noqa: S102

        mut_mod = ast.Module(body=[mutant.mutated_node], type_ignores=[])  # type: ignore[list-item]
        ast.fix_missing_locations(mut_mod)
        mut_code = compile(mut_mod, "<mutant>", "exec")
        mut_ns: dict[str, Any] = {}
        exec(mut_code, mut_ns)  # noqa: S102

        func_name = func_node.name
        orig_fn = orig_ns.get(func_name)
        mut_fn = mut_ns.get(func_name)

        if orig_fn is None or mut_fn is None:
            return False

        boundary_inputs = _generate_boundary_inputs(func_node)
        successful_comparisons = 0

        for args in boundary_inputs:
            orig_exc = mut_exc = None
            orig_result = mut_result = None
            try:
                orig_result = orig_fn(*args)
            except Exception as e:
                orig_exc = e
            try:
                mut_result = mut_fn(*args)
            except Exception as e:
                mut_exc = e

            # One raises and the other doesn't → NOT equivalent
            if (orig_exc is None) != (mut_exc is None):
                return False
            # Both returned values → compare
            if orig_exc is None:
                if orig_result != mut_result:
                    return False
                successful_comparisons += 1
            # Both raised → check exception type matches
            elif type(orig_exc) is not type(mut_exc):
                return False

        # Only declare equivalent if we got at least one real comparison.
        # If ALL inputs raised, we have no evidence of equivalence.
        return successful_comparisons > 0

    except Exception:
        return False


# ── Multi-Pass Convergence ───────────────────────────────────────


def run_function_converged(
    func_node: ast.FunctionDef,
    func_key: str,
    categories: set[MutationCategory],
    test_functions: list[Callable[..., None]],
    original_func: Callable[..., Any] | None,  # kept for API symmetry
    budget_ms: float = 5000,
    max_per_category: int | None = None,
    per_mutant_timeout_ms: float = 500,
    passes: int = 1,
    category_order: list[MutationCategory] | None = None,
    full_matrix: bool = False,
    source_path: str | None = None,
    scope_tests: bool = True,
    trace_budget_s: float | None = DEFAULT_TRACE_BUDGET_S,
    trace_progress: Callable[[int, int, float], None] | None = None,
    trace_session_budget_s: float | None = DEFAULT_TRACE_SESSION_BUDGET_S,
) -> ProfilingResult:
    """Multi-pass convergence with integrated equivalence detection.

    ``scope_tests`` defaults to True: test-impact selection is the intended design,
    not an optimisation bolted on. Only a test that EXECUTES the mutated line can
    distinguish the mutant; the rest behave identically under the mutation, so
    running them buys nothing. ``test_scoped_and_unscoped_verdicts_agree`` pins that
    the two produce the SAME verdict against a suite that kills everything — scoping
    is verdict-exact, which is what makes it free.

    This default was False for a period, preserving an older path's behaviour, and
    that was doing real damage on both axes:

    ACCURACY. Measured on prism/economics.py::analyze (identical 130-mutant set):
    unscoped credited 130 kills, 107 of them to ``test_nudge_contains_tool_count`` —
    a test in another module that never references ``analyze`` and fails identically
    on the UNMUTATED original. A test that cannot distinguish anything was credited
    with killing everything. ``trace_line_coverage`` correctly records 0 covered
    lines for it, so scoping drops it; the unscoped number was simply inflated.

    COST, and it dominated the reports. ``evaluate_mutant`` returns on the FIRST
    assertion kill but scans the WHOLE set before conceding a survivor — so unscoped,
    every would-be survivor pays for the entire suite. Against ``per_mutant_timeout_ms``
    (500ms, sized for a scoped handful) that is not a budget any real suite can meet:
    profiling Detective (306 tests) turned 1093 of 1305 unspecified dimensions into
    TIMEOUTS, and ModelAtlas (1000 tests) 2602 of 2801 — with ZERO true survivors.
    Those runs measured suite speed, not specification. Scoped, the same work is a
    handful of tests per mutant and the 500ms budget is generous (1.8s vs 33.6s above).

    A remaining defect is upstream of this flag and unaffected by it: a test that
    fails identically on the unmutated original must never be credited with a kill,
    whatever the reason it fails. ``failing_on_baseline`` only counts
    ``AssertionError``, so a missing-fixture ``TypeError`` slips past it. Scoping
    happens to drop such tests when their coverage is empty, but that is a side
    effect, not the fix.

    Returns ``ProfilingResult`` with full kill matrix, survivor/killed
    records, and gateability — compatible with downstream consumers
    (gap classifiers, convergence engines, cross-channel gates).

    ``max_per_category=None`` (the default) is DOF mode: each category's budget
    is the function's own :func:`dimension_budget`, so a SINGLE pass covers every
    behavioral dimension exactly once — full DOF coverage at the fewest mutants
    that can achieve it. Additional passes then deepen WITHIN already-covered
    dimensions (a second mutant per dimension, a third, …), which buys kill
    evidence but no new DOF; hence ``passes=1`` by default. A positive
    ``max_per_category`` pins an explicit per-pass budget instead, and each pass p
    takes the next window of the greedy order, extending the coverage prefix
    rather than re-rolling a random subset. Surviving mutants are checked for
    semantic equivalence via boundary input evaluation.

    When ``category_order`` is provided (from Layer 2 predictive priors),
    mutants are generated in priority order within each pass. If budget
    runs out mid-pass, high-prior categories have already been tested.

    Coverage depth:
      - "profiled" if all possible mutants were tested
      - "converged" if passes > 1
      - "sampled" otherwise
    """
    start = time.monotonic()
    universe = estimate_universe_size(func_node, categories)
    dof_total = dof_universe(func_node, categories)
    dims_covered: set[str] = set()
    dims_pinned: set[str] = set()
    qualname = (
        func_key.split("::", 1)[1]
        if "::" in func_key
        else getattr(func_node, "name", None)
    )
    # func_key = 'rel/path.py::Qualname' — the (project-relative) source path callers know but do not
    # pass to evaluate_mutant (original_func is stubbed by some callers, so its co_filename is useless).
    # An EXPLICIT source_path wins: this line used to overwrite the parameter unconditionally, so a
    # caller that passed one had it silently discarded — two individually-correct commits (the param,
    # then the derivation) colliding where they met. No caller passes it today, which is exactly why
    # it went unnoticed; the next one would have debugged the wrong thing.
    source_path = source_path or (
        func_key.split("::", 1)[0] if "::" in func_key else None
    )

    # Test-impact scoping (shared with run_function_profiling — one implementation, so
    # the two paths cannot drift on soundness). Engages only when ``original_func`` is a
    # real callable to trace against; callers that stub it get the full test set, which
    # is always sound, just slower.
    _trace_truncated: set[str] = set()
    _tests_for, line_cov, exec_lines, failing = _build_test_scope(
        func_node,
        test_functions,
        original_func,
        scope_tests,
        None,
        qualname,
        trace_budget_s,
        _trace_truncated,
        trace_progress,
        trace_session_budget_s,
    )

    seen: dict[str, MutantResult] = {}
    kill_matrix: dict[str, list[str]] = {}
    survivor_records: list[dict] = []
    killed_records: list[dict] = []

    for pass_idx in range(passes):
        if _elapsed(start) > budget_ms:
            break
        mutants = generate_mutants(
            func_node,
            categories,
            max_per_category=max_per_category,
            pass_index=pass_idx,
            category_order=category_order,
        )
        for mutant in mutants:
            if mutant.mutant_id in seen:
                continue
            if _elapsed(start) > budget_ms:
                break

            # Only the tests that EXECUTE this mutant's line can kill it; the rest
            # behave identically under the mutation, so running them is pure cost.
            scoped = _tests_for(mutant)
            result = evaluate_mutant(
                mutant,
                scoped,
                original_func,  # type: ignore[arg-type]
                # Full-matrix mode runs every test, so budget for the whole
                # suite (~50ms/test) rather than the first-killer per-mutant cap.
                timeout_ms=(
                    max(per_mutant_timeout_ms, 50.0 * len(scoped))
                    if full_matrix
                    else per_mutant_timeout_ms
                ),
                qualname=qualname,
                record_all_killers=full_matrix,
                source_path=source_path,
            )

            # Integrated equivalence: check survivors immediately
            if not result.killed:
                if check_equivalent(func_node, mutant):
                    result = MutantResult(
                        mutant=mutant,
                        killed=False,
                        equivalent=True,
                        elapsed_ms=result.elapsed_ms,
                    )

            seen[mutant.mutant_id] = result
            if mutant.dimension and not _is_dead(mutant.dimension):
                dim_key = f"{mutant.category.value}\x00{mutant.dimension}"
                dims_covered.add(dim_key)
                # A dimension counts as PINNED only when a test DISTINGUISHED the mutant's
                # value — an assertion kill. A crash or timeout kill proves the tests RAN the
                # mutated code, not that any of them checked what it returned, so it pins
                # nothing: `CategoryResult.value_survived` already states this outright
                # ("Value-unspecified DOF: survivors PLUS crash/timeout kills. For
                # specification these are equivalent — none pins the return value"), and
                # `value_killed` is `killed_by_assertion` alone.
                #
                # This is the difference between a mutation score and a specification
                # measurement, and getting it wrong is not conservative — it inflates. Counting
                # every kill made Wesker report 98% on its own ci.py, where Detective (which
                # reads value_killed) reports 29% for the same code: mutating the machinery the
                # tests drive makes them fall over, and every one of those crashes was being
                # credited as pinned behavior. An equivalent mutant pins nothing either, and is
                # excluded from the denominator elsewhere rather than credited here.
                if result.killed and result.killed_by == "assertion":
                    dims_pinned.add(dim_key)

            # Build kill matrix and records for downstream consumers.
            # ``mutated_line`` and ``dimension`` are what let a survivor be reported AT the
            # source line it lives on, naming the behavior no test pins — an editor annotation
            # or a SARIF result, rather than a count. Both are carried verbatim from the
            # mutant; ``mutated_line`` is None when the mutator could not report a position.
            record = {
                "mutant_id": mutant.mutant_id,
                "mutant": mutant.description,
                "category": mutant.category.value,
                "mutated_line": mutant.mutated_line,
                "dimension": mutant.dimension,
                "change": _mutant_change(mutant),
                "diff_summary": _mutant_diff(mutant),
                "elapsed_ms": round(result.elapsed_ms, 1),
            }
            if result.killed:
                record["killed_by"] = result.killed_by
                record["test"] = result.test_name
                killed_records.append(record)
                # First-killer mode records the single killer; full-matrix mode
                # records every test that kills this mutant (the per-test
                # attribution a greedy-convergence analysis needs).
                if full_matrix and result.killed_by_tests:
                    kill_matrix.setdefault(mutant.description, []).extend(
                        result.killed_by_tests
                    )
                elif result.test_name:
                    kill_matrix.setdefault(mutant.description, []).append(
                        result.test_name
                    )
            elif result.equivalent:
                record["equivalent"] = True
                survivor_records.append(record)
            else:
                survivor_records.append(record)

    # Aggregate by category
    results_by_cat: dict[MutationCategory, CategoryResult] = {}
    for result in seen.values():
        cat = result.mutant.category
        cr = results_by_cat.setdefault(cat, CategoryResult(category=cat))
        cr.total += 1
        if result.killed:
            cr.killed += 1
            if result.killed_by == "assertion":
                cr.killed_by_assertion += 1
            elif result.killed_by == "exception":
                # Its own counter, never folded into assertion: `value_killed` needs it to
                # count, and a reader needs to still see WHICH contract pinned the mutant.
                cr.killed_by_exception += 1
            elif result.killed_by == "crash":
                cr.killed_by_crash += 1
            elif result.killed_by == "timeout":
                cr.timed_out += 1
        elif result.equivalent:
            cr.equivalent += 1
            cr.survived += 1
        else:
            cr.survived += 1

    per_cat = list(results_by_cat.values())
    total = sum(cr.total for cr in per_cat)
    killed = sum(cr.killed for cr in per_cat)
    equiv = sum(cr.equivalent for cr in per_cat)
    survived = total - killed
    budget_exhausted = _elapsed(start) > budget_ms

    # Determine coverage depth
    if total >= universe > 0:
        depth = "profiled"
    elif passes > 1:
        depth = "converged"
    else:
        depth = "sampled"

    return ProfilingResult(
        function_key=func_key,
        categories_tested=len(per_cat),
        total_mutants=total,
        total_killed=killed,
        total_survived=survived,
        total_equivalent=equiv,
        universe_size=universe,
        survival_rate=survived / total if total > 0 else 0.0,
        dof_total=dof_total,
        dof_covered=len(dims_covered),
        dof_pinned=len(dims_pinned),
        coverage_depth=depth,
        is_gateable=depth == "profiled",
        per_category=per_cat,
        kill_matrix=kill_matrix,
        survivor_records=survivor_records,
        killed_records=killed_records,
        budget_exhausted=budget_exhausted,
        elapsed_ms=_elapsed(start),
        trace_truncated=sorted(_trace_truncated),
    )
