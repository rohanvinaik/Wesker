"""Monty Hall filtering — exclude irrelevant mutation categories (§6.1).

Layer 1 (exclusionary): If a function has no comparisons, boundary mutants
cannot survive (there's nothing to mutate), so generating them wastes budget.
The filter reveals which "doors" have no prize before opening them.

Layer 2 (predictive priors): When cached mutation data exists, use historical
per-category survival rates to prioritize categories most likely to have
surviving mutants, directing budget where it matters most.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from Wesker.engine import MutationCategory, estimate_universe_size


@dataclass
class CategoryPrior:
    """A mutation category with its expected survival probability."""

    category: MutationCategory
    prior: float  # 0.0 = never survives, 1.0 = always survives


@dataclass
class _FunctionSignals:
    """Structural signals extracted from a function AST for mutation filtering."""

    param_count: int = 0
    has_comparisons: bool = False
    has_self_assigns: bool = False
    has_global_nonlocal: bool = False
    has_isinstance: bool = False
    has_arithmetic: bool = False
    has_logical: bool = False
    has_deletable_stmt: bool = False
    has_return_value: bool = False
    has_exception: bool = False


def _classify_signal_node(node: ast.AST, signals: _FunctionSignals) -> None:
    """Set the structural-signal flags implied by a single AST node."""
    if isinstance(node, ast.Compare):
        signals.has_comparisons = True
    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            signals.has_self_assigns = True
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        signals.has_global_nonlocal = True
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
    ):
        signals.has_isinstance = True
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
    ):
        signals.has_arithmetic = True
    elif isinstance(node, ast.AugAssign) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
    ):
        signals.has_arithmetic = True
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        signals.has_arithmetic = True
    elif isinstance(node, ast.BoolOp):
        signals.has_logical = True
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        signals.has_logical = True
    elif isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
        signals.has_deletable_stmt = True
    # A bare ``raise`` (re-raise) carries no type to swap, so it is not a target — in
    # lockstep with ``_ExceptionMutator.visit_Raise``, which skips ``exc is None``.
    elif isinstance(node, ast.Raise) and node.exc is not None:
        signals.has_exception = True
    elif isinstance(node, ast.ExceptHandler):
        signals.has_exception = True
    # In lockstep with ``_count_state_return_target`` / ``_StateMutator.visit_Return``:
    # a return_none target is a ``return`` WITH a value (a bare ``return`` already
    # returns None, so replacing it is a no-op, not a degree of freedom).
    elif isinstance(node, ast.Return) and node.value is not None:
        signals.has_return_value = True


def _collect_signals(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> _FunctionSignals:
    """Walk the function AST to collect structural signals."""
    signals = _FunctionSignals(param_count=len(func_node.args.args))
    for node in ast.walk(func_node):
        _classify_signal_node(node, signals)
    # STMT deletability is a FUNCTION-level property (is this name already bound before
    # this statement?), so the per-node classifier above cannot decide it — it can only
    # see the ``ast.Expr`` case. Asking the engine keeps ONE definition of "deletable":
    # a divergence here would gate the category off for a function whose only deletable
    # statements are rebindings or aliased writes, silently dropping them from the
    # universe while the engine stood ready to mutate them.
    from Wesker.engine import _deletable_stmt_ids

    if _deletable_stmt_ids(func_node):
        signals.has_deletable_stmt = True
    return signals


def filter_categories(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_pure: bool = False,
) -> set[MutationCategory]:
    """Layer 1: Exclusionary filtering (§6.1).

    Returns the set of categories relevant to this function.
    Categories where the function has no structural support are excluded.
    """
    sig = _collect_signals(func_node)
    relevant: set[MutationCategory] = {MutationCategory.VALUE}

    # SWAP is asked of the MUTATOR, not guessed from the signature. `_SwapMutator` does not
    # touch formal parameters at all — it transposes adjacent positional ARGUMENTS at call
    # sites, unwraps used calls, and substitutes curated callee duals. Gating on
    # `param_count >= 2` therefore answered a different question than the one that decides
    # whether a target exists, and answered it wrong in both directions: `def square(x):
    # return pow(x, 2)` has one formal parameter and two live SWAP targets, so the category was
    # dropped and `pow(x, 2) -> pow(2, x)` never entered the universe. A suite testing only
    # x == 2 then reported `✓ COMPLETE (operator universe) · 3/3 killed` — unqualified, not even
    # "modulo unproven-equivalent" — while failing to distinguish a mutant that differs at every
    # other x. That is a false SC = 1, which is the one error the completeness claim cannot
    # survive. Counting the real targets is the same walk `estimate_universe_size` already does.
    if estimate_universe_size(func_node, {MutationCategory.SWAP}):
        relevant.add(MutationCategory.SWAP)
    if sig.has_comparisons:
        relevant.add(MutationCategory.BOUNDARY)
    # STATE carries two orthogonal sub-modes, and only ONE of them is about state:
    #   - remove_assign  — a real state mutation; needs a state-WRITING impure function.
    #   - return_none    — "does any test notice if this returns None?", a pure VALUE
    #                      distinction, and the most basic question you can ask of a
    #                      function whose behaviour IS its return value.
    # Gating the whole category on state-writing hid return_none from every function that
    # does not assign to self/global — i.e. almost all of them, pure or not (a filesystem
    # helper is impure yet writes no state). Each sub-mode is selected against its own
    # target count in ``_generate_state_mutants``, so a function with returns but no state
    # writes picks up return_none ONLY: remove_assign counts zero and contributes nothing.
    if sig.has_return_value or (
        not is_pure and (sig.has_self_assigns or sig.has_global_nonlocal)
    ):
        relevant.add(MutationCategory.STATE)
    if sig.has_isinstance:
        relevant.add(MutationCategory.TYPE)
    if sig.has_arithmetic:
        relevant.add(MutationCategory.ARITHMETIC)
    if sig.has_logical:
        relevant.add(MutationCategory.LOGICAL)
    if sig.has_deletable_stmt:
        relevant.add(MutationCategory.STMT)
    # NOT gated on purity: raising and catching are behavior regardless of whether the
    # function is otherwise side-effect free, and a "pure" function that raises still
    # owes its callers a pinned exception type.
    if sig.has_exception:
        relevant.add(MutationCategory.EXCEPTION)

    return relevant


# ── Layer 2: Predictive priors (§6.2) ────────────────────────────────


_DEFAULT_PRIOR = 0.5  # uniform when no history


def prioritize_categories(
    relevant: set[MutationCategory],
    cached_state: dict | None = None,
) -> list[CategoryPrior]:
    """Layer 2: Predictive priors from cached mutation data.

    Takes the Layer 1 exclusionary output and annotates each category
    with a survival prior derived from previous profiling runs. Returns
    categories ordered by descending prior (highest-survival first),
    so budget-limited runs test the most informative categories first.

    When no cached data exists, all priors are uniform (0.5).

    Note: ``per_category`` in cached mutation state is a *list* of dicts
    (``[{"category": "VALUE", "total": 10, "survived": 3}, ...]``),
    not a dict keyed by category name.
    """
    # Build lookup from the list format used by mutation engine output
    cat_lookup: dict[str, dict] = {}
    if cached_state:
        raw = cached_state.get("per_category", [])
        if isinstance(raw, list):
            for entry in raw:
                cat_name = entry.get("category", "")
                if cat_name:
                    cat_lookup[cat_name] = entry
        elif isinstance(raw, dict):
            cat_lookup = raw  # defensive: handle dict format too

    priors: list[CategoryPrior] = []
    for cat in relevant:
        cat_data = cat_lookup.get(cat.value, {})
        if cat_data:
            total = cat_data.get("total", 0)
            survived = cat_data.get("survived", 0)
            prior = survived / total if total > 0 else _DEFAULT_PRIOR
        else:
            prior = _DEFAULT_PRIOR
        priors.append(CategoryPrior(category=cat, prior=round(prior, 3)))

    # Sort by prior descending — highest survival first for budget efficiency
    priors.sort(key=lambda p: p.prior, reverse=True)
    return priors
