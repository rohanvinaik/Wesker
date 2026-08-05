"""Monty Hall filtering — exclude irrelevant mutation categories (§6.1).

Layer 1 (exclusionary): a function with no comparisons has no BOUNDARY
universe, so generating BOUNDARY mutants wastes budget. The filter reveals
which "doors" have no prize before opening them — and it asks the ENGINE
which doors those are (target counts), never a parallel structural guess.

Layer 2 (predictive priors): when cached mutation data exists, use historical
per-category survival rates to prioritize categories most likely to have
surviving mutants, directing budget where it matters most.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# _STATE_SUB_MODES/_count_state_targets are engine-private on purpose: the
# filter's one job is to relay the engine's own target counts, and importing
# the counting machinery keeps ONE definition of "has a target" (the same
# reasoning that moved STMT onto _deletable_stmt_ids and SWAP onto
# estimate_universe_size before this file stopped keeping signals at all).
from Wesker.engine import (
    MutationCategory,
    _count_state_targets,
    _STATE_SUB_MODES,
    estimate_universe_size,
)


@dataclass
class CategoryPrior:
    """A mutation category with its expected survival probability."""

    category: MutationCategory
    prior: float  # 0.0 = never survives, 1.0 = always survives


def filter_categories(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_pure: bool = False,
) -> set[MutationCategory]:
    """Layer 1: Exclusionary filtering (§6.1).

    A category is relevant exactly when the engine counts at least one target
    for it. Eligibility IS the target count — the same number
    ``estimate_universe_size`` reports and generation iterates — so this layer
    is lossless by construction: it can only exclude a category whose universe
    is empty, and skipping an empty category loses exactly nothing.

    It used to read a parallel set of structural signals instead, and a second
    predicate is how issue #9 shipped a false ``✓ COMPLETE``: SWAP's proxy
    (``param_count >= 2``) answered a different question than the mutator,
    dropped ``pow(x, 2) -> pow(2, x)`` from the universe, and the loss was
    invisible from the report. The same shape was still live in STATE's gate
    until this rewrite: a function with ``break``/``continue`` but no return
    value and no self-assign had live loop_flow targets and no STATE category,
    so ``break`` ↔ ``continue`` never entered its universe.

    ``is_pure`` is a policy overlay, not a structural fact: a caller asserting
    purity asserts that ``self.x = ...`` writes are unobservable, so the
    remove_assign sub-mode's targets stop counting toward STATE's relevance.
    return_none and loop_flow are value/control-flow questions and count
    either way.
    """
    relevant: set[MutationCategory] = set()
    for cat in MutationCategory:
        if cat is MutationCategory.STATE:
            count = sum(
                _count_state_targets(func_node, mode)
                for mode, _desc in _STATE_SUB_MODES
                if not (is_pure and mode == "remove_assign")
            )
        else:
            count = estimate_universe_size(func_node, {cat})
        if count:
            relevant.add(cat)
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
