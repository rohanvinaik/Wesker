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


def operator_disposition(generated: int, withheld: int) -> str:
    """What the policy DID to one operator on one target (issue #22).

    An internal assertion that record-mode count equals generator count proves the two
    implementations agree; it cannot prove they agree on the RIGHT set, because both can omit
    the same site class and still match. Auditability needs the census to say what happened to
    every candidate, and the three answers are not interchangeable:

    * ``generated``      — at least one mutant was produced. The universe covers this operator.
    * ``withheld``       — candidate sites EXIST and policy suppressed all of them. The only
      case a reader must be able to see and disagree with, because it is a judgement rather
      than a structural fact: `filter_categories`' ``is_pure`` overlay asserts that
      ``self.x = ...`` writes are unobservable and stops remove_assign counting. If that
      assertion is wrong the behaviour is real and unmeasured, and today nothing says so.
    * ``not_applicable`` — no candidate site exists at all. Nothing was decided and nothing is
      missing; skipping an empty category loses exactly nothing.

    ``generated`` wins when both are positive: policy suppressed SOME sites and the operator is
    still represented, so the universe is not missing it. The withheld count still travels
    alongside for a reader who wants the partial suppression.
    """
    if generated > 0:
        return "generated"
    if withheld > 0:
        return "withheld"
    return "not_applicable"


def _census_row(
    generated: int, withheld: int, sub_modes: dict[str, dict] | None = None
) -> dict[str, object]:
    """One census entry, built in one place so both branches produce the same shape.

    A row is heterogeneous BY DESIGN — int counts, a str disposition, an optional nested
    sub-mode map — and it was built by two separate dict literals that a checker then inferred
    separately: `dict[str, int]` in the non-STATE arm, so assigning the `str` disposition
    afterwards fit neither. Annotating the variable does not fix that, because the annotation is
    narrowed by whichever literal was assigned last.

    The disposition is computed from the ARGUMENTS, not read back out of the row. Re-reading a
    bag to derive a value the caller just computed is the measurement/decision gap in miniature:
    the bag answers with whatever it happens to hold, including something a later edit put there.

    Key order is preserved (``generated``, ``withheld``, ``sub_modes``, ``disposition``) because
    this census is serialised.
    """
    row: dict[str, object] = {"generated": generated, "withheld": withheld}
    if sub_modes is not None:
        row["sub_modes"] = sub_modes
    row["disposition"] = operator_disposition(generated, withheld)
    return row


def filter_categories(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_pure: bool = False,
    two_sign: bool = False,
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
    return {
        cat
        for cat, row in category_census(func_node, is_pure, two_sign).items()
        if row["disposition"] == "generated"
    }


def category_census(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    is_pure: bool = False,
    two_sign: bool = False,
) -> dict[MutationCategory, dict]:
    """What the policy did to EVERY candidate operator on this target (issue #22).

    The engine could say which categories are in the universe and not why the others are out,
    and those are different questions. A category absent because the function has no candidate
    site is nothing to worry about; a category absent because a policy overlay SUPPRESSED its
    sites is a judgement the reader may want to disagree with — and the two were indistinguishable
    from the report.

    `filter_categories` is DERIVED from this rather than computing the counts alongside it. The
    repo treats a second predicate for one fact as a defect class, not a style question: issue #9
    shipped a false ✓ COMPLETE exactly that way, when SWAP's structural proxy answered a different
    question than its mutator and silently dropped `pow(x, 2) -> pow(2, x)` from the universe.
    A census that could disagree with the filter it explains would be the same defect wearing a
    report.

    The withheld count is per SUB-MODE for STATE, because that is the granularity policy acts at:
    ``is_pure`` suppresses only ``remove_assign`` (a caller asserting purity asserts that
    ``self.x = ...`` writes are unobservable), while ``return_none`` and ``loop_flow`` are
    value/control-flow questions and count either way. Naming the suppressed alternatives is what
    makes the assertion reviewable — if purity was asserted wrongly, that behaviour is real,
    unmeasured, and currently invisible.
    """
    census: dict[MutationCategory, dict] = {}
    for cat in MutationCategory:
        if cat is MutationCategory.STATE:
            sub_modes: dict[str, dict] = {}
            generated = withheld = 0
            for mode, _desc in _STATE_SUB_MODES:
                targets = _count_state_targets(func_node, mode)
                suppressed = bool(is_pure and mode == "remove_assign")
                if suppressed:
                    withheld += targets
                else:
                    generated += targets
                sub_modes[mode] = {
                    "targets": targets,
                    # Empty unless policy actually removed something: a reason attached to a
                    # sub-mode with no sites would read as a suppression that never happened.
                    "withheld_by": "purity_overlay" if (suppressed and targets) else "",
                }
            row = _census_row(generated, withheld, sub_modes)
        elif cat is MutationCategory.OUTPUT and not two_sign:
            # μ⁻ is off under the one-sign default policy: candidate return sites may exist,
            # but the policy withholds them. Shown as withheld (issue #22 auditability) — the
            # reader sees the negative operator is available and disabled, not absent.
            row = _census_row(0, estimate_universe_size(func_node, {cat}))
        else:
            generated, withheld = estimate_universe_size(func_node, {cat}), 0
            row = _census_row(generated, withheld)
        census[cat] = row
    return census


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
