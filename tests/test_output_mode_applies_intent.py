"""μ⁻ Fork 2 is SOUND BY RESTRICTION — a type-conditional OUTPUT perturbation is generated only when
EVERY observed return type is applicable (Def. 11.10 / Prop. 11.11).

The engine used `observed & types` (any-intersection): a heterogeneous return like `int | str` observed
as `{int, str}` generated `return_negate` (applicable to `{int, float}`) because `int` was present — and
the STATIC return-site rewrite (`return X` -> `return -X`) then fired on the `str` branch too and RAISED.
That mis-typed perturbation is Def. 7.3's source-(b): the engine scored the raise as a crash-kill and
reported a phantom "crash-only value gap" for a perturbation that simply does not apply. `observed <=
applicable` (`output_mode_applies`) makes the paper's claim true — the perturbation is never generated
where it could raise, so no `undefined` disposition is ever needed. A heterogeneous return forgoes the
type-conditional perturbation entirely (only the always-applicable Fork-1 set), the honest restriction.

Measured (the reproduced case, `classify(x)` returning `x*2` on the positive branch and `'neg'` on the
negative): before, `converge --two-sign` reported `✓ COMPLETE modulo 1 unproven-equivalent · 2 crash-only
value gaps · 24/25`; after, a clean `✓ COMPLETE · 19/19` — the 6 mis-typed OUTPUT perturbations gone.
"""

from __future__ import annotations

from Wesker.engine import (
    _OUTPUT_TYPE_CONDITIONAL,
    _output_sub_modes,
    output_mode_applies,
)

_NUMERIC = frozenset({"int", "float"})


# ── output_mode_applies: the pinned applicability predicate (isolation ✓ COMPLETE 8/8) ───────────────
def test_output_mode_applies_requires_every_observed_type_applicable():
    assert (
        output_mode_applies(frozenset(), _NUMERIC) is False
    )  # nothing observed → not applicable
    assert (
        output_mode_applies(frozenset({"int"}), _NUMERIC) is True
    )  # subset → applicable
    assert output_mode_applies(_NUMERIC, _NUMERIC) is True  # equal → applicable
    # heterogeneous: a str return is NOT numeric, so the numeric perturbation must NOT be generated
    assert output_mode_applies(frozenset({"int", "str"}), _NUMERIC) is False
    assert (
        output_mode_applies(frozenset({"str"}), _NUMERIC) is False
    )  # disjoint → not applicable


# ── _output_sub_modes: the mis-typed perturbation is never emitted for a heterogeneous return ────────
def _mode_names(observed) -> set[str]:
    return {name for name, _desc in _output_sub_modes(observed)}


def test_homogeneous_numeric_return_keeps_its_applicable_perturbations():
    modes = _mode_names(frozenset({"int"}))
    assert "return_negate" in modes  # int is applicable to the sign fence…
    assert "return_abs" in modes
    assert (
        "return_nan" not in modes
    )  # …but NaN is float-only, so it stays out (relevance)


def test_heterogeneous_return_forgoes_the_raising_perturbation():
    # {int, str}: return_negate would raise on the str branch (`-'neg'`), so it must not be generated —
    # the whole point. Every RAISING type-conditional (negate / abs / reorder) is excluded.
    modes = _mode_names(frozenset({"int", "str"}))
    for raising in ("return_negate", "return_abs", "return_reorder"):
        assert raising not in modes, (
            f"{raising} must not be generated for a heterogeneous return"
        )


def test_string_return_keeps_the_applicable_container_fence():
    modes = _mode_names(frozenset({"str"}))
    assert "return_empty" in modes  # str IS in return_empty's applicable set
    assert "return_negate" not in modes  # but not the numeric ones


def test_unobserved_return_emits_only_fork_one_always_applicable():
    # No observed types (None / empty) → not applicable to ANY type-conditional; only the Fork-1 set.
    conditional = {name for name, _desc, _types in _OUTPUT_TYPE_CONDITIONAL}
    assert _mode_names(None).isdisjoint(conditional)
    assert _mode_names(frozenset()).isdisjoint(conditional)
