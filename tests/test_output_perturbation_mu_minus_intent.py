"""μ⁻ Form A — output-space perturbation, the negative sign of the two-sign teaching set.

Intent test authored from the design (docs/theory/NEGATIVE_SPECIFICATION.md §3, §11): the
independence pair (→const, →identity) fences what NO positive operator can — that the output
DEPENDS ON THE INPUT (¬const) and is a NON-TRIVIAL TRANSFORM of it (¬identity). A suite that
passes when the function is replaced by a constant leaves input-dependence UNPINNED, and μ⁻'s
→const survivor is exactly that negative degree of freedom. These are characterization-
independent claims about what the operator must catch, so they are hand-written, not generated.

μ⁻ is OPT-IN: it belongs to the two-sign policy σ(P, μ ∪ μ⁻); the default one-sign policy
generates no OUTPUT mutant. Engine-core cannot self-profile, so evaluation is a hand-written
unit test by design, on the ``test_kill_attribution.py`` pattern (on-disk module, real
``co_filename`` for the module-qualified patch).
"""

from __future__ import annotations

import ast
import importlib.util
import sys

from Wesker.engine import (
    MutationCategory,
    _output_sub_modes,
    estimate_universe_size,
    evaluate_mutant,
    generate_mutants,
)
from Wesker.filter import filter_categories


def _load(tmp_path, name, src):
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, str(path)


def _output_mutants(src):
    node = ast.parse(src).body[0]
    return generate_mutants(node, {MutationCategory.OUTPUT}, max_per_category=0)


def test_fork2_sub_modes_are_type_directed():
    # μ⁻ Fork 2: the type-conditional perturbations are emitted only for an OBSERVED codomain type.
    assert {m[0] for m in _output_sub_modes(None)} == {
        "return_none",
        "return_const",
        "return_identity",
    }
    assert "return_negate" in {m[0] for m in _output_sub_modes(frozenset({"int"}))}
    assert "return_empty" in {m[0] for m in _output_sub_modes(frozenset({"str"}))}
    assert "return_reorder" in {m[0] for m in _output_sub_modes(frozenset({"list"}))}
    # bool is NOT int: no numeric perturbation for a bool return — the silent-coercion hole
    # Fork 1 could not close, closed by keying on the exact observed type name.
    assert "return_negate" not in {m[0] for m in _output_sub_modes(frozenset({"bool"}))}


def test_fork2_generation_is_typed_and_count_equals_generation():
    # count == generation (the issue-#9 invariant) must hold under an OBSERVED type, or the
    # coverage denominator lies about the type-conditional universe.
    node = ast.parse("def f(x):\n    return x * 2\n").body[0]
    obs = frozenset({"int"})
    muts = generate_mutants(
        node, {MutationCategory.OUTPUT}, max_per_category=0, observed_return_types=obs
    )
    assert {"return_negate", "return_abs"} <= {_sub_mode(m) for m in muts}
    assert len(muts) == estimate_universe_size(
        node, {MutationCategory.OUTPUT}, observed=obs
    )


def test_fork2_typed_perturbation_kills_by_assertion_never_a_crash(tmp_path):
    # A type-directed perturbation on its observed type never raises: a discriminating test kills
    # it by ASSERTION (a sign distinction), not a perturbation crash. This is why observation
    # replaces the UNDEFINED-from-mis-typing hazard.
    src = "def f(x):\n    return x * 2\n"
    mod, path = _load(tmp_path, "mu_neg", src)
    try:
        (m,) = [
            m
            for m in generate_mutants(
                ast.parse(src).body[0],
                {MutationCategory.OUTPUT},
                max_per_category=0,
                observed_return_types=frozenset({"int"}),
            )
            if m.dimension.split(":")[1] == "return_negate"
        ]

        def case():
            assert mod.f(3) == 6

        r = evaluate_mutant(m, [case], mod.f, qualname="f", source_path=path)
        assert r.killed and r.killed_by == "assertion"
    finally:
        sys.modules.pop("mu_neg", None)


def _sub_mode(m):
    # dimension is "OUTPUT:<mode>:<lineno>"
    return m.dimension.split(":")[1] if m.dimension else ""


def test_default_policy_generates_no_output_mutant():
    # μ⁻ is opt-in: the one-sign default filter must not surface OUTPUT, so nothing changes
    # for existing consumers until the two-sign policy is explicitly requested.
    node = ast.parse("def f(x):\n    return x * 2\n").body[0]
    assert MutationCategory.OUTPUT not in filter_categories(node)
    assert MutationCategory.OUTPUT in filter_categories(node, two_sign=True)


def test_form_a_generates_the_three_always_applicable_submodes():
    modes = {_sub_mode(m) for m in _output_mutants("def f(x):\n    return x.upper()\n")}
    assert modes == {"return_none", "return_const", "return_identity"}


def test_a_kill_is_always_by_assertion_never_a_perturbation_crash(tmp_path):
    # The Fork-1 soundness claim: →none/→const/→identity never RAISE on application, whatever
    # the codomain type. On a str-returning function a discriminating test kills every
    # perturbation by ASSERTION (a value distinction) — never "crash"/"exception", which would
    # be a perturbation that could not apply and must be UNDEFINED, not a kill.
    src = "def f(x):\n    return x.upper()\n"
    mod, path = _load(tmp_path, "mu_str", src)
    try:

        def case():
            assert mod.f("ab") == "AB"

        for m in _output_mutants(src):
            r = evaluate_mutant(m, [case], mod.f, qualname="f", source_path=path)
            assert r.constructed and r.installed, _sub_mode(m)
            assert r.killed and r.killed_by == "assertion", (_sub_mode(m), r.killed_by)
    finally:
        sys.modules.pop("mu_str", None)


def test_const_survives_a_degenerate_suite_and_dies_on_a_dependence_pin(tmp_path):
    # The independence claim, both directions. f(0)==0 cannot distinguish `return n*2` from
    # `return 0`, so →const SURVIVES — the suite has not pinned that the output depends on the
    # input. f(3)==6 pins it, so →const is value-KILLED.
    src = "def f(n):\n    return n * 2\n"
    mod, path = _load(tmp_path, "mu_dep", src)
    try:
        (m,) = [m for m in _output_mutants(src) if _sub_mode(m) == "return_const"]

        def degenerate():
            assert mod.f(0) == 0

        def pins_dependence():
            assert mod.f(3) == 6

        assert not evaluate_mutant(
            m, [degenerate], mod.f, qualname="f", source_path=path
        ).killed
        killed = evaluate_mutant(
            m, [pins_dependence], mod.f, qualname="f", source_path=path
        )
        assert killed.killed and killed.killed_by == "assertion"
    finally:
        sys.modules.pop("mu_dep", None)


def test_identity_catches_a_missing_transform(tmp_path):
    # →identity fences non-triviality: f("") == "" cannot tell `x.upper()` from `x`, so
    # →identity SURVIVES; f("a") == "A" pins the transform and KILLS it.
    src = "def f(x):\n    return x.upper()\n"
    mod, path = _load(tmp_path, "mu_id", src)
    try:
        (m,) = [m for m in _output_mutants(src) if _sub_mode(m) == "return_identity"]

        def degenerate():
            assert mod.f("") == ""

        def pins_transform():
            assert mod.f("a") == "A"

        assert not evaluate_mutant(
            m, [degenerate], mod.f, qualname="f", source_path=path
        ).killed
        assert evaluate_mutant(
            m, [pins_transform], mod.f, qualname="f", source_path=path
        ).killed
    finally:
        sys.modules.pop("mu_id", None)


def test_a_noop_perturbation_is_not_a_target():
    # A perturbation syntactically identical to the original return is a guaranteed-equivalent
    # and must NOT be generated — the universe carries no built-in survivor (the discipline
    # _ExceptionMutator applies to already-pass handlers).
    for src, absent in [
        ("def f(x):\n    return None\n", "return_none"),
        ("def f(x):\n    return 0\n", "return_const"),
        ("def f(x):\n    return x\n", "return_identity"),
    ]:
        modes = {_sub_mode(m) for m in _output_mutants(src)}
        assert absent not in modes, (src, absent, modes)


def test_a_nested_functions_return_is_not_the_targets_codomain():
    # Only the target function's own returns are its codomain; a return inside a nested def is
    # that helper's, and must not be perturbed.
    src = "def f(x):\n    def g(y):\n        return y + 1\n    return g(x)\n"
    muts = _output_mutants(src)
    # The outer `return g(x)` is line 4; the nested `return y + 1` is line 3 — never a target.
    lines = {int(m.dimension.split(":")[2]) for m in muts if m.dimension}
    assert lines == {4}, lines
