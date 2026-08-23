"""§18 Q3 intent — Form-B (runtime-wrapper) μ⁻ equivalence.

Form A resolves equivalence by COMPILING the perturbation-as-mutant (Prop. 11.5); a Form-B perturbation is
a runtime ``wrapper_factory`` with no compilable mutant, so `check_equivalent` could never see it — the
synthetic marker node excepts on compile → a false NOT-equivalent. These pin the pure verdict and the
runtime mirror: a generator that yields nothing on the boundary inputs makes truncate/drop/duplicate/empty
no-ops → a runtime candidate-equivalent; a generator that yields items is distinguished.
"""

from __future__ import annotations

import ast

from Wesker.engine import (
    _generate_output_wrappers,
    check_equivalent,
    form_b_equivalence,
)


def _node(src):
    return ast.parse(src).body[0]


# ─── the pure verdict ───
def test_form_b_equivalence_names_the_three_outcomes():
    assert form_b_equivalence(["match", "match"]) == "equivalent"
    assert (
        form_b_equivalence(["match", "differ"]) == "distinguished"
    )  # any differ → a real kill
    assert form_b_equivalence(["differ"]) == "distinguished"
    assert (
        form_b_equivalence(["raised", "raised"]) == "no_evidence"
    )  # all raised → unclaimable
    assert form_b_equivalence([]) == "no_evidence"
    # `distinguished` outranks: a witness is decisive even with raises and matches also present.
    assert form_b_equivalence(["raised", "differ", "match"]) == "distinguished"


# ─── the runtime dispatch through check_equivalent (the negative mirror) ───
def test_form_b_perturbation_on_an_empty_generator_is_equivalent():
    # yields NOTHING on the boundary inputs (none > 100) → every yield perturbation is a no-op on [].
    g = _node("def g(x):\n    if x > 100:\n        yield x\n")
    muts = _generate_output_wrappers(g)
    assert muts and all(check_equivalent(g, m) for m in muts)


def test_form_b_perturbation_on_a_yielding_generator_is_distinguished():
    # yields items on the boundary inputs → truncate/drop/duplicate/empty all change the sequence.
    h = _node("def h(x):\n    yield x\n    yield 1\n")
    muts = _generate_output_wrappers(h)
    assert muts and not any(check_equivalent(h, m) for m in muts)
