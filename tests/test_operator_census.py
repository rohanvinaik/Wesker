"""The universe must say what it OMITTED and why (issue #22).

The engine could report which categories are in the universe and not why the others are out.
Absent-because-no-candidate-site and absent-because-policy-suppressed-it are different facts:
the first loses nothing, the second is a judgement the reader may want to disagree with. They
were indistinguishable from the report.

`is_pure` is the live case. A caller asserting purity asserts that ``self.x = ...`` writes are
unobservable, so remove_assign's targets stop counting — but a constructor's receiver state is
OUTPUT, not an internal side effect, so on `__init__` that assertion is doing real work and
nobody could see it. An internal assertion that record-mode count equals generator count cannot
catch this: both agree, and both omit the same sites.
"""

from __future__ import annotations

import ast

from Wesker.engine import MutationCategory
from Wesker.filter import category_census, filter_categories, operator_disposition

# The issue's own example: partial initialisation is a distinct observable behaviour.
_CTOR = "class P:\n    def __init__(self, v):\n        self.left = v\n        self.right = v\n"


def _ctor_node():
    return ast.parse(_CTOR).body[0].body[0]


def test_suppressed_constructor_state_is_named_not_merely_absent():
    """The defect. Under an asserted purity these targets vanish from the universe; the census
    has to say they existed and that a policy overlay removed them."""
    row = category_census(_ctor_node(), is_pure=True)[MutationCategory.STATE]
    assert row["disposition"] == "withheld"
    assert row["withheld"] == 2
    assert row["sub_modes"]["remove_assign"]["withheld_by"] == "purity_overlay"


def test_the_same_sites_are_generated_when_purity_is_not_asserted():
    """The control: the sites are real. If they vanished either way the census would be
    reporting a policy decision that never had anything to decide."""
    row = category_census(_ctor_node(), is_pure=False)[MutationCategory.STATE]
    assert row["disposition"] == "generated"
    assert row["generated"] == 2
    assert row["withheld"] == 0
    assert row["sub_modes"]["remove_assign"]["withheld_by"] == ""


def test_a_category_with_no_sites_is_not_applicable_not_withheld():
    """Nothing was decided and nothing is missing. Conflating this with `withheld` would bury
    the one case worth reviewing under dozens that are not."""
    # No return statement, no self-write, no loop — so all three STATE sub-modes are empty.
    # `def f(n): return n` is NOT this case: `return_none` has a target there, which is the
    # point of the sub-mode breakdown and was worth getting wrong once to confirm.
    node = ast.parse("def f(n):\n    print(n)\n").body[0]
    row = category_census(node)[MutationCategory.STATE]
    assert row["disposition"] == "not_applicable"
    assert row["generated"] == 0 and row["withheld"] == 0


def test_the_filter_agrees_with_the_census_by_construction():
    """`filter_categories` is DERIVED from the census rather than computing counts alongside it.
    Issue #9 shipped a false COMPLETE precisely because a second predicate answered a different
    question than the mutator; a census that could disagree with the filter it explains would be
    that defect wearing a report."""
    for pure in (False, True):
        node = _ctor_node()
        census = category_census(node, pure)
        derived = {c for c, row in census.items() if row["disposition"] == "generated"}
        assert derived == filter_categories(node, pure)


def test_partial_suppression_still_counts_as_generated():
    """`generated` wins when both are positive: policy removed some sites and the operator is
    still represented, so the universe is not missing it — while the withheld count travels
    alongside for a reader who wants the detail."""
    assert operator_disposition(3, 2) == "generated"
    assert operator_disposition(0, 2) == "withheld"
    assert operator_disposition(0, 0) == "not_applicable"
