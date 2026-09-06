"""next_routing_action: the item-incremental widen's stop/continue rule (#15, Fix B).

INTENT: the widen must trace only as far as it needs to. It COMPLETES the instant every proof
obligation is discharged (so the low-stratum unknowns are never traced — the efficiency win); it
declares a GAP only once the ELIGIBLE set is EXHAUSTED — the strata the DRIVER handed over as
`widen_tests`, which is the APPLICABLE set, not the whole collection (Detective hands over the
caller-reaching stratum only and discloses the rest as not consulted; the negative conclusion is
sound relative to that set, and the driver states the boundary); and a mid-widen containment loss
yields a typed UNRESOLVED, never a false gap.
"""

from __future__ import annotations

from Wesker.engine import next_routing_action


def test_discharged_obligations_complete_immediately():
    """The efficiency win: nothing open -> stop now, leaving any remaining unknowns untraced."""
    assert next_routing_action(False, True, False) == "complete"
    assert next_routing_action(False, False, False) == "complete"


def test_completion_dominates_a_late_containment_loss():
    """A fully-discharged run is complete even if containment was lost — nothing negative rests on
    the incomplete trace, so a settled positive verdict stands."""
    assert next_routing_action(False, True, True) == "complete"


def test_an_open_obligation_with_items_left_traces_next():
    """A survivor / provisional-equivalent / uncovered line remains and an eligible test could still
    discharge it — widen one more stratum, never concede yet."""
    assert next_routing_action(True, True, False) == "trace_next"


def test_an_open_obligation_with_no_items_left_is_the_honest_gap():
    """Every eligible unknown was traced and an obligation still stands — THIS is the specification
    gap, sound relative to the eligible set: the strata the driver handed over (its APPLICABLE set,
    whose boundary the driver discloses), now exhausted."""
    assert next_routing_action(True, False, False) == "gap"


def test_containment_loss_with_open_obligations_is_unresolved_not_a_gap():
    """A budget cut / truncation / uncontained worker crossed mid-widen with an obligation still
    open: a negative conclusion is invalid under an unfinished trace, so it is a typed non-gateable
    UNRESOLVED — the false gap #15 closeout #6 forbids — whether or not items remain."""
    assert next_routing_action(True, True, True) == "unresolved"
    assert next_routing_action(True, False, True) == "unresolved"
