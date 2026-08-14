"""_unknown_stratum_rank: order the widen (unknown) stratum most-likely-reacher first (#15 C).

INTENT: the item-incremental widen must trace the strongest remaining signal first, so it discharges
obligations before paying for the weak ones. file_peer < unknown_dynamic < unknown_no_path, and an
unrecognised code sorts LAST (never ahead of a real signal).
"""

from __future__ import annotations

from Wesker.ci import _unknown_stratum_rank


def test_the_stratum_ranks_are_exact_and_ordered():
    """file_peer (its file names the target) is the strongest widen signal, then dynamic-uncertain,
    then no-signal-at-all. Exact ranks, so the sort is total and stable."""
    assert _unknown_stratum_rank("file_peer") == 0
    assert _unknown_stratum_rank("unknown_dynamic") == 1
    assert _unknown_stratum_rank("unknown_no_path") == 2


def test_an_unrecognised_code_sorts_last():
    """A code this rank does not know (a future stratum, or a candidate that never reaches the widen)
    sorts LAST — it can never jump ahead of a real unknown signal."""
    assert _unknown_stratum_rank("candidate_caller") == 3
    assert _unknown_stratum_rank("") == 3
