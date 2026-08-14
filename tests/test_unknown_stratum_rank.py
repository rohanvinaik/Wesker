"""_unknown_stratum_rank: order the widen (unknown) stratum most-likely-reacher first (#15 C/B).

INTENT: the item-incremental widen must trace the strongest remaining signal first, so it discharges
obligations before paying for the weak ones. caller_reaches (a real call path, #15 B) < file_peer <
unknown_dynamic < unknown_no_path, and an unrecognised code sorts LAST (never ahead of a real signal).
"""

from __future__ import annotations

from Wesker.ci import _unknown_stratum_rank


def test_the_stratum_ranks_are_exact_and_ordered():
    """caller_reaches (the item names a production caller that reaches the target) is the strongest
    widen signal, then file_peer (its file names the target), then dynamic-uncertain, then
    no-signal-at-all. Exact ranks, so the sort is total and stable."""
    assert _unknown_stratum_rank("caller_reaches") == 0
    assert _unknown_stratum_rank("file_peer") == 1
    assert _unknown_stratum_rank("unknown_dynamic") == 2
    assert _unknown_stratum_rank("unknown_no_path") == 3


def test_an_unrecognised_code_sorts_last():
    """A code this rank does not know (a candidate that never reaches the widen, or a future stratum)
    sorts LAST — it can never jump ahead of a real unknown signal."""
    assert _unknown_stratum_rank("candidate_static") == 4
    assert _unknown_stratum_rank("") == 4
