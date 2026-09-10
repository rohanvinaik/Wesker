"""Intent pins for the orphaned-execution-lock refusal (#19).

THE DEFECT, measured 2026-09-10. `converge` on a Wesker-internal target never completed --
8/8 runs, byte-identical stacks, ~2.1s of CPU consumed and then nothing for 600s. It was not a
grind: one live thread, blocked acquiring `_EXECUTION_LOCK`, which a thread that no longer
existed still owned (`lock_repr` named a dead tid). The documented 300s aggregate deadline never
fired, because a cooperative budget cannot be polled by a thread that has stopped executing.

HOW A DEAD THREAD OWNS IT, and why it is deterministic rather than a race: `interrupt`'s own
BOUNDARY says an `abandon` injection cannot land on a thread blocked outside the interpreter
until that call returns ON ITS OWN. A contended `acquire()` is such a call -- so the injection is
already pending while the worker sleeps, the acquire eventually succeeds, and the exception
arrives the instant it does. The thread dies OWNING the lock, before any `with` cleanup exists.

These tests are written from what the decision must do, not from what it currently returns; the
generated suite beside them is a characterization and would pin a wrong answer wrong.
"""

from __future__ import annotations

from Wesker.engine import execution_lock_disposition

MAIN = 1000
WORKER = 2000
GONE = 3000


def test_holding_it_outranks_every_bookkeeping_question():
    """`acquired` wins whatever the owner record says.

    Once the caller holds the lock the owner bookkeeping is irrelevant, and a stale record must
    never be able to turn a successful acquire into a refusal.
    """
    assert execution_lock_disposition(True, GONE, MAIN, (MAIN,)) == "acquired"
    assert execution_lock_disposition(True, 0, MAIN, (MAIN, WORKER)) == "acquired"


def test_the_record_channel_names_a_dead_owner():
    """An owner was recorded and that thread is gone -- nobody will ever release it."""
    assert execution_lock_disposition(False, GONE, MAIN, (MAIN, WORKER)) == "orphaned"


def test_the_sole_survivor_channel_stands_without_any_record():
    """The caller is the only live thread and still cannot acquire.

    Then the holder is definitionally not alive, whatever the record says -- and the record is
    exactly what is missing here (`owner_tid=0`), because the measured defect kills the thread in
    the window between `acquire()` returning and the record being written. This channel is the
    one that catches it; a detection resting on the record alone reports `free_but_unacquired`
    and waits out a lock that is never coming back.
    """
    assert execution_lock_disposition(False, 0, MAIN, (MAIN,)) == "orphaned"


def test_the_two_orphan_channels_are_independent():
    """Neither is derived from the other, so each must fire alone.

    Collapsing them into one rule would lose whichever case the survivor does not cover.
    """
    record_only = execution_lock_disposition(False, GONE, MAIN, (MAIN, WORKER))
    survivor_only = execution_lock_disposition(False, 0, MAIN, (MAIN,))
    assert record_only == survivor_only == "orphaned"


def test_orphaned_and_contended_never_share_a_signifier():
    """The load-bearing property: opposite remedies must not collapse.

    A live owner will release, so waiting is correct. A dead owner never will, so waiting is
    futile and a longer bound only lengthens the hang. One sign for both states is how a hang
    gets talked past.
    """
    contended = execution_lock_disposition(False, WORKER, MAIN, (MAIN, WORKER))
    orphaned = execution_lock_disposition(False, GONE, MAIN, (MAIN, WORKER))
    assert contended == "held_by_live_thread"
    assert orphaned == "orphaned"
    assert contended != orphaned


def test_zero_means_no_record_not_a_thread_id():
    """0 is the absence of a record, and must never be matched against live ids as if it were one.

    With other threads alive and nothing recorded, the honest answer is that neither remedy is
    established -- not an orphan claim the evidence does not support.
    """
    assert (
        execution_lock_disposition(False, 0, MAIN, (MAIN, WORKER))
        == "free_but_unacquired"
    )
    # and 0 must not be read as "a thread that is gone" merely because it is absent from live_tids
    assert execution_lock_disposition(False, 0, MAIN, (MAIN, WORKER)) != "orphaned"


def test_every_code_is_reachable_and_they_are_all_distinct():
    """Four states, four remedies. A code nothing can produce is a state nobody can act on."""
    seen = {
        execution_lock_disposition(True, 0, MAIN, (MAIN,)),
        execution_lock_disposition(False, GONE, MAIN, (MAIN, WORKER)),
        execution_lock_disposition(False, WORKER, MAIN, (MAIN, WORKER)),
        execution_lock_disposition(False, 0, MAIN, (MAIN, WORKER)),
    }
    assert seen == {
        "acquired",
        "orphaned",
        "held_by_live_thread",
        "free_but_unacquired",
    }
