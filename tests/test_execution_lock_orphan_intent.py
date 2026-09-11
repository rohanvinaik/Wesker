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

import threading

from Wesker.engine import (
    _lock_owner_from_repr,
    execution_lock_disposition,
    parse_lock_owner,
)

MAIN = 1000
WORKER = 2000
GONE = 3000
REPR_GONE = 4000


def test_rlock_repr_still_names_its_owner():
    """THE CONTRACT CHECK for the third channel, and the reason it is allowed to exist.

    `_thread.RLock.__repr__` is not documented, so the repr channel rests on an observation
    rather than a promise. This constructs a genuinely held lock and asserts the parse reads its
    owner back, so a CPython format change fails LOUDLY here instead of silently degrading orphan
    detection to the two channels measured insufficient on 2026-09-10.

    If this ever fails, the repr channel is gone -- that is the finding, and the answer is to say
    so, not to relax the assertion.
    """
    lock = threading.RLock()
    assert parse_lock_owner(repr(lock)) == 0, "an unheld lock names no owner"
    with lock:
        assert parse_lock_owner(repr(lock)) == threading.get_ident()


def test_the_repr_channel_reads_the_live_engine_lock():
    """The accessor abstains rather than raising, and reads 0 when nothing holds the lock."""
    assert _lock_owner_from_repr() == 0


def test_holding_it_outranks_every_bookkeeping_question():
    """`acquired` wins whatever the owner record says.

    Once the caller holds the lock the owner bookkeeping is irrelevant, and a stale record must
    never be able to turn a successful acquire into a refusal.
    """
    assert execution_lock_disposition(True, GONE, 0, MAIN, (MAIN,)) == "acquired"
    assert execution_lock_disposition(True, 0, 0, MAIN, (MAIN, WORKER)) == "acquired"


def test_the_record_channel_names_a_dead_owner():
    """An owner was recorded and that thread is gone -- nobody will ever release it."""
    assert (
        execution_lock_disposition(False, GONE, 0, MAIN, (MAIN, WORKER)) == "orphaned"
    )


def test_the_sole_survivor_channel_stands_without_any_record():
    """The caller is the only live thread and still cannot acquire.

    Then the holder is definitionally not alive, whatever the record says -- and the record is
    exactly what is missing here (`owner_tid=0`), because the measured defect kills the thread in
    the window between `acquire()` returning and the record being written.
    """
    assert execution_lock_disposition(False, 0, 0, MAIN, (MAIN,)) == "orphaned"


def test_the_repr_channel_catches_what_the_other_two_cannot():
    """THE MEASURED CASE, 2026-09-10 -- and the reason a third channel exists at all.

    Sampled live: `owner=6147403776` held the lock, that thread was NOT among `live_tids`, the
    record read 0 (the thread died inside the unrecordable window), and TWO threads were alive so
    sole-survivor correctly saw nothing. Both original channels blind; the repr named it.

    Verifying the fix against a target where only main survived is what let this gap ship -- the
    incompleteness was written in the docstring and then never exercised.
    """
    assert (
        execution_lock_disposition(False, 0, REPR_GONE, MAIN, (MAIN, WORKER))
        == "orphaned"
    )


def test_the_three_orphan_channels_are_independent():
    """None is derived from another, so each must fire alone.

    Collapsing any two would lose whichever case the survivor does not cover -- which is exactly
    what happened when the record and sole-survivor were the whole of it.
    """
    record_only = execution_lock_disposition(False, GONE, 0, MAIN, (MAIN, WORKER))
    survivor_only = execution_lock_disposition(False, 0, 0, MAIN, (MAIN,))
    repr_only = execution_lock_disposition(False, 0, REPR_GONE, MAIN, (MAIN, WORKER))
    assert record_only == survivor_only == repr_only == "orphaned"


def test_orphaned_and_contended_never_share_a_signifier():
    """The load-bearing property: opposite remedies must not collapse.

    A live owner will release, so waiting is correct. A dead owner never will, so waiting is
    futile and a longer bound only lengthens the hang. One sign for both states is how a hang
    gets talked past.
    """
    contended = execution_lock_disposition(False, WORKER, 0, MAIN, (MAIN, WORKER))
    orphaned = execution_lock_disposition(False, GONE, 0, MAIN, (MAIN, WORKER))
    assert contended == "held_by_live_thread"
    assert orphaned == "orphaned"
    assert contended != orphaned


def test_zero_means_no_record_not_a_thread_id():
    """0 is the absence of a record, and must never be matched against live ids as if it were one.

    With other threads alive and nothing recorded, the honest answer is that neither remedy is
    established -- not an orphan claim the evidence does not support.
    """
    assert (
        execution_lock_disposition(False, 0, 0, MAIN, (MAIN, WORKER))
        == "free_but_unacquired"
    )
    assert execution_lock_disposition(False, 0, 0, MAIN, (MAIN, WORKER)) != "orphaned"


def test_every_code_is_reachable_and_they_are_all_distinct():
    """Four states, four remedies. A code nothing can produce is a state nobody can act on."""
    seen = {
        execution_lock_disposition(True, 0, 0, MAIN, (MAIN,)),
        execution_lock_disposition(False, GONE, 0, MAIN, (MAIN, WORKER)),
        execution_lock_disposition(False, WORKER, 0, MAIN, (MAIN, WORKER)),
        execution_lock_disposition(False, 0, 0, MAIN, (MAIN, WORKER)),
    }
    assert seen == {
        "acquired",
        "orphaned",
        "held_by_live_thread",
        "free_but_unacquired",
    }
