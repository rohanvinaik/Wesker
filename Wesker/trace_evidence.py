"""Outcome-qualified per-TestId baseline evidence — the proof ledger (#17).

The baseline trace observes which test executed which line, and which branch EDGE: an arc is the
``(prev_line, cur_line)`` transition a line event completes inside one frame, so the two sides of a
conditional, short-circuit operands and loop-zero-vs-entry — which line execution collapses — stay
apart. "A trace observed this line" and "a baseline-green, contained test proves this line under the
session" are different facts, and unioning coverage BEFORE qualifying it by outcome is how a failing
test's reach came to close a line ledger (Detective #59's counterexample). This module holds the
per-item ledger those two views derive from, without loss through early unioning:

  * ``observed`` reach — every test that executed the line or arc, conservative routing/diagnostic
    evidence (Wesker #15 may consume it, but it is not proof);
  * ``admissible`` reach — only baseline-green, contained, non-truncated, freshly observed items,
    the only evidence that may discharge a statement OR arc obligation.

Each item carries what that qualification needs, and nothing is inferred from absence:

  * its typed baseline OUTCOME (:func:`baseline_outcome`) — ``passed | failed | skipped | xfailed |
    error``, pytest's own categories. ``baseline_passed`` is kept beside it for existing readers and
    is True only for ``passed``: a skipped or expected-failure item does not pass on the original,
    and its reach proves nothing (it used to read green, so an xfail test's reach closed lines);
  * its ARCS — recorded since ``4158ee4``. The session baseline captures them on every trace (the
    trace cache's v4 cell carries them, so a warm session has them too) and ``arcs_from_trace``
    keeps one function's; the per-function and precomputed baseline paths carry none, and an empty
    ``arcs`` reads "no arc evidence", never "no branch reached";
  * its REACH completeness — whether everything the item executed was VISIBLE to the tracer. The
    tracer follows threads a test starts while it runs; a thread that outlives the test's traced
    window may run target code nobody observed, so that item's reach is ``incomplete_thread`` and
    the lines it lacks are UNKNOWN, never a negative.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: The typed baseline outcomes (#17): pytest's own terminal categories for one item's run on the
#: UNMUTATED program. Only ``passed`` is green.
BASELINE_OUTCOMES = ("passed", "failed", "skipped", "xfailed", "error")


def item_run_status(setup: str, call: str, teardown: str) -> str:
    """pytest's terminal category for ONE run of a live item, read off its phase reports (#17, pure
    — pinned).

    The live-session runner raises only when a report FAILED, so a skipped item and an expected
    failure (``xfail``) both returned normally — and the baseline read them as PASSED. Measured: an
    xfail test asserting the wrong value on the false branch closed that branch's line in the
    admissible ledger, which is Detective #59's failing-only counterexample with a marker on it. The
    reports carry the distinction; this reads it the way pytest's own ``pytest_report_teststatus``
    does, restricted to the baseline vocabulary.

    Each argument is that phase's report outcome — ``passed`` / ``failed`` / ``skipped``, or
    ``xfailed`` for a skipped report carrying ``wasxfail`` (the expected failure happened) and
    ``xpassed`` for a passed one carrying it (a non-strict unexpected pass); ``""`` when the phase
    did not run. Precedence, earliest decisive phase first:

    * ``error``   — setup failed (the test never ran), or only teardown failed: pytest's ERROR.
    * ``failed``  — the call failed: its expectation does not hold on the original. A strict XPASS
      arrives here too, because pytest itself reports it failed. Checked before a teardown error,
      because the call's verdict is the one the test states.
    * ``xfailed`` — an expected failure (in any phase, ``xfail(run=False)`` included).
    * ``skipped`` — a skip in any phase: the test did not run to a verdict.
    * ``passed``  — the call passed. A non-strict XPASS is ``passed``: every assertion in it held on
      the original, which is all a proof needs; the stale marker is a fact about the test file.
    * ``unobserved`` — no call verdict and nothing decisive (no reports at all): the caller decides
      from its own channel.
    """
    if setup == "failed":
        return "error"
    if call == "failed":
        return "failed"
    if teardown == "failed":
        return "error"
    if "xfailed" in (setup, call, teardown):
        return "xfailed"
    if "skipped" in (setup, call, teardown):
        return "skipped"
    if call in ("passed", "xpassed"):
        return "passed"
    return "unobserved"


def baseline_outcome(run_code: str | None, item_status: str) -> str:
    """The typed baseline outcome of ONE test on the UNMUTATED program (#17, pure — pinned).

    Two channels describe one run and neither is enough alone. ``run_code`` is the engine's
    (``_run_test_with_timeout``: None for a clean return, else ``assertion`` / ``exception`` /
    ``crash`` / ``timeout`` / ``uncontained``); it cannot tell a skip from a pass, because the
    live-session runner returns normally for both. ``item_status`` is pytest's own category for the
    item (:func:`item_run_status`), or ``unobserved`` for a callable that is not a live pytest item.

    * ``error`` when the run never reached a verdict (``timeout`` / ``uncontained``): whatever a
      report says about a run that was stopped is partial. Checked first.
    * pytest's ``error`` / ``failed`` when it reported one — it alone sees which PHASE failed, so a
      fixture's AssertionError is an error, not a wrong expectation.
    * a RAISING callable with no report: ``assertion`` / ``exception`` (a declared failure) is
      ``failed``; anything else is ``error``, never ``failed`` — under the direct-call contract a
      crash is ambiguous (a missing fixture argument), the rule ``failing_on_baseline`` keeps by not
      accusing such a test of a wrong expectation. An unrecognised code is ``error`` too: a new kill
      reason is by construction "did not pass".
    * a clean return: pytest's ``skipped`` / ``xfailed`` when it reported one, else ``passed``.
    """
    if run_code in ("timeout", "uncontained"):
        return "error"
    if item_status in ("error", "failed"):
        return item_status
    if run_code is not None:
        return "failed" if run_code in ("assertion", "exception") else "error"
    if item_status in ("skipped", "xfailed"):
        return item_status
    return "passed"


def trace_admissibility(
    baseline_passed: bool,
    truncated: bool,
    contained: bool,
    fresh: bool = True,
    outcome: str = "unrecorded",
) -> str:
    """Whether a baseline observation may discharge a proof obligation (#17/#20, pure — pinned).

    Only a baseline-GREEN, CONTAINED, non-TRUNCATED, FRESHLY-observed trace is admissible; every
    other case names WHY, because "the test failed", "it was skipped", "its trace was cut", "the
    measurement escaped containment", and "this reach was replayed from cache" are different facts a
    certificate and a user must keep apart. Raw reachability stays available as the observed view —
    this decides only what may be PROOF.

    Order encodes precedence, and it is not arbitrary:

    * ``refuse_uncontained`` — containment is ABSORBING: a measurement the harness could not
      contain (a worker it could not stop) may have perturbed every observation in it, so nothing
      it saw is proof. Checked first because it invalidates the whole measurement, not one item.
    * ``refuse_truncated`` — this item's trace hit the budget and was CUT, so its line set is
      UNDER-counted; an under-counted trace cannot be read as "did not reach", which is the false
      completeness a truncated union would manufacture.
    * ``refuse_error`` / ``refuse_skipped`` / ``refuse_xfailed`` — the typed ``outcome``
      (:func:`baseline_outcome`) says the item did not run to a green verdict: it errored in setup
      or teardown, was skipped, or failed as expected. Each is named, never folded into
      ``refuse_failed``: "your expectation is wrong" and "this test never ran to a verdict" have
      different remedies.
    * ``refuse_failed`` — the item is baseline-RED: its expectation does not hold on the unmutated
      program, so its execution proves nothing about the code under test. Also the answer whenever
      ``baseline_passed`` is False with no typed outcome to say more, and for an outcome this does
      not recognise — only an affirmative ``passed`` is green.
    * ``refuse_replayed`` — the reach was served from the trace CACHE, not measured this session
      (#20). Cached reach is keyed by source, not by the full fixture/conftest/plugin/config
      context, so a context change can make it stale; it is useful ROUTING (shortlist a test) but
      "structurally incapable of being mistaken for fresh admissible coverage". A proof obligation
      must rest on a trace observed THIS session, never a replay. Checked last: a replay of a green,
      contained, whole test is still only routing.
    * ``admissible`` — green, contained, whole, freshly observed: the evidence a proof may rest on.

    ``fresh`` defaults True so a caller that does not track provenance keeps the pre-#20 meaning;
    ``outcome`` defaults ``unrecorded`` so a caller with only the boolean keeps the pre-typing one.
    """
    if not contained:
        return "refuse_uncontained"
    if truncated:
        return "refuse_truncated"
    if outcome == "error":
        return "refuse_error"
    if outcome == "skipped":
        return "refuse_skipped"
    if outcome == "xfailed":
        return "refuse_xfailed"
    if not baseline_passed or outcome not in ("passed", "unrecorded"):
        return "refuse_failed"
    if not fresh:
        return "refuse_replayed"
    return "admissible"


@dataclass(frozen=True)
class TraceEvidence:
    """One test item's baseline observation, outcome-qualified (#17).

    Keyed by the exact ``test_id`` (Wesker #16 node identity), so duplicate function names and
    parametrized cases keep SEPARATE ownership — the whole point of not unioning early.
    """

    test_id: str
    lines: tuple[int, ...]
    baseline_passed: bool
    truncated: bool
    contained: bool
    #: :func:`trace_admissibility` code — ``admissible`` or the reason it is not.
    reason: str
    #: The (prev_line, cur_line) branch edges this item executed inside the target (#17), so a
    #: consumer can distinguish the two sides of a conditional that ``lines`` alone collapses.
    #: Populated on the session-baseline path, which captures arcs on every trace; empty on the
    #: per-function and precomputed paths, where it means "no arc evidence", never "no branch
    #: reached". Governed by the SAME admissibility as ``lines``: a failing/truncated/uncontained
    #: owner's arcs prove nothing either.
    arcs: tuple[tuple[int, int], ...] = ()
    #: Where this reach came from (#20): ``fresh`` (traced THIS session) or ``replayed`` (served from
    #: the source-keyed trace cache). A replay is routing-usable but proof-INADMISSIBLE — the reason
    #: is then ``refuse_replayed`` — so cache reuse is structurally incapable of becoming fresh
    #: admissible coverage. Default ``fresh`` keeps every pre-#20 construction admissible.
    provenance: str = "fresh"
    #: The typed baseline outcome (#17, :func:`baseline_outcome`): ``passed | failed | skipped |
    #: xfailed | error``. ``unrecorded`` for a row built without one — a hand construction, or a row
    #: rehydrated from a cache written before outcomes were typed — where ``baseline_passed`` alone
    #: decides, as it always did.
    baseline_outcome: str = "unrecorded"
    #: Whether everything this item executed was VISIBLE to the tracer (#17): ``complete``, or
    #: ``incomplete_thread`` when a thread it started outlived its traced window, so code that thread
    #: ran afterwards was never observed. It qualifies ABSENCE only: the lines that were observed
    #: stay exactly as admissible as ``reason`` says, while a line missing from ``lines`` is
    #: "not observed", never "not reached".
    reach: str = "complete"

    @property
    def admissible(self) -> bool:
        """Whether this observation may discharge a statement OR arc obligation."""
        return self.reason == "admissible"


def build_trace_ledger(
    line_coverage: Mapping[str, Iterable[int]],
    failed_ids: Iterable[str],
    truncated_ids: Iterable[str],
    contained: bool,
    arc_coverage: Mapping[str, Iterable[tuple[int, int]]] | None = None,
    replayed_ids: Iterable[str] | None = None,
    outcomes: Mapping[str, str] | None = None,
    incomplete_reach: Mapping[str, str] | None = None,
) -> tuple[TraceEvidence, ...]:
    """The per-TestId ledger over every item with observed line coverage (#17).

    A test with no coverage owns no line obligation, so only the covering items are recorded —
    but a covering item that is baseline-red or truncated IS kept, marked inadmissible with its
    reason, so a consumer sees "observed but does not prove" rather than the item silently
    vanishing (which is what an early ``admissible_line_coverage`` filter did, losing the fact
    that the line WAS reached, just not admissibly).

    ``arc_coverage`` (optional) supplies each item's branch edges — from a trace run with arc
    capture; when omitted, ``arcs`` is empty and only the statement view is populated. Arcs carry
    the SAME per-item admissibility as lines, since they are the same observation seen finer.

    ``outcomes`` (optional) is each item's typed baseline outcome (:func:`baseline_outcome`). An
    item it names is green only when its outcome is ``passed`` AND it is not in ``failed_ids``; an
    item it does not name is ``unrecorded`` and ``failed_ids`` alone decides, as before.
    ``incomplete_reach`` (optional) names the items whose reach the tracer could not fully observe,
    with the code saying why; every other item's reach is ``complete``.
    """
    failed = set(failed_ids)
    truncated = set(truncated_ids)
    replayed = set(replayed_ids or ())
    arcs = arc_coverage or {}
    typed = outcomes or {}
    unseen = incomplete_reach or {}
    ledger: list[TraceEvidence] = []
    for test_id in sorted(line_coverage):
        outcome = typed.get(test_id, "unrecorded")
        passed = test_id not in failed and outcome in ("passed", "unrecorded")
        is_truncated = test_id in truncated
        # #20: reach served from the cache is `replayed`, and `trace_admissibility` refuses it for
        # proof — routing may still use it, but it may not close an admissible obligation.
        fresh = test_id not in replayed
        ledger.append(
            TraceEvidence(
                test_id=test_id,
                lines=tuple(sorted(set(line_coverage[test_id]))),
                baseline_passed=passed,
                truncated=is_truncated,
                contained=contained,
                reason=trace_admissibility(
                    passed, is_truncated, contained, fresh, outcome
                ),
                arcs=tuple(sorted(set(arcs.get(test_id, ())))),
                provenance="fresh" if fresh else "replayed",
                baseline_outcome=outcome,
                reach=unseen.get(test_id, "complete"),
            )
        )
    return tuple(ledger)
