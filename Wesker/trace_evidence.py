"""Outcome-qualified per-TestId baseline evidence — the proof ledger (#17).

The baseline trace observes which test executed which line. "A trace observed this line" and "a
baseline-green, contained test proves this line under the session" are different facts, and
unioning coverage BEFORE qualifying it by outcome is how a failing test's reach came to close a
line ledger (Detective #59's counterexample). This module holds the per-item ledger those two
views derive from, without loss through early unioning:

  * ``observed`` reach — every test that executed the line, conservative routing/diagnostic
    evidence (Wesker #15 may consume it, but it is not proof);
  * ``admissible`` reach — only baseline-green, contained, non-truncated observations, the only
    evidence that may discharge a statement (later: arc) obligation.

ARCS ARE NOT YET RECORDED HERE. Line execution cannot distinguish the two edges of a conditional,
short-circuit operands, or loop-zero-vs-entry — the branch obligations #17 also asks for. Adding
them needs the tracer to record arc events, a separate change; this ledger is statement-level and
carries the outcome qualification the proof view was missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def trace_admissibility(
    baseline_passed: bool, truncated: bool, contained: bool, fresh: bool = True
) -> str:
    """Whether a baseline observation may discharge a proof obligation (#17/#20, pure — pinned).

    Only a baseline-GREEN, CONTAINED, non-TRUNCATED, FRESHLY-observed trace is admissible; every
    other case names WHY, because "the test failed", "its trace was cut", "the measurement escaped
    containment", and "this reach was replayed from cache" are different facts a certificate and a
    user must keep apart. Raw reachability stays available as the observed view — this decides only
    what may be PROOF.

    Order encodes precedence, and it is not arbitrary:

    * ``refuse_uncontained`` — containment is ABSORBING: a measurement the harness could not
      contain (a worker it could not stop) may have perturbed every observation in it, so nothing
      it saw is proof. Checked first because it invalidates the whole measurement, not one item.
    * ``refuse_truncated`` — this item's trace hit the budget and was CUT, so its line set is
      UNDER-counted; an under-counted trace cannot be read as "did not reach", which is the false
      completeness a truncated union would manufacture.
    * ``refuse_failed`` — the item is baseline-RED (or skipped/errored): it does not pass on the
      unmutated program, so its execution proves nothing about the code under test.
    * ``refuse_replayed`` — the reach was served from the trace CACHE, not measured this session
      (#20). Cached reach is keyed by source, not by the full fixture/conftest/plugin/config
      context, so a context change can make it stale; it is useful ROUTING (shortlist a test) but
      "structurally incapable of being mistaken for fresh admissible coverage". A proof obligation
      must rest on a trace observed THIS session, never a replay. Checked last: a replay of a green,
      contained, whole test is still only routing.
    * ``admissible`` — green, contained, whole, freshly observed: the evidence a proof may rest on.

    ``fresh`` defaults True so a caller that does not track provenance keeps the pre-#20 meaning.
    """
    if not contained:
        return "refuse_uncontained"
    if truncated:
        return "refuse_truncated"
    if not baseline_passed:
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
    #: Empty unless the trace was run with arc capture (opt-in — it doubles the hot callback).
    #: Governed by the SAME admissibility as ``lines``: a failing/truncated/uncontained owner's
    #: arcs prove nothing either.
    arcs: tuple[tuple[int, int], ...] = ()
    #: Where this reach came from (#20): ``fresh`` (traced THIS session) or ``replayed`` (served from
    #: the source-keyed trace cache). A replay is routing-usable but proof-INADMISSIBLE — the reason
    #: is then ``refuse_replayed`` — so cache reuse is structurally incapable of becoming fresh
    #: admissible coverage. Default ``fresh`` keeps every pre-#20 construction admissible.
    provenance: str = "fresh"

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
    """
    failed = set(failed_ids)
    truncated = set(truncated_ids)
    replayed = set(replayed_ids or ())
    arcs = arc_coverage or {}
    ledger: list[TraceEvidence] = []
    for test_id in sorted(line_coverage):
        passed = test_id not in failed
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
                reason=trace_admissibility(passed, is_truncated, contained, fresh),
                arcs=tuple(sorted(set(arcs.get(test_id, ())))),
                provenance="fresh" if fresh else "replayed",
            )
        )
    return tuple(ledger)
