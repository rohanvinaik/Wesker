#!/usr/bin/env python3
"""Knowability II, Appendix A — two checks that run in under a second with no dependencies.

    python docs/theory/knowability_ii/local_guards_check.py

CHECK 1 (Cor. 3.2 of KNOWABILITY_II.md). Theorem 3.2 of the knowability paper, run at the point of
mutation. Take a comparison ``a ? b``. Its LOCAL return type is the three regions
R_s = {L (a<b), E (a==b), G (a>b)}. Each replacement operator has a local footprint: the regions where
its truth value differs from the original's. The theorem says the singleton footprints (the local value
guards) are the minimal sufficient regime, and every other replacement's footprint must be a union of
guard footprints it contains. The guards it produces are EXACTLY the Kaminski–Ammann–Offutt sufficient
ROR replacements (J. Syst. Softw. 86(8), 2013), for all six operators.

CHECK 2 (Cor. 3.3). The same theorem applied to Wesker's live ``_BoundaryMutator`` table
(``Wesker/engine.py``, class ``_BoundaryMutator``: ``_SWAP`` / ``_DIRECTION`` / ``_EQUALITY`` plus the
True/False predicate constants). For ``<`` and ``>`` the ``!=`` guard is missing, so a suite can kill
every BOUNDARY mutant Wesker emits for ``a < b`` while never once exercising the ``a > b`` side: observe
only L and E and all five die; the ``!=`` mutant (footprint {G}) is non-equivalent and survives. For
``==`` and ``!=`` on ORDERABLE operands the one-sided guards are missing too (``<=``/``>=`` for ``==``,
``<``/``>`` for ``!=``); on unorderable operands those replacements raise, so that half of the repair is
type-conditional (NEGATIVE_SPECIFICATION Def. 11.10, Fork 2).

Both checks are exhaustive over the 3-region local type, so they are proofs for that type, not samples.
Exit 0 iff both checks hold as stated above.
"""

from __future__ import annotations

import sys

REGIONS = "LEG"
TRUTH = {
    "<": (1, 0, 0), "<=": (1, 1, 0), ">": (0, 0, 1), ">=": (0, 1, 1),
    "==": (0, 1, 0), "!=": (1, 0, 1), "True": (1, 1, 1), "False": (0, 0, 0),
}
ORIGINALS = ["<", "<=", ">", ">=", "==", "!="]

# Kaminski, Ammann & Offutt (2013), Table: the sufficient ROR replacements per operator.
KAMINSKI = {
    "<": {"<=", "!=", "False"}, "<=": {"<", "==", "True"}, ">": {">=", "!=", "False"},
    ">=": {">", "==", "True"}, "==": {"<=", ">=", "False"}, "!=": {"<", ">", "True"},
}

# Wesker/engine.py ``_BoundaryMutator`` as read on 2026-11-07 (commit 2fbd029).
WESKER_SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
WESKER_DIRECTION = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
WESKER_EQUALITY = {"<": "==", ">": "==", "<=": "==", ">=": "=="}


def footprint(orig: str, op: str) -> frozenset[str]:
    """Local footprint: the regions where ``op`` and ``orig`` disagree."""
    return frozenset(r for r, (x, y) in zip(REGIONS, zip(TRUTH[orig], TRUTH[op])) if x != y)


def sufficient(regime: set[str], target: str, orig: str) -> bool:
    """Theorem 3.2's condition at the site: every b in Mov(target) lies in some regime footprint
    that is contained in Mov(target)."""
    fp_t = footprint(orig, target)
    return all(any(b in footprint(orig, p) and footprint(orig, p) <= fp_t for p in regime) for b in fp_t)


def check_1() -> bool:
    ok = True
    print("CHECK 1 — Theorem 3.2 at the point of mutation reproduces the Kaminski sufficient ROR set")
    for orig in ORIGINALS:
        alts = {op for op in TRUTH if op != orig}
        guards = {op for op in alts if len(footprint(orig, op)) == 1}
        holds = all(sufficient(guards, t, orig) for t in alts)
        match = guards == KAMINSKI[orig]
        ok &= holds and match
        fps = ", ".join(f"{op}:{{{''.join(sorted(footprint(orig, op)))}}}" for op in sorted(alts))
        print(f"  {orig:>2}: guards={sorted(guards)}  sufficient-for-all={holds}  matches-Kaminski={match}   [{fps}]")
    return ok


def check_2() -> bool:
    print("CHECK 2 — the same theorem on Wesker's BOUNDARY table (uncertified targets = the gap)")
    expected_gap = {"<": ["!="], ">": ["!="], "==": ["<", "<=", ">", ">="], "!=": ["<", "<=", ">", ">="],
                    "<=": [], ">=": []}
    ok = True
    for orig in ORIGINALS:
        regime = {WESKER_SWAP[orig], "True", "False"}
        if orig in WESKER_DIRECTION:
            regime.add(WESKER_DIRECTION[orig])
        if orig in WESKER_EQUALITY:
            regime.add(WESKER_EQUALITY[orig])
        alts = {op for op in TRUTH if op != orig}
        gap = sorted(t for t in alts - regime if not sufficient(regime, t, orig))
        missing = sorted({op for op in alts if len(footprint(orig, op)) == 1} - regime)
        ok &= gap == expected_gap[orig]
        print(f"  {orig:>2}: wesker={sorted(regime)}  missing-guards={missing}  uncertified-targets={gap}")
    print("  separating case for '<': I = {L,E,G}, O = {L,E}: all five Wesker mutants die; '!=' survives.")
    return ok


if __name__ == "__main__":
    good = check_1() and check_2()
    print("OK" if good else "MISMATCH — the document's Cor. 3.2 / 3.3 no longer describe this table")
    sys.exit(0 if good else 1)
