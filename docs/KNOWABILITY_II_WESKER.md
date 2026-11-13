# Knowability II — Wesker's side

The theory and the full plan live in Detective, which is the planning home for the pair:

- theory: `Detective/docs/theory/knowability_ii/KNOWABILITY_II.md` (what a survivor of the wider regime can say)
- handoff block (carry verbatim): `Detective/docs/theory/knowability_ii/CONSTRAINT_BLOCK.md`
- the staged plan for BOTH repositories: `Detective/openspec/changes/implement-knowability-ii/tasks.md`

This file is the engine-side summary so a session opened in Wesker alone has what it needs. If it ever
disagrees with `tasks.md`, `tasks.md` wins.

## What Wesker does in this change

Wesker **measures**; Detective decides and speaks. Every addition here is annotation on records that
already exist (`ProfilingResult.survivor_records` / `killed_records`), defaulting absent, with one
exception: W1 changes the operator universe and moves the policy id.

| Stage | What | Why (theory) | Default |
|---|---|---|---|
| **W1** | BOUNDARY gains the `!=` guard for `<`/`>` (`BOUNDARY:<Op>~far`) and type-conditional one-sided guards for `==`/`!=` on orderable operands; policy 10; DOF counts 3 questions per ordering comparison (the three local guards), the other alternatives marked `subsumed_by` | the knowability paper's Theorem 3.2 run at the point of mutation (KII Thm 3.1, Cor 3.2) reproduces the Kaminski sufficient ROR set and shows the current table is incomplete (Cor 3.3): a suite can kill all five mutants of `a < b` without ever exercising `a > b` | on (universe change) |
| **W2** | site identity (`site_id`), reach per survivor from existing admissible trace evidence, and an opt-in infection probe (`site_trace=True`) recording per test: reached / infected / propagated | DeMillo–Offutt's three conditions measured separately; infection is the one the engine does not observe today | off |
| **W3** | `Wesker/constraints.py`: a reach formula (path guards over parameters, single-assignment bindings inlined) and an infect formula (per-operator table) per mutant, with a kind: boolean / linear / polynomial / opaque | the conditions as constraints Detective's exact solver can decide (KII §4) | on (static, cheap) |
| **W4** | surprise counts per survivor with a MEASURED propagation base q (this function's own killed infected pairs); `unmeasured` when no base exists, never 0 | KII Def 7.4; SIGNIFICANCE_WEIGHTING §2: the base is read off the real data, never defaulted | off (needs W2) |
| **W5** | report schema 2 (`site`, `constraint`, `surprise`, `subsumed_by`, `q_repo`), `--site-trace`, docs; re-measure the README headline under policy 10 | — | — |

## The check you can run now

`python docs/local_guards_check.py` — exhaustive over the 3-region local type; exit 0 iff the Kaminski
set is reproduced (check 1) and the gap in the live `_BoundaryMutator` table is exactly as described
(check 2). After W1, update the `WESKER_*` dicts and `expected_gap` in the script and in
`tests/test_boundary_local_guards_intent.py` (task 0.2 / W1.4) so the script keeps describing the live
table.

## Invariants (from the constraint block)

- `SCORED_DISPOSITIONS` and `mutant_disposition` unchanged; a `site` block is evidence about why a
  survivor survived, never a disposition.
- With `site_trace=False`, every `killed`, `killed_by`, disposition and count is byte-identical
  (`tests/test_site_trace_invariance_intent.py`, W2.4).
- The probe is semantics-preserving: returns its argument unchanged, never copies, never calls `__eq__`.
- No new dependency.
- A policy id moves only when the universe or the observation rule changes (W1), as the CHANGELOG
  records for 8 → 9; W2–W5 do not move it.

## Where to start

Task 0.2 in `tasks.md`: write `tests/test_boundary_local_guards_intent.py` pinning the gap against the
live table, so W1 can flip it. Then W1.1.
