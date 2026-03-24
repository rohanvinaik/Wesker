# Wesker

**Mutation testing that tells you *what* your tests don't specify, not just *how much*.**

`Zero dependencies · In-process AST mutation · 5 semantic categories · Runs in CI`

Your test suite passes. Coverage is 90%. You feel safe. Then a one-character change breaks production — because coverage measures which lines *execute*, not which behaviors are *specified*.

Wesker finds the difference. It rewrites your code's AST, runs your existing tests against each mutant, and tells you exactly which behavioral degrees of freedom your tests leave unconstrained. Not "line 42 isn't covered" but "the constant `0.5` on line 42 can be changed to `0.0` and no test notices."

## What it finds

```bash
wesker src/
```

```
Wesker — profiling 12 files

  query.py          18/18  ✓  1200ms
  patterns.py       14/15     890ms
    └─ _gradient_decay: VALUE survived — return 0.5 can become 0.0 (line 73)
  scoring.py        45/45  ✓  2100ms

──────────────────────────────────────────────────
Kill rate: 97% (77/79)
Functions: 24
Elapsed: 4190ms
```

Two mutants survived. Both are VALUE mutations — constants your tests don't pin. That's not a coverage gap. It's a *specification* gap: your tests prove the function runs, but not what it computes.

## Five categories of mutation

Every surviving mutant tells you something different about what your tests don't constrain:

| Category | What it mutates | What survival means |
|----------|----------------|-------------------|
| **VALUE** | Constants (`0` → `1`, `True` → `False`) | Tests don't pin exact outputs |
| **BOUNDARY** | Comparisons (`<` → `<=`, `>=` → `>`) | Tests don't exercise edge cases |
| **SWAP** | Argument order in function calls | Tests don't verify which argument is which |
| **STATE** | State mutations (`self.x = ...` → removed) | Tests don't verify side effects |
| **TYPE** | Type checks (`isinstance(x, T)` → `True`) | Tests don't exercise type discrimination |

The category IS the diagnosis. VALUE survivors mean "pin your constants." BOUNDARY survivors mean "test your edges." SWAP survivors mean "your arguments are interchangeable as far as the tests know." Each category prescribes a specific fix — not "write more tests" but "write *this* test."

## Why it's fast

Traditional mutation testing spawns a subprocess per mutant, rewrites files on disk, and takes hours. Wesker:

- **Mutates in-process.** AST rewriting + `exec()` in a sandboxed namespace. No files touched.
- **Filters before generating.** No comparisons in the function? Skip BOUNDARY. No `self.x = ...`? Skip STATE. Only generate mutants in categories that are structurally present.
- **Discovers tests in 3 layers.** Convention matching (`src/query.py` → `tests/test_query.py`), then static AST impact analysis (which tests reference this function?), then full fallback. Most functions resolve at layer 1.
- **Budgets per function.** Default 10s per file. Never blocks CI.

A 50-function module profiles in under 30 seconds.

## Quick start

```bash
pip install wesker
wesker src/
```

### GitHub Action

```yaml
- uses: rohanvinaik/wesker@v1
  with:
    targets: src/
    threshold: 90        # Fail if kill rate < 90%
```

### Options

```
wesker [targets...] [options]

  --threshold N            Exit 1 if kill rate < N%
  --mcdc FILE::FUNC ...    MC/DC verification on specific functions
  --json                   JSON output (for CI parsing)
  --budget MS              Per-file time budget (default: 10000ms)
  --max-per-category N     Max mutants per category (default: 3)
  --exclude FILE ...       Files to skip
  --quiet                  Minimal output
```

## Equivalent mutant detection

When a BOUNDARY mutant survives (`<` → `<=`), Wesker checks whether it's *semantically equivalent* — whether any input can distinguish the original from the mutant. If `f(boundary_value)` produces the same result for both, no test *can* kill it. These are marked as equivalent and excluded from the kill rate, so your score reflects real specification gaps, not false positives.

## MC/DC verification

For functions where you need to prove that every condition independently affects the output — the [DO-178C Level A](https://en.wikipedia.org/wiki/DO-178C) standard for flight-critical software — Wesker verifies Modified Condition/Decision Coverage by flipping each comparison operator independently.

```bash
wesker --mcdc src/scoring.py::_bank_score
```

## The idea behind the categories

Mutation testing is usually treated as a test quality metric: "what percentage of mutants does your test suite kill?" Wesker treats it as a **specification completeness** metric. A surviving mutant is a program that behaves differently from yours but passes all your tests — meaning your tests don't specify which behavior is correct.

The five categories decompose *why* the specification is incomplete. This matters because each category has a different fix:

| Survival pattern | What's underspecified | The fix |
|---|---|---|
| VALUE survivors | Output values | Assert exact expected values, not just shapes |
| BOUNDARY survivors | Edge behavior | Test at the boundary, not just near it |
| SWAP survivors | Argument semantics | Test with asymmetric inputs |
| STATE survivors | Side effects | Assert state after mutation |
| TYPE survivors | Type contracts | Test with wrong-type inputs |

Knowing *that* tests are insufficient is coverage. Knowing *how* they're insufficient is specification. Wesker gives you the second one.

The theoretical foundation — specification complexity as a Blum complexity measure, the five-field identification theorem, 15,000 lines of Lean 4 proofs — lives in [LintGate](https://github.com/rohanvinaik/LintGate). Wesker is the engine, extracted to run standalone.

---

No config files. No subprocess spawning. No external dependencies beyond your test framework.

MIT — Rohan Vinaik
