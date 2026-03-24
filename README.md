# Wesker

**Mutation testing just went from "overnight batch job" to "runs in CI on every commit."**

`Zero dependencies · In-process AST mutation · 100-500x faster · Fully deterministic`

Mutation testing is the gold standard of test quality — mutate the code, check if tests catch it, find the gaps nothing else finds. Everyone knows this. Almost nobody runs it. Because it takes hours: spawn a subprocess per mutant, rewrite files on disk, run the full test suite each time, repeat a thousand times.

Wesker makes it operational. A 50-function module profiles in under 30 seconds. Your CI runs it on every commit. The cost dropped by two orders of magnitude and the soundness didn't change — every mutant is fully evaluated, no sampling, no approximation.

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

4.2 seconds. 24 functions. 79 mutants generated, evaluated, and reported. Two survived — both VALUE mutations, both telling you exactly which constant on which line isn't pinned by any test. That's not a coverage gap. Coverage said 90%. This is a *specification* gap: your tests prove the code runs, but not what it computes.

---

## How it gets 100-500x faster without losing soundness

The cost reduction is multiplicative across three layers. Each is provably safe — no information is lost.

**Layer 1: In-process AST mutation.** Traditional tools spawn a subprocess per mutant and rewrite source files on disk. Wesker compiles mutants from AST, monkey-patches them into a sandboxed namespace, and evaluates directly. Same process. No I/O. No startup overhead. (~10x)

**Layer 2: Categorical filtering.** Before generating a single mutant, Wesker walks the function's AST and checks which mutation categories are structurally possible. No comparisons? BOUNDARY mutants can't exist — skip them. No `self.x = ...`? STATE mutants can't exist — skip them. This is provably sound: if the syntactic target doesn't exist, the mutation can't be generated, so skipping it loses nothing. (~2-5x)

**Layer 3: Targeted test discovery.** Traditional tools run the full test suite against every mutant. Wesker discovers which tests are relevant in three stages: (1) naming convention (`src/query.py` → `tests/test_query.py`), (2) static AST impact analysis (which test files reference this function?), (3) full fallback only if layers 1-2 found nothing. Most functions resolve at layer 1. Each mutant runs against 3-10 tests, not 500. (~5-10x)

**Combined: 100-500x.** And the result is identical to exhaustive mutation testing on the functions and categories that survive filtering. The speedup comes from *not doing work that provably can't produce results*, not from approximating the work.

---

## Five categories tell you *what kind* of gap you have

Not just "a mutant survived" but *which behavioral dimension* your tests leave unconstrained:

| Category | What it mutates | What survival means | The fix |
|----------|----------------|-------------------|---------|
| **VALUE** | Constants (`0` → `1`, `True` → `False`) | Tests don't pin exact outputs | Assert expected values, not just shapes |
| **BOUNDARY** | Comparisons (`<` → `<=`, `>=` → `>`) | Tests don't exercise edge cases | Test at the boundary, not near it |
| **SWAP** | Argument order in function calls | Tests can't tell which arg is which | Test with asymmetric inputs |
| **STATE** | `self.x = ...` → removed, `return x` → `return None` | Tests don't verify side effects | Assert state after mutation |
| **TYPE** | `isinstance(x, T)` → `True` | Tests don't exercise type guards | Test with wrong-type inputs |

The category IS the diagnosis. Every surviving mutant points to a specific kind of test to write — not "write more tests" but "write *this* test for *this* reason."

---

## Equivalent mutant detection

The classic problem with mutation testing: some mutants are *semantically equivalent* — no input can distinguish them from the original. These inflate the denominator and make your score look worse than it is.

When a BOUNDARY mutant survives (`<` → `<=`), Wesker evaluates both versions on boundary inputs. If `f(0) == f_mutant(0)` for all boundary values, the mutant is equivalent — no test *can* kill it. Marked and excluded. Your kill rate reflects real specification gaps, not false positives.

## MC/DC verification

For safety-critical functions where you need to prove that every condition independently affects the output — the [DO-178C Level A](https://en.wikipedia.org/wiki/DO-178C) standard for flight-critical software — Wesker verifies Modified Condition/Decision Coverage by flipping each comparison operator independently against the full test suite.

```bash
wesker --mcdc src/scoring.py::_bank_score
```

---

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

### CLI

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

---

## The bigger picture

Mutation testing has been the gold standard since DeMillo, Lipton, and Sayward formalized it in 1978. For 48 years, it's been too expensive to use routinely. The tools that exist (mutmut, cosmic-ray, PIT) are faithful implementations of the original idea — and they inherit its cost: O(mutants × test suite). On real codebases, that's hours.

Wesker doesn't approximate mutation testing. It restructures the computation so the expensive parts don't happen when they provably can't contribute. The AST tells you which categories are possible. The test graph tells you which tests are relevant. The filtering is sound — nothing is skipped that could have produced a result. The evaluation is exact — every generated mutant is fully tested.

The result: mutation testing at the speed of linting, with the diagnostic power of formal verification. Every commit. Every CI run. No config.

The theoretical foundation — specification complexity as a Blum complexity measure, the five-field identification theorem, and the connection between mutation pressure and algebraic decomposability — lives in [LintGate](https://github.com/rohanvinaik/LintGate). Wesker is the engine, extracted to run standalone.

---

No config files. No subprocess spawning. No external dependencies beyond your test framework.

MIT — Rohan Vinaik
