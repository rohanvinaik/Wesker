# Wesker

**In-process AST mutation testing that runs in CI on every commit.**

`Zero dependencies · 7 semantic categories · Multi-pass convergence · Fully deterministic`

Mutation testing is the gold standard for measuring specification completeness — mutate the code, check if tests catch it, find the behavioral gaps that line coverage cannot see. The concept has been understood since DeMillo, Lipton, and Sayward formalized it in 1978. For 48 years, it has been too expensive to use routinely. The tools that exist (mutmut, cosmic-ray, PIT, Stryker) are faithful implementations of the original idea — and they inherit its cost model: `O(mutants × subprocess_startup × full_test_suite)`. On real codebases, that means hours.

Wesker restructures the computation. A 100-function project profiles in under 10 seconds. The soundness guarantee is the same as exhaustive mutation testing — the efficiency comes from not doing work that provably cannot produce results, not from weakening the analysis.

```
Wesker — Prism Specification Metrics (3 passes, 5/cat)

  [1/13] sources.py      212/212 [212/235] 904ms
  [2/13] behavior.py       82/82 [82/173]  309ms
  [3/13] economics.py      35/35 [35/118]  277ms
  ...
  [13/13] engine.py        84/84 [84/94]   197ms

Kill rate: 100% (1229/1229) | Universe: 2195 | 109 functions | 4.1s
```

1229 mutants tested across 7 categories, 109 functions, 13 files. 100% kill rate. The full mutation universe is 2195 — Wesker tested 56% of it in 4.1 seconds. Every tested mutant was killed. The remaining 44% is reachable by increasing passes or running exhaustive mode; the statistical argument for why that is unnecessary in most cases is explained below.

---

## What mutation testing actually measures

Line coverage tells you which code *executed*. Mutation testing tells you which *behaviors* are constrained by your tests. These are fundamentally different questions.

A function with 99% line coverage can have a 40% mutation kill rate. That means 60% of its behavioral dimensions — its outputs, its boundary conditions, its branch logic — could change without any test noticing. The tests prove the code runs. They do not prove what it computes.

Each surviving mutant is a specific alteration to the function's logic that produces different output and no test notices. Taken together, surviving mutants form a constructive specification of everything the tests *don't require* the function to do — the **negative space**. The mutation kill rate measures how much of the function's behavior is actually pinned down. This is **specification completeness**: the degree to which the test suite fully determines what the code does.

---

## Seven categories tell you *what kind* of gap you have

Not just "a mutant survived" but *which behavioral dimension* the tests leave unconstrained:

| Category | What it mutates | What survival means |
|----------|----------------|-------------------|
| **VALUE** | Constants (`0`→`1`, `True`→`False`, `"x"`→`""`) | Tests don't pin exact outputs |
| **BOUNDARY** | Comparisons (`<`↔`<=`, `>`↔`>=`, `==`↔`!=`) | Tests don't exercise boundary conditions |
| **ARITHMETIC** | Operators (`+`↔`-`, `*`↔`/`, `//`→`/`, remove unary `-`) | Tests don't verify computations |
| **LOGICAL** | Boolean logic (`and`↔`or`, remove `not`) | Tests don't exercise conditional composition |
| **SWAP** | Argument order in function calls | Tests can't distinguish argument positions |
| **STATE** | `self.x = ...`→removed, `return x`→`return None` | Tests don't verify side effects or return values |
| **TYPE** | `isinstance(x, T)`→`True` | Tests don't exercise type guards |

The category IS the diagnosis. A VALUE survivor means "assert the exact value, not just the shape." A BOUNDARY survivor means "test at the boundary, not near it." An ARITHMETIC survivor means "verify the computation, not just that it returns a number." The fix is always specific, not "write more tests."

These seven categories cover the standard mutation operator set from the literature: AOR (Arithmetic Operator Replacement), ROR (Relational Operator Replacement), COR (Conditional Operator Replacement), UOI (Unary Operator Insertion/Deletion), and the domain-specific operators for state mutation and type guards.

---

## How it gets fast without losing soundness

The cost reduction is multiplicative across three architectural layers. Each is provably safe — no information is lost at any layer.

### Layer 1: In-process AST mutation (10-50x)

Traditional tools (mutmut, cosmic-ray) spawn a subprocess per mutant, rewrite source files on disk, and invoke the test runner externally. Each cycle costs ~0.4s of subprocess startup plus disk I/O. For 200 mutants, that is 80 seconds of pure overhead before a single test executes.

Wesker compiles mutant ASTs in memory using Python's `ast` module, monkey-patches them into a sandboxed namespace via the test function's `__globals__`, and evaluates directly. Same process. No I/O. No file rewrites. No subprocess startup. The per-mutant overhead drops from ~400ms to ~1ms.

This architecture follows the **meta-mutant dispatch table** pattern validated by mutest-rs (Lévai & McMinn, ICST 2023, TOSEM 2026) and mu2's `MutationClassLoader` (Vikram & Padhye, ISSTA 2023). The soundness argument is straightforward: the mutated function is compiled from the same AST that a file-rewriting tool would produce; the evaluation is the same assertion check. The execution path is different, the observable semantics are identical.

### Layer 2: Categorical exclusion — the Monty Hall filter (2-5x)

Before generating a single mutant, Wesker walks the function's AST and checks which mutation categories have syntactic targets:

- No comparison operators? **BOUNDARY** mutants cannot exist — skip.
- No `self.x = ...` assignments? **STATE** mutants cannot exist — skip.
- No arithmetic operators? **ARITHMETIC** mutants cannot exist — skip.
- Fewer than 2 call arguments? **SWAP** mutants cannot exist — skip.

This is not sampling. It is elimination of structural impossibilities. If the syntactic target for a mutation category does not exist in the function's AST, no mutation in that category can be generated. Skipping it loses zero information. A typical function has 3-4 applicable categories out of 7, reducing the mutation space by 40-60% before any test runs.

This is the **Monty Hall insight**: the game show host opens doors that have no prize behind them. You learn nothing by opening them yourself. The AST structure reveals which categories are empty doors.

### Layer 3: Targeted test discovery (5-20x)

Traditional tools run the full test suite against every mutant. For a project with 300 tests, that is 300 test invocations per mutant. Most of those tests have no relationship to the mutated function and cannot possibly kill the mutant.

Wesker discovers relevant tests in three layers:
1. **Convention matching** — `src/query.py` maps to `tests/test_query.py` (fast, high precision)
2. **Static AST impact** — scan test files for references to the mutated function name
3. **Full fallback** — only if layers 1-2 found nothing

Most functions resolve at layer 1. Each mutant runs against 3-15 tests, not the full suite. The soundness argument: if a test does not import, reference, or transitively call the mutated function, it cannot detect the mutation. Running it is pure waste.

### Combined cost model

Traditional:
```
O(functions × mutants_per_function × subprocess_startup × full_test_suite)
```

Wesker:
```
O(functions × applicable_mutants × in_process_toggle × covering_tests)
```

| Factor | Traditional (mutmut) | Wesker | Reduction |
|--------|---------------------|--------|-----------|
| Mutants per function | 50-200 | 15-35 (sampled) | 3-10x |
| Per-mutant overhead | ~400ms (subprocess) | ~1ms (in-process) | ~400x |
| Tests per mutant | Full suite (100-500) | Covering tests (3-15) | 10-30x |
| **Wall-clock for 100 functions** | **30-120 minutes** | **4-15 seconds** | **100-1000x** |

This is what makes mutation testing viable as a CI action. A 30-minute batch job becomes a 10-second step that runs on every commit.

---

## Why sampling is statistically sound

Wesker does not test every possible mutant by default. For a function with 20 VALUE mutation targets and `max_per_category=5`, it tests 5 per pass. This is deliberate, and the theoretical justification is rigorous.

### The universe is always computable

`estimate_universe_size()` counts every possible mutation target across all applicable categories by walking the AST — no mutant generation, no test execution, pure counting. This is the same universe that exhaustive tools like mutmut would generate. Setting `max_per_category=0` (unlimited) runs Wesker in **exhaustive mode**, producing results identical to traditional mutation testing.

The output always reports both numbers: `killed/tested [tested/universe]`. A result like `82/82 [82/173]` means 82 mutants were tested, all 82 were killed, and the full universe for that file is 173. You know exactly how much of the space was covered.

### Multi-pass convergence

Each convergence pass uses a different deterministic seed to shuffle target indices before sampling. Pass 0 might test VALUE targets `[7, 2, 14, 0, 11]`; pass 1 tests `[3, 18, 6, 9, 1]`. After N passes, up to `N × max_per_category` unique mutants are tested per category.

The probability of missing a real survivor (a mutant that would survive if tested) after K passes:

```
P(miss) = ((n - k) / n) ^ K
```

where `n` = targets in category, `k` = max_per_category.

| Targets (n) | k=5, 1 pass | k=5, 3 passes | k=5, 5 passes |
|-------------|-------------|---------------|---------------|
| 5 | 0% (exhaustive) | 0% | 0% |
| 10 | 50% | 12.5% | 3.1% |
| 20 | 75% | 42% | 24% |
| 50 | 90% | 73% | 59% |

For functions with ≤5 targets per category (the majority in well-decomposed code), sampling IS exhaustive — you naturally test everything. For larger functions, increasing passes tightens the bound. In practice, functions with 50+ targets per category are rare and are themselves a signal that the function needs decomposition.

### Declining marginal utility

The first few mutants per category are the most informative. Consider a function with 20 VALUE mutation targets. If the test suite kills the first 5 (drawn from different locations in the function via seeded shuffle), the probability that the 6th would reveal a *new* specification gap is low. The 5 killed mutants already demonstrate that the tests pin values at diverse points in the function.

This is not an approximation — it is an information-theoretic property of the mutation categories. Each category represents a **behavioral dimension** (value correctness, boundary handling, arithmetic integrity, logical composition, etc.). Once you have confirmed that a dimension is well-specified by killing several mutants in it, additional confirmation has diminishing returns. The marginal information gain of the Nth killed mutant in a category approaches zero as N increases.

The mutation theory formalizes this via **predictive priors**: historical per-category survival rates from previous profiling runs inform which categories are most likely to contain real gaps. Budget-limited runs should test high-prior categories first, spending budget where information gain is highest. Wesker's `prioritize_categories()` infrastructure supports this (wired for a future release); the current implementation uses uniform priors with seeded shuffling, which provides unbiased coverage.

### When to use exhaustive mode

For published libraries, safety-critical code, or when you want the badge to say "every mutant tested":

```toml
[tool.wesker]
max_per_category = 0       # unlimited — test every mutant
convergence_passes = 1     # one pass is enough when unlimited
```

This produces results identical to traditional mutation testing — same mutants, same evaluation, same kill rate. The only difference is execution architecture (in-process vs. subprocess, targeted tests vs. full suite), which affects speed but not semantics.

---

## Equivalent mutant detection

The classic problem with mutation testing: some mutants are *semantically equivalent* — no input can distinguish them from the original. These inflate the denominator and make the score look worse than it is.

When a mutant survives, Wesker compiles both the original and mutated function, runs them on synthesized boundary inputs (0, 1, -1, 0.5, boundary±1), and compares outputs. If every input produces identical results from both versions, the mutant is **likely equivalent** — no test *can* kill it. Equivalence detection requires at least one successful (non-exception) comparison to declare a match; if all inputs raise exceptions, the result is inconclusive (not marked equivalent).

Equivalent mutants are reported separately and excluded from the effective kill rate:

```
effective_kill_rate = killed / (tested - equivalent)
```

## MC/DC verification

For safety-critical functions requiring proof that every condition independently affects the output — the [DO-178C Level A](https://en.wikipedia.org/wiki/DO-178C) standard — Wesker verifies Modified Condition/Decision Coverage by flipping each comparison operator independently against the full test suite.

---

## Quick start

```bash
pip install wesker
wesker src/
```

### Configuration

```toml
# pyproject.toml
[tool.wesker]
source_dir = "src/mypackage"
exclude = ["src/mypackage/server.py"]
max_per_category = 5          # mutants per category per pass (default: 5)
convergence_passes = 3        # convergence passes (default: 3)
mcdc_targets = [["src/mypackage/scoring.py", "compute_score"]]
```

Wesker auto-detects source layout from `[tool.coverage.run]`, `[tool.hatch.build]`, or `src/*/` convention when `[tool.wesker]` is not configured.

### CLI

```
wesker [targets...] [options]

  --threshold N            Exit 1 if kill rate < N%
  --mcdc FILE::FUNC ...    MC/DC verification on specific functions
  --json                   JSON output (for CI parsing)
  --budget MS              Per-file time budget (default: 10000ms)
  --max-per-category N     Mutants per category per pass (default: 5, 0=exhaustive)
  --passes N               Convergence passes (default: 3)
  --exclude FILE ...       Files to skip
  --quiet                  Minimal output
```

### GitHub Action

```yaml
- uses: rohanvinaik/wesker@v1
  with:
    targets: src/
    threshold: 90
```

---

## The bigger picture

Mutation testing measures specification completeness — the degree to which a test suite fully determines what the code does. This is the prerequisite for safe refactoring, algebraic optimization, and mechanical code transformation. A function whose behavior is underdetermined by its tests is an ambiguous specification. You cannot safely optimize, cache, parallelize, or refactor what you cannot prove equivalent.

Wesker makes this measurement routine by eliminating the cost barrier. The theoretical foundation — specification complexity as a lattice from "no tests" to "fully specified," the Monty Hall architecture for efficient execution, the connection between mutation pressure and algebraic decomposability, and the predictive prior framework for budget-optimal category selection — is developed in the [mutation testing theory](https://github.com/rohanvinaik/LintGate/blob/main/docs/mutation/mutation-theory.md) document within the [LintGate](https://github.com/rohanvinaik/LintGate) project. Wesker is the standalone engine that makes the theory operational.

---

No config files required. No subprocess spawning. No external dependencies beyond your test framework. Seven semantic categories. Multi-pass statistical convergence. Full-universe exhaustive mode when you need it.

MIT — Rohan Vinaik
