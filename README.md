# Wesker

**Mutation testing that counts the behavioral questions your tests leave open — one mutant per question, not the forty ways to ask it.**

<p align="center">
  <a href="https://github.com/rohanvinaik/Wesker/actions/workflows/ci.yml"><img src="https://github.com/rohanvinaik/Wesker/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=rohanvinaik_Wesker"><img src="https://sonarcloud.io/api/project_badges/measure?project=rohanvinaik_Wesker&amp;metric=alert_status" alt="Quality Gate"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-3367d6.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-3367d6.svg" alt="Python 3.10+"></a>
</p>

`9 semantic categories · 1.00 mutants per behavioral dimension · Zero dependencies · Fully deterministic`

Point it at Detective's 25 files and it finds 2,795 behavioral questions your tests could leave open. A traditional tool asks those same questions **8,657 times**:

```
Detective · 25 files · 271 functions

  the questions            2,795
  ─────────────────────────────────
                   mutants  per dim
  mutmut 3           8,657     3.10
  Wesker exhaustive  5,693     2.04
  Wesker DOF         2,795     1.00
```

Same 25 files, same functions. One tool writes 8,657 mutants; the other writes exactly one per question and stops.

---

## The number that is an artifact

Mutation testing measures what your tests actually pin down. Change the code, see whether a test complains, and every change that slips through silently is a behavior your suite never constrained — a gap line coverage cannot see. DeMillo, Lipton, and Sayward formalized it in 1978. The objection was always cost, and the tools have answered it: mutmut 3 pools workers and clears thousands of mutants a minute. But speed was never the whole problem.

The other half is the denominator, and it is an artifact. Enumerate every change an operator can express and you count one gap many times over. An unconstrained return value can be phrased forty ways, so you get forty survivors for a single fact. The number that comes out is not weighted by your code. It is weighted by the operator's enumeration. Worse, most of the universe is one fact restated: of mutmut's 8,657 mutants on Detective, **4,212 (49%) sit in code with no test at all** — "this is untested," reported four thousand times. Neither the repetition nor the untested count is a property of your code.

Wesker counts the questions instead of the phrasings. A function has some number of *behavioral dimensions* — places where its behavior could genuinely differ — and that is the real denominator. Exhaustive mode is the ceiling: every mutant, **2.04 per question**. DOF mode is the claim: **2,795 mutants for 2,795 questions, 1.00.** One mutant per dimension, on a real repository, and it was not tuned toward — the cover sets here are singletons, so greedy selection is *exactly* optimal rather than the 1−1/e that bounds the general case. It costs nothing to trust: DOF reproduces the exhaustive verdict with **98.26% agreement and zero false "specified" claims**, and where it differs it *under*-claims. Run `--complete` and the ceiling comes back unchanged — not a fallback, a receipt.

---

## A surviving mutant is a specification gap, and its category is the fix

Line coverage tells you which code *executed*. Wesker tells you which *behaviors* your tests constrain — the more useful question. A function at 99% line coverage can still have a 40% kill rate, meaning 60% of its behavioral dimensions could change without a single test noticing. The tests prove the code runs. They do not prove what it computes.

Each surviving mutant is one specific alteration that changes behavior and goes unnoticed. Together the survivors are a constructive map of everything the tests do *not* require the function to do — its **negative space** — and the kill rate is how much of that behavior is pinned: **specification completeness.** And each survivor arrives tagged with the dimension it came from, so you get the diagnosis, not just the symptom:

| Category | What it mutates | What survival means |
|----------|----------------|---------------------|
| **VALUE** | Constants (`0`→`1`, `True`→`False`, `"x"`→`""`) | Tests don't pin exact outputs |
| **BOUNDARY** | Comparisons (`<`↔`<=`, `>`↔`>=`, `==`↔`!=`) | Tests don't exercise boundary conditions |
| **ARITHMETIC** | Operators (`+`↔`-`, `*`↔`/`, `//`→`/`, unary `-`) | Tests don't verify computations |
| **LOGICAL** | Boolean logic (`and`↔`or`, drop `not`) | Tests don't exercise conditional composition |
| **SWAP** | Argument order in calls | Tests can't distinguish argument positions |
| **STATE** | `self.x = …`→dropped, `return x`→`return None` | Tests don't verify side effects or return values |
| **TYPE** | `isinstance(x, T)`→`True` | Tests don't exercise type guards |
| **STMT** | Deletes a statement (`items.append(y)`, `cfg[k] = v`) | Tests don't notice the statement's effect at all |
| **EXCEPTION** | Raised type, handler body→`pass`, caught type→`BaseException` | Tests don't pin what raises, what's caught, or what a handler does |

The category *is* the diagnosis. A VALUE survivor says *assert the exact value, not the shape*. A BOUNDARY survivor says *test at the boundary, not near it*. The fix is always specific, never "write more tests." Together the categories cover the standard operator set from the literature — AOR, ROR, COR, UOI — plus **SDL** (statement deletion, which Delamaro and Offutt rank among the highest-value operators because it catches what operator-*replacement* structurally cannot), and domain operators for state, type guards, and exception behavior. `total = abs(total)` deleted a rebinding no replacement operator would touch; `cfg[k] = v` mutated a caller's object where `STATE` only ever targeted `self.x`; extraction across a `try` boundary changed what raises where. Each is a real refactor that passes every assertion a suite had, and none was a survivor before, because none was in the universe.

---

## How it gets fast without losing soundness

The cost drops multiplicatively across three layers, and no information is lost at any one of them.

**Layer 1 · In-process AST mutation.** Traditional tools spawn a subprocess per mutant, rewrite source on disk, and shell out to the runner — roughly 400 ms of overhead before a single test runs. Wesker compiles mutant ASTs in memory, patches them into a sandboxed namespace through the test's `__globals__`, and evaluates in-process: no subprocess, no disk I/O. The mutated function is compiled from the same AST a file-rewriting tool would produce and judged by the same assertion — the execution path differs, the observable semantics do not. This is the **meta-mutant dispatch** pattern validated by mutest-rs (Lévai & McMinn, ICST 2023) and mu2 (Vikram & Padhye, ISSTA 2023).

**Layer 2 · Categorical exclusion — the Monty Hall filter.** Before generating a single mutant, Wesker walks the AST and asks which categories even *have* a target. No comparison operators → no BOUNDARY mutant can exist; no `self.x = …` → no STATE; fewer than two call arguments → no SWAP. This is not sampling. It is elimination of structural impossibilities, and skipping an absent category loses exactly nothing. A typical function has 3–4 applicable categories out of 9, cutting the space 40–60% before any test runs.

**Layer 3 · Targeted test discovery.** A test that neither imports, references, nor transitively calls the mutated function cannot detect the mutation; running it is pure waste. Wesker resolves covering tests in three tiers — convention (`src/query.py` → `tests/test_query.py`), then static AST impact, then full fallback only if the first two find nothing — so each mutant runs against 3–15 tests rather than the whole suite. This is the one comparison with nothing confounded: `scope_tests=False` *is* classical mutation testing, and it was Wesker's own default until these numbers were taken.

| Repo | Tests | Every test | Covering only | |
|------|-------|-----------|---------------|--|
| ModelAtlas | 1,000 | 1,625 s | **348 s** | **4.7×** |
| Detective | 306 | 966 s | **141 s** | **6.9×** |
| Prism · `economics.py::analyze` | 421 | 33.6 s | **1.8 s** | **18.7×** |

The ratio is the smaller half of it: at the same budget the classical arm *did not finish*, truncating 9 functions on Detective and 7 on ModelAtlas, so its number is a sample of whatever was cheap to reach.

The two cost models sit side by side. Classical mutation testing costs $O(\text{functions} \times \text{mutants/fn} \times \text{subprocess-startup} \times \text{full-suite})$; Wesker costs $O(\text{functions} \times \text{applicable mutants} \times \text{in-process toggle} \times \text{covering tests})$. Every factor after the first is smaller, and none of the three reductions changes a verdict.

---

## Provably the least work a sound profiler can do

The speedup is not a bag of heuristics that happen to work. Every layer is either lossless or optimal within computational hardness, so the pipeline is the minimal-work sound mutation profiler up to an NP-hardness ceiling.

The three reductions change no verdict. In-process evaluation runs the same AST and the same assertion a file-rewriting tool would; categorical exclusion skips only mutants that *cannot be generated*; targeted discovery skips only tests that *cannot kill*. None can flip a single kill-or-survive result.

The selection layer spends the remaining budget optimally. Choosing which mutants to test is a maximum-coverage problem over behavioral dimensions. Give each mutant a cover set — the dimensions it pins — and Wesker picks greedily, taking the mutant with the largest marginal coverage each time. Three facts, each **machine-checked in Lean against `Mathlib`**, make that the right move: coverage is submodular (`coverage_submodular`), so marginal coverage is antitone (`marginal_antitone`) — a dimension already covered is never worth re-covering — and greedy therefore attains

$$\mathrm{opt} - g_k \;\ge\; \big(1 - e^{-1}\big)\,\mathrm{opt} \qquad (\text{proved: } \texttt{greedy\_coverage\_bound}).$$

That number is not a convenient bound. Maximum coverage is NP-hard, and 1 − 1/e ≈ 0.632 is the best ratio *any* polynomial-time algorithm can guarantee unless P = NP (Feige, 1998). Greedy attains the ceiling. Set `max_per_category=0` and the budget disappears: the classical result comes back unchanged. The selection layer decides only *which* mutants a bounded run tests — never *what a mutant means*.

---

## The honest ledger

The numbers above are the ones that matter, and these are the ones that do not:

- **Wesker is not faster per mutant.** mutmut 3 runs a forking worker pool and did all 8,657 mutants in 81 s on Detective — about 9 ms each. In-process evaluation removes subprocess overhead, but a modern pooled runner has largely removed it too. Any claim of a ~400× per-mutant advantage would be measured against a tool that no longer exists.
- **Suite time is the real variable.** Detective's suite runs in 0.3 s, the best case for a per-mutant runner. The covering-tests reduction compounds as suite time grows, so the gap widens on slow suites and narrows to nothing on fast ones.
- **The claim is knowledge per mutant, not mutants per second.** 1.00 is not a speed record. It is the proof that no mutant was spent twice on the same question.

The reduction is in redundancy, not in rigour. Wesker does not test fewer things. It tests each thing once.

---

## Equivalent mutants, and MC/DC

Some mutants are *semantically equivalent* — no input distinguishes them from the original — and they inflate the denominator and make a score look worse than it is. When a mutant survives, Wesker compiles both versions, runs them on synthesized boundary inputs (`0, 1, -1, 0.5, boundary ± 1`), and compares outputs. If every input agrees, the mutant is **likely equivalent** — no test *can* kill it — and it is reported separately and removed from the effective rate:

$$\text{effective kill rate} \;=\; \frac{\text{killed}}{\text{tested} - \text{equivalent}}.$$

Detection needs at least one non-exception comparison; if every input raises, the result is inconclusive, not "equivalent." For safety-critical functions that must prove every condition independently affects the outcome — [DO-178C Level A](https://en.wikipedia.org/wiki/DO-178C) — Wesker verifies Modified Condition/Decision Coverage by flipping each comparison operator independently against the full suite.

---

## Install and run

```bash
uv add wesker          # or: pip install wesker
```

```bash
wesker src/                    # profile a directory
wesker src/ --threshold 90     # fail CI below 90%
```

Zero dependencies beyond the standard library and your test framework. Python 3.10+. As a GitHub Action, one step, and survivors land on the diff as code-scanning alerts:

```yaml
- uses: actions/checkout@v4
  with: {fetch-depth: 0}
- uses: rohanvinaik/Wesker@v0.7.1
  with:
    base-ref: ${{ github.event.pull_request.base.sha }}
    sarif: .wesker/wesker.sarif
- uses: github/codeql-action/upload-sarif@v3
  with: {sarif_file: .wesker/wesker.sarif, category: wesker}
```

It requires pytest and a suite that passes on unmutated code, and it fails loudly rather than publish a number it cannot stand behind — a non-pytest runner, a suite broken before any mutant exists, or a budget-truncated run each fail with the diagnosis and the fix.

By default Wesker samples, and three things keep that honest. The universe is always computable: `estimate_universe_size()` counts every possible target by walking the AST, with no generation and no execution, so `82/82 [82/173]` means 82 tested and killed out of a 173-mutant universe — you always know how much of the space you covered. Multi-pass convergence deepens rather than re-rolls: each pass takes the next window of the greedy order, so N passes test N × `max_per_category` *unique* mutants per category, not a fresh random subset. And category order comes from predictive priors — each run writes per-category survival rates to `.wesker/mutation_report.json`, and the next run tests the historically weak categories first. Set `max_per_category = 0` and the budget disappears: every mutant, identical to classical mutation testing.

**[Full usage → `docs/usage.md`](docs/usage.md)** — CLI reference, library API, configuration, action inputs and outputs, and the scheduled whole-repo workflow.

---

## What the measurement is for

A function is fully specified exactly when the set of behaviors its tests leave open collapses to the function itself — when the conditional entropy of the function given its tests hits zero:

$$\mathrm{SC}(f)=1 \quad\Longleftrightarrow\quad H\big(f \mid \mathrm{tests}\big)=0.$$

Every surviving mutant is one bit of behavior your tests never asked for. And specification completeness is the prerequisite for everything you would want to do to code and *prove* you did not break it — safe refactoring, algebraic optimization, mechanical transformation. A function whose behavior its tests underdetermine is an ambiguous specification: you cannot safely optimize, cache, parallelize, or refactor what you cannot prove equivalent. That proof is what [Detective](https://github.com/rohanvinaik/Detective) builds on top of Wesker; the theory — specification complexity as a lattice from *no tests* to *fully specified*, the Monty Hall architecture, and the link between mutation pressure and algebraic decomposability — is in the [mutation-testing theory](https://github.com/rohanvinaik/LintGate/blob/main/docs/mutation/mutation-theory.md) document.

For decades, mutation testing was something you ran overnight, if you ran it at all. Wesker makes it a line in your CI log — and a proof that the line is honest.

---

MIT — Rohan Vinaik. Zero dependencies, fully deterministic, and the `(1−1/e)` selection bound is machine-checked in Lean against Mathlib.
