# Wesker

In-process AST mutation testing for Python. Zero external dependencies.

Wesker generates mutants by rewriting your code's AST, evaluates them by running your existing tests in-process, and reports which mutations survived — revealing gaps in your test suite's specification coverage.

## Why Wesker

**Traditional mutation testing** spawns a subprocess per mutant, rewrites files on disk, and takes hours on real codebases. Wesker runs entirely in-process: mutants are compiled from AST, monkey-patched into test namespaces, and evaluated directly. A 50-function module profiles in under 30 seconds.

**What it tells you:** Not "your coverage is 87%." Instead: "this specific constant on line 42 is not pinned by any test" and "this boundary comparison at line 78 produces identical behavior whether it's `<` or `<=`."

## Quick Start

```bash
pip install Wesker
wesker src/
```

Output:
```
Wesker — profiling 12 files

  query.py          18/18  ✓  1200ms
  patterns.py       14/15     890ms
  scoring.py        45/45  ✓  2100ms

──────────────────────────────────────────────────
Kill rate: 97% (77/79)
Functions: 24
Elapsed: 4190ms
```

## GitHub Action

```yaml
- uses: rohanvinaik/Wesker@v1
  with:
    targets: src/
    threshold: 90        # Fail if kill rate < 90%
    mcdc: src/scoring.py::_bank_score  # Optional MC/DC verification
```

## Concepts

### Mutation Categories

Wesker generates five categories of AST mutations:

| Category | What it does | What survival means |
|----------|-------------|-------------------|
| **VALUE** | Replace constants (`0` → `1`, `True` → `False`) | Tests don't pin exact output values |
| **BOUNDARY** | Flip comparisons (`<` → `<=`, `>=` → `>`) | Tests don't exercise boundary conditions |
| **SWAP** | Transpose function call arguments | Tests don't verify argument ordering |
| **STATE** | Remove `self.x = ...` or replace `return x` with `return None` | Tests don't verify state mutations |
| **TYPE** | Replace `isinstance(x, T)` with `True` | Tests don't exercise type discrimination |

### Monty Hall Filtering

Before generating mutants, Wesker checks which categories are structurally relevant. No comparisons? Skip BOUNDARY. No `self.x = ...`? Skip STATE. This is like the Monty Hall problem — eliminating doors with no prize before you choose.

### 3-Layer Test Discovery

1. **Convention** (fast): `src/query.py` → `tests/test_query.py`
2. **Static impact** (precise): AST scan for function name references across all test files
3. **Full fallback**: All `test_*.py` files under `tests/`

### Equivalent Mutant Detection

When a BOUNDARY mutant survives, Wesker checks if it's semantically equivalent by evaluating both original and mutant on boundary inputs. If `f(0) == f_mutant(0)` for all boundary values, the mutant is equivalent — no test *can* distinguish them.

### MC/DC Verification (DO-178C Level A)

For safety-critical scoring functions, Wesker verifies Modified Condition/Decision Coverage via text-level operator swaps. Each comparison operator is flipped independently and the full test suite re-run. 100% MC/DC means every condition independently affects the output.

## CLI Reference

```
wesker [targets...] [options]

Targets:
  src/                     Profile all .py files under src/
  src/core.py src/utils.py Profile specific files

Options:
  --threshold N            Exit 1 if kill rate < N%
  --mcdc FILE::FUNC ...    MC/DC verification on specific functions
  --json                   JSON output (for CI parsing)
  --budget MS              Per-file time budget (default: 10000ms)
  --max-per-category N     Max mutants per category (default: 3)
  --exclude FILE ...       Files to skip
  --quiet                  Minimal output
```

## Architecture

```
Wesker/
├── engine.py    # AST mutators, mutant generation, evaluation, patching
├── filter.py    # Monty Hall category filtering + predictive priors
├── ci.py        # 3-layer test discovery, equivalent mutant detection, profiling
├── metrics.py   # CI runner: mutation + MC/DC + sigma + badge metrics
└── cli.py       # CLI entry point
```

No external dependencies. No config files. No subprocess spawning. Just AST manipulation and your existing tests.

## License

MIT
