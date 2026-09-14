# Wesker — usage

The [README](../README.md) is the argument: what specification completeness is, and why one mutant per behavioral dimension is a theorem rather than a tuning result. This is the manual.


## CLI

```bash
wesker src/                          # profile a directory
wesker src/scoring.py src/query.py   # profile specific files
wesker src/ --threshold 90           # fail CI if kill rate < 90%
wesker src/ --max-per-category 0 --passes 1   # exhaustive mode
wesker src/ --json                   # JSON output for CI
wesker --mcdc src/scoring.py::compute_score   # MC/DC verification
```

Full option reference:

```
wesker [targets...] [options]      # no targets: the current directory, recursively

  --version                Print the installed version and exit
  --threshold N            Exit 1 if kill rate < N%
  --mcdc FILE::FUNC ...    MC/DC verification on specific functions
  --json                   JSON output (for CI parsing)
  --budget MS              Per-file time budget (default: 10000ms)
  --max-per-category N     Mutants per category per pass (default: derived per function from its
                           degrees of freedom; 0=exhaustive)
  --passes N               Convergence passes (default: 1; extra passes deepen within covered
                           dimensions)
  --exclude FILE ...       Files to skip
  --quiet                  Minimal output
  --purge                  Delete regeneratable .wesker/ files from old runs and exit
```

## As a library

```python
from Wesker.ci import profile_function

result = profile_function(".", "src/scoring.py", "compute_score")
print(result["kill_matrix"])        # which test killed which mutant
print(result["survivor_records"])   # survivors, with category + location
print(result["is_gateable"])        # True if every mutant was tested
```

Whole file, or whole codebase:

```python
from Wesker.ci import profile_file, profile_codebase

for r in profile_file(".", "src/scoring.py", passes=3):
    print(f"{r['function_key']}: {r['total_killed']}/{r['total_mutants']}")

result = profile_codebase(".", ["src/scoring.py", "src/query.py"])
print(result["kill_pct"], result["per_category"])
```

## Configuration

```toml
# pyproject.toml
[tool.wesker]
source_dir = "src/mypackage"
exclude = ["src/mypackage/server.py"]
max_per_category = 0          # mutants per category per pass; unset (the default) derives it
                              # per function from its degrees of freedom, 0 = exhaustive
convergence_passes = 1        # convergence passes (default: 1)
mcdc_targets = [["src/mypackage/scoring.py", "compute_score"]]
```

Layout is auto-detected from `[tool.coverage.run]`, `[tool.hatch.build]`, or an `src/*/` convention when `[tool.wesker]` is absent.

## GitHub Action

Two workflows. The first runs on every pull request and is scoped to the diff, so it costs
seconds. The second measures the whole codebase on a schedule and writes the badge. On a public
repo GitHub's standard runners are free, so both cost nothing but wall-clock nobody is waiting on.

**On pull requests** — survivors land as code-scanning alerts, inline on the diff:

```yaml
name: Specification
on: pull_request
permissions:
  contents: read
  security-events: write        # required to upload SARIF
jobs:
  spec:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0        # the diff scope needs the merge base
      - uses: rohanvinaik/Wesker@v1.1.1
        with:
          base-ref: ${{ github.event.pull_request.base.sha }}
          sarif: .wesker/wesker.sarif
      - uses: github/codeql-action/upload-sarif@v3
        if: always() && hashFiles('.wesker/wesker.sarif') != ''
        with:
          sarif_file: .wesker/wesker.sarif
          category: wesker
```

**On a schedule** — the whole codebase, for the badge:

```yaml
name: Specification (full)
on:
  schedule: [{cron: "0 6 * * 1"}]
  workflow_dispatch:
jobs:
  spec:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: rohanvinaik/Wesker@v1.1.1
        id: spec
        with:
          budget: "30000"
      - run: echo "${{ steps.spec.outputs.spec-pct }}% of ${{ steps.spec.outputs.dimensions-total }} dimensions"
```

See [`.github/workflows/`](../.github/workflows/) for the versions this repo runs on itself,
including badge generation.

| Input | Default | |
|-------|---------|--|
| `base-ref` | — | scope to files changed since this ref; omit for the whole codebase |
| `targets` | auto | explicit files; otherwise discovered from `[tool.wesker]` or layout |
| `budget` | `15000` | per-file budget, ms |
| `threshold` | — | fail if `spec-pct` is below this |
| `sarif` | `.wesker/wesker.sarif` | where to write the SARIF report |
| `allow-truncation` | `false` | report a budget-limited sample rather than failing |

Outputs: `spec-pct`, `dimensions-total`, `dimensions-pinned`, `dimensions-unspecified`,
`kill-pct`, `total-mutants`, `survivors`, `execution-mode`, `report`, `sarif`.

**Requires pytest, and a suite that passes on unmutated code.** The action would rather fail
loudly than publish a number it cannot stand behind: it refuses to report if the run falls back
to the non-pytest runner, if most of the suite fails before any mutant is introduced (a broken
environment reports `0%` and looks identical to unspecified code), or if the budget truncated
the run (a sample is not a completeness measurement). Each refusal prints the one-line diagnosis
and the fix.

### Badge

The scheduled workflow writes SVGs to a `badges` branch, so the README stays untouched:

```markdown
![Specification](https://raw.githubusercontent.com/OWNER/REPO/badges/.github/badges/specification.svg)
```
