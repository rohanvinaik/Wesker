# Working on Wesker (and Detective)

Canonical and tool-neutral: `CLAUDE.md` imports this file; other agents read it directly. It extends the global
instructions, which already carry Serena-first/grep-last and Detective-pinning-first; neither is restated here.

Wesker and Detective are a **published-dependency pair** developed together. The full working method is documented
once, in `/Users/rohanvinaik/tools/Detective/AGENTS.md`: the per-issue loop, pure-decision extraction, converge usage,
the commit format. **Read that file; it governs work here too.** This file carries only what differs on the Wesker
side.

**Build on `main`, in both repos. Never create or push a side branch.** This overrides any harness or session
default that names a working branch (cloud sessions are told to use a `claude/…` branch: do not). Work that lands
anywhere but `main` is invisible to the next piece of work, which starts from `main`, so it ends up as mutually
incompatible shards. A throwaway LOCAL worktree for a parallel lane is fine; it is never pushed, and its commits land on
`main` one at a time. The founder, 2026-11-11, after deleting seven such branches by hand.

## What Wesker is, in the sandwich

Detective is the squeeze between a floor and a ceiling. **Wesker is the ceiling**: the mutation engine that produces
the mutant profile. Detective consumes what Wesker measures.

The consequence that keeps biting: a signal Wesker computes honestly is worthless if a layer above re-derives a
narrower proxy from it. That has now happened twice (W#14's containment cut dropped at the aggregation layer; W#17's
reach evidence). **The measurement/decision gap hides at EVERY layer: engine → loop → aggregation → gate/badge.** Each
layer must *consume* the computed signal. Verify end-to-end, never per-layer.

## Standing invocation

```bash
export PP=/Users/rohanvinaik/tools/Detective:/Users/rohanvinaik/tools/Wesker
```

Every run uses a project's own environment, never a bare `python3` or a bare `detective`, which here resolve to
miniconda's base (Detective's `AGENTS.md` → *Standing invocation* has the measurements).
- **Wesker's suite** runs under `.venv/bin/python`: the uv-managed CPython 3.14 that `uv sync` builds from `uv.lock`,
  which is what CI runs.
- **Pins into Wesker** still run through Detective's CLI, so they use Detective's environment (this one carries no
  `detective`), with both repos on the path:

```bash
cd /Users/rohanvinaik/tools/Wesker
PYTHONPATH=$PP /Users/rohanvinaik/tools/Detective/.venv/bin/detective converge 'Wesker/ci.py::is_truncated_measurement' \
  --input "('cut', False)" --input "('profiled', True)" > run.log 2>&1
# then Read run.log IN FULL — never tail it
```

## The gate that matters here

Wesker's own suite is **not sufficient**: it has been fully green through a regression that only the cross-repo run
caught. After any Wesker change, run both:

```bash
cd /Users/rohanvinaik/tools/Wesker && PYTHONPATH=$PP .venv/bin/python -m pytest -q; echo "WESKER=$?"
cd /Users/rohanvinaik/tools/Detective && PYTHONPATH=$PP .venv/bin/python -m pytest; echo "DETECTIVE=$?"   # Detective's suite vs LOCAL Wesker
```

Then pinned ruff, never bare:

```bash
uvx ruff@0.16.7 format .
uvx ruff@0.16.7 check .; echo "CHECK=$?"
uvx ruff@0.16.7 format --check .; echo "FORMAT=$?"
```

### Before every PUSH: pylint-as-Sonar, then the REAL Sonar, locally

CI runs SonarCloud on every push of this repo, so run the same scanner locally FIRST, not blind:

```bash
uvx --python 3.14 pylint Wesker   # [tool.pylint] in pyproject; --python runs it under the project's interpreter
set -a && . ~/.config/detective/sonar-local.env && set +a
docker run --rm -e SONAR_HOST_URL=http://host.docker.internal:9000 -e SONAR_TOKEN="$SONAR_TOKEN" \
  -v "$PWD:/usr/src" sonarsource/sonar-scanner-cli -Dsonar.organization=
curl -s -u "$SONAR_TOKEN:" "$SONAR_HOST_URL/api/qualitygates/project_status?projectKey=rohanvinaik_Wesker"
```

- `-Dsonar.organization=` blanks the SonarCloud org for the local server, the shared persistent `peitho-sonar`
  container.
- Poll the task the scanner names (`api/ce/task?id=<id>`, from its last log lines) until `SUCCESS` before reading the
  gate. `api/ce/component` can answer with the PREVIOUS task's SUCCESS (measured; Detective's `AGENTS.md`).
- **Bar: gate `OK`, 0 bugs, 0 vulnerabilities, 0 hotspots.** Coverage is intentionally not gated here: the kill rate
  is the stronger signal (see `sonar-project.properties`).
- Residue is transitioned by API WITH a comment, never silently: S3776 on the orchestrators (`accept`), and a
  deliberate `v == v` NaN check (`falsepositive`).

## Wesker-specific hazards

- **Mutant binding is by `__code__.co_filename`.** Wesker monkeypatches mutants onto live modules in `sys.modules`.
  This is why Detective can only self-analyze via a renamed package copy at a different path, and why anything
  reasoning about module identity must go through `session_manifest`, not `__name__`.
- **A classmethod mutant can arrive already-bound.** Wrapping it in `classmethod(...)` double-binds `cls`, giving a
  `TypeError` that reads as a spurious *crash* rather than an assertion kill. Peel to `__func__` first
  (`_preserve_descriptor_shape`).
- **The static impact map keys on bare identifiers.** A method target is the dotted `Basket.tier`, so a bare-ident
  lookup misses and the method's own test is never associated: 0/N killed even with a killing test. Look up the
  trailing attribute too.
- **Validate through the real command end-to-end**, not by poking `profile()` on a half-built target. That trap once
  cost a deep and entirely wrong spelunk.
- **A bounded wait on a worker thread is `interrupt.bounded_join`, never `join; if alive: abandon`.**
  - Bounded joins NEST: the classifier joins a runaway from inside a test that the traced baseline is running in a
    worker under its own budget.
  - When the joiner is abandoned while parked in a join, the injection lands at the first bytecode AFTER the join,
    before the abandon. The runaway it was bounding is orphaned, and hogs the GIL for the rest of the process.
    Measured 2026-09-06: a 13-minute stall against a 300 s wall.
  - The `finally` in `bounded_join` is the whole fix; all four sites in the pair use it.

## Scope

Publishing, version bumps and publish order are the user's domain. Wesker and Detective release as a coordinated pair,
and never automatically.
