# Changelog

Notable changes, newest first. Dates are the commit dates.

## 1.1.1 — 2026-09-14

A patch release. The mutation policy is unchanged (`7.a73c76cd1d65`): no question in the universe moved.
What changed is how a survivor is judged equivalent, a descriptor leak and a warning in the pytest
plumbing, what a truncated Action run tells you, how the published Action handles its inputs, and the
documentation.

### Correctness

- **The equivalence probe can tell argument positions apart.** For a function of three or more
  parameters every probe row repeated one value in every position — `(0, 0, 0)`, `(1, 1, 1)`, … — so a
  SWAP mutant such as `pow(a, c)` → `pow(c, a)` agreed with the original on every row, was reported
  "likely equivalent", and left the effective kill rate. The rows now add rotations in which every
  position holds a different value, and keep the uniform rows a BOUNDARY mutant needs. This is the
  equivalence-side twin of 1.1.0's generation-side fix. **Effective kill rates on functions of three or
  more parameters can go down**: survivors that were being discounted as equivalent are now counted.
- **Isolated workers close their pipes on every path.** Only `communicate()` closed a worker's pipes; a
  worker reaped or closed any other way leaked the descriptor, which surfaced as
  `ResourceWarning: unclosed file` in a consumer that treats warnings as errors.
- **The makereport hook no longer re-raises in its own teardown.** When the wrapped call raised — an
  abandoned test unwinding — the old-style wrapper re-raised it inside its teardown and pluggy emitted
  `PluggyTeardownRaisedWarning`. It now lets pluggy propagate the original exception. No verdict moves.
- **A truncated run names what was cut, and why.** The Action refused a cut run with a count and two
  possible remedies, and named no function; the report that could have been read is written only after
  the gates pass. The refusal, and the CLI's warning, now list each cut function with its elapsed time
  and the mutants it evaluated, under the remedy its cause needs. A worker that could not be stopped
  comes first, because no budget fixes it. The report carries the list as `truncated_functions`.
- **A worker that could not be stopped is named.** Under an uncontained cut the refusal now names the
  test to bound or isolate, and the mutant it was running (or, for the baseline pass, the test whose
  trace could not be stopped). `evaluate_mutant` and the isolated path record that test on the mutant's
  result; both profiling paths carry the list as `containment_lost`, which `to_dict` emits and each
  `truncated_functions` entry repeats. Before this, the engine knew both and reduced them to one bool.
- `budget` is described as what it is, a budget per function. The action input, both `--budget` help
  texts and `docs/usage.md` called it per-file.

### Security

- **`action.yml` passes its inputs through the environment** rather than expanding them into the step's
  shell script, which was a template-injection path. Workflows pin actions to commit SHAs, run with
  read-only tokens, do not persist checkout credentials, and are audited by zizmor in CI; CodeQL runs on
  push, pull request and weekly.
- CI installs with `uv sync --locked` and runs every later step with `--no-sync --no-build`, so nothing
  is resolved, installed or built after the install step; zizmor installs with `--no-build`.
  `spec-pr.yml` grants its write permissions to its one job rather than to the whole workflow.
- **Releases are signed.** Publishing a GitHub release builds the tagged commit, refuses when the tag is
  not `__version__`, checks the sdist against git, and attaches the files with their Sigstore bundles.
- `SECURITY.md` (private reporting through the Security tab) and `CONTRIBUTING.md`.

### Documentation

- README: `--complete` never existed (exhaustive mode is `--max-per-category 0`); the default category
  count; the equivalence inputs as they are; "fails loudly" describes the Action, not the CLI.
  `docs/usage.md`: the real defaults of `--max-per-category` and `--passes`, the missing `--version` and
  `--purge`, and a dead link. The Action examples pin `@v1.1.1`.

### Housekeeping

- ruff 0.16.7 with its default rules worked through rather than pinned away; ty clean; pytest runs with
  warnings as errors.
- The float-perturbation NaN guard reads `math.isnan(v)` rather than `v == v`. Same behaviour; the old
  spelling reads as a typo to a reviewer and to SonarCloud alike.
- The real-runaway `bounded_join` test now uses the engine's thread shape: the loop in a called function,
  the handler in the thread target. On Python 3.13 an injected stop can skip a handler written in the
  same frame as the loop (python/cpython#139622; 25 of 30 locally), which failed the test under
  warnings-as-errors. The engine's own paths already had the safe shape and leaked nothing on 3.11–3.13.

## 1.1.0 — 2026-09-11

**Mutation policy 6 → 7.** SWAP asks about every PAIR of positional arguments, not only adjacent
ones. Every verdict under policy 6 is a claim about a universe that never asked whether
non-adjacent argument positions are distinguished.

### The defect

Measured through the real CLI. For `f(a, b, c) = g(a, b, c)` with the hand-written test
`f(1, 2, 1) == 8`, the suite read COMPLETE and `verify-rewrite` returned PRESERVED for the rewrite
`g(c, b, a)` — which returns 10 where the original returns 14 at `(1, 2, 3)`. `_alternatives` looped
`for i in range(len(args) - 1)`, so the first-and-third transposition was never a question, and a
suite whose inputs happened to repeat a value across positions 0 and 2 read complete while that
rewrite passed it. An unasked question is not a passed one.

### What changed

- **Every pair is a question**, selected greedily (each pair is its own behavioural dimension, so
  marginal coverage ties and nearest-first is the deterministic tie-break) under a **hard
  per-call-site budget of 10** — every pair of a call with up to five positional arguments.
- **What the budget declines is WITHHELD, counted, and reported** in the operator census, from the
  same `swap_plan` call that drives generation, so the counter and the generator cannot disagree.
  "Never asked" and "asked, and no distinguishing input was found" stay different states.
- **Neighbour labels are byte-identical to policy 6** and keep their emission position, so no
  per-site prefix moves; farther pairs take a new `~p<i>,<j>` spelling.
- The declared surface carries the budget, the ordering rule and the withholding, so changing the
  budget moves the policy id by construction. The fingerprint corpus gained `fp_swap_wide` —
  every other corpus call has at most two positional arguments, where "every pair" and "every
  adjacent pair" are the same set.

Policy id: `6.13a1fd436d29` → `7.a73c76cd1d65`. Every existing fingerprint row is unmoved.

### Not changed

No rotations or multi-position reorderings, and no blanket skip for "symmetric" builtins —
`max(1, 1.0)` is `1` and `max(1.0, 1)` is `1.0`, equal under `==` and different in type, so any
regression test for that case must check TYPE or IDENTITY. Argument order *within* a starred
expansion remains an unresolved surface: pairs are over the AST's positional entries.

### Quality

Cleared the local SonarQube new-code gate to OK / 0 bugs / 0 vulnerabilities / 0 hotspots. Three
`except BaseException` handlers narrowed to `(Exception, Abandoned)` — `Abandoned` derives from
BaseException by design so a test's own handler cannot swallow a stop, which is why those catches
reached that low; naming the pair keeps abandonment and lets a real KeyboardInterrupt end the run.

## 1.0.0 — 2026-09-09

First stable release. The API and the verdict vocabulary are what Detective 1.0.0 is built against,
and are now under a compatibility commitment rather than a version number that happened to be
recent.

Nothing in the release commit changes behaviour: it is the version, and the maturity classifier
moving from Beta to Production/Stable.

### Correctness

- **Kill attribution follows the asserting test**, and isolation preserves collected package
  identity. A value kill was being credited to whichever test ran first past the mutant rather than
  to the one whose assertion distinguished it, which made a suite's own coverage read as a gap. This
  is the fix behind Detective's Finding B: `audit` reported "3 killable mutants no test kills" over a
  suite that demonstrably killed two of them.
- **A foreign module's hostile `__getattr__` no longer blinds kill detection.** A module that
  answers every attribute lookup can make a mutant look alive.
- **A bounded join leaves no runaway behind on any exit path** (`bounded_join`, at both timed
  paths), and a broken pipe raises rather than asserting. The claim is pinned on a fake thread,
  deliberately: a real runaway is not a deterministic oracle, so a test that waits for one is a test
  that sometimes passes for the wrong reason.

### Performance

- **The widen persists its trace cache once per widen**, not once per single-test step. The gap it
  reports stays sound over the driver's applicable set.

### Housekeeping

- CI matrix across Python versions and operating systems; agent files excluded from the sdist.
- **The sdist is checked against git rather than trusted to an exclude list**, via
  `scripts/check_sdist.py`, in CI and before every publish. Wesker's sdist was already clean, which
  is a fact about what happens to be in the tree rather than a property of the build: the note in
  `pyproject.toml` claiming hatchling excludes VCS-ignored files by default has been corrected,
  because that holds for the **root** `.gitignore` only and a nested one is not consulted. Detective
  hit the real version of this and nearly published 110 MB of gitignored Lean build output.
- `pylint`-as-Sonar configuration baked into `pyproject.toml` so the pre-push gate is reproducible
  rather than reconstructed per session; the local SonarQube recipe recorded.
- `session_manifest.conflicting_module_names` gained the canonicalisation caveat it needed:
  `/var/x.py` and `/private/var/x.py` are one file behind a symlink, and comparing spellings without
  canonicalising them reports a conflict where there is none — a **false refusal**, which is the
  worse direction. Carried over from a duplicate of this decision that lived in Detective and has
  been deleted; Wesker owns the manifest and consumes it.

---

Earlier releases predate this file. `git log` is the record.
