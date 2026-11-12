# Changelog

Notable changes, newest first. Dates are the commit dates.

## Unreleased

Development source after 1.2.2; no version or release is implied. The operator universe is unchanged —
policy 8, the same categories and the same fingerprint counts — but the observation rule it declares
changed twice, so the policy id moved twice.

### Correctness

- **The wall that bounds a mutant's liveness follows the machine, and a fired wall is retried once.**
  The isolated per-mutant wall was a fixed `max(per_mutant_timeout_ms, 10 s)`, so work no loop budget
  counts (a loop in a helper, a C builtin, another thread) read COMPLETE on a fast machine and
  `undetermined` on a slow one. The bound is now a compute budget (10 reference seconds;
  `WESKER_COMPUTE_BUDGET_S` sets it, and an invalid value is refused, not ignored) times this machine's
  measured slowdown on a fixed reference workload, taken once per process (`monitoring.harness_slowdown`,
  `liveness_bound_ms`, whose code names the deciding term). A bound that fires is retried once on a fresh
  worker, with the machine measured again, and a second fire is `undetermined`. The unarmed in-process
  allowance's floor and cap and the original's counted-run wall follow the same slowdown. Policy id moves
  (the observation clause names the bound; the universe and `POLICY_VERSION` do not): one-sign
  `9.008828e5f6a9` -> `9.10f8e602bf48`, two-sign `9+neg.1a13fddd1d5f` -> `9+neg.de8c78c57407`. (W#27)
- **A trace row says when its reach is unknown, and what kind of non-pass a route cannot type.**
  A test that reaches the target through a thread alive before it started (a pool warmed at import), or only
  in a child process, used to read `reach=complete lines=()`: an admissible "reached nothing". Its
  `TraceEvidence.reach` is now `incomplete_preexisting_thread` or `incomplete_subprocess` (causes join with
  `+`, with `incomplete_thread`); the unobservable lines are never attributed and such a trace is never
  cached. On the legacy direct-call route (no pytest phase) a non-pass that cannot be told apart (skip,
  `xfail()`, setup error) or a callable carrying skip/xfail marks is the named outcome `untyped`, refused as
  `refuse_untyped`, where it was typed `error` or, for an xfail-marked test, named a wrong expectation.
  The `trace_evidence` docstring no longer claims same-line short-circuit operands stay apart. A failed
  subscription setup whose rollback also fails no longer leaves its registration in the broker. Adds
  fields and codes; no mutant's disposition changes and the policy id does not move. (W#17, W#32)
- **One profile run can carry the per-test kill relation.** `run_function_profiling(full_matrix=True)`
  runs every test that reaches a mutant and returns, per killed mutant, every test that kills it with that
  test's own kill reason (`ProfilingResult.kill_relation`; `kill_relation_standing` says whether it is
  whole, and is `not_requested` when it was not asked for), in both execution modes: in-process through
  `evaluate_mutant`'s scan, isolated through a worker that no longer stops at the first value kill. The
  #55 original control still voids a node the original also fails, and a loop budget, a wall bound or a
  refused measurement is an unresolved cell, never a kill. It adds a field and changes no verdict: every
  mutant's killed / survived / unscored disposition and `killed_by` are what they are with it off, so
  the policy id does not move. Default off, because it runs tests a first-killer run skips. Detective
  paid for this relation by profiling once per test (audit 0.9 s to 3.6 s). (Detective D#145)
- **A test exception never selects a second invocation.** The unpatched fallback called
  `test_fn(mutated_func)` and, on any `TypeError`, called `test_fn()` again, so a body that changed
  state and raised could run twice and read as a survivor. The calling convention is now chosen by
  signature binding before the one bounded invocation. An unavailable signature keeps one injected
  attempt, and a `TypeError` raised during execution is that test's crash, never a retry. (`ca215e4`,
  W#47)
- **A forwarding wrapper keeps its advertised call.** A `functools.wraps` wrapper whose own signature is
  `(*args, **kwargs)` but which advertises zero arguments received the injected mutant and failed
  before its test ran. Binding now honours the advertised signature where the actual one also binds,
  and on 3.14 reads parameter shape without evaluating deferred annotations. (`3e98ad8`, W#47)

- **A kill is a difference from the original on the same, authored input.** One isolated worker runs
  every mutant in one process, and a module-level `parametrize` row is the same object for all of
  them, so a mutant that wrote into its argument altered the row for every later mutant. Those later
  failures counted as kills (a mutant identical to the original read "killed by assertion" and
  Detective printed `✓ COMPLETE` over it), and in-process the altered row lost real crash kills.
  Now, in both modes: (1) every evaluation, the original's and each mutant's, gets a fresh deep copy
  of the test's parametrize values (direct, indirect and fixture params) from a snapshot taken before
  any test body ran; an argument that cannot be copied stays shared and is named in the profile
  (`shared_arguments`), and so is an item whose test passes on the authored objects but fails on a copy
  (it depends on object identity: `identity_sensitive`), measured on the original's first failure;
  (2) a test that fails under a mutant is re-run against the original in the
  same process state, and a failure the original reproduces is the unscored outcome
  `original_also_fails`, never a kill and never a survivor. Both modes read one decision,
  `engine.original_control_standing`. The in-process control used to cover only the tests the mutant
  could not be patched into; it now covers every failure. (W#55; Detective #127)
- **A stale kill needs two calls in one test.** The stale wrapper (Form B `return_stale`) held the
  previous call's return for the whole evaluation, across tests. Two tests that each call once, and
  each pass alone, killed it together, the kill was credited to whichever ran second, and the verdict
  depended on test order and on which subset ran. Def. 11.8 (i) says stale stands for an output that
  is not a pure function of this call; a test detects it by calling twice and seeing two results. The
  wrapper's memory now ends at every test boundary, in both modes: two calls with different results
  in one test kill it by assertion and are credited to that test; two tests with one call each
  survive, in either order. Detective's pins that write one test per `--input` therefore no longer
  kill it; a pin must call the function twice in one test. (W#49)

Policy id: `8.dcc355186875` → `8.68a525594c46` → `8.51a0c2a545b5`; two-sign
`8+neg.671d835828bf` → `8+neg.40ae45e28e39` → `8+neg.8d34cddbefca`. Detective's verdict cache keys
on the id, so rows recorded under an older observation rule are not reused.

W#55 moves it again, on the policy-9 universe (categories and counts unchanged): `9.261d14e34119` →
`9.415d3755221e` → `9.008828e5f6a9` (copies are a measured precondition, see above); two-sign
`9+neg.f5a15364c11d` → `9+neg.001902a9950c` → `9+neg.89326e7f549c` (W#49) → `9+neg.1a13fddd1d5f`. Receipts and cached verdicts
recorded under the old id are not reused. W#49 alone moves only the two-sign id: its rule is a two-sign Form B operator's, so the one-sign id and its receipts stand.

### The two-sign policy (μ⁻)

- **The rest of Def. 11.8's output operators are generated.** A two-sign run generated nine of the 38
  operators in Detective's negative-specification paper. It now generates 23 more Form A operators
  under Fork 2, each where every observed return type admits it:
  - universal: zero-of, type;
  - boolean: not, true, false;
  - numeric: ±1, ±ε, scale, round, inf;
  - string: trunc, case, ws, rev;
  - container: drop, dup, singleton, ctype, dict's drop-field and default-field;
  - ordered: adj, sortbreak.

  Two more come from Form B: `return_stale` on every non-generator, and `raise_swallow` where the
  function's own body raises. Each rewrite evaluates the return expression exactly once, never
  raises on an applicable type, and chooses deterministically across processes. The nine operators
  already built keep their ids. (W#44; Detective #116–#123)
- **Yield mutants reach the isolated worker.** That worker compiles source alone. A yield mutant's
  node was a marker that defines nothing, so the worker installed the original function, and all four
  yield mutants of a generator read as survivors under `--isolated`. Every Form B mutant is now
  source: the definition with `Wesker.mu_minus.form_b(<mode>)` applied innermost. The in-process
  yield factory binds a classmethod's `cls` once and keeps an async generator async. (W#44)
- **The catalog.** `docs/MU_MINUS_CATALOG.md` lists every Def. 11.8 operator: 30 built, 2 partial, 2
  realized by another operator, 4 unsupported, each with its reason and its evidence. A test checks
  it against `Wesker.mu_minus.MU_MINUS_CATALOG` and the engine tables. The two-sign manifest declares
  the Fork 2 table, the Form B modes and the unbuilt operators. (W#44)

The one-sign universe and its id are unchanged by this. Two-sign id `9+neg.aaff90df3dbf` →
`9+neg.8144c37b1638`, and the two-sign fingerprint's OUTPUT column rises on every corpus function.
Still open: stale needs two calls with different results, and Detective's witness search reaches some
functions with a single input, where stale reads as candidate-equivalent.

### The Action

- **A stalled badge run leaves evidence of where it stalled.** `WESKER_STACK_DUMP_S`, a positive whole
  number of seconds, arms a periodic all-thread stack dump on a copy of stderr duplicated before
  profiling starts. Unset arms nothing; a malformed value is named rather than read as unset.
  `spec-full.yml` sets it to 900, bounds the job at 180 minutes (liveness, never the measurement's
  budget) and lets a newer push supersede a running measurement. Every push of 2026-10-24 had run
  that job for six hours with no output after the file count. (`9362bc5`, W#29)

## 1.2.2 — 2026-10-24

Paired with Detective 1.2.2. The engine is the one released as 1.2.1, and the mutation policy is
unchanged (`8.dcc355186875`).

- **Release artifacts are built and signed only after exact-commit CI.** The release workflow refuses
  unless push CI for the exact checked-out release commit completed successfully. A pending, missing,
  failed or wrong-commit run does not satisfy it, and an older green run cannot mask a newer failed
  one. (`60df89b`)
- The README shows live release and quality signals; the Action example documentation is updated.
  (`6de166d`)

## 1.2.1 — 2026-10-24

The mutation policy is unchanged (`8.dcc355186875`).

- **Cold opcode observations survive a co-resident coverage tool.** Global discovery is installed
  before local event masks, so a tool that disables PY_START — coverage.py's `SysMonitor` core on
  3.14 — no longer costs instruction events on a code object's first execution. (`8aa2b4e`, W#32)
- The stale-source tests state the fingerprint contract across interpreter versions and exercise
  tokenizer failures explicitly. (`8aa2b4e`)

## 1.2.0 — 2026-10-24

**Mutation policy 7 → 8.** Annotation subtrees — parameter and return annotations, type parameters
and type aliases — are outside the runtime mutation universe (option A of W#30). Definition defaults,
decorators and annotated-assignment values remain behavioural targets. Under PEP 563, and under PEP 649
absent introspection, an annotation mutant is inert to every test that does not evaluate annotations,
so it could only land in the candidate-equivalent residual. The fingerprint corpus gained
`fp_annotations`. Policy 8 also declares the execution-outcome rules below.

Policy id: `7.a73c76cd1d65` → `8.dcc355186875`.

Requires Python 3.12 or newer. 3.14 is the standard interpreter, and the Action defaults to it.
(`1561067`)

### Execution outcomes

- **A timeout is not a kill.** A wall-clock bound that fires before any kill is established records
  `undetermined`; an explicit step budget that is exceeded records `diverged`. Both are named
  dispositions outside the scored denominator — never a kill, never survival evidence.
  `Wesker.monitoring.step_budget` is the deterministic counter; the engine does not yet arm it for
  every test. (`a24171e`, `daad6cf`, W#27)
- **A test that runs the engine inside a held evaluation is refused by name**, `nested_measurement`:
  unscored and non-gateable, instead of deadlocking on the execution lock and orphaning it.
  (`b2d790c`, W#28)
- **A mutant that was installed but never entered names why** — `not_reached`, `decorator_wrapper`,
  `captured_at_import`, `pre_bound_reference`, `holder_not_found` or `no_reach_data` — with the reach
  and the holders that keep the original. (`7176e9a`, `23ab488`, W#18)
- **Baselines are typed.** Every trace-evidence row carries `baseline_outcome` (passed, failed,
  skipped, xfailed, error), and a skipped, xfailed or errored test is never proof. A thread a test
  starts is traced while it runs; one that outlives its test marks the row `incomplete_thread` —
  unknown, never negative. (`d8c228e`, `e212f7e`, `8ee8beb`, W#17)
- **The whole-pool fallback runs only baseline-screened tests.** An unscreened remainder is recorded as
  typed unknown on the survivor record and is never credited as a kill. (`87ebdd3`, W#31)

### Interpreter instrumentation

- **One owner of `sys.monitoring`.** Subscriptions with thread attribution and run tokens replace
  per-site tool registration; INSTRUCTION events are dispatched with thread attribution and token
  retirement; unresolved execution facts are retained rather than dropped. (`a24171e`, `daad6cf`,
  `0884e46`, W#32)
- Shape-hazard records keep their provenance (detected, declared, override) and their test-level
  records. (`6b49414`, `34711f6`, W#19)

### The live session

- The session baseline names the files it traces (Detective #103). (`39be82d`)
- A test whose file was edited after import fingerprints by name and never crashes the run
  (Detective #104). (`9e5023c`)

### Performance

- **No failure report is formatted that nobody reads.** The in-process measurement session and the
  isolated worker give every collected item a cheap `repr_failure`; the verdict is pytest's exit code
  and the kill reason comes from the raw `excinfo`. Verdicts, generated suites and certificates are
  identical. (`49f7e91`, `3718b0a`, `3e2a88a`, W#34)

### The CLI

- `--isolated` runs mutant evaluation in isolated worker subprocesses and `--mem-limit-mb` sets their
  memory limit (default 512). The address-space cap is applied where the OS accepts it, which Linux
  does and macOS often does not. (`daad6cf`; acceptance tests `8b6f222`; W#45)

### Documentation

- `AGENTS.md` is Wesker's one instruction base and `CLAUDE.md` imports it; the instruction files read
  Detective's output whole. (`1ce3449`, `ed2256a`)

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
