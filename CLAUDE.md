# Working on Wesker (and Detective)

Extends the global CLAUDE.md — Serena-first/grep-last and Detective-pinning-first are
assumed there and not repeated.

Wesker and Detective are a **published-dependency pair** developed together. The full
working method — the per-issue loop, pure-decision extraction, converge usage, commit
format, Serena cautions — is documented once, in
`/Users/rohanvinaik/tools/Detective/CLAUDE.md`. **Read that file; it governs work here too.**
This file carries only what differs on the Wesker side.

## What Wesker is, in the sandwich

Detective is the squeeze between a floor and a ceiling. **Wesker is the ceiling** — the
mutation engine that produces the mutant profile. Detective consumes what Wesker measures.

The consequence that keeps biting: a signal Wesker computes honestly is worthless if a layer
above re-derives a narrower proxy from it. That has now happened twice (W#14's containment
cut dropped at the aggregation layer; W#17's reach evidence). **The measurement/decision gap
hides at EVERY layer — engine → loop → aggregation → gate/badge.** Each layer must *consume*
the computed signal. Verify end-to-end, never per-layer.

## Standing invocation

```bash
export PP=/Users/rohanvinaik/tools/Detective:/Users/rohanvinaik/tools/Wesker
```

Pins into Wesker still run through Detective's CLI, with both repos on the path:

```bash
cd /Users/rohanvinaik/tools/Wesker
PYTHONPATH=$PP detective converge 'Wesker/ci.py::is_truncated_measurement' \
  --input "('cut', False)" --input "('profiled', True)" 2>&1 | tail -8
```

## The gate that matters here

Wesker's own suite is **not sufficient**. It has been fully green through a regression that
only the cross-repo run caught. After any Wesker change, run both:

```bash
cd /Users/rohanvinaik/tools/Wesker && python3 -m pytest -q 2>&1 | tail -3
PYTHONPATH=$PP python3 -m pytest 2>&1 | tail -3   # Detective's suite vs LOCAL Wesker
```

Plus pinned ruff, never bare:

```bash
uvx ruff@0.14.10 format . 2>&1 | tail -1 \
  && uvx ruff@0.14.10 check . 2>&1 | tail -2 \
  && uvx ruff@0.14.10 format --check . 2>&1 | tail -1
```

## Wesker-specific hazards

- **Mutant binding is by `__code__.co_filename`.** Wesker monkeypatches mutants onto live
  modules in `sys.modules`. This is why Detective can only self-analyze via a renamed package
  copy at a different path, and why anything reasoning about module identity must go through
  `session_manifest`, not `__name__`.
- **A classmethod mutant can arrive already-bound.** Wrapping it in `classmethod(...)`
  double-binds `cls` → `TypeError` read as a spurious *crash* rather than an assertion kill.
  Peel to `__func__` first (`_preserve_descriptor_shape`).
- **The static impact map keys on bare identifiers.** A method target is the dotted
  `Basket.tier`, so a bare-ident lookup misses and the method's own test is never associated
  — 0/N killed even with a killing test. Look up the trailing attribute too.
- **Validate through the real command end-to-end**, not by poking `profile()` on a
  half-built target. That trap once cost a deep and entirely wrong spelunk.

## Scope

Publishing, version bumps, and publish order are the user's domain — Wesker and Detective
release as a coordinated pair, and never automatically.
