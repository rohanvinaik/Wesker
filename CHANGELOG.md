# Changelog

Notable changes, newest first. Dates are the commit dates.

## Unreleased

### The language-surface census — a denominator for the policy

`mutation_policy()` has always declared its categories and, in `exclusions`, several deliberate
withholdings. That list is honest, and it is a **numerator**: what somebody thought to name. It
had no denominator, so the question *"is there a semantic surface of Python that no category
mentions at all?"* had no mechanical answer.

`Wesker/language_surface.py` is the denominator. It is derived from CPython's own `ast` grammar
(generated from ASDL), and every concrete node and semantic field carries exactly one disposition
— `covered`, `withheld`, `not_applicable`, `unsupported`. A test checks the census against the
**running interpreter**, so Wesker's CI matrix makes the claim hold across 3.11–3.13 rather than
on one machine. Python adds a node, the build fails until somebody decides what that node means.

It found two things immediately, which is the argument for having built it:

- **Six binary operators (`@`, `<<`, `>>`, `&`, `|`, `^`) and two unary ones (`~`, unary `+`) have
  no operator family at all** — `_BIN_SWAP` covers seven of thirteen. They were never withheld and
  never declared; they were simply absent, and absence is indistinguishable from coverage until
  something counts. Comparisons, by contrast, turn out to be **complete**: all ten `cmpop` nodes
  are in BOUNDARY's table.
- **The first census was exhaustive on 3.14 and failed on 3.11–3.13**, which still carry the
  pre-3.8 constant aliases. A census that holds only on the author's interpreter is exactly the
  subspace error the project exists to refuse, caught before it could be believed.

`policy_id` is **unchanged** (`5.751e8e9f4f11`), and that is deliberate: the fault model did not
move, only the bookkeeping about it, so folding the census into the manifest digest would
invalidate every receipt and cached verdict in existence to record a documentation change. The
census carries its own `surface_id` instead, exposed as `MutationPolicy.language_surface` with the
`unsupported` slots listed **by name** — a consumer deciding whether a target sits inside the
model needs the slots, not a count.

Includes a test that checks the census against the engine's real operator tables, so `covered`
cannot drift into a comfortable lie.

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
