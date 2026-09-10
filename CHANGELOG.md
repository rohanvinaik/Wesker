# Changelog

Notable changes, newest first. Dates are the commit dates.

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
- `pylint`-as-Sonar configuration baked into `pyproject.toml` so the pre-push gate is reproducible
  rather than reconstructed per session; the local SonarQube recipe recorded.
- `session_manifest.conflicting_module_names` gained the canonicalisation caveat it needed:
  `/var/x.py` and `/private/var/x.py` are one file behind a symlink, and comparing spellings without
  canonicalising them reports a conflict where there is none — a **false refusal**, which is the
  worse direction. Carried over from a duplicate of this decision that lived in Detective and has
  been deleted; Wesker owns the manifest and consumes it.

---

Earlier releases predate this file. `git log` is the record.
