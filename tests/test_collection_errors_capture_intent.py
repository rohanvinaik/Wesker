"""The LIVE session must SURFACE collection errors, never swallow them — the "degrade loudly" floor.

Defect (found in Detective's dogfood on a torch-heavy repo): a test file that fails to COLLECT (an
ImportError at collection — a torch dep, a broken conftest) is silently absent from the routed suite.
``run_in_session`` collected the good siblings, discarded pytest's collection-error exit code, and handed
back only a diagnostic string — so a mutant only that file's tests would kill read as candidate-equivalent
while the certificate stood COMPLETE. ``last_collection_errors()``, fed by the live session's own
``_CollectionErrorCapture``, is the signal Detective's ``normalize_validity`` stamps ``collection_incomplete``
on.

The LIVE session is the authority: the discovery helper (``collect_pytest_callables``) seeds ``sys.path``
differently (the #15/#58 import-identity issue), so its ``from <target> import ...`` failures are spurious;
only the session that actually measures reports trustworthy collection errors.
"""

from __future__ import annotations

from Wesker.pytest_discovery import last_collection_errors
from Wesker.pytest_runner import run_in_session


def _write(tmp_path, tag: str, *, broken: bool) -> None:
    # Files in the ROOT (no subdir), unique names per case: the live session seeds the root on sys.path,
    # so ``from m<tag> import f`` resolves; unique names avoid an in-process same-basename mismatch.
    (tmp_path / f"m{tag}.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / f"test_good_{tag}.py").write_text(
        f"from m{tag} import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )
    if broken:
        (tmp_path / f"test_broken_{tag}.py").write_text(
            "import totally_missing_dep_xyz  # ImportError at COLLECTION\n\n\ndef test_e():\n    assert True\n"
        )


def _errors_during(tmp_path) -> tuple[str, ...]:
    seen: dict = {}

    def body(callables, _session):
        # Read INSIDE the session, where the ContextVar the collection hook set is live.
        seen["errors"] = last_collection_errors()
        return callables

    run_in_session(str(tmp_path), body)
    return seen.get("errors", ("<body-never-ran>",))


def test_the_live_session_surfaces_a_collection_error(tmp_path):
    _write(tmp_path, "brk", broken=True)
    errors = _errors_during(tmp_path)
    assert errors and errors != ("<body-never-ran>",), (
        "the live session must surface a collection error"
    )
    assert any("test_broken_brk" in e for e in errors), errors


def test_a_clean_live_session_reports_no_errors(tmp_path):
    _write(tmp_path, "cln", broken=False)
    assert _errors_during(tmp_path) == (), (
        "a clean collection must report zero errors, fresh per session"
    )
