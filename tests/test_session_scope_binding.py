"""#26 — a proof-facing manifest must belong to the exact live session that measured it.

Wesker captured a PytestSessionManifest during collect-only discovery and stored it as a
process-global "last manifest". The live measurement path never bound a manifest to its own
session, so a prior project's collection was consumed as if it described the current live run —
``collection_identity_standing`` would authorize project B's certificate using project A's
collection. Reproduced before the fix (A's rootpath, standing ``confirmed``, read inside B's
session).

These are INTENT tests: the defect is a wrong authorization, so a characterization of current
output cannot catch it. Each asserts a property from the issue's own regression list, driven
through the real entries (``collect_pytest_callables`` for discovery, ``run_with_live_suite`` for
a live session), never by poking a global.
"""

from __future__ import annotations

import os

import pytest

from Wesker.ci import run_with_live_suite
from Wesker.pytest_discovery import (
    collect_pytest_callables,
    current_measurement_scope,
    last_session_manifest,
    live_measurement_scope,
)
from Wesker.session_manifest import manifest_admissibility


def _project(root: str, name: str) -> str:
    os.makedirs(root, exist_ok=True)
    (root_file := os.path.join(root, f"test_{name}.py"))
    with open(root_file, "w") as fh:
        fh.write(f"def test_{name}():\n    assert True\n")
    return os.path.realpath(root)


# ── the pure decision: admit only THIS session's manifest ────────────────────────


def test_admissible_only_when_both_scopes_present_and_equal():
    """The one admit case: a positive id shared by the manifest and the consuming session."""
    assert manifest_admissibility(7, 7) == "admit"


def test_a_manifest_from_another_session_is_refused():
    """The leak, in one line: two live sessions get distinct ids, so a manifest minted by one is
    never admissible to the other — even when every other field coincides."""
    assert manifest_admissibility(1, 2) == "refuse"


def test_an_unstamped_manifest_is_refused():
    """A collect-only or pre-#26 manifest carries scope 0 — it describes a collection, not this
    measurement, and must not be read as proof evidence for the run consuming it."""
    assert manifest_admissibility(0, 5) == "refuse"


def test_a_manifest_is_refused_when_no_session_is_measuring():
    """No live scope means nothing is measuring; a manifest lingering in the ContextVar is not
    evidence for a run that is not happening."""
    assert manifest_admissibility(5, 0) == "refuse"


# ── end-to-end: the A→B leak is closed ───────────────────────────────────────────


def test_a_live_session_for_B_does_not_consume_project_As_manifest(tmp_path):
    """Collect A, then run a LIVE session for B in the same process. B's session must carry B's
    own manifest and derive its identity standing from B — never from A's lingering collection."""
    a_root = _project(str(tmp_path / "A"), "alpha")
    b_root = _project(str(tmp_path / "B"), "beta")

    collect_pytest_callables(a_root)  # _LAST_MANIFEST now names A (scope 0)
    assert os.path.realpath(last_session_manifest().rootpath) == a_root

    seen = {}

    def _reader():
        from Wesker.engine import _live_collection_identity

        m = last_session_manifest()
        seen["rootpath"] = os.path.realpath(m.rootpath) if m and m.rootpath else None
        seen["scope"] = getattr(m, "scope", None)
        seen["standing"], _ = _live_collection_identity()

    run_with_live_suite(b_root, _reader)

    assert seen["rootpath"] == b_root, "B's live session leaked A's manifest"
    assert seen["scope"] and seen["scope"] > 0, (
        "the live manifest was not scope-stamped"
    )
    # A clean single-project session, observed by its own runner, confirms identity — and it does
    # so from B's collection, which is the whole point.
    assert seen["standing"] == "confirmed"


def test_scope_and_manifest_are_restored_after_a_live_session(tmp_path):
    """A sequential run cannot see the prior session's scope. After the live session exits, no
    scope is bound, so a later direct consumer cannot mistake a leftover manifest for its own."""
    b_root = _project(str(tmp_path / "B"), "beta")
    assert current_measurement_scope() is None
    run_with_live_suite(b_root, lambda: None)
    assert current_measurement_scope() is None


def test_an_exception_in_a_session_still_resets_the_scope():
    """The session owns the scope; an exception mid-session must not leave it bound for the next
    run (every session-owned ContextVar resets in ``finally``). Exercised on the scope primitive
    directly — a live ``run_with_live_suite`` verifies the SAME reset on its real path, but a
    ``pytest.main`` nested inside this test swallows the inner body's exception, so the primitive
    is the deterministic surface. (Standalone: the live path propagates the body error AND leaves
    the scope None, confirmed outside pytest.)"""
    assert current_measurement_scope() is None
    with pytest.raises(ValueError, match="boom"):
        with live_measurement_scope():
            assert current_measurement_scope() is not None
            raise ValueError("boom")
    assert current_measurement_scope() is None


def test_nested_scopes_are_distinct_and_restore_the_outer(tmp_path):
    """Nested live sessions get distinct ids and the inner restores the outer on exit."""
    with live_measurement_scope() as outer:
        assert current_measurement_scope() == outer
        with live_measurement_scope() as inner:
            assert inner != outer
            assert current_measurement_scope() == inner
        assert current_measurement_scope() == outer
    assert current_measurement_scope() is None
