"""A certificate's proof basis is the RUNNER'S node IDs, captured under the session scope (issue #58).

Detective's receipt used to freeze a file set re-derived from the kill matrix — which omits a test
that owns only a line/arc obligation — by reading pytest config a second way, which can pass under a
regime the consumer's real pytest cannot even collect. #58 replaces that with the runner's own answer:
`PytestSessionManifest.items` = each collected test by `(node_id, content digest)`.

The subtle part is TIMING, and it is why this rides the session baseline rather than a result-assembly
read. The manifest is admissible only inside the live measurement scope; the per-mutant collect-only
discoveries later overwrite `_LAST_MANIFEST` with a scope-0 collection, so a read at result assembly
finds `unobserved`. `build_session_baseline` runs ONCE under the scope, so it captures the admissible
identity + basis there, and every function's profile reads that stored answer.
"""

from __future__ import annotations

import os
import sys

import pytest

# Both tests here drive Wesker through Detective's `profile` (cross-repo integration). Wesker's own CI
# is zero-dependency and does not install its consumer, so the whole module skips there and runs where
# both are present (local dev via PYTHONPATH, or Detective's CI which installs Wesker).
pytest.importorskip("Detective")


def test_proof_basis_is_the_runner_node_id_basis_through_the_session_baseline(tmp_path):
    """The payoff: a profile under a live session surfaces the runner's node-ID basis — each item as
    (node_id, non-empty content digest) — on `ProfilingResult.proof_basis`, with `collection_conflicts`
    empty (the session CONFIRMED its module identity). Not re-derived; read from the scoped manifest."""
    (tmp_path / "pbapp.py").write_text(
        "def grade(n):\n    if n >= 90:\n        return 'A'\n    return 'F'\n"
    )
    (tmp_path / "test_pbapp.py").write_text(
        "from pbapp import grade\n\n\n"
        "def test_a():\n    assert grade(95) == 'A'\n\n\n"
        "def test_f():\n    assert grade(10) == 'F'\n"
    )
    for name in ("pbapp", "test_pbapp"):
        sys.modules.pop(name, None)

    from Wesker.ci import run_with_live_suite

    seen: dict = {}

    def _body():
        from Detective.engine import profile

        r = profile("pbapp.py", "grade", str(tmp_path))
        seen["basis"] = r.proof_basis
        seen["conflicts"] = r.collection_conflicts

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        run_with_live_suite(str(tmp_path), _body, target_files=["pbapp.py"])
    finally:
        os.chdir(cwd)

    basis = seen["basis"]
    names = {nid.rsplit("::", 1)[-1] for nid, _ in basis}
    assert names == {"test_a", "test_f"}, (
        "the runner's node-ID basis is missing its items"
    )
    assert all(dig for _nid, dig in basis), (
        "each basis entry must carry a non-empty content digest"
    )
    # A CONFIRMED collection (no shadowed module name) surfaces the basis; conflicts empty.
    assert seen["conflicts"] == ()


def test_proof_basis_is_empty_for_a_profile_outside_a_live_session(tmp_path):
    """The guard: no live session → no admissible manifest → an EMPTY basis, never a false frozen one.
    A certificate must not rest on a basis reconstructed outside the scope that measured it."""
    (tmp_path / "solo.py").write_text("def f(n):\n    return n + 1\n")
    (tmp_path / "test_solo.py").write_text(
        "from solo import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )
    for name in ("solo", "test_solo"):
        sys.modules.pop(name, None)

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        from Detective.engine import profile

        r = profile("solo.py", "f", str(tmp_path))
    finally:
        os.chdir(cwd)

    assert r.proof_basis == (), (
        "a profile outside a live session must not surface a frozen basis"
    )
