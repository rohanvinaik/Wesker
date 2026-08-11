"""Portable pin for ``resolve_test_id`` — the suite-wide identity contract, from intent.

This replaces the Detective-generated golden ``test_Wesker_ci_resolve_test_id_synth.py``, which
converge kept re-minting with MACHINE-COUPLED witnesses: its equivalence search captured real call
sites (`test_refreshing_...` calling ``resolve_test_id`` with an absolute pytest tmp origin and
``project_root="."``), so the pinned string held the author's ``/private/var/folders/.../pytest-1106/``
tmp path. That is green on the machine that generated it and RED on any CI runner with a different
directory depth (`os.path.relpath` against ``.`` is cwd-relative). A pure function that relativizes a
path cannot be pinned by capturing a real absolute path; it must be pinned on ABSTRACT paths whose
relationship is fixed. So this is hand-written from the contract, covering the same behaviour portably.
"""

from __future__ import annotations

import pytest

from Wesker.ci import resolve_test_id


@pytest.mark.parametrize(
    "node_id, display_name, origin, project_root, case, expected",
    [
        # A real pytest nodeid (`::`) is already unique + root-relative → returned untouched.
        (
            "tests/test_x.py::test_f[case0]",
            "test_f",
            "/abs/tests/test_x.py",
            "/abs",
            "case0",
            "tests/test_x.py::test_f[case0]",
        ),
        # Synthesized id: `legacy:` namespace, origin relativized against an ABSOLUTE parent root.
        (
            "test_shared",
            "test_shared",
            "/root/tests/test_other.py",
            "/root",
            "",
            "legacy:tests/test_other.py::test_shared",
        ),
        # origin one level of nesting deeper — the relative path keeps that structure, portably.
        (
            "t",
            "t",
            "/deep/a/b/test_other.py",
            "/deep",
            "",
            "legacy:a/b/test_other.py::t",
        ),
        # project_root is NOT a parent of origin → a fixed `..` relative path (portable across hosts,
        # because both operands are absolute and abstract — this is the case the tmp-path witness
        # tested with `project_root="."`, made deterministic).
        (
            "test_x",
            "test_x",
            "/other/tests/test_x.py",
            "/root",
            "",
            "legacy:../other/tests/test_x.py::test_x",
        ),
        # origin == project_root → relpath is ".".
        ("abc", "abc", "abc", "abc", "abc", "legacy:.::abc[abc]"),
        # The `#display_name` suffix (#16 correctness floor): kept when display_name is NOT the final
        # dotted segment of the qualname — two closures sharing a qualname must not collapse.
        (
            "_heavy.<locals>.t",
            "heavy_1",
            "/root/tests/test_b.py",
            "/root",
            "",
            "legacy:tests/test_b.py::_heavy.<locals>.t#heavy_1",
        ),
        ("t", "other", "/root/x.py", "/root", "", "legacy:x.py::t#other"),
        # ...and NOT added for an ordinary function where the two agree.
        ("t", "t", "/root/x.py", "/root", "", "legacy:x.py::t"),
        # The `[case]` suffix is appended when a parametrize case is present.
        ("t", "t", "/root/x.py", "/root", "c1", "legacy:x.py::t[c1]"),
        # A parametrize row already in the node_id is stripped from the base.
        ("t[row]", "t", "/root/x.py", "/root", "", "legacy:x.py::t"),
        # Empty origin → `?` (unique, visibly non-portable rather than a crash).
        ("t", "t", "", "/root", "", "legacy:?::t"),
        # No project_root → origin is NOT relativized; the absolute path stands.
        ("t", "t", "/root/x.py", None, "", "legacy:/root/x.py::t"),
    ],
)
def test_resolve_test_id_contract(
    node_id, display_name, origin, project_root, case, expected
):
    assert (
        resolve_test_id(node_id, display_name, origin, project_root, case) == expected
    )
