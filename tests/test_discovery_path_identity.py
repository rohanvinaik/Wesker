"""Scoping must compare test origins CANONICALLY, or it silently returns nothing (issue #15).

`discover_test_callables` narrows the live suite by comparing each item's origin against the
scoped file list. The two sides come from different places: a live item's origin comes from
pytest, which canonicalises it, while `scoped` carries whatever spelling the caller typed. On
any symlinked root — `/var` -> `/private/var` on macOS, a symlinked checkout, a case-insensitive
rename — an `abspath` comparison matches NOTHING.

The failure is silent and points the wrong way. An empty discovery reads downstream as "no test
reaches this target", which is the synthesize path: the run stops measuring the suite it has and
starts inventing one, and every mutant the real suite kills is reported as an unpinned
behaviour. A filter that can return the empty set for a spelling difference is worse than no
filter at all.
"""

from __future__ import annotations

import os

import pytest

from Wesker import ci


def _make_test_callable(origin: str):
    """A stand-in for a live pytest item: the contract only requires the origin tag."""

    def case() -> None:
        pass

    case.__wesker_origin__ = origin  # type: ignore[attr-defined]
    return case


@pytest.fixture
def symlinked(tmp_path):
    """A project reachable by two spellings of the same directory."""
    real = tmp_path / "real_root"
    (real / "tests").mkdir(parents=True)
    (real / "shipping.py").write_text("def shipping_cost(w):\n    return w * 2\n")
    (real / "tests" / "test_shipping.py").write_text("def test_c():\n    assert True\n")
    link = tmp_path / "linked_root"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (
        OSError,
        NotImplementedError,
    ):  # pragma: no cover — platform without symlinks
        pytest.skip("symlinks unavailable")
    return real, link


def test_the_live_filter_survives_a_symlinked_root(symlinked):
    """The defect. The item's origin is the REAL path, the scoped list the LINKED one — the same
    file by two names. Compared literally they never match. Exercised in ``conservative`` mode,
    where a `candidate` verdict DEPENDS on the origin matching the scoped list — the default keeps
    everything as `unknown` (#15), which would make this pass even if canonicalisation were broken.
    So conservative mode is what keeps this test discriminating."""
    real, link = symlinked
    item = _make_test_callable(str(real / "tests" / "test_shipping.py"))

    token = ci._LIVE_SUITE.set([item])
    try:
        got = ci.discover_test_callables(
            str(link), "shipping.py", ["shipping_cost"], conservative=True
        )
    finally:
        ci._LIVE_SUITE.reset(token)

    assert got, (
        "the live suite emptied because two spellings of one path did not compare equal"
    )
    assert ci.callable_origin(got[0]) == str(real / "tests" / "test_shipping.py")


def test_an_unrelated_test_is_still_excluded(symlinked):
    """The control, and the guard against 'fixing' the symlink defect by widening. Canonicalising
    must make the same file compare equal — not make different files compare equal. In
    ``conservative`` mode the unrelated test (no name, no fixture edge → `unknown`) is dropped
    while the scoped test is kept, so this asserts narrowing still discriminates once the origin is
    canonicalised. (By default #15 KEEPS the unrelated test — that is the one-sided guarantee, and
    it has its own test in test_route_test_item.py.)"""
    real, link = symlinked
    (real / "tests" / "test_unrelated.py").write_text(
        "def test_arith():\n    assert 2 + 2 == 4\n"
    )
    scoped_item = _make_test_callable(str(real / "tests" / "test_shipping.py"))
    other_item = _make_test_callable(str(real / "tests" / "test_unrelated.py"))

    token = ci._LIVE_SUITE.set([scoped_item, other_item])
    try:
        got = ci.discover_test_callables(
            str(link), "shipping.py", ["shipping_cost"], conservative=True
        )
    finally:
        ci._LIVE_SUITE.reset(token)

    origins = {os.path.basename(ci.callable_origin(c) or "") for c in got}
    assert "test_unrelated.py" not in origins
    assert "test_shipping.py" in origins
