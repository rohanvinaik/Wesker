"""Tests for Wesker's CI orchestration layer (ci.py).

Covers the deterministic, high-mutant units: the terminal color threshold,
the convention-matching predicate LintGate extracted, the three test-discovery
layers, AST function walking, source hashing, and the JSON caches. The heavy
end-to-end profiling entry points run real mutation testing and are exercised
by an integration smoke test rather than pinned mutant-by-mutant.
"""

from __future__ import annotations

import ast
import os
import textwrap

import Wesker.ci as ci
from Wesker.ci import (
    _build_static_impact_map,
    callable_origin,
    _discover_all_test_files,
    _discover_by_convention,
    _load_cached_state,
    _name_matches_convention,
    _pct_color,
    discover_tests,
    walk_functions,
)


# ── _pct_color: threshold boundaries ─────────────────────────────


def test_pct_color_thresholds(monkeypatch):
    # Distinct sentinels so the >=/== boundary mutants change the output.
    monkeypatch.setattr(ci, "_GREEN", "G")
    monkeypatch.setattr(ci, "_YELLOW", "Y")
    monkeypatch.setattr(ci, "_RED", "R")
    assert _pct_color(100) == "G"  # exact 100 (kills == → !=)
    assert _pct_color(99) == "Y"  # just below 100
    assert _pct_color(80) == "Y"  # boundary >= 80 (kills >= → >)
    assert _pct_color(79) == "R"  # below 80
    assert _pct_color(0) == "R"


# ── _name_matches_convention: the extracted predicate guard ──────


def _match(
    name,
    base="query",
    base_stripped=None,
    generated="test_query.py",
    parent_dir="src",
    parent_qualified=None,
    partial_stems=frozenset(),
):
    return _name_matches_convention(
        base,
        base_stripped or base,
        generated,
        name,
        parent_dir,
        parent_qualified,
        set(partial_stems),
    )


def test_match_exact_generated_name():
    assert _match("test_query.py", generated="test_query.py") is True


def test_match_exact_stem():
    assert _match("test_query.py", generated="test_other.py") is True


def test_match_prefix():
    assert _match("test_query_helpers.py", generated="test_other.py") is True


def test_match_partial_stem():
    # compound source query_navigate → partial stems match test_navigate.py
    assert (
        _match(
            "test_navigate.py",
            base="query_navigate",
            generated="test_x.py",
            partial_stems={"query", "navigate"},
        )
        is True
    )


def test_match_parent_qualified():
    assert (
        _match(
            "test_wiki_config.py",
            base="config",
            generated="test_x.py",
            parent_dir="wiki",
            parent_qualified="wiki_config",
        )
        is True
    )


def test_no_match():
    assert (
        _match(
            "test_unrelated.py",
            base="query",
            generated="test_query.py",
            partial_stems={"query"},
        )
        is False
    )


# ── Convention discovery on a real tree ──────────────────────────


def _make_project(tmp_path, source_rel, test_names):
    src = tmp_path / source_rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f():\n    return 1\n")
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    for tn in test_names:
        (tests / tn).write_text("def test_x():\n    assert True\n")
    return src


def test_discover_by_convention_finds_matching(tmp_path):
    src = _make_project(tmp_path, "query.py", ["test_query.py", "test_unrelated.py"])
    found = _discover_by_convention(str(tmp_path), str(src))
    names = {p.rsplit("/", 1)[-1] for p in found}
    assert "test_query.py" in names
    assert "test_unrelated.py" not in names


def test_discover_by_convention_none_when_absent(tmp_path):
    src = _make_project(tmp_path, "query.py", ["test_other.py"])
    assert _discover_by_convention(str(tmp_path), str(src)) == []


def test_discover_all_test_files_only_test_prefixed(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text("")
    (tests / "helper.py").write_text("")  # not test_-prefixed
    (tests / "test_b.py").write_text("")
    found = {p.rsplit("/", 1)[-1] for p in _discover_all_test_files(str(tmp_path))}
    assert found == {"test_a.py", "test_b.py"}


def test_discover_all_test_files_missing_dir(tmp_path):
    assert _discover_all_test_files(str(tmp_path)) == []


# ── Static impact map ────────────────────────────────────────────


def test_build_static_impact_map(tmp_path):
    tf = tmp_path / "test_x.py"
    tf.write_text("from mod import foo\ndef test_a():\n    foo()\n    obj.bar()\n")
    impact = _build_static_impact_map([str(tf)])
    assert str(tf) in impact.get("foo", [])  # Name reference
    assert str(tf) in impact.get("bar", [])  # Attribute reference


def test_build_static_impact_map_skips_unparseable(tmp_path):
    bad = tmp_path / "test_bad.py"
    bad.write_text("def (( this is not python\n")
    assert _build_static_impact_map([str(bad)]) == {}


# ── 3-layer discovery: convention hit is ordered first ───────────


def test_discover_tests_layers(tmp_path):
    src = _make_project(tmp_path, "query.py", ["test_query.py", "test_misc.py"])
    # test_misc references function 'f' via impact, test_query via convention.
    (tmp_path / "tests" / "test_misc.py").write_text("def test_m():\n    f()\n")
    found = [
        p.rsplit("/", 1)[-1] for p in discover_tests(str(tmp_path), str(src), ["f"])
    ]
    assert found[0] == "test_query.py"  # convention first
    assert set(found) == {
        "test_query.py",
        "test_misc.py",
    }  # impact/fallback add the rest


# ── walk_functions: qualnames incl. nested + methods ─────────────


def test_walk_functions_qualnames():
    tree = ast.parse(
        textwrap.dedent("""
        def top():
            def inner():
                pass
        class C:
            def method(self):
                pass
    """)
    )
    names = {q for q, _ in walk_functions(tree)}
    assert "top" in names
    assert "C.method" in names


# ── Caches: roundtrip, missing, corrupt ──────────────────────────


def test_load_cached_state_missing_is_none(tmp_path):
    assert _load_cached_state(str(tmp_path)) is None


def test_load_cached_state_reads_report(tmp_path):
    wdir = tmp_path / ".wesker"
    wdir.mkdir()
    (wdir / "mutation_report.json").write_text(
        '{"per_category": [{"category": "VALUE"}]}'
    )
    state = _load_cached_state(str(tmp_path))
    assert state is not None
    assert state["per_category"][0]["category"] == "VALUE"


def test_load_cached_state_corrupt_is_none(tmp_path):
    wdir = tmp_path / ".wesker"
    wdir.mkdir()
    (wdir / "mutation_report.json").write_text("{ not json")
    assert _load_cached_state(str(tmp_path)) is None


# ── callable_origin: the one origin contract for every callable shape ──


def test_callable_origin_resolution_order():
    """Tag outranks __wrapped__ outranks the raw code object; nothing → None.

    Module-level import on purpose: a function-local `from Wesker.ci import
    callable_origin` binds at call time straight from the module, so mutant
    patching (which installs into the TEST module's globals) never reaches it
    and every mutant of this function reads as an unkillable survivor.
    """

    def real():  # pragma: no cover — a value, never called
        pass

    real_origin = os.path.abspath(real.__code__.co_filename)

    def tagged():  # pragma: no cover
        pass

    tagged.__wesker_origin__ = "/somewhere/test_tagged.py"
    tagged.__wrapped__ = real
    # The tag wins even with __wrapped__ present.
    assert callable_origin(tagged) == "/somewhere/test_tagged.py"

    def wrapper():  # pragma: no cover
        pass

    wrapper.__wrapped__ = real
    # __wrapped__ wins over the wrapper's own code object.
    assert callable_origin(wrapper) == real_origin

    # A plain function falls through to its code object.
    assert callable_origin(real) == real_origin

    # Nothing to read — no tag, no wrap, no code — resolves to None, not a crash.
    assert callable_origin(1) is None
