"""split_live_callables: the target-first seed is a STRICT subset (Fix B routing).

`_route_live_callables` keeps every reachable-file test as a candidate because it routes at FILE
granularity. `split_live_callables` refines that: a test whose file statically NAMES the target is
a seed candidate; a test in a reachable file that does not name it (and has no fixture edge) is an
unknown, deferred to the widen pass. The union is unchanged — the split only decides trace ORDER —
so nothing eligible is dropped, which is the one-sided soundness the whole routing rests on.
"""

from __future__ import annotations

from Wesker.ci import split_live_callables


def _mk(origin, fixtures=()):
    def t():
        pass

    t.__wesker_origin__ = str(origin)
    if fixtures:
        t.__wesker_fixture_origins__ = tuple(str(f) for f in fixtures)
    return t


def _names(items):
    return {getattr(c, "__wesker_origin__", "?") for c in items}


def test_a_test_naming_the_target_is_a_candidate_the_rest_are_unknown(tmp_path):
    (tmp_path / "mod.py").write_text("def target(x):\n    return x + 1\n")
    names_f = tmp_path / "test_names.py"
    names_f.write_text(
        "from mod import target\n\ndef test_it():\n    assert target(1) == 2\n"
    )
    silent_f = tmp_path / "test_silent.py"
    # imports the module but never references `target` — reaches the file, not (statically) the fn.
    silent_f.write_text("import mod\n\ndef test_other():\n    assert mod is not None\n")

    live = [_mk(names_f), _mk(silent_f)]
    scoped = [str(names_f), str(silent_f)]
    candidates, unknowns = split_live_callables(live, scoped, "target", ["target"])

    assert _names(candidates) == {str(names_f)}
    assert _names(unknowns) == {str(silent_f)}
    # The union is exactly the kept suite — nothing eligible dropped.
    assert _names(candidates) | _names(unknowns) == {str(names_f), str(silent_f)}


def test_an_attribute_reference_also_names_the_target(tmp_path):
    (tmp_path / "mod.py").write_text("def target(x):\n    return x\n")
    attr_f = tmp_path / "test_attr.py"
    # `mod.target(...)` — an Attribute reference, not a bare Name import.
    attr_f.write_text("import mod\n\ndef test_it():\n    assert mod.target(3) == 3\n")

    live = [_mk(attr_f)]
    candidates, unknowns = split_live_callables(
        live, [str(attr_f)], "target", ["target"]
    )
    assert _names(candidates) == {str(attr_f)}
    assert unknowns == []


def test_an_unparseable_file_is_treated_as_naming_the_target(tmp_path):
    # Never rule a test OUT on a parse failure — one-sided soundness.
    bad_f = tmp_path / "test_bad.py"
    bad_f.write_text("def test_it(:\n    syntax error here\n")
    live = [_mk(bad_f)]
    candidates, unknowns = split_live_callables(
        live, [str(bad_f)], "target", ["target"]
    )
    assert _names(candidates) == {str(bad_f)}
    assert unknowns == []
