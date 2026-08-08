"""The tracer matches target files by IDENTITY, never by spelling.

co_filename carries the spelling a module was imported under; the target path
carries what the caller typed. A case-insensitive filesystem or a symlink makes
the two open the same file while comparing unequal — the dispatch then never
attached, and a converge could report "80/80 killed" and a body-wide line gap
about the same function (measured on wesker/engine.py vs Wesker/engine.py).
Spellings are never rewritten: matching is (st_dev, st_ino), not case-folding.
"""

from __future__ import annotations

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Wesker.line_coverage import (  # noqa: E402
    _target_matcher,
    _trace_one,
    _trace_one_multi,
    coverage_from_trace,
    executable_lines,
)


def _write_module(tmp_path, name="idmod"):
    p = tmp_path / f"{name}.py"
    p.write_text(
        textwrap.dedent(
            """
            def double(x):
                y = x * 2
                return y
            """
        )
    )
    return p


def _load(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matcher_matches_a_symlink_spelling_to_its_target(tmp_path):
    real = _write_module(tmp_path)
    link = tmp_path / "alias.py"
    link.symlink_to(real)
    match = _target_matcher({str(link)})
    # co_filename spelling (the real path) resolves to the caller's spelling
    assert match(str(real)) == str(link)
    # memoized second call agrees
    assert match(str(real)) == str(link)
    assert match("<string>") is None


def test_trace_one_attaches_through_a_differently_spelled_target(tmp_path):
    real = _write_module(tmp_path)
    mod = _load(real)  # co_filename == str(real)
    link = tmp_path / "alias.py"
    link.symlink_to(real)

    def t():
        assert mod.double(3) == 6

    hits, *_ = _trace_one(t, str(link), {2, 3, 4, 5})
    assert hits  # the old string-equality dispatch collected nothing here


def test_trace_one_multi_keys_hits_by_the_callers_spelling(tmp_path):
    real = _write_module(tmp_path)
    mod = _load(real)
    link = tmp_path / "alias.py"
    link.symlink_to(real)

    def t():
        assert mod.double(2) == 4

    hits, *_ = _trace_one_multi(t, {str(link)})
    assert set(hits) == {str(link)}  # never the co_filename spelling
    assert hits[str(link)]


def test_coverage_from_trace_falls_back_to_identity_lookup(tmp_path):
    real = _write_module(tmp_path)
    link = tmp_path / "alias.py"
    link.symlink_to(real)
    # a persisted trace keyed under the OTHER spelling of the same file
    traced = {"t_one": {str(real): {2, 3}}}
    got = coverage_from_trace(traced, str(link), {2, 3, 4})
    assert got == {"t_one": [2, 3]}
    # a genuinely different file still misses
    other = _write_module(tmp_path, name="unrelated")
    assert coverage_from_trace(traced, str(other), {2, 3}) == {"t_one": []}


def test_executable_lines_and_trace_agree_on_a_method_target(tmp_path):
    p = tmp_path / "cls.py"
    p.write_text(
        textwrap.dedent(
            """
            class Box:
                @staticmethod
                def pick(v):
                    out = v + 1
                    return out
            """
        )
    )
    mod = _load(p)
    import ast as _ast

    tree = _ast.parse(p.read_text())
    func = tree.body[0].body[0]
    exec_lines = executable_lines(func)

    def t():
        assert mod.Box.pick(1) == 2

    link = tmp_path / "cls_alias.py"
    link.symlink_to(p)
    hits, *_ = _trace_one(t, str(link), exec_lines)
    assert hits == {5, 6}
