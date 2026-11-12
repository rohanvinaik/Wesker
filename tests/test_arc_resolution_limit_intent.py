"""#17 — what an ARC resolves, and the one thing it does not, stated rather than implied.

THE DEFECT (measured 2026-10-26). The `trace_evidence` module docstring said short-circuit operands
"stay apart". Measured: `if a or b:` on ONE line records the arc `(2, 3)` whether `a` or `b`
decided it, because an arc is a `(prev_line, cur_line)` pair and both operands are on line 2. A
page reading that sentence would believe a short-circuit path was an obligation the ledger can
discharge. It is not (no instruction-level branch edge is recorded), so the docstring now says so
and these tests pin both sides: the lines-apart cases that ARE resolved, and the same-line case
that is NOT — so the limit cannot silently turn into a claim in either direction.
"""

from __future__ import annotations

import importlib.util
import os

from Wesker import trace_evidence
from Wesker.line_coverage import trace_suite


def _arcs_of(tmp_path, name, source, calls):
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real = os.path.realpath(str(path))
    tests = []
    for i, c in enumerate(calls):

        def t(c=c):
            c(module)

        t.__name__ = t.__qualname__ = f"t_{name}_{i}"
        tests.append(t)
    arcs: dict = {}
    trace_suite(tests, {real}, arcs_out=arcs)
    return [
        {a for aset in files.values() for a in aset} for files in arcs.values()
    ], real


_ONE_LINE = "def f(a, b):\n    if a or b:\n        return 1\n    return 0\n"


def test_same_line_short_circuit_operands_share_one_arc(tmp_path):
    (a_decides, b_decides), _ = _arcs_of(
        tmp_path,
        "scl_one",
        _ONE_LINE,
        [lambda m: m.f(True, False), lambda m: m.f(False, True)],
    )
    assert a_decides == b_decides, (
        "arcs now separate same-line operands: the limit moved, update the docstring"
    )


def test_the_module_docstring_states_that_limit(tmp_path):
    doc = trace_evidence.__doc__
    assert "short-circuit operands stay apart" not in doc.replace("\n", " ")
    assert "one line" in doc.replace("\n", " ") or "ONE line" in doc.replace("\n", " ")
    assert "not an obligation this ledger can discharge" in doc.replace("\n", " ")


def test_the_two_edges_of_a_conditional_on_different_lines_stay_apart(tmp_path):
    src = "def f(a):\n    if a:\n        return 1\n    return 0\n"
    (t_true, t_false), _ = _arcs_of(
        tmp_path, "scl_two", src, [lambda m: m.f(True), lambda m: m.f(False)]
    )
    assert (2, 3) in t_true and (2, 3) not in t_false
    assert (2, 4) in t_false and (2, 4) not in t_true


def test_loop_entry_and_zero_iterations_stay_apart(tmp_path):
    src = "def f(xs):\n    for x in xs:\n        pass\n    return 0\n"
    (entered, empty), _ = _arcs_of(
        tmp_path, "scl_loop", src, [lambda m: m.f([1]), lambda m: m.f([])]
    )
    assert (2, 3) in entered and (2, 3) not in empty
    assert (2, 4) in empty
