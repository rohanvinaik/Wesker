"""Test-impact scoping must be verdict-EXACT, not merely fast.

Regression cover for a scoping defect that silently reported ~27% of Wesker's own
mutants (28.6% of Prism's) as survivors regardless of suite quality:

``executable_lines`` emitted only statement-START lines, while a mutator records its
fire site as the mutated NODE's line — a sub-expression line for any multi-line
statement. ``_trace_one`` intersects traced hits with that denominator, so those
lines never appeared in the coverage map; ``_tests_for`` then found no covering test
and evaluated the mutant against NOTHING, scoring it a survivor. The safe fallback
only fired for ``mutated_line is None``, never for "line the data cannot describe".
"""

import ast

from Wesker.line_coverage import executable_lines


def _fn(src: str) -> ast.FunctionDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


MULTILINE_SRC = '''
def f(items, flag):
    """doc"""
    if flag:
        return sum(
            x * 2
            for x in ("a", "b")
        )
    return max(
        len(items), 1
    )
'''


def test_executable_lines_span_multiline_statements():
    """The denominator must include EVERY line a statement occupies.

    A statement-start-only denominator drops the sub-expression lines CPython
    actually reports "line" events for, which is what orphaned mutants.
    """
    node = _fn(MULTILINE_SRC.strip())
    lines = executable_lines(node)
    # The genexp body/`for` clause and the continuation of the final return all
    # carry code, and a mutant can land on any of them.
    assert len(lines) >= 6
    assert max(lines) - min(lines) >= 5, f"denominator collapsed to {sorted(lines)}"


def test_every_mutant_line_is_in_the_denominator():
    """No mutant may sit on a line the coverage denominator cannot describe.

    This is the invariant that keeps scoping sound: if a mutated line is absent
    from ``executable_lines``, no coverage entry can ever mention it, so scoping
    would select zero tests and manufacture a false survivor.
    """
    from Wesker.engine import generate_mutants
    from Wesker.filter import filter_categories

    node = _fn(MULTILINE_SRC.strip())
    cats = filter_categories(node)
    lines = executable_lines(node)
    mutants = generate_mutants(node, cats, max_per_category=0)
    assert mutants, "expected mutants for this function"
    orphans = [
        (m.category.value, m.mutated_line)
        for m in mutants
        if m.mutated_line not in lines
    ]
    assert not orphans, f"mutants on lines outside the denominator: {orphans}"


def test_docstring_line_excluded_from_denominator():
    node = _fn(MULTILINE_SRC.strip())
    lines = executable_lines(node)
    assert node.lineno not in lines, "the def line is not reachable behavior"


def test_converged_scoping_default_is_pinned():
    """Pin the converged path's scoping default ON, so it cannot flip back unnoticed.

    Test-impact selection is the DESIGN, not an optimisation. Evaluating every mutant
    against every test is plain mutation testing — the expensive thing this engine's
    whole architecture exists to avoid. Off, this path was both slower and wronger:

      - COST: ``evaluate_mutant`` scans the entire suite before conceding a survivor,
        so unscoped, every would-be survivor pays for the whole suite. Against the
        500ms per-mutant budget (sized for a scoped handful) that turned 83% of
        Detective's and 92% of ModelAtlas's unspecified dimensions into TIMEOUTS,
        with zero true survivors. Those runs measured suite speed, not specification.
      - ACCURACY: unscoped credited 107 kills on prism/economics.py::analyze to a test
        that never references the function and fails identically on the UNMUTATED
        original — a test that distinguishes nothing, credited with killing
        everything. Scoping drops it, because its traced coverage is empty.

    This default was previously pinned OFF as the status quo, with a note asking
    whoever revisited it to do so deliberately. This is that revisit.
    ``test_scoped_and_unscoped_verdicts_agree`` below is the warrant: same verdict,
    less work.
    """
    import inspect

    from Wesker.engine import run_function_converged

    sig = inspect.signature(run_function_converged)
    assert sig.parameters["scope_tests"].default is True, (
        "the converged path's scope_tests default flipped OFF — that is plain "
        "mutation testing: every test against every mutant, which blows the "
        "per-mutant budget into mass timeouts and credits non-distinguishing tests "
        "with kills"
    )


def test_scoped_and_unscoped_verdicts_agree():
    """The whole justification for scoping: identical verdicts, less work.

    Profiles a real function against a suite that kills every mutant, both scoped
    and unscoped. Any divergence means scoping invented survivors.

    DISPOSITION-EXACT (A0, #40): equal totals do not prove agreement — killing mutant
    A instead of B leaves ``total_killed`` unchanged yet is a scoping soundness bug
    (the kind ``c8ee1c0`` records as "caught by the oracle, not by inspection"). So this
    also compares the per-mutant OUTCOME map, not just the counts.
    """
    from Wesker.engine import MutationCategory, run_function_profiling

    src = (
        "def scoreit(a, b, flag):\n"
        '    """doc"""\n'
        "    if flag:\n"
        "        return sum(\n"
        "            v * 2\n"
        "            for v in (a, b)\n"
        "        )\n"
        "    return max(\n"
        "        a, b\n"
        "    )\n"
    )
    node = _fn(src)
    ns: dict = {}
    exec(compile(ast.parse(src), "<scoretest>", "exec"), ns)
    original = ns["scoreit"]

    def test_flag_true():
        assert scoreit(1, 2, True) == 6  # noqa: F821

    def test_flag_false():
        assert scoreit(1, 2, False) == 2  # noqa: F821

    def test_flag_true_other():
        assert scoreit(3, 4, True) == 14  # noqa: F821

    tests = [test_flag_true, test_flag_false, test_flag_true_other]
    for t in tests:
        t.__globals__["scoreit"] = original

    cats = {MutationCategory.VALUE, MutationCategory.ARITHMETIC, MutationCategory.SWAP}
    unscoped = run_function_profiling(
        node,
        "<scoretest>::scoreit",
        cats,
        tests,
        original,
        max_per_category=0,
        scope_tests=False,
    )
    scoped = run_function_profiling(
        node,
        "<scoretest>::scoreit",
        cats,
        tests,
        original,
        max_per_category=0,
        scope_tests=True,
    )
    assert scoped.total_mutants == unscoped.total_mutants
    assert scoped.total_killed == unscoped.total_killed, (
        f"scoping changed the verdict: {unscoped.total_killed} killed unscoped vs "
        f"{scoped.total_killed} scoped — scoping is not verdict-exact"
    )

    # Disposition-exact: every mutant's IDENTITY must map to the same OUTCOME under both
    # scopes. Keyed on mutant_id; the outcome is `killed_by` (assertion / exception / crash /
    # timeout) for a kill, "survived" otherwise. WHICH tests killed a mutant (the kill_matrix
    # values) legitimately shrinks under scoping and is NOT part of the disposition.
    def _disposition(r) -> dict:
        d = {rec["mutant_id"]: rec.get("killed_by") for rec in r.killed_records}
        d.update({rec["mutant_id"]: "survived" for rec in r.survivor_records})
        return d

    assert _disposition(scoped) == _disposition(unscoped), (
        "scoping changed WHICH mutants died, not just how many:\n"
        f"  unscoped: {_disposition(unscoped)}\n"
        f"  scoped:   {_disposition(scoped)}"
    )


BRANCHY_SRC = '''
def g(x):
    """doc"""
    if x <= 2:
        r = 1
    elif x <= 10:
        r = 2
    else:
        r = 3
    return r
'''


def test_keyword_only_lines_excluded_from_denominator():
    """A bare ``else:`` carries no bytecode, so no trace event can EVER mention it.

    Regression cover for the inverse defect of the statement-start one above:
    spanning full statement extents swept keyword-only lines (``else:``,
    ``finally:``) into the denominator. ``_trace_one`` intersects traced hits
    with the denominator, so such a line read as a permanent coverage gap no
    test could ever close — and callers asked for input after input to reach a
    line that is not code.
    """
    src = BRANCHY_SRC.strip()
    node = _fn(src)
    lines = executable_lines(node)
    else_line = next(
        i for i, text in enumerate(src.splitlines(), start=1) if text.strip() == "else:"
    )
    assert else_line not in lines, "keyword-only line inflates the denominator forever"
    # The branch bodies on either side of the keyword ARE reachable behavior.
    assert else_line - 1 in lines
    assert else_line + 1 in lines


def test_no_mutant_orphaned_by_traceability_filter():
    """The co_lines intersection must not evict any mutant's fire-site line.

    Same invariant as ``test_every_mutant_line_is_in_the_denominator``, asserted
    against the branchy sample the filter actually prunes: soundness requires
    pruning ONLY lines no mutant and no trace event can ever name.
    """
    from Wesker.engine import generate_mutants
    from Wesker.filter import filter_categories

    node = _fn(BRANCHY_SRC.strip())
    cats = filter_categories(node)
    lines = executable_lines(node)
    mutants = generate_mutants(node, cats, max_per_category=0)
    assert mutants, "expected mutants for this function"
    orphans = [
        (m.category.value, m.mutated_line)
        for m in mutants
        if m.mutated_line not in lines
    ]
    assert not orphans, f"mutants on lines outside the denominator: {orphans}"
