"""ENTRY is a precondition for reading an outcome as evidence about the SUITE (issue #18).

Installing a mutant and the test CALLING it are different events, and the engine could only
observe the first. `_patch_mutant_into_test` returns a bool meaning "an attribute was rebound
somewhere"; every reference taken BEFORE that rebinding still points at the original. A
registry built at import is the simplest exact case, and decorators, `functools.partial`,
`lru_cache` wrappers, closure cells, object fields and callback lists are the same defect
wearing different clothes.

The test then passes for the honest reason that nothing changed, and the mutant is reported
as a surviving specification gap — a claim about the user's tests derived from a fact about
ours. `_preserve_descriptor_shape` records the mirror image from the other side (issue #25:
a double-bound classmethod raised TypeError "which the runner reads as a spurious crash").

Engine-core cannot self-profile, so this is a hand-written unit test by design — the same
reason `test_kill_attribution.py` is.
"""

from __future__ import annotations

import ast
import importlib.util
import sys

import pytest

from Wesker.engine import (
    SCORED_DISPOSITIONS,
    evaluate_mutant,
    generate_mutants,
    mutant_disposition,
)
from Wesker.filter import filter_categories

# `HANDLERS` captures the function OBJECT at import. Rebinding the module global `handle` —
# which is what every install strategy the engine has ultimately does — cannot reach a
# reference that was already taken.
_SRC = "def handle(n):\n    return n * 2\n\n\nHANDLERS = [handle]\n"


@pytest.fixture
def target(tmp_path):
    """A real module on disk imported under its own name, so `co_filename` is a real path the
    module-qualified patch can match — exactly as in a consumer repo."""
    path = tmp_path / "capture_mod.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("capture_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capture_mod"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod, str(path)
    finally:
        sys.modules.pop("capture_mod", None)


def _mutants():
    node = ast.parse(_SRC).body[0]
    return generate_mutants(
        node, filter_categories(node, True), max_per_category=0, pass_index=0
    )


def _results(tests, mod, path):
    return [
        evaluate_mutant(m, tests, mod.handle, qualname="handle", source_path=path)
        for m in _mutants()
    ]


def test_a_captured_callable_is_observed_as_never_entered(target):
    """The defect. The registry holds the original, so no mutant is ever called — and before
    #18 that was indistinguishable from a suite that ran the mutant and failed to notice."""
    mod, path = target

    def case():
        assert mod.HANDLERS[0](5) == 10

    results = _results([case], mod, path)
    assert results, "no mutants generated — the fixture is not exercising the engine"
    assert all(r.entered is False for r in results), (
        f"entry flags: {[r.entered for r in results]}"
    )


def test_a_never_entered_mutant_is_kept_out_of_the_denominator(target):
    """The consequence that matters: a mutant nothing called is not a behaviour the tests
    failed to pin, so it belongs on neither side of the score."""
    mod, path = target

    def case():
        assert mod.HANDLERS[0](5) == 10

    for r in _results([case], mod, path):
        disposition = mutant_disposition(
            r.constructed, r.installed, r.entered, True, r.killed
        )
        assert disposition == "not_entered"
        assert disposition not in SCORED_DISPOSITIONS


def test_a_directly_called_target_is_observed_as_entered(target):
    """The control, and the guard against the cheap fix. A probe that reported `False`
    unconditionally would satisfy the two tests above and destroy every real measurement;
    this is the same call through the module binding the patch DOES reach."""
    mod, path = target

    def case():
        assert mod.handle(5) == 10

    results = _results([case], mod, path)
    assert results
    assert all(r.entered is True for r in results), (
        f"entry flags: {[r.entered for r in results]}"
    )
