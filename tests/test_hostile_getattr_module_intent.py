"""A module that imports `torch` used to read as `0/N killed` — every mutant an un-evaluable
survivor — SILENTLY. `_patch_module_qualified` scans every module in `sys.modules` and asks each
one's attribute for `__code__`, to match the function under test by `co_filename`. `torch._classes`
returns a PROXY for ANY attribute name (so `getattr(mod, name, None)` is never None — the default
cannot fire), and that proxy raises a RuntimeError — not AttributeError — the moment it is asked for
`__code__` (`Tried to instantiate class '<name>.__code__'`). The exception escaped the scan, escaped
`evaluate_mutant`, and the enclosing loop scored every mutant an un-evaluable survivor. A whole class
of real code — anything importing torch — read as fully surviving, which is a false clean bill: the
one outcome the engine must never produce.

This reproduces the SHAPE without torch: a module whose `__getattr__` hands back a hostile proxy.
Before the fix (`_safe_code`), evaluating a mutant with such a module present raises RuntimeError;
after, the scan skips the hostile module and kills the mutant it should.

Engine-core cannot self-profile, so this is a hand-written unit test by design — the same reason
`test_mutant_entry.py` and `test_kill_attribution.py` are.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types

import pytest

from Wesker.engine import evaluate_mutant, generate_mutants
from Wesker.filter import filter_categories

_SRC = "def add(n):\n    return n + 1\n"


class _HostileProxy:
    """Like `torch._classes`' namespace proxy: constructible for ANY name, and raising (NOT
    AttributeError) the instant an attribute — here `__code__` — is read off it."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str):
        raise RuntimeError(
            f"Tried to instantiate class '{self._name}.{attr}', but it does not exist!"
        )


class _HostileModule(types.ModuleType):
    """A `sys.modules` entry that returns a proxy for every attribute name, torch._classes-style."""

    def __getattr__(self, name: str):  # noqa: D401 — any name resolves, never AttributeError
        return _HostileProxy(name)


@pytest.fixture
def hostile_module():
    mod = _HostileModule("hostile_ext._classes")
    sys.modules["hostile_ext._classes"] = mod
    try:
        yield mod
    finally:
        sys.modules.pop("hostile_ext._classes", None)


@pytest.fixture
def target(tmp_path):
    """A real on-disk module so `co_filename` is a path the module-qualified patch can match."""
    path = tmp_path / "add_mod.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("add_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["add_mod"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod, str(path)
    finally:
        sys.modules.pop("add_mod", None)


def _mutants():
    node = ast.parse(_SRC).body[0]
    return generate_mutants(
        node, filter_categories(node, True), max_per_category=0, pass_index=0
    )


def test_hostile_getattr_module_does_not_blind_kill_detection(target, hostile_module):
    """With a torch._classes-shaped module in `sys.modules`, mutants are still KILLED.

    The regression: `_patch_module_qualified` crashed on the hostile module, `evaluate_mutant`
    propagated the RuntimeError, and every mutant read as an un-evaluable survivor → `0/N killed`
    for the whole (torch-importing) module. The fix (`_safe_code`) treats a hostile `__code__`
    read as "not the function we are looking for" and skips it.
    """
    mod, path = target

    def test_add() -> None:
        assert mod.add(2) == 3  # kills any mutation of `n + 1` observable at n == 2

    results = [
        evaluate_mutant(m, [test_add], mod.add, qualname="add", source_path=path)
        for m in _mutants()
    ]

    assert results, "expected at least one mutant to evaluate"
    # No mutant may be dropped as an un-evaluable harness error caused by the hostile module.
    unconstructed = [r for r in results if not r.constructed]
    assert not unconstructed, (
        f"hostile module blinded construction of {len(unconstructed)} mutant(s)"
    )
    # And the point: at least one mutant of `n + 1` is actually caught.
    assert any(r.killed for r in results), (
        "no mutant killed — kill-detection was blinded"
    )
