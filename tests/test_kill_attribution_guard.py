"""The nameless-target guard on the attribution control.

``_outcome_on_original`` re-runs a test against the ORIGINAL to decide whether a mutant
really earned a kill. To do that it must first put the original bindings back. Every rebind
goes through ``setattr(target, func_name, saved)`` inside a bare ``except: continue``, so a
``func_name`` of ``None`` does not raise — it fails silently on every target, leaves the
MUTANT installed, and runs the control against the mutant itself. The two runs then agree
and the caller discards a real kill.

``func_name`` comes from ``getattr(mutant.mutated_node, "name", None)``, so the type says it
can be ``None`` even though today's transformers always yield a named node. The guard makes
the impossible case explicit and conservative: decline, and let the kill stand.
"""

from __future__ import annotations

import types

from Wesker.engine import _outcome_on_original


def _module_with(name: str, value):
    mod = types.ModuleType("attribution_probe")
    setattr(mod, name, value)
    return mod


def test_a_nameless_target_declines_instead_of_scoring_the_mutant_against_itself():
    """Returns ``None`` — which cannot equal the caller's outcome, so the kill survives —
    and does not touch the bindings it has no name to restore."""

    def _original():
        return "original"

    def _mutant():
        return "mutant"

    ran: list[str] = []

    def _test_fn(*_a, **_k):
        ran.append("ran")

    mod = _module_with("target", _mutant)  # the mutant is installed
    module_saved = [(mod, _original)]

    assert _outcome_on_original(_test_fn, _original, module_saved, None, 1000.0) is None
    # The control never ran: running it here would have executed against the mutant.
    assert ran == []
    # And nothing was rebound — the caller's own restore path still owns that.
    assert mod.target is _mutant


def test_a_named_target_restores_the_original_for_the_control_run_then_puts_it_back():
    """The contract the guard protects: during the control the module binding is the
    ORIGINAL, and afterwards it is whatever was live before (the mutant)."""

    def _original():
        return "original"

    def _mutant():
        return "mutant"

    seen: list[object] = []

    def _test_fn(*_a, **_k):
        seen.append(mod.target)

    mod = _module_with("target", _mutant)
    module_saved = [(mod, _original)]

    _outcome_on_original(_test_fn, _original, module_saved, "target", 1000.0)

    assert seen == [_original]  # the control saw the original, not the mutant
    assert mod.target is _mutant  # and the mutant is re-installed afterwards


def test_a_target_that_refuses_rebinding_does_not_take_the_control_run_down_with_it():
    """The restore loop swallows per-target failures on purpose. A class owner can expose the
    name as a read-only descriptor, and the control's verdict — already computed — must not be
    lost to a cleanup that cannot finish. The outcome is still returned."""

    def _original():
        return "original"

    def _test_fn(*_a, **_k):
        return None

    class _Frozen:
        # Settable once (the pre-control rebind), then permanently refuses — so the restore
        # in the `finally` raises and must be absorbed.
        def __init__(self):
            self._set = 0

        def __setattr__(self, name, value):
            if name == "_set":
                object.__setattr__(self, name, value)
                return
            if self._set:
                raise AttributeError(f"{name} is read-only")
            object.__setattr__(self, "_set", 1)
            object.__setattr__(self, name, value)

    target = _Frozen()
    object.__setattr__(target, "target", _original)

    assert (
        _outcome_on_original(
            _test_fn, _original, [(target, _original)], "target", 1000.0
        )
        is None
    )
