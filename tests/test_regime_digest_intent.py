"""The pytest regime is digested and carried non-forcingly on the session holder (#63, Detective side).

Detective's verdict cache must key on the pytest execution regime a verdict was measured under, or a
warm verdict crosses regimes. Wesker supplies that: `PytestSessionManifest.regime_digest` hashes the
regime IDENTITY, and `LazySessionBaseline` carries it readable WITHOUT forcing the baseline trace —
the same non-forcing contract as `budgets`, since a cache key that forced the trace it exists to skip
would defeat the laziness. Pinned from intent.
"""

from __future__ import annotations

from Wesker.engine import LazySessionBaseline, session_regime_digest
from Wesker.session_manifest import PytestSessionManifest as M


def test_regime_digest_tracks_identity_not_collection_or_order():
    """It changes with a real regime change (import mode, rootdir, plugin SET) and is stable across
    things that are not a regime change (plugin ORDER, the collected items, the session scope)."""
    base = M(
        pytest_version="8",
        python_version="3.12",
        import_mode="prepend",
        rootpath="/r",
        inipath="/r/tox.ini",
        plugins=("a", "b"),
    )
    assert (
        base.regime_digest
        != base.__class__(**{**base.__dict__, "import_mode": "importlib"}).regime_digest
    )
    assert (
        base.regime_digest
        != base.__class__(**{**base.__dict__, "rootpath": "/other"}).regime_digest
    )
    assert (
        base.regime_digest
        != base.__class__(**{**base.__dict__, "plugins": ("a", "b", "c")}).regime_digest
    )
    # Not a regime change: plugin order, scope, and the item set must NOT move the digest.
    assert (
        base.regime_digest
        == base.__class__(**{**base.__dict__, "plugins": ("b", "a")}).regime_digest
    )
    assert (
        base.regime_digest
        == base.__class__(**{**base.__dict__, "scope": 99}).regime_digest
    )


def test_holder_exposes_the_regime_without_forcing_the_build():
    """The non-forcing contract: reading `regime_digest` off the holder must NOT trigger the baseline
    trace — the whole reason it lives on the holder (like budgets) and not on the built baseline."""
    built: list[int] = []
    holder = LazySessionBaseline(
        lambda subset=None: built.append(1), regime_digest="rd-xyz"
    )
    assert holder.regime_digest == "rd-xyz"
    assert holder.built is False
    assert built == []


def test_accessor_is_empty_outside_a_live_session():
    """No live session -> no regime -> "", so the Detective side leaves its cache key unchanged rather
    than invent a regime and thrash the cache."""
    assert session_regime_digest() == ""
