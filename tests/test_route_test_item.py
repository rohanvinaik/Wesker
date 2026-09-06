"""#15 — Monty-Hall test routing: skip only the provably-impossible; keep `unknown`.

The selector is one-sided. Dropping a genuinely relevant test removes kills and covered lines, so
it OVERSTATES a specification gap; it can never manufacture a kill. The concrete bug this closes: a
test reaching the target only through an autouse/conftest fixture names nothing itself, so the
static scan dropped it and the run read "no test reaches this target" → needless synthesis
(reproduced before the fix: 0 callables returned for a fixture-reached suite).

INTENT tests: the defect is a wrong exclusion, so a characterization of current output cannot catch
it. The pure decisions assert the routing contract; the end-to-end tests drive the real live path
(`run_with_live_suite`) and assert the fixture-reached test survives.
"""

from __future__ import annotations

import os
import sys

from Wesker.ci import (
    discover_test_callables,
    route_admits,
    route_test_item,
    run_with_live_suite,
)


def _evict(*mods: str) -> None:
    """Drop stale in-process modules before a nested pytest session.

    Two end-to-end projects share the module names ``conftest`` and ``test_thing``; the first
    session leaves them in ``sys.modules`` pointing at a since-deleted tmp dir, and the second
    would import the stale copy (a harness artifact of running pytest inside pytest, not a routing
    fact). Wesker evicts ``test_*`` modules itself; ``conftest`` and the target module are not
    ``test_``-prefixed, so they are cleared here."""
    for name in mods:
        sys.modules.pop(name, None)


# ── the pure decision: only observed non-reach is impossible ──────────────────────


def test_a_static_miss_is_unknown_not_impossible():
    """The crux. No name, no fixture edge, no observation → `unknown`, which is KEPT. Classifying
    this as impossible is the false exclusion #15 exists to end."""
    assert route_test_item("none", False, False, "unseen", False) == "unknown_no_path"


def test_only_an_observed_non_reach_is_impossible():
    """Impossibility requires POSITIVE evidence — a prior trace that ran the node and did not touch
    the target. Nothing static can prove a negative here."""
    assert (
        route_test_item("none", False, False, "not_reached", False)
        == "impossible_observed"
    )
    assert (
        route_test_item("item", True, False, "reached", False) == "candidate_observed"
    )


def test_a_fixture_edge_is_a_candidate():
    """The autouse/conftest reach: the item names nothing but a fixture in its closure does."""
    assert route_test_item("none", True, False, "unseen", False) == "candidate_fixture"


def test_a_static_name_is_a_candidate():
    """The item's OWN body names the target — the direct-item stratum, a seed candidate."""
    assert route_test_item("item", False, False, "unseen", False) == "candidate_static"


def test_a_file_only_reference_is_a_file_peer_not_a_seed_candidate():
    """residual-1 (#15, per-item): only the item's FILE names the target — a sibling test does, the
    item's own body does not. That is a `file_peer`: tagged and kept in the pool, NOT a seed
    candidate, so one real test naming the target no longer drags its file-siblings into the eager
    seed. Whether a stratum is WIDENED is the driver's policy — Detective does not consult
    `file_peer` (a sibling's name is no evidence about this item) and discloses it as not consulted.
    The distinction file-vs-item is the whole point — a file bit conflated them."""
    assert route_test_item("file", False, False, "unseen", False) == "file_peer"


def test_a_caller_reach_is_a_widen_stratum_below_fixture_above_file_peer():
    """#15 B: the item names a production caller that reaches the target (a test of `resolve_roles`
    that calls `_compute_sets`) — a positive TRANSITIVE reach. It routes as `caller_reaches`: a widen
    stratum, below a fixture edge, above a file-peer, never a false drop and never eagerly seeded."""
    assert route_test_item("none", False, True, "unseen", False) == "caller_reaches"
    # A stronger own-body or fixture signal outranks it; an observed non-reach still dominates.
    assert route_test_item("item", False, True, "unseen", False) == "candidate_static"
    assert route_test_item("none", True, True, "unseen", False) == "candidate_fixture"
    assert (
        route_test_item("none", False, True, "not_reached", False)
        == "impossible_observed"
    )
    # It outranks a mere file-peer — a real call path beats a same-file coincidence.
    assert route_test_item("file", False, True, "unseen", False) == "caller_reaches"


def test_dynamic_uncertainty_widens_to_unknown_not_impossible():
    """Plugins / dynamic imports mean the static picture is incomplete — incomplete is not proof of
    irrelevance, so it widens rather than excludes."""
    assert route_test_item("none", False, False, "unseen", True) == "unknown_dynamic"


def test_admits_keeps_unknown_by_default_and_drops_it_when_conservative():
    """The one-sided guarantee is the default; `conservative` is the opt-in lossy narrowing."""
    assert route_admits("unknown_no_path", conservative=False) is True
    assert route_admits("unknown_no_path", conservative=True) is False
    # impossible is dropped either way; a candidate is kept either way.
    assert route_admits("impossible_observed", conservative=False) is False
    assert route_admits("candidate_fixture", conservative=True) is True
    # file_peer is a WEAK reason: admitted to the POOL by default, dropped only in conservative/lossy
    # mode. Admission to the pool is not a trace: whether a stratum is widened is the driver's policy.
    assert route_admits("file_peer", conservative=False) is True
    assert route_admits("file_peer", conservative=True) is False


# ── end-to-end: the reproduced autouse-fixture false-negative is closed ───────────


def _fixture_project(root: str, mod: str) -> str:
    """A project whose only path from the test to the target runs through an autouse fixture.

    ``mod`` is a UNIQUE target module name per test: two projects sharing ``target`` would collide
    in ``sys.modules`` across tests (the conftest's ``from target import target`` then resolves to
    a deleted tmp dir), which is a harness artifact, not a routing fact. Returns the source-file
    name to hand to discovery."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"{mod}.py"), "w") as fh:
        fh.write("def target(x):\n    return x + 1\n")
    with open(os.path.join(root, "conftest.py"), "w") as fh:
        fh.write(
            "import pytest\n"
            f"from {mod} import target\n\n"
            "@pytest.fixture(autouse=True)\n"
            "def _exercise():\n"
            "    assert target(1) == 2\n"
        )
    # the test body never mentions the target
    with open(os.path.join(root, "test_thing.py"), "w") as fh:
        fh.write("def test_thing():\n    assert 1 + 1 == 2\n")
    return f"{mod}.py"


def test_autouse_fixture_reached_test_is_kept_by_default(tmp_path):
    """The reproduced defect, closed: default routing keeps the fixture-reached test (unknown), so
    discovery does not falsely report an empty suite and send the caller to synthesize."""
    root = str(tmp_path / "proj")
    src = _fixture_project(root, "target_default")
    _evict("conftest", "test_thing", "target_default", "target_conservative")
    seen = {}

    def _probe():
        seen["names"] = sorted(
            getattr(c, "__name__", "?")
            for c in discover_test_callables(root, src, ["target"])
        )

    run_with_live_suite(root, _probe, target_files=[src])
    assert seen["names"] == ["test_thing"], (
        "the fixture-reached test was dropped — the #15 false gap"
    )


def test_conservative_mode_keeps_the_fixture_reached_test_on_its_fixture_edge(tmp_path):
    """Even the lossy narrowing must not drop a fixture-reached test: the fixture-definition file
    (conftest.py) references the target, so the item is a `candidate_fixture`, not `unknown`."""
    root = str(tmp_path / "proj")
    src = _fixture_project(root, "target_conservative")
    _evict("conftest", "test_thing", "target_default", "target_conservative")
    seen = {}

    def _probe():
        seen["names"] = sorted(
            getattr(c, "__name__", "?")
            for c in discover_test_callables(root, src, ["target"], conservative=True)
        )

    run_with_live_suite(root, _probe, target_files=[src])
    assert seen["names"] == ["test_thing"], (
        "conservative mode dropped a fixture-reached test — the fixture edge was not seen"
    )
