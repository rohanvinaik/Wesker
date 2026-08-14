"""partition_live_callables: the target-first seed is a STRICT subset, routed per TESTID (Fix B, #15).

`_route_live_callables` keeps every reachable-file test as a candidate because it routes at FILE
granularity. The seed router refines that PER ITEM: a test whose OWN body statically names the target
is a seed candidate; a file-PEER — the same file names the target but this item's body does not — is
KEPT (widened) as an unknown, never dragged into the seed. That per-item distinction is residual-1:
one real test naming the target used to promote its every file-sibling into the eager seed. The union
is unchanged — the split only decides trace ORDER — so nothing eligible is dropped, which is the
one-sided soundness the whole routing rests on.
"""

from __future__ import annotations

from Wesker.ci import callable_test_id, partition_live_callables


def _mk(origin, fixtures=()):
    """A mock whose own body does NOT name the target — its only association is its origin file."""

    def t():
        pass

    t.__wesker_origin__ = str(origin)
    if fixtures:
        t.__wesker_fixture_origins__ = tuple(str(f) for f in fixtures)
    return t


def _mk_caller(origin):
    """A mock whose OWN body names `resolve` — a production CALLER of the target, not the target
    itself (#15 B). Reaches the target transitively when `resolve` is passed in `caller_names`."""

    def t():
        return resolve  # noqa: F821 — names a caller of the target, never executed

    t.__wesker_origin__ = str(origin)
    return t


def _names(items):
    return {getattr(c, "__wesker_origin__", "?") for c in items}


def test_exact_observed_reach_populates_all_three_routing_buckets(tmp_path):
    """#15 intent: observed reach is consumed; absent reach stays unknown, never impossible."""
    names_f = tmp_path / "test_names.py"
    names_f.write_text("from mod import target\n")
    silent_f = tmp_path / "test_silent.py"
    silent_f.write_text("import mod\n")
    unseen_f = tmp_path / "test_unseen.py"
    unseen_f.write_text("import mod\n")
    reached, missed, unseen = _mk(names_f), _mk(silent_f), _mk(unseen_f)
    live = [reached, missed, unseen]
    observed = {
        callable_test_id(reached): "reached",
        callable_test_id(missed): "not_reached",
    }

    candidates, unknowns, impossible = partition_live_callables(
        live,
        [str(names_f), str(silent_f), str(unseen_f)],
        "target",
        ["target"],
        observed,
    )

    assert _names(candidates) == {str(names_f)}
    assert _names(unknowns) == {str(unseen_f)}
    assert _names(impossible) == {str(silent_f)}


def test_unknowns_are_ordered_file_peer_before_no_path(tmp_path):
    """#15 C: the widen stratum is ordered most-likely-reacher first — a file_peer (its file names
    the target) is traced before an item whose file has no signal, so the item-incremental widen
    tries the stronger candidate first and can stop earlier."""
    (tmp_path / "mod.py").write_text("def target(x):\n    return x + 1\n")
    peer_f = tmp_path / "test_peer.py"
    peer_f.write_text(
        "from mod import target\n\ndef test_it():\n    assert target(1) == 2\n"
    )
    blank_f = tmp_path / "test_blank.py"
    blank_f.write_text("def test_it():\n    assert 1 == 1\n")

    no_path = _mk(blank_f)  # file names nothing -> unknown_no_path
    file_peer = _mk(peer_f)  # file names target, body does not -> file_peer
    # Pass no_path FIRST so only the stratum sort can put file_peer ahead of it.
    candidates, unknowns, _imp = partition_live_callables(
        [no_path, file_peer], [str(peer_f), str(blank_f)], "target", ["target"]
    )
    assert candidates == []
    assert unknowns == [
        file_peer,
        no_path,
    ]  # file_peer sorted ahead despite input order


def test_a_caller_reaching_test_is_a_widen_stratum_ahead_of_a_file_peer(tmp_path):
    """#15 B: a test whose body names a PRODUCTION CALLER of the target (never the target itself)
    reaches it transitively — a `caller_reaches` widen stratum, ordered AHEAD of a file_peer. The
    caller set is passed as `caller_names={"resolve"}`, mirroring Detective's one-hop backward slice
    (`resolve` is a production function that calls the private `target`)."""
    (tmp_path / "mod.py").write_text("def target(x):\n    return x + 1\n")
    peer_f = tmp_path / "test_peer.py"
    peer_f.write_text(
        "from mod import target\n\ndef test_it():\n    assert target(1) == 2\n"
    )
    caller_f = tmp_path / "test_caller.py"
    caller_f.write_text(
        "from mod import resolve\n\ndef test_it():\n    assert resolve(1) == 2\n"
    )

    file_peer = _mk(peer_f)  # file names target, body does not -> file_peer
    caller = _mk_caller(
        caller_f
    )  # body names `resolve`, a caller of target -> caller_reaches
    # Pass file_peer FIRST so only the stratum sort can put the caller-reaching item ahead of it.
    candidates, unknowns, _imp = partition_live_callables(
        [file_peer, caller],
        [str(peer_f), str(caller_f)],
        "target",
        ["target"],
        None,
        {"resolve"},
    )
    assert (
        candidates == []
    )  # neither names the target in its own body — both are widen strata
    assert unknowns == [
        caller,
        file_peer,
    ]  # caller_reaches (rank 0) sorts ahead of file_peer
