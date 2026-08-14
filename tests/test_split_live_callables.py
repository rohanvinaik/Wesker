"""split_live_callables / partition_live_callables: the target-first seed is a STRICT subset, routed
per TESTID (Fix B routing, #15).

`_route_live_callables` keeps every reachable-file test as a candidate because it routes at FILE
granularity. The seed router refines that PER ITEM: a test whose OWN body statically names the target
is a seed candidate; a file-PEER — the same file names the target but this item's body does not — is
KEPT (widened) as an unknown, never dragged into the seed. That per-item distinction is residual-1:
one real test naming the target used to promote its every file-sibling into the eager seed. The union
is unchanged — the split only decides trace ORDER — so nothing eligible is dropped, which is the
one-sided soundness the whole routing rests on.
"""

from __future__ import annotations

from Wesker.ci import callable_test_id, partition_live_callables, split_live_callables


def _mk(origin, fixtures=()):
    """A mock whose own body does NOT name the target — its only association is its origin file."""

    def t():
        pass

    t.__wesker_origin__ = str(origin)
    if fixtures:
        t.__wesker_fixture_origins__ = tuple(str(f) for f in fixtures)
    return t


def _mk_naming(origin):
    """A mock whose OWN body statically references `target` (a bare Name). The per-item router reads
    its source through `callable_source` and seeds it as a candidate, not merely its file."""

    def t():
        return target  # noqa: F821 — a static Name ref in the item's OWN body; never executed

    t.__wesker_origin__ = str(origin)
    return t


def _mk_attr_naming(origin):
    """A mock whose OWN body references the target as an ATTRIBUTE (`mod.target`), not a bare Name."""

    def t():
        return mod.target  # noqa: F821 — a static Attribute ref in the item's OWN body; never run

    t.__wesker_origin__ = str(origin)
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


def test_only_the_item_naming_the_target_is_a_candidate_a_file_peer_is_unknown(
    tmp_path,
):
    """residual-1 (#15, per-item): a test whose OWN body names the target is a seed candidate; a
    file-PEER — same file references the target but this item's body does not — is KEPT (widened) as
    an unknown, never seeded. The union is the whole kept suite; only the seed shrinks."""
    (tmp_path / "mod.py").write_text("def target(x):\n    return x + 1\n")
    peer_f = tmp_path / "test_peer.py"
    # The FILE references the target, but a given item's body need not — a file-peer.
    peer_f.write_text(
        "from mod import target\n\ndef test_it():\n    assert target(1) == 2\n"
    )
    silent_f = tmp_path / "test_silent.py"
    silent_f.write_text("import mod\n\ndef test_other():\n    assert mod is not None\n")

    naming = _mk_naming(peer_f)  # its OWN body names target -> candidate
    peer = _mk(peer_f)  # same file names target, body does not -> file_peer -> unknown
    silent = _mk(silent_f)  # file does not name target -> unknown
    candidates, unknowns = split_live_callables(
        [naming, peer, silent], [str(peer_f), str(silent_f)], "target", ["target"]
    )

    assert candidates == [naming]
    assert set(unknowns) == {peer, silent}
    # The union is exactly the kept suite — nothing eligible dropped, only the seed narrowed.
    assert set(candidates) | set(unknowns) == {naming, peer, silent}


def test_an_attribute_reference_in_the_item_body_also_names_the_target(tmp_path):
    """The per-item scan counts an Attribute reference (`mod.target`) in the item's OWN body, not
    only a bare Name — the same two-form match `_files_referencing_target` uses."""
    (tmp_path / "mod.py").write_text("def target(x):\n    return x\n")
    attr_f = tmp_path / "test_attr.py"
    attr_f.write_text("import mod\n\ndef test_it():\n    assert mod.target(3) == 3\n")

    naming = _mk_attr_naming(attr_f)
    candidates, unknowns = split_live_callables(
        [naming], [str(attr_f)], "target", ["target"]
    )
    assert candidates == [naming]
    assert unknowns == []


def test_an_unparseable_file_is_kept_as_a_file_peer_never_dropped(tmp_path):
    """One-sided soundness on a parse failure: an unparseable file MIGHT reference the target, so its
    test is never ruled OUT. But a parse failure cannot prove the ITEM's body names the target, so it
    is a file_peer — KEPT (widened as unknown), not a seed candidate. Kept-not-dropped is the
    invariant; seeding is only an ordering optimization."""
    bad_f = tmp_path / "test_bad.py"
    bad_f.write_text("def test_it(:\n    syntax error here\n")
    live = [_mk(bad_f)]
    candidates, unknowns = split_live_callables(
        live, [str(bad_f)], "target", ["target"]
    )
    assert candidates == []  # body unprovable -> not a seed candidate
    assert _names(unknowns) == {str(bad_f)}  # but KEPT (widened), never dropped


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
