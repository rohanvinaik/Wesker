"""The live default routing is identity, so it must not compute the discarded impact map (#15 C2).

Grounding ([R-exec] probe against local Wesker): `_route_live_callables`, under the default
`conservative=False`, admits every per-item route it can produce (static_reach × fixture over
{item,none}×{T,F}, with caller/observed/dynamic fixed as the body passes them — only
candidate_static / candidate_fixture / unknown_no_path arise, all kept). So its result equals the
live suite verbatim. `discover_test_callables` used to compute `scoped = relevant_test_files(...)`
BEFORE that identity and then discard it on every live profiling call — the "computed every live
call, discarded" waste (TEST_BASIS §4.5).

Intent: skip that computation on the proven-identity path, and ONLY there. `conservative=True`
still narrows (it drops `unknown_no_path`), so it must keep routing — the boundary that keeps
`_route_live_callables` alive rather than deleted. These tests pin both the admission invariant the
short-circuit rests on and the observable skip.
"""

from __future__ import annotations

import Wesker.ci as ci
from Wesker.ci import (
    _LIVE_SUITE,
    discover_test_callables,
    route_admits,
    route_test_item,
)


def test_the_default_router_admits_every_production_route_but_conservative_narrows():
    # If any of these four flips to False, the identity short-circuit is no longer valid and must
    # be revisited — this is the invariant `if live and not conservative: return live` rests on.
    for static_reach in ("item", "none"):
        for fixture_reaches in (True, False):
            code = route_test_item(
                static_reach, fixture_reaches, False, "unseen", False
            )
            assert route_admits(code, conservative=False) is True, (
                static_reach,
                fixture_reaches,
                code,
            )
    # The boundary: conservative=True DOES drop the no-signal item — real narrowing, so the router
    # is not dead and cannot be inlined away.
    dropped = route_test_item("none", False, False, "unseen", False)
    assert route_admits(dropped, conservative=True) is False


def test_live_default_path_returns_the_suite_without_computing_the_impact_map(
    monkeypatch, tmp_path
):
    def _must_not_run(*args, **kwargs):
        raise AssertionError(
            "relevant_test_files must not run on the proven-identity path"
        )

    monkeypatch.setattr(ci, "relevant_test_files", _must_not_run)

    def t1():
        pass

    def t2():
        pass

    live = [t1, t2]
    token = _LIVE_SUITE.set(live)
    try:
        got = discover_test_callables(
            str(tmp_path), "mod.py", ["f"]
        )  # conservative=False default
    finally:
        _LIVE_SUITE.reset(token)
    assert (
        got == live
    )  # identity — and _must_not_run never fired, proving the map was skipped


def test_conservative_live_path_is_not_short_circuited(monkeypatch, tmp_path):
    called: list[int] = []

    def _tracking(*args, **kwargs):
        called.append(1)
        return []

    monkeypatch.setattr(ci, "relevant_test_files", _tracking)

    def t():
        pass

    token = _LIVE_SUITE.set([t])
    try:
        discover_test_callables(str(tmp_path), "mod.py", ["f"], conservative=True)
    finally:
        _LIVE_SUITE.reset(token)
    assert called, (
        "conservative=True must NOT short-circuit; it must compute scoped and route"
    )
