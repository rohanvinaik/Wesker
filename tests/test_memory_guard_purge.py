"""`purge_caches` must remove every cache `.wesker/` accumulates — including
``trace_cache.json``, which was absent from its hardcoded target list for the
file's whole early life. `purge` is the documented recovery path for a
poisoned cache; a purge that reports "clean" while the poisoned entry
survives is worse than none, because the user acts on the claim.
"""

from __future__ import annotations

import os

from Wesker.memory_guard import purge_caches


def test_purge_removes_trace_cache(tmp_path):
    wdir = tmp_path / ".wesker"
    wdir.mkdir()
    for name in ("mutation_report.json", "mcdc_report.json", "trace_cache.json"):
        (wdir / name).write_text("{}")

    removed, reclaimed = purge_caches(str(tmp_path))

    assert not (wdir / "trace_cache.json").exists()
    assert {os.path.basename(p) for p in removed} == {
        "mutation_report.json",
        "mcdc_report.json",
        "trace_cache.json",
    }
    assert reclaimed > 0


def test_purge_missing_dir_is_quiet(tmp_path):
    removed, reclaimed = purge_caches(str(tmp_path))
    assert removed == ()
    assert reclaimed == 0
