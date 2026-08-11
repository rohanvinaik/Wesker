"""Emit the collection's node-ID basis to JSON, for a SUBPROCESS verification run (#58).

The in-process manifest capture (`_Collect` / `_Driver`) sets a ContextVar the parent reads. A
certificate's FINAL verification, though, runs the whole proof suite (user + generated) under the
consumer's OWN pytest in a real subprocess — the one place the complete collection is addressed as
pytest addresses it — and a ContextVar does not cross a process boundary. So this loads as a `-p`
plugin in that subprocess and writes the runner's own answer to the file named by
``WESKER_MANIFEST_OUT``: each collected item as ``[node_id, content digest]``, plus the shadow-module
and collection-error facts a consumer must gate a frozen basis on.

Defensive throughout: describing a run must never break it. No output path, or any failure building
the manifest, writes nothing and raises nothing — the verification's own pass/fail verdict is the
product, this description of it is not.
"""

from __future__ import annotations

import json
import os
from typing import Any


def pytest_collection_modifyitems(session: Any, config: Any, items: list[Any]) -> None:
    """At collection finish, dump the runner's node-ID basis for the parent to freeze on (#58)."""
    out = os.environ.get("WESKER_MANIFEST_OUT")
    if not out:
        return
    try:
        from .session_manifest import capture_manifest

        manifest = capture_manifest(session, config, list(items))
        payload = {
            "node_basis": [
                [it.node_id, it.origin_digest] for it in manifest.items if it.node_id
            ],
            "conflicting_modules": list(manifest.conflicting_modules),
            "collection_errors": list(manifest.collection_errors),
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:  # noqa: BLE001 — a manifest that raises must not fail the verification
        pass
