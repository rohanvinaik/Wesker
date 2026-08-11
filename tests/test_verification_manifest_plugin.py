"""The verification-manifest plugin emits the runner's node-ID basis across a subprocess (issue #58).

A certificate's final verification runs the whole proof suite under the consumer's own pytest in a
real subprocess. The in-process manifest capture sets a ContextVar the parent reads — useless across
a process boundary. This plugin, loaded with `-p Wesker.verification_manifest`, writes each collected
item's `[node_id, content digest]` to the file named by `WESKER_MANIFEST_OUT`, so the parent can
freeze a receipt on the runner's own answer for the complete collection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def test_the_plugin_emits_the_node_basis_json(tmp_path):
    """The payoff: run a real pytest subprocess with the plugin and read back the node-ID basis — one
    item per collected test, each with a non-empty content digest, and a clean identity (no conflicts,
    no collection errors)."""
    (tmp_path / "m.py").write_text("def f(n):\n    return n + 1\n")
    (tmp_path / "test_m.py").write_text(
        "from m import f\n\n\ndef test_f():\n    assert f(1) == 2\n"
    )
    out = tmp_path / "manifest.json"
    env = {**os.environ, "WESKER_MANIFEST_OUT": str(out)}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            ".",
            "-p",
            "Wesker.verification_manifest",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        check=False,
    )
    data = json.loads(out.read_text())
    assert [n for n, _ in data["node_basis"]] == ["test_m.py::test_f"]
    assert data["node_basis"][0][1], (
        "each collected node must carry a non-empty content digest"
    )
    assert data["conflicting_modules"] == [] and data["collection_errors"] == []


def test_no_output_path_is_a_silent_no_op(tmp_path):
    """Without `WESKER_MANIFEST_OUT` the plugin writes nothing and does not fail — describing a run
    must never break it. A plain run collects and passes with the plugin loaded and no env set."""
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    env = {k: v for k, v in os.environ.items() if k != "WESKER_MANIFEST_OUT"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            ".",
            "-p",
            "Wesker.verification_manifest",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, (
        "the plugin must not disturb a normal run when it has no output path"
    )
