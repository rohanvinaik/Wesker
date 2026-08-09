"""Intent tests for the pytest session manifest (Detective #58).

Written from what the manifest is FOR, not from what it currently emits. The generated suites
alongside these are characterizations — they pin present behaviour, so a wrong value would be
pinned wrong. These state the contract instead.

The load-bearing one is `test_plugins_are_reproducible_across_sessions`. The manifest exists so
a certificate can name the regime it was measured under and a replay can check that regime is
unchanged. A field that differs every process defeats that by construction: the comparison
either always fails or gets ignored, and "ignored" is how a real regime change walks past a
proof.
"""

from __future__ import annotations

import pytest

from Wesker.session_manifest import (
    CollectedItem,
    PytestSessionManifest,
    _digest,
    capture_manifest,
    collection_identity_standing,
    conflicting_module_names,
)


class _Dist:
    def __init__(self, project_name: str, version: str) -> None:
        self.project_name = project_name
        self.version = version


class _PluginManager:
    """Enough of pytest's PluginManager to exercise the identity rules."""

    def __init__(self, name_plugin, distinfo=()) -> None:
        self._name_plugin = name_plugin
        self._distinfo = distinfo

    def list_name_plugin(self):
        return self._name_plugin

    def list_plugin_distinfo(self):
        return self._distinfo


class _Config:
    def __init__(self, pluginmanager) -> None:
        self.pluginmanager = pluginmanager
        self.rootpath = "/proj"
        self.inipath = "/proj/pyproject.toml"

    def getoption(self, _name):
        return "prepend"


def _capture(name_plugin, distinfo=()):
    return capture_manifest(None, _Config(_PluginManager(name_plugin, distinfo)), [])


def test_plugins_are_reproducible_across_sessions():
    """An anonymous plugin must not put a per-process value in the manifest.

    `list_name_plugin` names an instance-registered plugin — which includes Wesker's own
    collector — by stringifying its `id()`. Recorded raw, two identical sessions produced
    different manifests and any digest over `plugins` was unstable.
    """
    anon_a, anon_b = object(), object()
    first = _capture([("4372631232", anon_a), ("anyio", object())])
    second = _capture([("4407692992", anon_b), ("anyio", object())])
    assert first.plugins == second.plugins


def test_an_anonymous_plugin_is_counted_even_though_it_cannot_be_named():
    """Dropping it silently would hide "a plugin appeared that was not here before"."""
    manifest = _capture([("4372631232", object()), ("999", object())])
    assert manifest.plugins == ("<unnamed>:2",)


def test_a_distribution_backed_plugin_records_its_version():
    """#58 asks for plugin identities AND versions; a bare name is half the answer."""
    plugin = object()
    manifest = _capture(
        [("anyio", plugin)], distinfo=[(plugin, _Dist("anyio", "4.12.1"))]
    )
    assert manifest.plugins == ("anyio==4.12.1",)


def test_a_distribution_without_a_version_still_records_its_name():
    plugin = object()
    manifest = _capture([("odd", plugin)], distinfo=[(plugin, _Dist("odd", ""))])
    assert manifest.plugins == ("odd",)


def test_a_broken_pluginmanager_yields_no_plugins_rather_than_failing_collection():
    """The measurement is the product; describing it must never break it."""

    class _Exploding:
        def list_name_plugin(self):
            raise RuntimeError("boom")

        def list_plugin_distinfo(self):
            raise RuntimeError("boom")

    assert capture_manifest(None, _Config(_Exploding()), []).plugins == ()


def test_conflicting_modules_delegates_to_the_pinned_decision():
    """The property is an accessor. If it re-derived the rule, the two could disagree."""
    manifest = PytestSessionManifest(
        module_origins={"pkg.mod": ("/a/mod.py", "/b/mod.py"), "ok": ("/c.py",)}
    )
    assert manifest.conflicting_modules == conflicting_module_names(
        manifest.module_origins
    )
    assert manifest.conflicting_modules == ("pkg.mod",)


@pytest.mark.parametrize(
    "origins, expected",
    [
        ({"m": ("/a.py", "/b.py")}, ("m",)),
        ({"m": ("/a.py",)}, ()),
        ({"m": ("/a.py", "/a.py")}, ()),  # one file seen twice is not two files
        ({"m": ()}, ()),  # a name that never participated
        ({}, ()),
        ({"z": ("/1.py", "/2.py"), "a": ("/3.py", "/4.py")}, ("a", "z")),  # sorted
    ],
)
def test_shadowing_is_two_files_under_one_name_and_nothing_else(origins, expected):
    assert conflicting_module_names(origins) == expected


def test_an_unreadable_file_digests_to_empty_not_to_a_placeholder_hash():
    """Two unreadable files must not compare as identical content."""
    assert _digest("/nonexistent/definitely/not/here.py") == ""


def test_a_readable_file_digests_stably(tmp_path):
    target = tmp_path / "m.py"
    target.write_text("x = 1\n")
    same = tmp_path / "n.py"
    same.write_text("x = 1\n")
    other = tmp_path / "o.py"
    other.write_text("x = 2\n")
    assert _digest(str(target)) == _digest(str(same)) != ""
    assert _digest(str(target)) != _digest(str(other))


def test_collection_errors_are_empty_because_this_hook_cannot_see_them():
    """Not a stub to fill in later — a file that fails to IMPORT never reaches
    `pytest_collection_modifyitems`. Sourcing this field from anywhere else would make an
    always-empty list read as a clean session, which is the opposite of what it means."""
    assert _capture([]).collection_errors == ()


def test_an_item_without_a_path_is_recorded_without_inventing_one():
    class _Item:
        nodeid = "tests/t.py::x"
        path = None
        module = None

    manifest = capture_manifest(None, _Config(_PluginManager([])), [_Item()])
    assert manifest.items == (
        CollectedItem(node_id="tests/t.py::x", origin="", origin_digest="", module=""),
    )


# --------------------------------------------------------------------------------------
# The manifest finally decides something (Detective #58)
# --------------------------------------------------------------------------------------


def test_not_looking_and_looking_clean_are_different_answers():
    """THE reason this is three states and not a bool.

    `last_session_manifest()` had ZERO consumers in either repo — the capture ran every session
    and informed no decision. Wiring it to a gate makes the distinction load-bearing: a run with
    no manifest must keep its previous meaning, while a run whose collection CONFIRMED one file
    per name carries positive evidence the pre-flight prediction cannot give.
    """
    assert collection_identity_standing(False, ()) == "unobserved"
    assert collection_identity_standing(True, ()) == "confirmed"
    assert collection_identity_standing(False, ()) != collection_identity_standing(
        True, ()
    )


def test_a_name_resolving_to_two_files_is_ambiguous():
    assert collection_identity_standing(True, ("pkg.mod",)) == "ambiguous"


def test_conflicts_without_an_observation_cannot_refuse():
    """Defensive: conflicts can only come FROM a manifest, so this pairing should not arise —
    and if it ever does, "we did not look" is the honest answer rather than a refusal built on
    a value with no provenance."""
    assert collection_identity_standing(False, ("pkg.mod",)) == "unobserved"


def test_the_gate_consumes_it():
    """The wiring, not just the decision. `_measurement_gateable` is the single owner both
    ProfilingResult sites route through, so one conjunct covers the exhaustive and converged
    paths alike."""
    from Wesker.engine import _measurement_gateable

    assert _measurement_gateable(True, True, True, True) is True
    assert _measurement_gateable(True, True, True, False) is False
    # An unasked question must not become a refusal.
    assert _measurement_gateable(True, True, True) is True


def test_a_config_whose_invocation_params_lack_args_does_not_break_collection():
    """The module's contract: describing a measurement must never fail one.

    The read used to dereference one `getattr(config, "invocation_params")` call while
    guarding a SECOND, separate call — so the value checked was never the value used, and an
    object present but without `.args` raised from inside a collection hook.
    """

    class _Odd:
        invocation_params = object()  # exists, has no `.args`
        rootpath = "/proj"
        inipath = ""
        pluginmanager = _PluginManager([])

        def getoption(self, _name):
            return "prepend"

    assert capture_manifest(None, _Odd(), []).invocation_args == ()


def test_invocation_args_are_recorded_when_present():
    class _Params:
        args = ("-q", "tests")

    class _Cfg(_Config):
        invocation_params = _Params()

    manifest = capture_manifest(None, _Cfg(_PluginManager([])), [])
    assert manifest.invocation_args == ("-q", "tests")
