"""The execution-regime digest must bind the config file's CONTENT, not only its path (§2.2 a-1).

A replayed cached negative ("this test does not reach the target") is admissible only under the same
regime that measured it. Keying on the ini PATH alone cannot see an in-place edit to `addopts` /
markers / options at that same path — a config whose bytes changed is a different regime, and serving
a warm verdict across it is exactly the stale-cache soundness hole §2.2 rules out. The content is
captured at BUILD (`capture_manifest`, via `_digest`), so `regime_digest` stays a PURE hash of the
frozen snapshot rather than reading the filesystem from a property.
"""

from types import SimpleNamespace

from Wesker.session_manifest import PytestSessionManifest, capture_manifest


def test_regime_digest_binds_the_captured_config_content():
    # Two manifests identical but for the config-content digest MUST differ.
    common = dict(
        pytest_version="8", rootpath="/r", inipath="/r/pytest.ini", plugins=("p",)
    )
    a = PytestSessionManifest(inicontent_digest="aaaa", **common)
    b = PytestSessionManifest(inicontent_digest="bbbb", **common)
    assert a.regime_digest != b.regime_digest


def test_capture_reads_and_binds_config_content(tmp_path):
    # End to end: an in-place edit to the config file moves the captured digest AND the regime key.
    ini = tmp_path / "pytest.ini"
    ini.write_text("[pytest]\naddopts = -q\n")
    cfg = SimpleNamespace(rootpath=str(tmp_path), inipath=str(ini))
    before = capture_manifest(None, cfg, [])
    assert before.inicontent_digest  # the bytes were hashed into the field
    before_key = before.regime_digest

    ini.write_text(
        "[pytest]\naddopts = -q --strict-markers\n"
    )  # same PATH, different BYTES
    after = capture_manifest(None, cfg, [])
    assert after.inicontent_digest != before.inicontent_digest
    assert after.regime_digest != before_key


def test_no_config_is_still_cacheable():
    # An absent config is valid pytest configuration, not incomplete evidence: still a real key.
    assert PytestSessionManifest(inipath="", inicontent_digest="").regime_digest != ""
