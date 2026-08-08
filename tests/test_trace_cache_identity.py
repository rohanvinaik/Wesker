"""Two different files must never share a trace-cache key (issue #20).

`targets_fingerprint` keyed on `os.path.basename` plus a content digest, which is not an
identity. Two checkouts of one repo, a vendored copy, or a `src/`+`build/` pair hold
same-named files with byte-identical content — and every one of them collapsed onto a single
key, so a trace measured against one file was served for another. The line numbers look
plausible precisely because the content matched when it was cached.

Canonical-path keying also has to keep the converse: two SPELLINGS of one file (a symlink, a
case-insensitive rename) are the same file and must still agree, which is the identity
`coverage_from_trace` resolves by `st_dev`/`st_ino` when reading a persisted trace back.
"""

from __future__ import annotations

import pytest

from Wesker.trace_cache import targets_fingerprint

_SRC = "def f(n):\n    return n + 1\n"


def _write(directory, name="mod.py", text=_SRC):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text)
    return path


def test_same_named_files_with_identical_content_do_not_share_a_key(tmp_path):
    """The defect. Byte-identical `mod.py` in two checkouts is two files, not one."""
    a = _write(tmp_path / "checkout_a")
    b = _write(tmp_path / "checkout_b")
    assert a.read_bytes() == b.read_bytes(), "the fixture must hold content constant"
    assert targets_fingerprint({str(a)}) != targets_fingerprint({str(b)})


def test_two_spellings_of_one_file_share_a_key(tmp_path):
    """The converse, and the guard against 'fixing' this by keying on the raw path. A symlink
    is not a second file; keying spellings apart would cold-miss every aliased target forever."""
    real = _write(tmp_path / "pkg")
    link = tmp_path / "alias.py"
    try:
        link.symlink_to(real)
    except (
        OSError,
        NotImplementedError,
    ):  # pragma: no cover — platform without symlinks
        pytest.skip("symlinks unavailable")
    assert targets_fingerprint({str(real)}) == targets_fingerprint({str(link)})


def test_editing_a_target_still_voids_its_key(tmp_path):
    """The original contract, unchanged: content is still part of the key, because editing a
    target moves its line numbers and every entry naming it is void."""
    path = _write(tmp_path / "pkg")
    before = targets_fingerprint({str(path)})
    path.write_text(_SRC + "\n\ndef g():\n    return 2\n")
    assert targets_fingerprint({str(path)}) != before


def test_an_unreadable_target_keys_canonically_too(tmp_path):
    """The failure path used the raw path while the success path used a basename, so it was the
    MORE specific of the two — an asymmetry that would have masked the defect for any target
    that could not be read."""
    missing_a = tmp_path / "one" / "gone.py"
    missing_b = tmp_path / "two" / "gone.py"
    assert targets_fingerprint({str(missing_a)}) != targets_fingerprint(
        {str(missing_b)}
    )
