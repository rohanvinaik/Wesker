"""Wesker#46: the isolated worker's standalone fallback had no package.

The defect: when no collected test imports the target, ``_build_mutant`` loaded it as
``_wesker_mutant_target`` with no parent package. A module-level relative import failed while the module
loaded (the mutant was never built: ``harness_error`` for every mutant), and a function-level one built
and then raised ``ImportError: attempted relative import with no known parent package`` when called,
which reads as a crash kill that the harness caused.

The target is now imported under its real dotted name (a regular or a namespace package, or a flat
module); only a file with no dotted name AND no need of one is loaded standalone; a file that needs a
package it cannot have is not built (``constructed`` False), never a kill.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from Wesker._isolated_worker import standalone_load_standing

_DRIVE = (
    "import json,sys\n"
    "from Wesker._isolated_worker import _evaluate_full\n"
    "print(json.dumps(_evaluate_full(sys.argv[1], sys.argv[2], ['test_sub.py::test_f'], sys.argv[3])))\n"
)


def _run(cwd, target, name, mutant, extra_path=""):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [extra_path, *sys.path] if p and os.path.isdir(p)
    )
    run = subprocess.run(
        [sys.executable, "-c", _DRIVE, str(target), name, mutant],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=env,
    )
    return json.loads(run.stdout.strip().splitlines()[-1])


def _fixture(root, init=True):
    pkg = root / "pkg"
    pkg.mkdir()
    if init:
        (pkg / "__init__.py").write_text("")
    (pkg / "x.py").write_text("OFFSET = 2\n")
    (pkg / "target.py").write_text(
        "from . import x\n\n"
        "def f(a):\n    return a + x.OFFSET\n\n"
        "def g(a):\n    from . import x as y\n    return a * y.OFFSET\n"
    )
    # Reaches the target only through a subprocess: no collected test imports it in-process.
    (root / "test_sub.py").write_text(
        "import subprocess, sys\n\n"
        "def test_f():\n"
        "    out = subprocess.run([sys.executable, '-c', 'print(1)'], capture_output=True, text=True)\n"
        "    assert out.stdout.strip() == '1'\n"
    )
    return pkg / "target.py"


@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize(
    ("name", "mutant"),
    [
        ("f", "def f(a):\n    return a - x.OFFSET\n"),
        ("g", "def g(a):\n    from . import x as y\n    return a + y.OFFSET\n"),
    ],
)
def test_unimported_target_in_a_package_is_built_under_its_package(
    tmp_path, init, name, mutant
):
    target = _fixture(tmp_path, init)
    result = _run(tmp_path, target, name, mutant)
    assert result["constructed"] is True
    assert result["killed_by"] is None
    assert result["rc"] == 0


def test_function_level_relative_import_is_not_a_crash_kill(tmp_path):
    """Only a function-level relative import: it used to build, then raise ImportError when called."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "x.py").write_text("OFFSET = 2\n")
    (pkg / "target.py").write_text(
        "def g(a):\n    from . import x\n    return a * x.OFFSET\n"
    )
    (tmp_path / "test_sub.py").write_text(
        "def test_f():\n    from pkg.target import g\n    assert g(3) == 6\n"
    )
    result = _run(
        tmp_path,
        pkg / "target.py",
        "g",
        "def g(a):\n    from . import x\n    return a + x.OFFSET\n",
    )
    assert result["constructed"] is True
    assert result["killed_by"] == "assertion"


def test_package_not_importable_is_not_constructed_never_a_kill(tmp_path):
    """The package's root is on no sys.path: its name cannot be derived, so no mutant is built."""
    src = tmp_path / "src"
    src.mkdir()
    target = _fixture(src)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "test_sub.py").write_text((src / "test_sub.py").read_text())
    result = _run(elsewhere, target, "f", "def f(a):\n    return a - x.OFFSET\n")
    assert result["constructed"] is False
    assert result["killed_by"] is None
    assert result["installed"] is False


def test_flat_module_off_path_without_relative_imports_still_loads_standalone(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "flat.py").write_text("def f(a):\n    return a + 1\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "test_sub.py").write_text("def test_f():\n    assert True\n")
    result = _run(elsewhere, src / "flat.py", "f", "def f(a):\n    return a - 1\n")
    assert result["constructed"] is True


def test_flat_module_off_path_with_a_relative_import_is_not_constructed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "flat.py").write_text("def f(a):\n    from . import x\n    return a\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "test_sub.py").write_text("def test_f():\n    assert True\n")
    result = _run(
        elsewhere,
        src / "flat.py",
        "f",
        "def f(a):\n    from . import x\n    return a + 1\n",
    )
    assert result["constructed"] is False


@pytest.mark.parametrize(
    ("named", "in_package_dir", "relative", "code"),
    [
        (True, False, False, "real_name"),
        (True, True, True, "real_name"),
        (False, False, False, "standalone"),
        (False, True, False, "unconstructible"),
        (False, False, True, "unconstructible"),
        (False, True, True, "unconstructible"),
    ],
)
def test_standalone_load_standing(named, in_package_dir, relative, code):
    assert standalone_load_standing(named, in_package_dir, relative) == code
