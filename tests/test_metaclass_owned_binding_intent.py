"""W#38 — a mutant whose binding the owner refuses is NOT INSTALLED; it never ends the file's profile.

THE DEFECT (reproduced 2026-10-25 on Wesker 1addf57). toolz's ``Compose`` defines a property whose
getter is literally named ``__name__``. Installing a mutant of it ran
``setattr(Compose, "__name__", <function>)``; for a class that assignment is taken by ``type``'s own
``__name__`` data descriptor, which accepts only a str, so ``TypeError: can only assign string to
Compose.__name__, not 'function'`` escaped ``_patch_mutant_into_test`` through ``evaluate_mutant`` and
``profile_file`` and ended the whole file's profile with no results for any function in it.

THE CONTRACT. A name the class's metaclass owns as a data descriptor, or any owner that refuses the
assignment, means the mutant cannot be bound: it is not installed, the original stays bound, nothing
is restored that was never installed, there is no fall-through to the bare-name strategies (for
``__name__`` those would rebind the TEST module's own ``__name__``), and every other function in the
file is still measured. Isolated execution reads the mutant ``not_installed``.

W#50 (real toolz, both modes): the refused binding still read ``not_entered`` ("installed"), and
in-process the mutant was credited as a crash kill by ``test_curried.py::test_reduce``, a test that never
touches ``Compose``. The mechanism: a test module whose globals do NOT hold ``Compose`` has no owner to
refuse the binding, so the bare-name strategies ran and found the test module's OWN ``__name__`` global,
replaced it with the mutant, and reported "installed". Two defects follow, both closed here: the bare-name
strategies never bind a name every module namespace carries (``bare_binding_standing``), and the in-process
result records whether anything took the mutant (``install_standing``), so a refused binding is
``not_installed`` in BOTH modes and no unrelated test can be credited with it.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap

import pytest

from Wesker.engine import (
    MutationCategory,
    _execution_guard,
    _metaclass_owns_name,
    _patch_mutant_into_test,
    _unpatch_mutant,
    bare_binding_standing,
    evaluate_mutant,
    generate_mutants,
    install_standing,
)

_SOURCE = textwrap.dedent(
    """
    class Compose:
        def __init__(self, *funcs):
            self.funcs = funcs

        def __call__(self, x):
            for f in reversed(self.funcs):
                x = f(x)
            return x

        @property
        def __name__(self):
            try:
                return "_of_".join(f.__name__ for f in self.funcs)
            except AttributeError:
                return type(self).__name__


    def compose(*funcs):
        return Compose(*funcs)
    """
)


def _load(tmp_path, name="w38_compose"):
    path = tmp_path / f"{name}.py"
    path.write_text(_SOURCE)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, str(path)


def _name_test(module):
    """A test whose globals hold `Compose`, as a real test module's would."""
    namespace = {
        "Compose": module.Compose,
        "compose": module.compose,
        "__name__": "tests.test_w38",
    }
    exec(  # noqa: S102
        "def test_name():\n"
        "    inc = lambda x: x + 1\n"
        "    inc.__name__ = 'inc'\n"
        "    assert compose(inc).__name__ == 'inc'\n",
        namespace,
    )
    return namespace["test_name"], namespace


class _Meta(type):
    pass


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("__name__", True),  # type's data descriptor: the class's own name
        ("__qualname__", True),
        ("__call__", False),  # a non-data slot on type: rebindable like any method
        ("__init__", False),
        ("compose_helper", False),  # absent everywhere on the metaclass
    ],
)
def test_the_metaclass_owns_exactly_its_data_descriptor_names(name, expected):
    assert _metaclass_owns_name(type("Plain", (), {}), name) is expected
    assert _metaclass_owns_name(_Meta("Custom", (), {}), name) is expected


def test_a_non_class_owner_is_never_metaclass_owned(tmp_path):
    module, _ = _load(tmp_path)
    assert _metaclass_owns_name(module, "__name__") is False
    assert _metaclass_owns_name(module.Compose(), "__name__") is False


def test_a_metaclass_owned_name_is_not_installed_and_nothing_moves(tmp_path):
    module, _ = _load(tmp_path)
    original = module.Compose.__dict__["__name__"]
    test_fn, namespace = _name_test(module)
    with _execution_guard() as proof:
        patched, saved, target = _patch_mutant_into_test(
            proof, test_fn, "Compose.__name__", lambda self: "MUTANT"
        )
        _unpatch_mutant(proof, patched, saved, target, "__name__")
    assert (patched, saved, target) == (False, None, None)
    assert module.Compose.__dict__["__name__"] is original
    assert module.Compose.__name__ == "Compose"
    # No fall-through to the bare-name strategies: the test module's own `__name__` is untouched.
    assert namespace["__name__"] == "tests.test_w38"


def test_an_owner_that_refuses_the_assignment_is_not_installed():
    # `int` is immutable: no metaclass descriptor owns `bit_length`, but the assignment is refused.
    namespace = {"int": int}
    exec("def test_fn():\n    assert int(5).bit_length() == 3\n", namespace)  # noqa: S102
    test_fn = namespace["test_fn"]
    with _execution_guard() as proof:
        patched, saved, target = _patch_mutant_into_test(
            proof, test_fn, "int.bit_length", lambda self: 0
        )
    assert (patched, saved, target) == (False, None, None)
    assert (5).bit_length() == 3


def test_evaluate_mutant_returns_instead_of_ending_the_profile(tmp_path):
    module, path = _load(tmp_path)
    tree = ast.parse(_SOURCE)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    getter = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__name__"
    )
    mutant = next(iter(generate_mutants(getter, {MutationCategory.STATE})))
    test_fn, _ = _name_test(module)
    original = module.Compose.__dict__["__name__"]

    result = evaluate_mutant(
        mutant,
        [test_fn],
        original.fget,
        timeout_ms=2000,
        qualname="Compose.__name__",
        source_path=path,
    )

    assert result.killed is False, "a mutant that never ran cannot be killed"
    assert result.entered is False, "it was never installed, so never entered"
    assert module.Compose.__dict__["__name__"] is original
    assert module.Compose.__name__ == "Compose"


def test_the_file_profile_continues_and_isolated_reads_not_installed(
    tmp_path, monkeypatch
):
    from Wesker.ci import profile_file, run_with_live_suite

    (tmp_path / "w38_mod.py").write_text(_SOURCE)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_w38_mod.py").write_text(
        textwrap.dedent(
            """
            from w38_mod import Compose, compose


            def inc(x):
                return x + 1


            def double(x):
                return x * 2


            def test_name():
                assert compose(inc, double).__name__ == "inc_of_double"


            def test_call():
                assert compose(inc, double)(3) == 7
            """
        )
    )
    # The real-toolz shape (#50): a second module whose globals do not hold `Compose` and which never
    # touches it, as `test_curried.py::test_reduce` never touches toolz's.
    (tmp_path / "tests" / "test_w38_unrelated.py").write_text(
        textwrap.dedent(
            """
            def test_unrelated():
                assert isinstance(__name__, str)
                assert "w38" in __name__
            """
        )
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\ntestpaths = ["tests"]\n'
    )
    monkeypatch.chdir(tmp_path)

    def body():
        return {
            mode: {
                r["function_key"]: r
                for r in profile_file(".", "w38_mod.py", isolated=mode)
            }
            for mode in (False, True)
        }

    rows = run_with_live_suite(str(tmp_path), body, target_files=["w38_mod.py"])
    assert rows is not None, "no live session"
    for by_fn in rows.values():
        name_row = by_fn["w38_mod.py::Compose.__name__"]
        # Never a kill or a survivor: the mutants sit outside the denominator, and the run is refused.
        assert name_row["total_mutants"] == 0
        assert name_row["is_gateable"] is False
        # Every other function in the file was still measured.
        assert by_fn["w38_mod.py::Compose.__call__"]["total_killed"] > 0
        assert by_fn["w38_mod.py::compose"]["total_mutants"] > 0
        # Refused in both modes, under the same cause, and credited to no test (#50).
        assert set(name_row["unscored_by"]) == {"not_installed"}
        assert name_row["total_killed"] == 0


def test_a_test_whose_globals_lack_the_owner_does_not_lose_its_own_module_name():
    """#50: no owner in the namespace must not mean "rebind the module's own `__name__`"."""
    namespace = {"__name__": "tests.test_unrelated", "__doc__": "doc"}
    exec("def test_fn():\n    assert __name__\n", namespace)  # noqa: S102
    for name in ("__name__", "__doc__"):
        with _execution_guard() as proof:
            patched, saved, target = _patch_mutant_into_test(
                proof, namespace["test_fn"], f"Compose.{name}", lambda self: "MUTANT"
            )
        assert (patched, saved, target) == (False, None, None)
    assert namespace["__name__"] == "tests.test_unrelated"
    assert namespace["__doc__"] == "doc"


def test_a_bare_function_named_like_an_ordinary_global_is_still_bound():
    namespace = {"helper": lambda: 1}
    exec("def test_fn():\n    assert helper() == 1\n", namespace)  # noqa: S102
    mutant = lambda: 2
    with _execution_guard() as proof:
        patched, saved, target = _patch_mutant_into_test(
            proof, namespace["test_fn"], "helper", mutant
        )
        assert patched is True
        assert namespace["helper"] is mutant
        _unpatch_mutant(proof, patched, saved, target, "helper")
    assert namespace["helper"]() == 1


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("__name__", "module_reserved"),
        ("__doc__", "module_reserved"),
        ("__file__", "module_reserved"),
        ("__spec__", "module_reserved"),
        ("__call__", "bindable"),
        ("__init__", "bindable"),
        ("name", "bindable"),
        ("", "bindable"),
    ],
)
def test_bare_binding_standing(name, code):
    assert bare_binding_standing(name) == code


@pytest.mark.parametrize(
    ("ran", "module", "patched", "injected", "refused", "code"),
    [
        (0, 0, 0, 0, 0, "unobserved"),
        (0, 3, 2, 1, 1, "unobserved"),
        (2, 0, 0, 0, 1, "not_installed"),
        (2, 0, 0, 0, 0, "not_proven"),
        (2, 1, 0, 0, 1, "installed"),
        (2, 0, 1, 0, 0, "installed"),
        (2, 0, 0, 1, 1, "installed"),
        (-1, 0, 0, 0, 1, "unobserved"),
    ],
)
def test_install_standing(ran, module, patched, injected, refused, code):
    assert install_standing(ran, module, patched, injected, refused) == code
