"""Two concurrent profiles must not patch and restore across one another (issue #19).

Mutant evaluation monkey-patches a module global, runs a test against it, and restores it. That
is only correct if nothing else touches the same namespace in between, and nothing enforced it:
an MCP request and a CLI run, or two MCP requests, share one interpreter.

MEASURED BEFORE THE FIX, two threads evaluating different mutants of one function, 5/5 runs:

    run 1:  ('b', 0.5, 0.5)     <- b saw A's arithmetic mutant
    run 2:  ('b', 0.5, 2)       <- A's mutant, then the RESTORED ORIGINAL, mid-test
    run 3:  ('b', 0.5, 2)
    run 4:  ('b', 0.5, 0.5)
    run 5:  ('b', 0.5, 0.5)

Thread B's mutant is "replace return with None", so under its own mutant the target returns
None. It observed None in ZERO of five runs. Every verdict thread B recorded was about a body
that was never installed for it -- a survivor of a mutant that never ran, or a kill earned by
another thread's code. Nothing in the result says so; it is indistinguishable from a real
measurement, which is what makes it worse than a crash.

THE ASSERTION IS OWNERSHIP, NOT STABILITY. An earlier version of this test only checked that
the two reads within one test agreed, and it PASSED on the broken code three runs out of five --
because A's mutant sat installed across B's whole test, consistently wrong. Consistency is not
correctness: the question is whether a thread observed ITS OWN mutant.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import threading
import time

import pytest

from Wesker.engine import evaluate_mutant, generate_mutants
from Wesker.filter import filter_categories

_SRC = "def compute(n):\n    return n * 2\n"


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "race_mod.py"
    path.write_text(_SRC)
    spec = importlib.util.spec_from_file_location("race_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["race_mod"] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod, str(path)
    finally:
        sys.modules.pop("race_mod", None)


def _mutants():
    node = ast.parse(_SRC).body[0]
    return generate_mutants(
        node, filter_categories(node, True), max_per_category=0, pass_index=0
    )


def test_a_concurrent_profile_cannot_be_scored_against_another_mutants_body(target):
    """THE regression. Each thread must observe the mutant IT installed, for the whole of its
    own test -- not another thread's, and not the restored original."""
    mod, path = target
    mutants = _mutants()
    assert len(mutants) >= 2, "need two distinct mutants to race"
    a, b = mutants[0], mutants[1]

    seen: dict[str, list[object]] = {}
    guard = threading.Lock()

    def make_test(tag: str):
        def _test() -> None:
            first = mod.compute(1)
            time.sleep(0.05)  # a window another thread could patch or restore inside
            second = mod.compute(1)
            with guard:
                seen.setdefault(tag, []).extend([first, second])

        _test.__name__ = f"test_{tag}"
        return _test

    def worker(tag: str, mutant) -> None:
        evaluate_mutant(
            mutant, [make_test(tag)], mod.compute, qualname="compute", source_path=path
        )

    threads = [
        threading.Thread(target=worker, args=("a", a)),
        threading.Thread(target=worker, args=("b", b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert set(seen) == {"a", "b"}, f"both threads must have run: {seen}"
    for tag, values in seen.items():
        assert len(set(values)) == 1, (
            f"thread {tag} saw the function body CHANGE mid-test: {values} -- "
            "another thread patched or restored inside its window"
        )
    # Ownership: the two threads installed DIFFERENT mutants, so they must not have observed the
    # same body. Equal values mean one thread's patch was live during the other's test.
    assert seen["a"][0] != seen["b"][0], (
        f"both threads observed the same body ({seen['a'][0]!r}) -- "
        "one mutant was scored against the other's code"
    )


def test_the_original_is_restored_after_concurrent_evaluation(target):
    """Serialization must not leave the last mutant installed. The lock changes WHEN a patch
    happens, never WHETHER it is undone."""
    mod, path = target
    mutants = _mutants()

    def _noop() -> None:
        mod.compute(1)

    threads = [
        threading.Thread(
            target=evaluate_mutant,
            args=(m, [_noop], mod.compute),
            kwargs={"qualname": "compute", "source_path": path},
        )
        for m in mutants[:3]
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert mod.compute(1) == 2, (
        "the original body must be back once every evaluation returns"
    )
