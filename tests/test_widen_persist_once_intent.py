"""The widen persists its trace-cache cells ONCE, not once per single-test micro-batch.

The defect, measured (Detective, 2026-09-05): the item-incremental widen builds one test per step,
and every `build_session_baseline` call wrote the WHOLE persistent cache — all prior cells plus its
own — twice (a checkpoint and a final save, each a full `json.dump` + fsync). At 1,600 traced tests
one save is ~96 ms, so a suite-scale widen spent ~5 minutes rewriting an unchanged file: O(n²) in
the widen's length, for a value that changed by one cell per step.

What the fix is FOR, stated from intent:

- `build_session_baseline(persist=False)` writes nothing and carries its fresh cells and outcomes on
  the returned baseline (`pending_persist`); `LazySessionBaseline.expand(more, persist=False)`
  accumulates them; `flush()` writes them in ONE save — and the disk ends byte-equivalent in content
  to per-step persistence (same cells, same outcomes, same failing/inert names);
- the widen loops in `run_function_profiling` / `run_function_converged` use exactly that: zero
  saves during the widen, one at its end — however many steps it took;
- a build closure that cannot hold persistence back (no `persist`/`carry` parameters — a test's
  bare closure, an older seam) degrades to the per-step save, never to a lost cell; `flush()` on
  nothing pending is a no-op.
"""

from __future__ import annotations

import ast
import importlib.util
import json

import Wesker.trace_cache as trace_cache
from Wesker.engine import (
    _SESSION_BASELINE,
    LazySessionBaseline,
    MutationCategory,
    build_session_baseline,
    run_function_profiling,
)

# ---------------------------------------------------------------- fixtures


def _target_module(tmp_path):
    """A real target module on disk (the trace keys cells by target FILE), imported for the tests."""
    mod = tmp_path / "mod.py"
    mod.write_text(
        "def target(x):\n    if x > 0:\n        return x * 2\n    return -x\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("persist_once_mod", mod)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(mod), module


def _tests(module):
    def test_pos():
        assert module.target(3) == 6

    def test_neg():
        assert module.target(-4) == 4

    def test_zero():
        assert module.target(0) == 0

    def test_unrelated():
        assert 1 + 1 == 2

    return test_pos, test_neg, test_zero, test_unrelated


def _batched_build(root, target_files, all_tests):
    """The production closure's shape (Wesker.ci): forwards `persist` and `carry`."""

    def build(subset=None, fresh=False, persist=True, carry=None):
        return build_session_baseline(
            list(subset) if subset is not None else list(all_tests),
            target_files,
            project_root=str(root),
            fresh=fresh,
            regime_digest="regime-a",
            persist=persist,
            carry=carry,
        )

    return build


def _bare_build(root, target_files, all_tests):
    """A closure WITHOUT the batching parameters — an older seam, or a test's own closure."""

    def build(subset=None, fresh=False):
        return build_session_baseline(
            list(subset) if subset is not None else list(all_tests),
            target_files,
            project_root=str(root),
            fresh=fresh,
            regime_digest="regime-a",
        )

    return build


def _spy_saves(monkeypatch):
    saves: list[str] = []
    real = trace_cache.save

    def spy(project_root, *args, **kwargs):
        saves.append(project_root)
        return real(project_root, *args, **kwargs)

    monkeypatch.setattr(trace_cache, "save", spy)
    return saves


def _blob(root) -> dict:
    return json.loads(
        (root / ".wesker" / "trace_cache.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------- the holder


def test_batched_expands_write_nothing_and_flush_writes_once_with_the_same_content(
    tmp_path, monkeypatch
):
    saves = _spy_saves(monkeypatch)
    # Root A: batched. Root B: per-step (today's default). Both trace the same target module.
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    mod_path, module = _target_module(tmp_path)
    t_pos, t_neg, t_zero, t_un = _tests(module)
    files = {mod_path}

    holder_a = LazySessionBaseline(
        _batched_build(root_a, files, [t_pos, t_neg, t_zero, t_un])
    )
    holder_a.seed([t_pos])
    after_seed = len(saves)  # the seed persists as before (a real trace to protect)
    assert holder_a.expand([t_neg], persist=False) is True
    assert holder_a.expand([t_zero], persist=False) is True
    assert holder_a.expand([t_un], persist=False) is True
    assert len(saves) == after_seed, "a batched widen step must not touch the disk"
    assert holder_a.flush() is True
    assert len(saves) == after_seed + 1, (
        "the flush is ONE save, however many steps were batched"
    )
    assert holder_a.flush() is False  # nothing pending now

    holder_b = LazySessionBaseline(
        _batched_build(root_b, files, [t_pos, t_neg, t_zero, t_un])
    )
    holder_b.seed([t_pos])
    for t in (t_neg, t_zero, t_un):
        assert holder_b.expand([t]) is True  # per-step persistence, the default

    a, b = _blob(root_a), _blob(root_b)
    # The same cells, the same outcome ledger — batching changed WHEN the disk was written, not WHAT.
    assert set(a["entries"]) == set(b["entries"]) and a["entries"] == b["entries"]
    assert set(a["outcomes_observed"]) == set(b["outcomes_observed"])
    assert a["outcome_fingerprints"] == b["outcome_fingerprints"]
    assert a["failing"] == b["failing"] and a["inert_names"] == b["inert_names"]
    # And the widened cells are really there: every traced test that reached the target has a cell.
    assert len(a["entries"]) >= 3


def test_a_bare_closure_degrades_to_per_step_persistence_never_to_a_lost_cell(
    tmp_path, monkeypatch
):
    saves = _spy_saves(monkeypatch)
    mod_path, module = _target_module(tmp_path)
    t_pos, t_neg, _t_zero, _t_un = _tests(module)
    holder = LazySessionBaseline(_bare_build(tmp_path, {mod_path}, [t_pos, t_neg]))
    holder.seed([t_pos])
    after_seed = len(saves)
    assert holder.expand([t_neg], persist=False) is True
    assert len(saves) > after_seed, (
        "without the batching parameters the step persists itself"
    )
    assert holder.flush() is False  # nothing was held back
    assert len(_blob(tmp_path)["entries"]) >= 2


def test_flush_on_an_unbuilt_or_idle_holder_is_a_noop(tmp_path, monkeypatch):
    saves = _spy_saves(monkeypatch)
    mod_path, module = _target_module(tmp_path)
    t_pos, *_ = _tests(module)
    holder = LazySessionBaseline(_batched_build(tmp_path, {mod_path}, [t_pos]))
    assert holder.flush() is False and saves == []
    holder.seed([t_pos])
    n = len(saves)
    assert holder.flush() is False and len(saves) == n


# ---------------------------------------------------------------- through the widen loop

_CATS = {
    MutationCategory.VALUE,
    MutationCategory.ARITHMETIC,
    MutationCategory.SWAP,
    MutationCategory.BOUNDARY,
}
_SRC = "def scoreit(a, b, flag):\n    if flag:\n        return a * 2 + b\n    return a - b\n"


def test_the_widen_loop_persists_once_at_its_end(tmp_path, monkeypatch):
    """Three widen steps (two unrelated tests first, then the one that discharges the obligations):
    zero saves during the widen, exactly one when it ends — the seed's own saves aside."""
    saves = _spy_saves(monkeypatch)
    node = ast.parse(_SRC).body[0]
    assert isinstance(node, ast.FunctionDef)
    # The oracle's shape (`test_widen_matches_full`): exec the source into a namespace and let the
    # tests call the BARE name, so Wesker's mutant installation reaches what the tests resolve. The
    # code object's filename is a REAL file on disk so the trace cache has a target to key cells by.
    src_file = tmp_path / "scoreit_mod.py"
    src_file.write_text(_SRC, encoding="utf-8")
    ns: dict = {}
    exec(compile(_SRC, str(src_file), "exec"), ns)  # noqa: S102 — test fixture source
    original = ns["scoreit"]

    def test_true():
        assert scoreit(1, 2, True) == 4  # noqa: F821 — flag=True branch only

    def test_false():
        assert scoreit(5, 3, False) == 2  # noqa: F821 — flag=False branch only

    def test_ua():
        assert 1 == 1

    def test_ub():
        assert "x".upper() == "X"

    tests = [test_true, test_false, test_ua, test_ub]
    for t in tests:
        t.__globals__["scoreit"] = original

    def _run(holder, **kw):
        tok = _SESSION_BASELINE.set(holder)
        try:
            return run_function_profiling(
                node,
                f"{src_file}::scoreit",
                _CATS,
                tests,
                original,
                max_per_category=0,
                **kw,
            )
        finally:
            _SESSION_BASELINE.reset(tok)

    def _matrix(result):
        return {
            "killed": result.total_killed,
            "survivors": sorted(r.get("mutant_id") for r in result.survivor_records),
            # Killer ids are relativized against each run's own root (`legacy:../..::name`), so two
            # roots spell the same killer differently: compare by the test NAME, the substance.
            "kill_matrix": {
                m: sorted(k.rsplit("::", 1)[-1] for k in ks)
                for m, ks in result.kill_matrix.items()
            },
            "covered_lines": sorted(
                {ln for lines in result.line_coverage.values() for ln in lines}
            ),
        }

    # The FULL baseline, on its own root (a separate cache), as the disposition oracle.
    full_root = tmp_path / "full"
    full_root.mkdir()
    full = _run(LazySessionBaseline(_batched_build(full_root, {str(src_file)}, tests)))

    holder = LazySessionBaseline(_batched_build(tmp_path, {str(src_file)}, tests))
    holder.seed([test_true])
    after_seed = len(saves)
    seeded = _run(holder, widen_tests=[test_ua, test_ub, test_false])
    assert len(saves) == after_seed + 1, (
        "the widen must persist exactly once, at its end"
    )
    # Batching the persistence changed nothing about the verdict: seed + widen still equals full.
    assert _matrix(seeded) == _matrix(full)
    # And the widened cells are on disk after the single flush (the seed's cell and the killer's).
    assert len(_blob(tmp_path)["entries"]) >= 2
