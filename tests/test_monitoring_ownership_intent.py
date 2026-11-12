"""#32 / B01 task 2.3 — clauses of the broker's ownership contract that had no test.

THE DEFECT (Detective `openspec/changes/complete-paired-repair/tasks.md` 2.3): "Verify
concurrent/same-event overlap, both cleanup orders, global-mask changes with existing subscribers,
setup rollback, startup cancellation, retirement/inflight delivery and thread-ID reuse." Read against
the tests that exist: both cleanup orders, startup cancellation (delivery admission and discovery
abandonment) and retirement/inflight delivery are tested elsewhere
(`test_monitoring_subscriptions_intent.py`, `test_unstranded_lock_intent.py`,
`test_discovery_failure_contained_intent.py`). Four clauses were built and untested, and are pinned
here: setup ROLLBACK (the code's own docstring promised it; nothing exercised it), CONCURRENT
subscription of the same event on the same code, GLOBAL-mask changes beside existing subscribers, and
thread-ID REUSE under the tracer's ownership rule. Each asserts the process-wide monitoring state is
what it was before, because a leaked mask or tool claim is the failure these clauses guard.
"""

from __future__ import annotations

import sys
import threading

import pytest

from Wesker import monitoring
from Wesker.monitoring import registered_thread, subscribe, tool_id


def _fn(a):
    return a + 1


def _fn_b(a):
    return a * 2


def _state():
    """Everything a leaked subscription would change."""
    tool = tool_id()
    return (
        sys.monitoring.get_events(tool),
        sys.monitoring.get_local_events(tool, _fn.__code__),
        sys.monitoring.get_local_events(tool, _fn_b.__code__),
        dict(monitoring._active_subscriptions),
        {c: list(s) for c, s in monitoring._subscriptions_by_code.items()},
    )


def test_a_failure_part_way_through_setup_rolls_back_every_effect(monkeypatch):
    before = _state()
    real = monitoring._recompute_mask
    calls = {"n": 0}

    def flaky(tool, code):
        calls["n"] += 1
        if calls["n"] == 2:  # the SECOND code object: the first is already applied
            raise RuntimeError("injected setup failure")
        return real(tool, code)

    monkeypatch.setattr(monitoring, "_recompute_mask", flaky)
    with pytest.raises(RuntimeError, match="injected setup failure"):
        subscribe(
            [_fn.__code__, _fn_b.__code__],
            sys.monitoring.events.PY_START,
            lambda *a, **k: None,
        )
    monkeypatch.setattr(monitoring, "_recompute_mask", real)
    assert _state() == before, "a half-installed subscription survived its own failure"
    # The broker still works afterwards, and the failed attempt left no registration behind.
    seen = []
    with subscribe(
        [_fn.__code__],
        sys.monitoring.events.PY_START,
        lambda code, *a, **k: seen.append(code),
    ):
        _fn(1)
    assert seen == [_fn.__code__]
    assert _state() == before


def test_a_setup_failure_beside_a_live_subscriber_leaves_that_subscriber_untouched(
    monkeypatch,
):
    seen = []
    live = subscribe(
        [_fn.__code__],
        sys.monitoring.events.PY_START,
        lambda code, *a, **k: seen.append("live"),
    )
    try:
        mask = sys.monitoring.get_local_events(tool_id(), _fn.__code__)
        real = monitoring._recompute_mask

        def failing(tool, code):
            raise RuntimeError("injected")

        monkeypatch.setattr(monitoring, "_recompute_mask", failing)
        with pytest.raises(RuntimeError):
            subscribe(
                [_fn.__code__], sys.monitoring.events.PY_RETURN, lambda *a, **k: None
            )
        monkeypatch.setattr(monitoring, "_recompute_mask", real)
        # The rollback itself met the same fault; it must still have removed the failed
        # subscription, leaving the live one alone in the registry.
        assert len(monitoring._active_subscriptions) == 1
        assert len(monitoring._subscriptions_by_code[_fn.__code__]) == 1
        assert sys.monitoring.get_local_events(tool_id(), _fn.__code__) == mask
        _fn(1)
        assert seen == ["live"]
    finally:
        live.close()


def test_threads_subscribing_the_same_event_on_the_same_code_all_get_delivery():
    barrier = threading.Barrier(6)
    counts = [0] * 6
    handles = []
    errors = []

    def worker(i):
        try:
            barrier.wait(5)

            def cb(code, *a, **k):
                counts[i] += 1

            handles.append(
                subscribe([_fn.__code__], sys.monitoring.events.PY_START, cb)
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    try:
        assert errors == []
        assert len(handles) == 6
        _fn(1)
        assert counts == [1] * 6, (
            "a concurrently installed passive observer missed its event"
        )
    finally:
        for h in handles:
            h.close()
    tool = tool_id()
    assert sys.monitoring.get_local_events(tool, _fn.__code__) == 0
    _fn(1)
    assert counts == [1] * 6, "delivery continued after every handle closed"


def test_global_mask_changes_leave_existing_subscribers_delivering(tmp_path):
    """A selector subscription turns PY_START on process-wide; adding and removing it must not
    disturb a local subscriber that was already there, and closing it restores the global mask."""
    tool = tool_id()
    global_before = sys.monitoring.get_events(tool)
    seen = []
    local = subscribe(
        [_fn.__code__],
        sys.monitoring.events.PY_START,
        lambda code, *a, **k: seen.append("local"),
    )
    try:
        local_mask = sys.monitoring.get_local_events(tool, _fn.__code__)
        path = tmp_path / "sel.py"
        path.write_text("def g():\n    return 1\n")
        sel = subscribe(
            [], sys.monitoring.events.LINE, lambda *a, **k: None, selectors=[str(path)]
        )
        assert sys.monitoring.get_events(tool) & sys.monitoring.events.PY_START
        _fn(1)
        assert seen == ["local"], "adding a global subscriber disturbed an existing one"
        sel.close()
        assert sys.monitoring.get_events(tool) == global_before
        assert sys.monitoring.get_local_events(tool, _fn.__code__) == local_mask
        _fn(1)
        assert seen == ["local", "local"]
    finally:
        local.close()
    assert sys.monitoring.get_local_events(tool, _fn.__code__) == 0


def test_a_thread_that_reuses_a_dead_threads_id_is_owned_by_the_test_that_started_it(
    tmp_path,
):
    """Thread ids are reused by the OS. The tracer owns a thread by OBJECT (not in the set alive
    before the test), never by id, so a new thread that lands on a finished thread's id is traced.
    The stimulus is best effort (the id is only known to be reused after the fact); the assertion
    holds either way."""
    import importlib.util
    import os

    from Wesker.line_coverage import trace_suite

    src = tmp_path / "reuse_target.py"
    src.write_text("def f(x):\n    return x + 1\n")
    spec = importlib.util.spec_from_file_location("reuse_target", str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = os.path.realpath(str(src))

    dead_ids = set()
    for _ in range(50):
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        dead_ids.add(t.ident)
    assert all(registered_thread(i) is None for i in dead_ids), (
        "a dead id is still registered"
    )

    reused = []

    def t_reuse():
        for _ in range(200):
            box = []
            t = threading.Thread(target=lambda b=box: b.append(mod.f(1)))
            t.start()
            t.join()
            if t.ident in dead_ids:
                reused.append(t.ident)
                break
        assert box == [2]

    incomplete: dict = {}
    traced = trace_suite([t_reuse], {path}, incomplete_reach=incomplete)
    (test_id,) = traced
    assert traced[test_id][path] == {2}, "a thread on a reused id was not traced"
    assert incomplete == {}
