"""A replayed cached negative must not be admitted for a Hypothesis test (§2.2 a-2).

`replayed_negative_admission` (the pure decision) is pinned by its generated golden; these assert the
IMPURE half — `_is_property_test`, the detector that feeds it. A property test's covered lines vary
with the example seed, so a cached non-reach is not safely replayable and must degrade to `unknown`
(re-trace). Hypothesis is not a test dependency here, so we assert against SYNTHETIC markers — which
also pins exactly what the detector inspects, independent of Hypothesis's own API.
"""

from Wesker.trace_cache import _is_property_test, replayed_negative_admission


def _plain():
    pass


def test_plain_function_is_not_a_property_test():
    assert _is_property_test(_plain) is False


def test_is_hypothesis_test_marker_detected():
    def fn():
        pass

    fn.is_hypothesis_test = True  # Hypothesis sets this on the @given wrapper
    assert _is_property_test(fn) is True


def test_hypothesis_handle_attr_detected():
    def fn():
        pass

    fn.hypothesis = object()  # Hypothesis stashes its handle here
    assert _is_property_test(fn) is True


def test_marker_detected_through_a_wrapper_chain():
    def inner():
        pass

    inner.is_hypothesis_test = True

    def outer():
        pass

    outer.__wrapped__ = inner  # a discovery wrapper OUTSIDE the Hypothesis wrapper
    assert (
        _is_property_test(outer) is True
    )  # callable_source unwraps only one level; we walk all


def test_plain_wrapper_chain_is_not_a_property_test():
    def inner():
        pass

    def outer():
        pass

    outer.__wrapped__ = inner
    assert _is_property_test(outer) is False


def test_a_wrapped_cycle_terminates():
    def a():
        pass

    def b():
        pass

    a.__wrapped__ = b
    b.__wrapped__ = a  # a pathological cycle must not hang the walk
    assert _is_property_test(a) is False


def test_a_property_test_negative_is_never_admitted():
    """The end the detector serves: even a completed outcome at a matching fingerprint degrades."""
    assert replayed_negative_admission(True, True, is_property_test=True) == "unknown"
    assert (
        replayed_negative_admission(True, True, is_property_test=False) == "not_reached"
    )
