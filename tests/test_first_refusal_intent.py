"""#55 (refactor): the refusal an evaluation reports names the FIRST test refused.

The rule used to sit inline in ``evaluate_mutant``'s loop, two nesting levels deep; it is now
``first_refusal``. These pin it from intent: a later refused test never overwrites the name of the
first, and the first one is named with its own code.
"""

from Wesker.engine import ORIGINAL_ALSO_FAILS, first_refusal


def test_the_first_refused_test_is_named_with_its_code():
    assert first_refusal("", None, ORIGINAL_ALSO_FAILS, "t.py::a") == (
        ORIGINAL_ALSO_FAILS,
        "t.py::a",
    )


def test_a_later_refusal_never_overwrites_the_first():
    assert first_refusal(
        "nested_measurement", "t.py::a", ORIGINAL_ALSO_FAILS, "t.py::b"
    ) == ("nested_measurement", "t.py::a")


def test_an_empty_first_name_is_still_a_name():
    # A refused test whose id is "" has been refused: absence is None, not an empty string.
    assert first_refusal("x", "", "y", "t.py::b") == ("x", "")
