import pytest


def test_can_call_variety_characters():
    import src.features

    try:
        src.features.variety_characters()
    except AttributeError:
        pytest.fail("Couldn't call variety_characters")
    except Exception:
        pass


def test_variety_characters_no_filtering_pass():
    from src.features import variety_characters

    input_values = [
        "abc",
        "ooo",
        "look",
        "facebook",
    ]

    expect_outputs = [
        1.0,
        1.0,
        0.5,
        0.25,
    ]

    for i, text in enumerate(input_values):
        result = variety_characters(text)
        try:
            assert(result == expect_outputs[i])
        except AssertionError:
            pytest.fail(f"variety_characters did not return expected result: {expect_outputs[i]} got {result}")


def test_variety_characters_no_filtering_fail():
    from src.features import variety_characters

    input_values = [
        "abc",
        "ooo",
        "look",
        "facebook",
    ]

    for text in input_values:
        result = variety_characters(text)
        try:
            assert(result != text)
        except AssertionError:
            return
        except Exception:
            pytest.fail("variety_characters did not fail as expected")


def test_variety_characters_filtering_pass():
    from src.features import variety_characters

    input_values = [
        "abc",
        "ooo",
        "look",
        "facebook",
    ]

    expect_outputs = [
        1.0,
        0.0,
        1.0,
        1.0,
    ]

    for i, text in enumerate(input_values):
        result = variety_characters(text, ["a", "e", "i", "o", "u", "y"])
        try:
            assert(result == expect_outputs[i])
        except AssertionError:
            pytest.fail(
                f"variety_characters with filtering did not return expected result: {expect_outputs[i]} got {result}"
            )


def test_variety_characters_no_side_effects():
    from src.features import variety_characters
    from copy import copy

    text_in = "hello"
    text_in_copy = copy(text_in)

    variety_characters(text_in)
    try:
        assert(text_in == text_in_copy)
    except AssertionError:
        pytest.fail("variety_characters modified input")


def test_variety_characters_filtering_no_side_effects():
    from src.features import variety_characters
    from copy import copy

    text_in = "hello"
    text_in_copy = copy(text_in)

    variety_characters(text_in, ["e"])
    try:
        assert(text_in == text_in_copy)
    except AssertionError:
        pytest.fail("variety_characters modified input")
