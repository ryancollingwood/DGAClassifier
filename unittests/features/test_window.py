import pytest


def test_can_call_window():
    import src.features
    try:
        src.features.window()
    except AttributeError:
        pytest.fail("Failed to call window")
    except Exception:
        pass


def test_window_two_size_pass():
    from src.features import window

    input_value = "helloworld"
    expected_output = [
        ("h", "e"),
        ("e", "l"),
        ("l", "l"),
        ("l", "o"),
        ("o", "w"),
        ("w", "o"),
        ("o", "r"),
        ("r", "l"),
        ("l", "d"),
    ]

    result = list(window(input_value, 2))
    try:
        assert(result == expected_output)
    except AssertionError:
        pytest.fail("window didn't return expected result with window size of 2")


def test_window_two_size_fail():
    from src.features import window

    input_value = "helloworld"

    result = list(window(input_value, 2))
    try:
        assert(result == input_value)
    except AssertionError:
        return
    except Exception:
        pytest.fail("window didn't fail as expected")


def test_window_raise_value_error_on_empty_input():
    from src.features import window

    input_value = list()

    try:
        with pytest.raises(ValueError):
            list(window(input_value))
    except AssertionError:
        pytest.fail("window didn't raise ValueError for empty input iterable")


def test_window_five_size_pass():
    from src.features import window

    input_value = "helloworld"
    expected_output = [
        ("h", "e", "l", "l", "o"),
        ("e", "l", "l", "o", "w"),
        ("l", "l", "o", "w", "o"),
        ("l", "o", "w", "o", "r"),
        ("o", "w", "o", "r", "l"),
        ("w", "o", "r", "l", "d")
    ]

    result = list(window(input_value, 5))
    try:
        assert(result == expected_output)
    except AssertionError:
        pytest.fail("window didn't return expected result with window size of 5")
