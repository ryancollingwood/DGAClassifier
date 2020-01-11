import pytest
import numpy as np


def test_can_call_character_distance_left_right():
    import src.features
    try:
        src.features.character_distance_left_right()
    except AttributeError:
        pytest.fail("Couldn't call character_distance_left_right")
    except Exception:
        pass


def test_character_distance_left_right_raise_value_error_on_empty_text():
    from src.features import character_distance_left_right

    input_text = ""
    input_ords_to_search_for = [ord(x) for x in ["a", "e", "i", "o", "u", "y"]]

    try:
        with pytest.raises(ValueError, match ="Must pass in valid value for `text`"):
            assert(character_distance_left_right(input_text, input_ords_to_search_for))
    except AssertionError:
        pytest.fail("character_distance_left_right did not raise ValueError for empty text")


def test_character_distance_left_right_raise_value_error_on_empty_ords_to_search_for():
    from src.features import character_distance_left_right

    input_text = "hello"
    input_ords_to_search_for = list()

    try:
        with pytest.raises(ValueError, match = "Must pass in valid value for `ords_to_search_for`"):
            assert(character_distance_left_right(input_text, input_ords_to_search_for))
    except AssertionError:
        pytest.fail("character_distance_left_right did not raise ValueError for empty ords_to_search_for")


def test_character_distance_left_right_raise_value_error_on_non_numeric_ords_to_search_for():
    from src.features import character_distance_left_right

    input_text = "hello"
    input_ords_to_search_for = [x for x in ["e", "o"]]

    try:
        with pytest.raises(TypeError, match = "Invalid data type in `ords_to_search_for` expected Number"):
            assert(character_distance_left_right(input_text, input_ords_to_search_for))
    except AssertionError:
        pytest.fail("character_distance_left_right did not raise ValueError for non numeric ords_to_search_for")


def test_character_distance_left_right_return_len_of_text_if_no_matches():
    from src.features import character_distance_left_right

    input_text = "hello"
    input_ords_to_search_for = [ord(x) for x in ["z", "q"]]
    expected_result = np.array([5])

    result = character_distance_left_right(input_text, input_ords_to_search_for)

    try:
        assert(result == expected_result)
    except AssertionError:
        pytest.fail("character_distance_left_right didn't return length of input text on no matches")


def test_character_distance_left_right_pass():
    from src.features import character_distance_left_right

    input_text = "hello"
    input_ords_to_search_for = [ord(x) for x in ["a", "e", "i", "o", "u", "y"]]

    expected_result = np.array([1, 0, 2, 1, 0])

    result = character_distance_left_right(input_text, input_ords_to_search_for)

    try:
        assert(all(result == expected_result))
    except AssertionError:
        pytest.fail("character_distance_left_right did not return expected value")


def test_character_distance_left_right_fail():
    from src.features import character_distance_left_right

    input_text = "hello"
    input_ords_to_search_for = [ord(x) for x in ["a", "e", "i", "o", "u", "y"]]

    unexpected_result = np.zeros(5) - 1

    result = character_distance_left_right(input_text, input_ords_to_search_for)
    try:
        assert(np.all(result == unexpected_result))
    except AssertionError:
        return
    except Exception:
        pytest.fail("character_distance_left_right did not fail as expected")


def test_character_distance_left_right_no_more_forward_matches():
    from src.features import character_distance_left_right

    input_text = "qwertsdfzxcv"
    input_ords_to_search_for = [ord(x) for x in ["e"]]

    expected_result = np.array([2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    result = character_distance_left_right(input_text, input_ords_to_search_for)

    try:
        assert(all(result == expected_result))
    except AssertionError:
        pytest.fail("character_distance_left_right did not return expected value for no more forward matches")

