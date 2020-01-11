import pytest
import pandas as pd
import numpy as np


def test_can_call_count_characters_in_list():
    try:
        import src.features
        src.features.count_column_characters_in_list()
    except AttributeError:
        pytest.fail("Couldn't call count_characters_in_list")
    except Exception:
        pass


def test_count_characters_in_list_pass():
    from src.features import count_column_characters_in_list

    input_values = ["boy", "toy", "girl", "look", "   book  ", np.nan, "", None, 786]
    input_series = pd.Series(input_values)

    expected_values = [1, 1, 0, 2, 2, 0, 0, 1, 0]
    expected_series = pd.Series(expected_values)

    looking_for = ["o"]

    try:
        result = count_column_characters_in_list(input_series, looking_for)
        assert(result.equals(expected_series))
    except AssertionError:
        pytest.fail("count_characters_in_list didn't produce expected results")


def test_count_characters_in_list_no_side_effects():
    from src.features import count_column_characters_in_list
    from copy import copy

    input_values = ["boy", "toy", "girl", "look", "   book  ", np.nan, "", None, 786]
    input_series = pd.Series(input_values)
    input_series_copy = input_series.copy(True)

    looking_for = ["o"]
    looking_for_copy = copy(looking_for)

    try:
        count_column_characters_in_list(input_series, looking_for)
        assert(input_series.equals(input_series_copy))
        assert(looking_for == looking_for_copy)
    except AssertionError:
        pytest.fail("count_characters_in_list altered input data")


def test_count_characters_in_list_fail():
    from src.features import count_column_characters_in_list

    input_values = ["boy", "toy", "girl", "look", "   book  ", np.nan, "", None, 786]
    input_series = pd.Series(input_values)

    looking_for = ["o"]

    try:
        result = count_column_characters_in_list(input_series, looking_for)
        assert(result.equals(input_series))
    except AssertionError:
        return
    except Exception:
        pytest.fail("count_characters_in_list did not fail as expected")


def test_count_characters_in_list_looking_for_digits():
    from src.features import count_column_characters_in_list

    input_values = [786, 123, 1, 3, 8, 7, 73]
    input_series = pd.Series(input_values)

    expected_values = [1, 1, 0, 1, 0, 1, 2]
    expected_series = pd.Series(expected_values)

    looking_for = [7, 3]

    try:
        result = count_column_characters_in_list(input_series, looking_for)
        assert(result.equals(expected_series))
    except AssertionError:
        pytest.fail("count_characters_in_list didn't produce expected results when looking for digits")

