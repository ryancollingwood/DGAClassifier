import pytest
import pandas as pd
import numpy as np


def test_can_call_ratio_of_characters_in_list():
    import src.features
    try:
        src.features.ratio_of_characters_in_list()
    except AttributeError:
        pytest.fail("Failed to call ratio_of_characters_in_list")
    except Exception:
        pass


def test_ratio_of_characters_in_list_pass():
    from src.features import ratio_of_characters_in_list

    input_values = ["boy", "toy", "girl", "look", "   book  ", np.nan, "", None, 786, "OOO"]
    input_series = pd.Series(input_values)

    expected_values = [0.33, 0.33, 0.0, 0.5, 0.22, 0, 0, 0.25, 0, 0]
    expected_series = pd.Series(expected_values)

    looking_for = ["o"]

    try:
        result = ratio_of_characters_in_list(input_series, looking_for)
        np.testing.assert_almost_equal(result.values, expected_series.values, decimal = 2, verbose = True)
    except AssertionError:
        pytest.fail("ratio_of_characters_in_list did not produce expected results")


def test_ratio_of_characters_in_list_no_side_effects():
    from src.features import ratio_of_characters_in_list

    input_values = ["boy", "toy", "girl", "look", "   book  ", np.nan, "", None, 786, "OOO"]
    input_series = pd.Series(input_values)
    input_series_copy = input_series.copy(True)

    looking_for = ["o"]

    try:
        ratio_of_characters_in_list(input_series, looking_for)
        assert(input_series.equals(input_series_copy))
    except AssertionError:
        pytest.fail("ratio_of_characters_in_list altered input data")


def test_ratio_of_characters_in_list_failed():
    from src.features import ratio_of_characters_in_list

    input_values = ["boy", "toy", "girl", "look", "   book  ", np.nan, "", None, 786, "OOO"]
    input_series = pd.Series(input_values)

    looking_for = ["o"]

    try:
        result = ratio_of_characters_in_list(input_series, looking_for)
        result.equals(input_values)
    except AssertionError:
        return
    except Exception:
        pytest.fail("ratio_of_characters_in_list did not fail in expected way")
