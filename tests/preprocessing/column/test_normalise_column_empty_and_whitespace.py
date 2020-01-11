import pytest
import pandas as pd
import numpy as np


def test_can_call_normalise_column_empty_and_whitespace_values():
    try:
        import src.preprocessing.column
        src.preprocessing.column.normalise_column_empty_and_whitespace()
    except AttributeError:
        pytest.fail("Couldn't call normalise_column_empty_and_whitespace")
    except Exception:
        pass


def test_normalise_column_empty_and_whitespace_pass():
    from src.preprocessing.column import normalise_column_empty_and_whitespace

    input_vaues = [
        "hello world", np.nan, None, "AllGood", "    hey! "
    ]
    input_series = pd.Series(input_vaues)

    expected_values = [
        "helloworld", "", "", "AllGood", "hey!"
    ]
    expected_series = pd.Series(expected_values)

    try:
        result = normalise_column_empty_and_whitespace(input_series)
        assert(all(result == expected_series))
    except AssertionError:
        pytest.fail("normalise_column_empty_and_whitespace did not produce expected result")


def test_normalise_column_empty_and_whitespace_no_side_effects():
    from src.preprocessing.column import normalise_column_empty_and_whitespace

    input_vaues = [
        "hello world", np.nan, None, "AllGood", "    hey! "
    ]
    input_series = pd.Series(input_vaues)
    input_series_copy = input_series.copy(True)

    try:
        normalise_column_empty_and_whitespace(input_series)
        assert(input_series.equals(input_series_copy))
    except AssertionError:
        pytest.fail("normalise_column_empty_and_whitespace altered input data")


def test_normalise_column_empty_and_whitespace_fail():
    from src.preprocessing.column import normalise_column_empty_and_whitespace

    input_vaues = [
        "hello world", np.nan, None, "AllGood", "    hey! "
    ]
    input_series = pd.Series(input_vaues)

    try:
        result = normalise_column_empty_and_whitespace(input_series)
        assert(all(result == input_vaues))
    except AssertionError:
        return
    except Exception:
        pytest.fail("normalise_column_empty_and_whitespace did not fail as expected")

