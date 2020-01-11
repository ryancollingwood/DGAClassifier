import pytest
import pandas as pd
import numpy as np


def test_can_call_features_length():
    try:
        import src.features
        src.features.length()
    except AttributeError:
        pytest.fail("Couldn't call features.length")
    except Exception:
        pass


def test_features_length_pass():
    from src.features import length

    input_values = ["Ryan", "baz      ", 123, np.nan, "    ", "", "z"]
    input_series = pd.Series(input_values)

    expected_values = [4, 9, 3, 3, 4, 0, 1]
    expected_series = pd.Series(expected_values)

    try:
        result = length(input_series)
        assert(result.equals(expected_series))
    except AssertionError:
        pytest.fail("features.length didn't return expected result")


def test_features_length_no_side_effects():
    from src.features import length

    input_values = ["Ryan", "baz      ", 123, np.nan, "    ", "", "z"]
    input_series = pd.Series(input_values)
    input_series_copy = input_series.copy(True)

    try:
        length(input_series)
        assert(input_series.equals(input_series_copy))
    except AssertionError:
        pytest.fail("features.length altered input data")


def test_features_length_fail():
    from src.features import length

    input_values = ["Ryan", "baz      ", 123, np.nan, "    ", "", "z"]
    input_series = pd.Series(input_values)

    try:
        result = length(input_series)
        assert(result.equals(input_values))
    except AssertionError:
        return
    except Exception:
        pytest.fail("features.length didn't fail in expected way")

