import pytest
import pandas as pd
import numpy as np


def test_can_call_vowel_distance_left_right_in_column():
    import src.features
    try:
        src.features.vowel_distance_ratio_left_right_in_column()
    except AttributeError:
        pytest.fail("Couldn't call vowel_distance_ratio_left_right_in_column")
    except Exception:
        pass


def test_vowel_distance_left_right_in_column_raise_value_error_if_distance_function_not_callable():
    from src.features import vowel_distance_ratio_left_right_in_column

    try:
        with pytest.raises(TypeError, match = "`distance_function` must be callable"):
            assert(vowel_distance_ratio_left_right_in_column(pd.Series(["hello"]), "foo"))
    except AssertionError:
        pytest.fail("vowel_distance_ratio_left_right_in_column didn't raise TypeError if noncallable distance_function")


def test_vowel_distance_left_right_in_column_sum_pass():
    from src.features import vowel_distance_ratio_left_right_in_column

    input_values = ["cat", "dog", "mouse", "lettuce"]
    input_series = pd.Series(input_values)

    expected_values = [
        0.6,
        0.6,
        0.4,
        0.7,
    ]
    expected_series = pd.Series(expected_values)

    result = vowel_distance_ratio_left_right_in_column(input_series, np.sum)
    try:
        np.testing.assert_almost_equal(result.values, expected_series.values, decimal=1, verbose=True)
    except AssertionError:
        pytest.fail("vowel_distance_ratio_left_right_in_column did not return expected summed values")


def test_vowel_distance_left_right_in_column_mean_pass():
    from src.features import vowel_distance_ratio_left_right_in_column

    input_values = [
        "cat", "dog", "mouse", "lettuce",
    ]
    input_series = pd.Series(input_values)

    expected_values = [
        0.22,
        0.22,
        0.08,
        0.10,
    ]
    expected_series = pd.Series(expected_values)

    result = vowel_distance_ratio_left_right_in_column(input_series, np.mean)
    try:
        np.testing.assert_almost_equal(result.values, expected_series.values, decimal=2, verbose=True)
    except AssertionError:
        pytest.fail("vowel_distance_ratio_left_right_in_column did not return expected summed values")


def test_vowel_distance_left_right_in_column_mean_clip_ratio():
    from src.features import vowel_distance_ratio_left_right_in_column

    input_values = [
        "zz", "zzzzzzz", "zzzzzzzzz", "zzzzzzzz",
        "zzzzzzzze"
    ]
    input_series = pd.Series(input_values)

    expected_values = [
        1.0,
        1.0,
        1.0,
        1.0,
        0.44
    ]
    expected_series = pd.Series(expected_values)

    result = vowel_distance_ratio_left_right_in_column(input_series, np.mean)
    try:
        np.testing.assert_almost_equal(result.values, expected_series.values, decimal=2, verbose=True)
    except AssertionError:
        pytest.fail("vowel_distance_ratio_left_right_in_column did not clip large ratio values")


def test_vowel_distance_left_right_in_column_mode_pass():
    from src.features import vowel_mode_distance_left_right_in_column

    input_values = [
        "cat", "dog", "mouse", "lettuce", "zzzzzzzze",
    ]
    input_series = pd.Series(input_values)

    expected_values = [
        0.33,
        0.33,
        0.00,
        0.00,
        1.00,
    ]
    expected_series = pd.Series(expected_values)

    result = vowel_mode_distance_left_right_in_column(input_series)
    try:
        np.testing.assert_almost_equal(result.values, expected_series.values, decimal=2, verbose=True)
    except AssertionError:
        pytest.fail("vowel_distance_ratio_left_right_in_column did not return expected summed values")
