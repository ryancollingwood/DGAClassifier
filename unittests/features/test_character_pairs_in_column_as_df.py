import pytest
import pandas as pd
import numpy as np


def test_can_call_character_pairs_in_column_as_df():
    import src.features
    try:
        src.features.binarize_character_pairs_in_column_as_df()
    except AttributeError:
        pytest.fail("Couldn't call binarize_character_pairs_in_column_as_df")
    except Exception:
        pass


def test_character_pairs_in_column_as_df_pass():
    from src.features import binarize_character_pairs_in_column_as_df
    input_values = ["dog", "book", "boy"]
    input_series = pd.Series(input_values)

    expected_output = pd.DataFrame(
        {
            "bo": [0, 1, 1],
            "do": [1, 0, 0],
            "og": [1, 0, 0],
            "ok": [0, 1, 0],
            "oo": [0, 1, 0],
            "oy": [0, 0, 1],
        },
        dtype = np.int32
    )

    result = binarize_character_pairs_in_column_as_df(input_series)
    try:
        pd.testing.assert_frame_equal(
            result, expected_output, check_dtype=False
        )
    except AssertionError:
        pytest.fail("binarize_character_pairs_in_column_as_df did not return expected values")


def test_character_pairs_in_column_as_df_with_subset_pass():
    from src.features import binarize_character_pairs_in_column_as_df
    input_values = ["dog", "book", "boy"]
    input_series = pd.Series(input_values)

    expected_output = pd.DataFrame(
        {
            "bo": [0, 1, 1],
            "zz": [0, 0, 0],
        },
        dtype = np.int32
    )

    result = binarize_character_pairs_in_column_as_df(input_series, prefix ="", subset_to_pairs = ["bo", "zz"])
    try:
        pd.testing.assert_frame_equal(
            result, expected_output, check_dtype=False
        )
    except AssertionError:
        pytest.fail("binarize_character_pairs_in_column_as_df did not return expected values")


def test_character_pairs_in_column_as_df_with_prefix_pass():
    from src.features import binarize_character_pairs_in_column_as_df
    input_values = ["dog", "book", "boy"]
    input_series = pd.Series(input_values)

    expected_output = pd.DataFrame(
        {
            "z_bo": [0, 1, 1],
            "z_do": [1, 0, 0],
            "z_og": [1, 0, 0],
            "z_ok": [0, 1, 0],
            "z_oo": [0, 1, 0],
            "z_oy": [0, 0, 1],
        },
        dtype = np.int32
    )

    result = binarize_character_pairs_in_column_as_df(input_series, "z")
    try:
        pd.testing.assert_frame_equal(
            result, expected_output, check_dtype=False
        )
    except AssertionError:
        pytest.fail("binarize_character_pairs_in_column_as_df did not return expected values")


def test_character_pairs_in_column_as_df_with_prefix_and_subset_pass():
    from src.features import binarize_character_pairs_in_column_as_df
    input_values = ["dog", "book", "boy"]
    input_series = pd.Series(input_values)

    expected_output = pd.DataFrame(
        {
            "z_bo": [0, 1, 1],
        },
        dtype = np.int32
    )

    result = binarize_character_pairs_in_column_as_df(input_series, "z", ["bo"])
    try:
        pd.testing.assert_frame_equal(
            result, expected_output, check_dtype = False
        )
    except AssertionError:
        pytest.fail("binarize_character_pairs_in_column_as_df did not return expected values")

