import logging
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.pipeline.steps import feature_generation
from src.logging import setup_logging

def test_feature_generation():
    """
    Verify that we generated the expected features after preprocessing

    :return:
    """
    setup_logging(logging.DEBUG)
    logging.debug("test_feature_generation")

    test_data_path = "integrationtests/test_preprocessing_expected.csv"
    expected_output_path = "integrationtests/test_feature_generation_expected.csv"

    logging.debug(f"test_data_path: {test_data_path}")
    logging.debug(f"expected_output_path: {expected_output_path}")
    logging.debug("Loading test data")

    in_data_df = pd.read_csv(test_data_path)

    pipeline = Pipeline([
        feature_generation(["domain"])
    ])

    logging.debug("Applying feature_generation pipeline")

    pipeline_output = pipeline.transform(in_data_df[["domain"]])

    try:
        assert(isinstance(pipeline_output, np.ndarray))
    except AssertionError:
        message = f"`pipeline_output` was not of expected type `np.ndarray`, got: {type(pipeline_output)}"
        logging.exception(message)
        pytest.fail(message)

    column_names = pipeline["feature_generation"].get_feature_names()

    logging.debug(f"column_names: {column_names}")

    expected_column_names = [
        'digit_ratio__domain_digit_ratio',
        'len__domain_len',
        'vowel_distance_mode_ratio__domain_vowel_distance_mode_ratio',
        'vowel_distance_std_ratio__domain_vowel_distance_std_ratio',
        'vowel_ratio__domain_vowel_ratio',
        'consonants_variety_ratio__domain_consonants_variety_ratio',
        'character_pairs__domain_character_pair_12',
        'character_pairs__domain_character_pair_36',
        'character_pairs__domain_character_pair_al',
        'character_pairs__domain_character_pair_an',
        'character_pairs__domain_character_pair_ar',
        'character_pairs__domain_character_pair_ct',
        'character_pairs__domain_character_pair_di',
        'character_pairs__domain_character_pair_en',
        'character_pairs__domain_character_pair_er',
        'character_pairs__domain_character_pair_es',
        'character_pairs__domain_character_pair_ff',
        'character_pairs__domain_character_pair_ga',
        'character_pairs__domain_character_pair_gg',
        'character_pairs__domain_character_pair_he',
        'character_pairs__domain_character_pair_in',
        'character_pairs__domain_character_pair_jj',
        'character_pairs__domain_character_pair_le',
        'character_pairs__domain_character_pair_li',
        'character_pairs__domain_character_pair_ma',
        'character_pairs__domain_character_pair_me',
        'character_pairs__domain_character_pair_ne',
        'character_pairs__domain_character_pair_on',
        'character_pairs__domain_character_pair_oo',
        'character_pairs__domain_character_pair_or',
        'character_pairs__domain_character_pair_pv',
        'character_pairs__domain_character_pair_qq',
        'character_pairs__domain_character_pair_ra',
        'character_pairs__domain_character_pair_re',
        'character_pairs__domain_character_pair_ss',
        'character_pairs__domain_character_pair_st',
        'character_pairs__domain_character_pair_te',
        'character_pairs__domain_character_pair_ti',
        'character_pairs__domain_character_pair_to',
        'character_pairs__domain_character_pair_ve',
        'character_pairs__domain_character_pair_vv',
        'character_pairs__domain_character_pair_wc',
        'character_pairs__domain_character_pair_we',
    ]

    logging.debug(f"expected_column_names: {expected_column_names}")

    try:
        assert(column_names == expected_column_names)
    except AssertionError:
        message = "`pipeline['feature_generation'].get_feature_names()` did not return expected values"
        logging.exception(message)
        pytest.fail(message)

    result_df = pd.DataFrame(
        pipeline_output,
        columns=column_names
    )

    logging.debug("Loading validation DataFrame")

    expected_df = pd.read_csv(expected_output_path)

    try:
        pd.testing.assert_frame_equal(
            result_df, expected_df, check_dtype=False
        )
    except AssertionError:
        message = "Feature Generation pipeline did not produce expected results"
        logging.exception(message)
        pytest.fail(message)

