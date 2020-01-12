import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.pipeline import pipeline_step_feature_generation


def test_feature_generation():
    print("\ntest_feature_generation")

    test_data_path = "integrationtests/test_preprocessing_expected.csv"
    expected_output_path = "integrationtests/test_feature_generation_expected.csv"

    in_data_df = pd.read_csv(test_data_path)

    pipeline = Pipeline([
        pipeline_step_feature_generation()
    ])

    pipeline_output = pipeline.transform(in_data_df[["domain_normed"]])

    try:
        assert(isinstance(pipeline_output, np.ndarray))
    except AssertionError:
        pytest.fail("`pipeline_output` was not of expected type `np.ndarray`")

    column_names = pipeline["feature_generation"].get_feature_names()

    expected_column_names = [
        'digit_ratio__domain_normed_digit_ratio',
        'len__domain_normed_len',
        'vowel_distance_mode_ratio__domain_normed_vowel_distance_mode_ratio',
        'vowel_distance_std_ratio__domain_normed_vowel_distance_std_ratio',
        'vowel_ratio__domain_normed_vowel_ratio',
        'consonants_variety_ratio__domain_normed_consonants_variety_ratio',
        'character_pairs__domain_normed_character_pair_12',
        'character_pairs__domain_normed_character_pair_36',
        'character_pairs__domain_normed_character_pair_al',
        'character_pairs__domain_normed_character_pair_an',
        'character_pairs__domain_normed_character_pair_ar',
        'character_pairs__domain_normed_character_pair_ct',
        'character_pairs__domain_normed_character_pair_di',
        'character_pairs__domain_normed_character_pair_en',
        'character_pairs__domain_normed_character_pair_er',
        'character_pairs__domain_normed_character_pair_es',
        'character_pairs__domain_normed_character_pair_ff',
        'character_pairs__domain_normed_character_pair_ga',
        'character_pairs__domain_normed_character_pair_gg',
        'character_pairs__domain_normed_character_pair_he',
        'character_pairs__domain_normed_character_pair_in',
        'character_pairs__domain_normed_character_pair_jj',
        'character_pairs__domain_normed_character_pair_le',
        'character_pairs__domain_normed_character_pair_li',
        'character_pairs__domain_normed_character_pair_ma',
        'character_pairs__domain_normed_character_pair_me',
        'character_pairs__domain_normed_character_pair_ne',
        'character_pairs__domain_normed_character_pair_on',
        'character_pairs__domain_normed_character_pair_oo',
        'character_pairs__domain_normed_character_pair_or',
        'character_pairs__domain_normed_character_pair_pv',
        'character_pairs__domain_normed_character_pair_qq',
        'character_pairs__domain_normed_character_pair_ra',
        'character_pairs__domain_normed_character_pair_re',
        'character_pairs__domain_normed_character_pair_ss',
        'character_pairs__domain_normed_character_pair_st',
        'character_pairs__domain_normed_character_pair_te',
        'character_pairs__domain_normed_character_pair_ti',
        'character_pairs__domain_normed_character_pair_to',
        'character_pairs__domain_normed_character_pair_ve',
        'character_pairs__domain_normed_character_pair_vv',
        'character_pairs__domain_normed_character_pair_wc',
        'character_pairs__domain_normed_character_pair_we',
    ]

    try:
        assert(column_names == expected_column_names)
    except AssertionError:
        pytest.fail("`pipeline['feature_generation'].get_feature_names()` did not return expected values")

    result_df = pd.DataFrame(
        pipeline_output,
        columns=column_names
    )

    expected_df = pd.read_csv(expected_output_path)

    try:
        pd.testing.assert_frame_equal(
            result_df, expected_df, check_dtype=False
        )
    except AssertionError:
        pytest.fail("Feature Generation pipeline did not produce expected results")

