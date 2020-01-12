import pytest
import pandas as pd
import numpy as np
from src.pipeline import prepare_model_inputs


def test_prepare_model_inputs():
    """
    Verify that our pipeline for creating model inputs works correctly. This pipeline is used both
    in training and prediction

    :return:
    """
    print("\ntest_prepare_model_inputs")

    test_data_path = "integrationtests/test_data.csv"

    in_data_df = pd.read_csv(test_data_path)
    print(in_data_df.head())

    feature_names, X_train, X_test, y_train, y_test = prepare_model_inputs(
        in_data_df, ["domain"], "class", test_size = 0.3, random_state_split=42
    )

    expected_feature_names = [
        'digit_ratio__domain_normed_digit_ratio', 'len__domain_normed_len',
        'vowel_distance_mode_ratio__domain_normed_vowel_distance_mode_ratio',
        'vowel_distance_std_ratio__domain_normed_vowel_distance_std_ratio', 'vowel_ratio__domain_normed_vowel_ratio',
        'consonants_variety_ratio__domain_normed_consonants_variety_ratio',
        'character_pairs__domain_normed_character_pair_12', 'character_pairs__domain_normed_character_pair_36',
        'character_pairs__domain_normed_character_pair_al', 'character_pairs__domain_normed_character_pair_an',
        'character_pairs__domain_normed_character_pair_ar', 'character_pairs__domain_normed_character_pair_ct',
        'character_pairs__domain_normed_character_pair_di', 'character_pairs__domain_normed_character_pair_en',
        'character_pairs__domain_normed_character_pair_er', 'character_pairs__domain_normed_character_pair_es',
        'character_pairs__domain_normed_character_pair_ff', 'character_pairs__domain_normed_character_pair_ga',
        'character_pairs__domain_normed_character_pair_gg', 'character_pairs__domain_normed_character_pair_he',
        'character_pairs__domain_normed_character_pair_in', 'character_pairs__domain_normed_character_pair_jj',
        'character_pairs__domain_normed_character_pair_le', 'character_pairs__domain_normed_character_pair_li',
        'character_pairs__domain_normed_character_pair_ma', 'character_pairs__domain_normed_character_pair_me',
        'character_pairs__domain_normed_character_pair_ne', 'character_pairs__domain_normed_character_pair_on',
        'character_pairs__domain_normed_character_pair_oo', 'character_pairs__domain_normed_character_pair_or',
        'character_pairs__domain_normed_character_pair_pv', 'character_pairs__domain_normed_character_pair_qq',
        'character_pairs__domain_normed_character_pair_ra', 'character_pairs__domain_normed_character_pair_re',
        'character_pairs__domain_normed_character_pair_ss', 'character_pairs__domain_normed_character_pair_st',
        'character_pairs__domain_normed_character_pair_te', 'character_pairs__domain_normed_character_pair_ti',
        'character_pairs__domain_normed_character_pair_to', 'character_pairs__domain_normed_character_pair_ve',
        'character_pairs__domain_normed_character_pair_vv', 'character_pairs__domain_normed_character_pair_wc',
        'character_pairs__domain_normed_character_pair_we'
    ]

    try:
        assert(feature_names == expected_feature_names)
    except AssertionError:
        pytest.fail("`prepare_model_inputs` did not return expected feature names")

    print(feature_names)

    try:
        for x in [X_train, X_test, y_train, y_test]:
            assert(isinstance(x, np.ndarray))
    except AssertionError:
        pytest.fail("`prepare_model_inputs` didn't return expected types")

    expected_X_train = np.loadtxt("integrationtests/test_prepare_model_inputs_X_train.csv", delimiter =',', dtype = np.float64)
    expected_X_test = np.loadtxt("integrationtests/test_prepare_model_inputs_X_test.csv", delimiter =',', dtype = np.float64)
    expected_y_train = np.loadtxt("integrationtests/test_prepare_model_inputs_y_train.csv", delimiter =',', dtype = str)
    expected_y_test = np.loadtxt("integrationtests/test_prepare_model_inputs_y_test.csv", delimiter =',', dtype = str)

    try:
        assert(np.all(X_train.ravel() == expected_X_train))
    except AssertionError:
        pytest.fail("Didn't produce expected `X_train`")

    try:
        assert(np.all(X_test.ravel() == expected_X_test))
    except AssertionError:
        pytest.fail("Didn't produce expected `X_test`")

    try:
        assert(np.all(y_train.ravel() == expected_y_train))
    except AssertionError:
        pytest.fail("Didn't produce expected `y_train`")

    try:
        assert(np.all(y_test.ravel() == expected_y_test))
    except AssertionError:
        pytest.fail("Didn't produce expected `y_test`")


