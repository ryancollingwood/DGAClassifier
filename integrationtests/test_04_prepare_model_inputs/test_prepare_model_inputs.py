import logging
import pytest
import pandas as pd
import numpy as np
from src.pipeline import prepare_model_inputs
from src.logging import setup_logging

def test_prepare_model_inputs():
    """
    Verify that our pipeline for creating model inputs works correctly. This pipeline is used both
    in training and prediction

    :return:
    """
    setup_logging(logging.DEBUG)
    logging.debug("test_prepare_model_inputs")

    test_data_path = "integrationtests/test_data.csv"

    logging.debug(f"test_data_path: {test_data_path}")
    logging.debug("Loading test data")

    in_data_df = pd.read_csv(test_data_path)
    print(in_data_df.head())

    logging.debug("Splitting into test and train sets by prepare_model_inputs")

    feature_names, X_train, X_test, y_train, y_test = prepare_model_inputs(
        in_data_df, ["domain"], "class", test_size = 0.3, random_state_split=42
    )

    logging.debug(f"feature_names: {feature_names}")

    expected_feature_names = [
        'digit_ratio__domain_digit_ratio', 'len__domain_len',
        'vowel_distance_mode_ratio__domain_vowel_distance_mode_ratio',
        'vowel_distance_std_ratio__domain_vowel_distance_std_ratio', 'vowel_ratio__domain_vowel_ratio',
        'consonants_variety_ratio__domain_consonants_variety_ratio',
        'character_pairs__domain_character_pair_12', 'character_pairs__domain_character_pair_36',
        'character_pairs__domain_character_pair_al', 'character_pairs__domain_character_pair_an',
        'character_pairs__domain_character_pair_ar', 'character_pairs__domain_character_pair_ct',
        'character_pairs__domain_character_pair_di', 'character_pairs__domain_character_pair_en',
        'character_pairs__domain_character_pair_er', 'character_pairs__domain_character_pair_es',
        'character_pairs__domain_character_pair_ff', 'character_pairs__domain_character_pair_ga',
        'character_pairs__domain_character_pair_gg', 'character_pairs__domain_character_pair_he',
        'character_pairs__domain_character_pair_in', 'character_pairs__domain_character_pair_jj',
        'character_pairs__domain_character_pair_le', 'character_pairs__domain_character_pair_li',
        'character_pairs__domain_character_pair_ma', 'character_pairs__domain_character_pair_me',
        'character_pairs__domain_character_pair_ne', 'character_pairs__domain_character_pair_on',
        'character_pairs__domain_character_pair_oo', 'character_pairs__domain_character_pair_or',
        'character_pairs__domain_character_pair_pv', 'character_pairs__domain_character_pair_qq',
        'character_pairs__domain_character_pair_ra', 'character_pairs__domain_character_pair_re',
        'character_pairs__domain_character_pair_ss', 'character_pairs__domain_character_pair_st',
        'character_pairs__domain_character_pair_te', 'character_pairs__domain_character_pair_ti',
        'character_pairs__domain_character_pair_to', 'character_pairs__domain_character_pair_ve',
        'character_pairs__domain_character_pair_vv', 'character_pairs__domain_character_pair_wc',
        'character_pairs__domain_character_pair_we'
    ]

    logging.debug(f"expected_feature_names: {expected_feature_names}")

    try:
        assert(feature_names == expected_feature_names)
    except AssertionError:
        message = "`prepare_model_inputs` did not return expected feature names"
        logging.exception(message)
        pytest.fail(message)

    testing_column = None
    try:
        for x in [X_train, X_test, y_train, y_test]:
            testing_column = x
            assert(isinstance(x, np.ndarray))
    except AssertionError:
        message = f"`prepare_model_inputs` didn't return expected types, got: {type(testing_column)}"
        logging.exception(message)
        pytest.fail(message)

    test_prepare_model_inputs_X_train = "integrationtests/test_prepare_model_inputs_X_train.csv"
    test_prepare_model_inputs_X_test = "integrationtests/test_prepare_model_inputs_X_test.csv"
    test_prepare_model_inputs_y_train = "integrationtests/test_prepare_model_inputs_y_train.csv"
    test_prepare_model_inputs_y_test = "integrationtests/test_prepare_model_inputs_y_test.csv"

    logging.debug("Loading validation split data sets")
    logging.debug(f"test_prepare_model_inputs_X_train: {test_prepare_model_inputs_X_train}")
    logging.debug(f"test_prepare_model_inputs_X_test: {test_prepare_model_inputs_X_test}")
    logging.debug(f"test_prepare_model_inputs_y_test: {test_prepare_model_inputs_y_test}")

    expected_X_train = np.loadtxt(test_prepare_model_inputs_X_train, delimiter =',', dtype = np.float64)
    expected_X_test = np.loadtxt(test_prepare_model_inputs_X_test, delimiter =',', dtype = np.float64)
    expected_y_train = np.loadtxt(test_prepare_model_inputs_y_train, delimiter =',', dtype = str)
    expected_y_test = np.loadtxt(test_prepare_model_inputs_y_test, delimiter =',', dtype = str)

    try:
        assert(np.all(X_train.ravel() == expected_X_train))
    except AssertionError:
        message = "Didn't produce expected `X_train`"
        logging.exception(message)
        pytest.fail(message)

    try:
        assert(np.all(X_test.ravel() == expected_X_test))
    except AssertionError:
        message = "Didn't produce expected `X_test`"
        logging.exception(message)
        pytest.fail(message)

    try:
        assert(np.all(y_train.ravel() == expected_y_train))
    except AssertionError:
        message = "Didn't produce expected `y_train`"
        logging.exception(message)
        pytest.fail(message)

    try:
        assert(np.all(y_test.ravel() == expected_y_test))
    except AssertionError:
        message = "Didn't produce expected `y_test`"
        logging.exception(message)
        pytest.fail(message)


