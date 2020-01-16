import logging
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.pipeline.steps import rescale
from src.logging import setup_logging

def test_rescale():
    """
    Verify our generated features rescale as we expect

    :return:
    """
    setup_logging(logging.DEBUG)
    logging.debug("\ntest_rescale")

    test_data_path = "integrationtests/test_feature_generation_expected.csv"
    expected_output_path = "integrationtests/test_rescale_expected.csv"

    logging.debug(f"test_data_path: {test_data_path}")
    logging.debug(f"expected_output_path: {expected_output_path}")
    logging.debug("Loading test data")

    in_data_df = pd.read_csv(test_data_path)

    pipeline = Pipeline([
        rescale()
    ])

    logging.debug("Applying rescale pipeline")

    pipeline_output = pipeline.fit_transform(in_data_df)

    try:
        assert(isinstance(pipeline_output, np.ndarray))
    except AssertionError:
        message = f"`pipeline_output` was not of expected type `np.ndarray`, got: {type(pipeline_output)}"
        logging.exception(message)
        pytest.fail(message)

    with pytest.raises(AttributeError):
        pipeline["rescale"].get_feature_names()

    logging.debug("Converting pipeline output to DataFrame")

    result_df = pd.DataFrame(
        pipeline_output,
        columns=in_data_df.columns
    )

    logging.debug("Loading validation data")

    expected_df = pd.read_csv(expected_output_path)

    try:
        pd.testing.assert_frame_equal(
            result_df, expected_df, check_dtype=False
        )
    except AssertionError:
        message = "Rescaling Pipeline didn't produce expected results"
        logging.exception(message)
        pytest.fail(message)
