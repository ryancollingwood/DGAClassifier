import logging
import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from src.data import load_data
from src.pipeline.steps import preprocess
from src.pipeline.steps import post_process_cleanup
from src.logging import setup_logging


def test_preprocessing_pipeline():
    """
    Verify we can read in and preprocess the data as expected.

    :return:
    """
    setup_logging(logging.DEBUG)
    
    logging.debug("test_preprocessing_pipeline")

    test_data_path = "integrationtests/test_data.csv"
    expected_output_path = "integrationtests/test_preprocessing_expected.csv"
    
    logging.debug(f"test_data_path: {test_data_path}")
    logging.debug(f"expected_output_path: {expected_output_path}")
    logging.debug("Loading test data")
    
    df = load_data(test_data_path, ["domain", "class"])

    pipeline = Pipeline([
        preprocess(),
    ])
    
    logging.debug("Applying pipeline transformations")
    
    pipeline_output = pipeline.transform(df)
    column_names = pipeline["preprocess"].get_feature_names()
    
    logging.debug("Pipeline transformation complete")
    logging.debug(f"column_names: {column_names}")

    try:
        assert(column_names == ['class', 'domain'])
    except AssertionError:
        message = f"Didn't get the expected `get_feature_names` from pipeline, got {column_names}"
        logging.exception(message)
        pytest.fail(message)

    try:
        assert(isinstance(pipeline_output, np.ndarray))
    except AssertionError:
        message = f"Didn't get expected type from pipeline, got {type(pipeline_output)}"
        logging.exception(message)
        pytest.fail(message)

    logging.debug(pipeline_output)
    logging.debug("Creating DataFrame from Pipeline output")

    result_df = pd.DataFrame(
        pipeline_output,
        columns=column_names
    )

    logging.debug("Applying `post_process_cleanup`")

    result_df = post_process_cleanup(result_df)

    logging.debug("Loading validation DataFrame")

    expected_df = pd.read_csv(expected_output_path)

    try:
        pd.testing.assert_frame_equal(
            result_df, expected_df, check_dtype=False
        )
    except AssertionError:
        message = "Data resulting from transformation did not match expected."
        logging.exception(message)
        pytest.fail(message)

    return pipeline
