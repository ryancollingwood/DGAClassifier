import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from src.data import load_data
from src.pipeline.steps import preprocess
from src.pipeline.steps import post_process_cleanup


def test_preprocessing_pipeline():
    """
    Verify we can read in and preprocess the data as expected.

    :return:
    """
    print("\ntest_preprocessing_pipeline")

    test_data_path = "integrationtests/test_data.csv"
    expected_output_path = "integrationtests/test_preprocessing_expected.csv"

    df = load_data(test_data_path, ["domain", "class"])

    pipeline = Pipeline([
        preprocess(),
    ])

    pipeline_output = pipeline.transform(df)
    column_names = pipeline["preprocess"].get_feature_names()
    print("column_names", column_names)

    try:
        assert(column_names == ['class', 'domain_normed'])
    except AssertionError:
        pytest.fail("Didn't get the expected `get_feature_names` from pipeline")

    try:
        assert(isinstance(pipeline_output, np.ndarray))
    except AssertionError:
        pytest.fail("Didn't get expected type from pipeline")

    print(pipeline_output)

    result_df = pd.DataFrame(
        pipeline_output,
        columns=column_names
    )

    result_df = post_process_cleanup(result_df)

    expected_df = pd.read_csv(expected_output_path)

    try:
        pd.testing.assert_frame_equal(
            result_df, expected_df, check_dtype=False
        )
    except AssertionError:
        pytest.fail("Data resulting from transformation did not match expected.")

    return pipeline
