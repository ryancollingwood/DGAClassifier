import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.pipeline import pipeline_step_rescale


def test_rescale():
    print("\ntest_rescale")

    test_data_path = "integrationtests/test_feature_generation_expected.csv"
    expected_output_path = "integrationtests/test_rescale_expected.csv"

    in_data_df = pd.read_csv(test_data_path)

    pipeline = Pipeline([
        pipeline_step_rescale()
    ])

    pipeline_output = pipeline.fit_transform(in_data_df)

    try:
        assert(isinstance(pipeline_output, np.ndarray))
    except AssertionError:
        pytest.fail("`pipeline_output` was not of expected type `np.ndarray`")

    with pytest.raises(AttributeError):
        pipeline["rescale"].get_feature_names()

    result_df = pd.DataFrame(
        pipeline_output,
        columns=in_data_df.columns
    )

    expected_df = pd.read_csv(expected_output_path)

    try:
        pd.testing.assert_frame_equal(
            result_df, expected_df, check_dtype=False
        )
    except AssertionError:
        pytest.fail("Rescaling Pipeline didn't produce expected results")
