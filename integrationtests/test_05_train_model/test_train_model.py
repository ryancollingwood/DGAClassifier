import pytest
import pandas as pd
from src.build import train_model
import os.path


def test_train_model():
    """
    Verify that our pipeline for creating model inputs works correctly. This pipeline is used both
    in training and prediction

    :return:
    """

    print("\ntest_train_model")
    path = "integrationtests"

    test_data_path = f"{path}/test_data.csv"

    in_data_df = pd.read_csv(test_data_path)
    
    print(in_data_df.head())

    train_model(
        test_data_path, ["domain"], "class",
        output_path=f"{path}/models",
        test_size=0.3,
        random_state=42,
        cross_validation_folds=2,
        verbose=2,
    )

    try:
        assert(os.path.exists(f"{path}/models/trained.model"))
    except AssertionError:
        pytest.fail("`train_model` did not save model to expected path")

