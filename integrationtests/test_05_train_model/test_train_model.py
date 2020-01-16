import logging
import pytest
import pandas as pd
from src.model import train_model
import os.path
from src.logging import setup_logging


def test_train_model():
    """
    Verify that our pipeline for creating model inputs works correctly. This pipeline is used both
    in training and prediction

    :return:
    """

    setup_logging(logging.DEBUG)
    logging.info("test_train_model")

    path = "integrationtests"
    test_data_path = f"{path}/test_data.csv"

    logging.info(f"Reading in data from: {test_data_path}")
    in_data_df = pd.read_csv(test_data_path)
    
    print(in_data_df.head())

    output_path = f"{path}/models"
    test_size = 0.3
    random_state = 42
    cross_validation_folds = 2

    logging.info("About to `train_model`")
    logging.info(f"output_path: {output_path}")
    logging.info(f"test_size: {test_size}")
    logging.info(f"random_state: {random_state}")
    logging.info(f"cross_validation_folds: {cross_validation_folds}")

    train_model(
        test_data_path, ["domain"], "class",
        output_path=f"{path}/models",
        test_size=test_size,
        random_state=random_state,
        cross_validation_folds=cross_validation_folds,
        verbose=2,
    )

    try:
        assert(os.path.exists(f"{path}/models/trained.model"))
    except AssertionError:
        message = "`train_model` did not save model to expected path"
        logging.exception(message)
        pytest.fail(message)

