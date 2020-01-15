import pytest
import pandas as pd
import os.path
from src.model import package_model


def test_package_model():
    """
    test that we can package the classifier program into a binary

    :return:
    """

    print("\ntest_package_model")
    path = "integrationtests"

    test_model_path = f"{path}/models/trained.model"

    package_model(test_model_path, "test_classifier")

