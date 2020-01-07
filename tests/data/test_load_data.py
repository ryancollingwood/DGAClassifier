import pytest
import pandas as pd


def test_can_call_load_data():
    try:
        import src.data
        src.data.load_data()
    except AttributeError:
        pytest.fail("Couldn't call load_data")
    except Exception:
        pass


def test_load_data_returns_dataframe():
    from src.data import load_data
    result = load_data("tests/data/mock_data.csv")

    try:
        assert(isinstance(result, pd.DataFrame))
    except AssertionError:
        pytest.fail("Expected Pandas DataFrame from load_data()")


def test_load_data_subset_pass():
    from src.data import load_data
    result = load_data("tests/data/mock_data.csv", ["number"])

    try:
        assert(list(result.columns) == ["number"])
    except AssertionError:
        pytest.fail("load_data didn't subset the return dataframe")

