import pytest


def test_can_create_dataframe_column_transformer():
    import src.pipeline

    try:
        src.pipeline.DataFrameColumnTransformer()
    except AttributeError:
        pytest.fail("Couldn't create DataFrameColumnTransformer")
    except Exception:
        pass


def test_dataframe_columns_transformer_in_columns_type():
    from src.pipeline import DataFrameColumnTransformer

    try:
        with pytest.raises(ValueError, match="`in_columns` must be a list"):
            assert(DataFrameColumnTransformer("suffix", None))
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer did not check type of `in_columns`")


def test_dataframe_columns_transformer_in_columns_has_entries():
    from src.pipeline import DataFrameColumnTransformer

    try:
        with pytest.raises(ValueError, match="`in_columns` cannot be empty"):
            assert(DataFrameColumnTransformer("suffix", list()))
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer did not check if `in_columns` was empty")


def test_dataframe_columns_transformer_suffix_type():
    from src.pipeline import DataFrameColumnTransformer

    try:
        with pytest.raises(ValueError, match="if specified must be str"):
            assert(DataFrameColumnTransformer(1, list("boo")))
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer did not check type of `column_suffix`")


def test_dataframe_columns_transformer_suffix_not_empty():
    from src.pipeline import DataFrameColumnTransformer

    try:
        with pytest.raises(ValueError, match="`column_suffix` must not be empty string"):
            assert(DataFrameColumnTransformer("           ", list("boo")))
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer did not check `column_suffix` is not empty")


def test_dataframe_columns_transformer_allow_none_suffix():
    from src.pipeline import DataFrameColumnTransformer

    result = DataFrameColumnTransformer(None, list("boo"))
    try:
        assert(result.column_suffix is None)
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer did set `column_suffix` to None")


def test_dataframe_columns_transformer_suffix_not_begin_underscore():
    from src.pipeline import DataFrameColumnTransformer

    try:
        with pytest.raises(ValueError, match="`column_suffix` must not begin with '_'"):
            assert(DataFrameColumnTransformer("_suffix", list("boo")))
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer did not check if `column_suffix` begin with '_'")


def test_dataframe_columns_transformer__transform_method_not_implemented():
    from src.pipeline import DataFrameColumnTransformer

    try:
        with pytest.raises(NotImplementedError):
            assert(DataFrameColumnTransformer("suffix", list("boo"))._transform(None))
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer did not return NotImplementedError for `_transform`.")


def test_dataframe_columns_transformer_transform_check_for_pandas_dataframe():
    from src.pipeline import DataFrameColumnTransformer

    try:
        with pytest.raises(ValueError, match="`in_df` must be pandas.DataFrame"):
            assert(DataFrameColumnTransformer("suffix", list("boo")).transform(None))
    except AssertionError:
        pytest.fail("DataFrameColumnTransformer.transform did not check for pandas.DataFrame")

