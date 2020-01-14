from typing import List
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(
            self, column_suffix: str,
            in_columns: List[str],
            **kwargs,
    ):
        """
        Base class for transformer steps to be used in a sklearn pipeline
        which works with Pandas Dataframes as `X` inputs

        :param column_suffix: Non-empty string or `None` if a non-empty string
            is specified this value will be appended to the column names
        :param in_columns: which columns of the Dataframe do we want to apply the
            operation to?
        :param kwargs:
        """
        self.in_columns = in_columns
        self.feature_names = None

        try:
            assert(isinstance(in_columns, List))
        except AssertionError:
            raise ValueError("`in_columns` must be a list")

        try:
            assert(len(in_columns) > 0)
        except AssertionError:
            raise ValueError("`in_columns` cannot be empty")

        if column_suffix:
            try:
                assert(isinstance(column_suffix, str))
            except AssertionError:
                raise ValueError("`column_suffix` if specified must be str")

        if column_suffix:
            self.column_suffix = column_suffix.strip()
        else:
            self.column_suffix = column_suffix

        try:
            assert(self.column_suffix != "")
        except AssertionError:
            raise ValueError("`column_suffix` must not be empty string")

        if column_suffix:
            try:
                assert(self.column_suffix[0] != "_")
            except AssertionError:
                raise ValueError("`column_suffix` must not begin with '_'")

        self.__dict__.update(kwargs)

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        raise NotImplementedError("`_transform` not defined derrived DataFrameColumnTransformer")

    def get_feature_names(self):
        return self.feature_names

    def transform(self, in_df: pd.DataFrame, y = None):

        if isinstance(in_df, np.ndarray):
            df = pd.DataFrame(in_df, columns=self.in_columns)
        else:
            try:
                assert(isinstance(in_df, pd.DataFrame))
            except AssertionError:
                raise ValueError(f"`in_df` must be pandas.DataFrame, got {type(in_df)}")

            df = in_df.copy()

        mutated_columns = list()

        for col in self.in_columns:
            if self.column_suffix:
                new_column = f"{col}_{self.column_suffix}"
            else:
                new_column = col


            transform_result, y = self._transform(df[col], y)

            if isinstance(transform_result, pd.Series) or isinstance(transform_result, np.ndarray):
                df[new_column] = transform_result

                mutated_columns.append(new_column)
            elif isinstance(transform_result, pd.DataFrame):
                transform_result_columns = [f"{new_column}_{x}" for x in transform_result.columns]
                transform_result.columns = transform_result_columns
                before_join_len = len(df)

                mutated_columns += transform_result_columns

                # fillna as we may have dropped rows due to thresh-holding
                df = df.join(transform_result, how ="left").fillna(0.0)

                assert(len(df) == before_join_len)
            else:
                raise ValueError(f"Unknown type {type(transform_result)} return from `_transform`")

            if self.column_suffix:
                df = df.drop(col, axis = 1)

        non_mutated_columns = [x for x in list(df.columns) if x not in mutated_columns]

        # keep the columns we modified to the right of the dataframe
        df = df[non_mutated_columns + mutated_columns]

        self.feature_names = list(df.columns)

        if y is not None:
            return df.values, y

        return df.values

    def fit(self, X, y = None):
        return self
