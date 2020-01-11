from typing import List
import numpy as np
import pandas as pd
from src.pipeline import DataFrameColumnTransformer
from src.features import count_column_characters_in_list


class CharacterCountTransformer(DataFrameColumnTransformer):
    def __init__(self, column_suffix: str, in_columns: List[str], looking_for: List[str] = None):
        self.looking_for = None

        super().__init__(
            column_suffix,
            in_columns,
            looking_for = looking_for
        )

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        if self.looking_for is not None:
            result = count_column_characters_in_list(series, self.looking_for)
        else:
            result = series.astype(str).apply(len)

        return result, y

    def fit(self, X, y = None):
        return self

