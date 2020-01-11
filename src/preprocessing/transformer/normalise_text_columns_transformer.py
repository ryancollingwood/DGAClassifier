from typing import List
import numpy as np
import pandas as pd
from src.pipeline import DataFrameColumnTransformer
from src.preprocessing.column import normalise_text_column


class NormaliseTextColumnsTransformer(DataFrameColumnTransformer):
    def __init__(self, column_suffix: str, in_columns: List[str]):

        super().__init__(
            column_suffix,
            in_columns,
        )

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        result = normalise_text_column(series)
        return result, y

    def fit(self, X, y = None):
        return self

