from typing import List
import numpy as np
import pandas as pd
from src.pipeline import DataFrameColumnTransformer


class TextLengthTransformer(DataFrameColumnTransformer):
    def __init__(self, column_suffix: str, in_columns: List[str], strip: bool = False):
        self.strip = strip

        super().__init__(
            column_suffix,
            in_columns,
        )

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        result = series.astype(str)

        if self.strip:
            result = result.apply(lambda x: x.strip())

        return result.apply(len), y

    def fit(self, X, y = None):
        return self
