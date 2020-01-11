from typing import List
import numpy as np
import pandas as pd
from src.pipeline import DataFrameColumnTransformer
from src.features import variety_characters


class CharacterVarietyRatioTransformer(DataFrameColumnTransformer):
    def __init__(self, column_suffix: str, in_columns: List[str], filter_out: List[str] = None):
        self.filter_out = None

        super().__init__(
            column_suffix,
            in_columns,
            filter_out = filter_out
        )

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        result = series.fillna("").astype(str).apply(lambda x: variety_characters(x, self.filter_out))
        result = result.values.round(2)
        return result, y

    def fit(self, X, y = None):
        return self
