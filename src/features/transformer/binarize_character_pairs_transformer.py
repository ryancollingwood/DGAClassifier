from typing import List
import numpy as np
import pandas as pd
from src.pipeline import DataFrameColumnTransformer
from src.features import binarize_character_pairs_in_column_as_df


class BinarizeCharacterPairsTransformer(DataFrameColumnTransformer):
    def __init__(self, column_suffix: str, in_columns: List[str], looking_for: List[str] = None):
        self.looking_for = looking_for

        super().__init__(
            column_suffix,
            in_columns,
        )

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        result = binarize_character_pairs_in_column_as_df(
            series,
            subset_to_pairs = self.looking_for
        )
        return result, y

    def fit(self, X, y = None):
        return self
