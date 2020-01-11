from typing import List
import numpy as np
import pandas as pd
from src.pipeline import DataFrameColumnTransformer
from src.features import ratio_of_characters_in_list


class CharacterRatioTransformer(DataFrameColumnTransformer):
    def __init__(self, column_suffix: str, in_columns: List[str],
                 looking_for: List[str]
                 ):

        self.looking_for = looking_for

        super().__init__(
            column_suffix,
            in_columns,
        )

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        result = ratio_of_characters_in_list(series, self.looking_for)
        result = result.values.round(2)
        return result, y

    def fit(self, df, y=None):
        return self
