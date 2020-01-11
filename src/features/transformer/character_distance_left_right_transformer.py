from typing import List, Callable
import numpy as np
import pandas as pd
from src.pipeline import DataFrameColumnTransformer
from src.features import character_distance_left_right, character_distance_left_right_ratio


class CharacterDistanceLeftRightTransformer(DataFrameColumnTransformer):
    def __init__(self, column_suffix: str, in_columns: List[str],
                 looking_for: List[str], agg_callback: Callable = None, as_ratio: bool = False
                 ):

        self.looking_for = [ord(x) for x in looking_for]

        if as_ratio and agg_callback is None:
            raise ValueError("If `as_ratio == True` then `agg_callback` must be specified.")

        self.as_ratio = as_ratio
        self.agg_callback = agg_callback

        super().__init__(
            column_suffix,
            in_columns,
        )

    def _transform(self, series: pd.Series, y: np.ndarray = None):
        agg_callback = self.agg_callback

        if not self.as_ratio:
            result = series.apply(lambda x: character_distance_left_right(x, self.looking_for))

            if agg_callback is not None:
                result = result.apply(agg_callback)
        else:
            result = character_distance_left_right_ratio(series, self.looking_for, agg_callback)

        return result, y

    def fit(self, X, y = None):
        return self
