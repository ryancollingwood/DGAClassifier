from typing import List
import pandas as pd
import numpy as np
from .character_count import count_column_characters_in_list
from .length import length


def ratio_of_characters_in_list(series: pd.Series, looking_for: List[str]) -> pd.Series:
    """
    For the given Series and list of characters to look for determine the ratio of the
    character to look for over the count of characters in the series returning this as a
    Series of numpy.float64

    For empty strings `0` is returned.

    Series containing non string data are evaluated as a Series of strings.

    :param series: pandas.Series
    :param looking_for: List[str]
    :return: pandas.Series
    """
    counts = count_column_characters_in_list(series, looking_for)
    lengths = length(series)

    return (counts / lengths).fillna(0).astype(np.float64)
