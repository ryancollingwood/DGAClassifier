import pandas as pd


def length(series: pd.Series) -> pd.Series:
    """
    For the given Series return a Series detailing the number of characters, regardless of
    original datatype.

    This doesn't account for Nothing values, i.e. `np.nan` will return `3`.
    Nor does it deal with leading or trailing whitespace, i.e. `" z "` will return `3`

    :param series: pandas.Series
    :return: pandas.Series
    """
    return series.astype(str).apply(len)

