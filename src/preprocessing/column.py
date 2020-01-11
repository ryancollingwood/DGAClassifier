import pandas as pd
import numpy as np
from .text import normalise_text_to_only_regex_matches
from .text import normalise_text_to_ascii
from .text import get_regex_for_az_digits_underscores
from .text import get_regex_for_non_whitespace


def normalise_column_to_lowercase(series: pd.Series) -> pd.Series:
    """
    Return the given Series as a String series that has been lower cased.
    This does not attempt to filter out Nothing values e.g. None, np.nan
    they will be returned as lower cased strings.

    :param series: pandas.Series
    :return: pandas.Series
    """
    return series.astype(str).str.lower()


def normalise_column_to_ascii(series: pd.Series) -> pd.Series:
    """
    Return the given Series as a String series where the characters have
    been reduced to ASCII characters only, removing inflections and other
    accents.
    This does not attempt to filter out Nothing values e.g. None, np.nan
    they will be returned as strings.

    :param series: pandas.Series
    :return: pandas.Series
    """
    return series.astype(str).apply(normalise_text_to_ascii)


def normalise_column_az_digits_underscores(series: pd.Series) -> pd.Series:
    """
    Return the given Series as a String Series with all characters that are not one of:
        - lower cased letter `a-z`, does not account for accented characters
        - digits `0-9`, does not account for `.`
        - hyphens `-`

    This does not attempt to filter out Nothing values e.g. None, np.nan
    they will be returned as strings.

    :param series: pandas.Series
    :return: pandas.Series
    """
    re_az_digits_underscores = get_regex_for_az_digits_underscores()

    return series.apply(lambda x: normalise_text_to_only_regex_matches(x, re_az_digits_underscores))


def normalise_column_empty_and_whitespace(series: pd.Series) -> pd.Series:
    """
    Return the given Series as a String Series replacing Nothing values with empty Strings and
    removing Whitespace and other non-printable characters

    :param series: pandas.Series
    :return: pandas.Series
    """
    result = series.copy(True).fillna("")

    re_non_whitespace = get_regex_for_non_whitespace()

    return result.apply(lambda x: normalise_text_to_only_regex_matches(x, re_non_whitespace))


def remove_column_duplicates(series: pd.Series) -> pd.Series:
    """
    Return the given Series with duplicate values removed, keeping the first occurrence of a
    duplicated value. The series is not converted to a specific data type, therefore 1 and "1"
    are treated as different values.

    :param series: pandas.Series
    :return: pandas.Series
    """
    return series.drop_duplicates(keep="first")


def normalise_text_column(series: pd.Series) -> np.ndarray:
    """
    Apply the required processing steps to the given Series
    :param series: pandas.Series
    :return: pandas.Series
    """
    result = series.copy()

    result = normalise_column_to_lowercase(result)
    result = result.apply(normalise_text_to_ascii)
    result = normalise_column_az_digits_underscores(result)
    result = normalise_column_empty_and_whitespace(result).replace({"", np.nan})

    return result.values
