from typing import List, Callable
from numbers import Number
from scipy.stats import mode
import numpy as np
import pandas as pd


def character_distance_left_right(text: str, ords_to_search_for: List[int]) -> np.array:
    """
    Going from left to right in `text` what is the distance to the next appearance of
    the ordinal values in `ords_to_search_for` in the case where there are no more forward matches
    the distance is computed as the distance from the last match.

    :param text:
    :param ords_to_search_for:
    :return: numpy.array
    """
    if len(text) == 0:
        raise ValueError("Must pass in valid value for `text`")

    if len(ords_to_search_for) == 0:
        raise ValueError("Must pass in valid value for `ords_to_search_for`")

    if not (all([isinstance(x, Number) for x in ords_to_search_for])):
        raise TypeError("Invalid data type in `ords_to_search_for` expected Number")

    ord_text = np.array([ord(x) for x in text])

    match_indexes = np.argwhere(np.isin(ord_text, ords_to_search_for)).flatten()

    if match_indexes.size == 0:
        return np.array(len(text))

    it = np.nditer(ord_text, flags=['f_index'])
    last_vowel_index = 0

    result = list()
    while not it.finished:
        result_matches = match_indexes - it.index
        forward_matches = result_matches[result_matches >= 0]

        if forward_matches.size > 0:
            result.append(forward_matches.min())
            last_vowel_index = it.index
        else:
            result.append(it.index - last_vowel_index)

        it.iternext()

    return np.array(result)


def character_distance_left_right_ratio(
        series: pd.Series,
        ords_to_search_for: List[int],
        distance_function: Callable) -> np.array:
    """
    Calculate the left to right `distance` in a Series of strings to any of ordinal values as in `ords_to_search_for`
    The `distance_function` must accept a numpy.array to calculate some measure of distance between the characters.

    Does not apply an string transformations to the series, it is assumed you've already preprocessed the Series

    :param series: pandas.Series
    :param ords_to_search_for: List[int] - e.g. [ord(x) for x in "hello"]
    :param distance_function: Callable - e.g. numpy.max
    :return: numpy.array
    """
    if not isinstance(distance_function, Callable):
        raise TypeError("`distance_function` must be callable")

    column_len = series.astype(str).apply(len)
    search_for_ords_distances = series.astype(str).apply(lambda x: character_distance_left_right(x, ords_to_search_for))
    search_for_ords_distances = search_for_ords_distances.apply(distance_function) / column_len
    search_for_ords_distances = search_for_ords_distances.clip(upper=1.0)
    search_for_ords_distances = search_for_ords_distances.fillna(1.0)

    return search_for_ords_distances


def vowel_distance_ratio_left_right_in_column(series: pd.Series, distance_function: Callable) -> pd.Series:
    """
    Calculate the left to right characters distance (as per `distance_function`) expressed as a ratio of the text length
    in a Series of strings looking for vowels:
        i.e. ["a", "e", "i", "o", "u", "y"]

    :param series: pandas.Series
    :param distance_function: distance_function: Callable - e.g. numpy.max
    :return: pd.Series
    """

    vowel_ords = [ord(x) for x in ["a", "e", "i", "o", "u", "y"]]

    return character_distance_left_right_ratio(series, vowel_ords, distance_function)


def get_mode(x: np.array) -> np.float64:
    """
    Helper function to return the mode of a numpy.array.
    If no mode can be determined numpy.nan is returned.

    :param x:
    :return:
    """
    result = mode(x)

    if result[1][0] > 1:
        return mode(x)[0][0]
    else:
        # there was no identifiable mode return np.nan
        return np.nan


def vowel_mode_distance_left_right_in_column(series: pd.Series) -> pd.Series:
    """
    Calculate the left to right characters mode distance expressed as a ratio of the text length
    in a Series of strings looking for vowels:
        i.e. ["a", "e", "i", "o", "u", "y"]

    :param series: pandas.Series
    :return: pandas.Series
    """
    return vowel_distance_ratio_left_right_in_column(series, get_mode)
