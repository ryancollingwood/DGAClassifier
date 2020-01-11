from typing import List
from collections import Counter
from copy import copy
import pandas as pd
import numpy as np


def count_column_characters_in_list(series: pd.Series, looking_for: List[str]) -> pd.Series:
    """
    For a Series return the count of characters that match the values in `looking_for`.
    While this is named for counting characters you could still pass in len > 1 items.

    :param series: pandas.Series
    :param looking_for: List[str]
    :return: pandas.Series
    """
    # because we don't trust inputs
    looking_for_as_str = [str(x) for x in looking_for]

    return series.astype(str).apply(lambda x: len([w for w in x if w in looking_for_as_str]))


def variety_characters(s: str, filter_out: List[str] = None) -> float:
    """
    Calculate the variety of characters in a string by calculating the ratio of
    the most frequently occurring character in `s` makes up in all of `s`.
    If all characters are different e.g. `cat` then 1.0 is returned
    If all characters are filtered out or if an empty string is passed 0.0 is returned

    :param s: str text to be evaluated
    :param filter_out: List[str] substring or characters to filter out e.g. vowels
    :return: float: the ratio the most frequently occurring character is of `s`
    """
    s_copy = copy(s)

    if filter_out is not None:
        s_copy = "".join([c for c in list(s_copy) if c not in filter_out])

    if len(s_copy) == 0:
        return 0.0

    counter = Counter(s_copy)
    counts = np.array(list(counter.values()))

    if np.min(counts) == np.max(counts):
        return 1.0

    return np.max(counts) / np.sum(counts)


def consonants_column_variety_ratio(series: pd.Series) -> pd.Series:
    """
    For the given series determine the ratio the most frequently occurring
    consonant makes of the entire word, having filtered out all vowels.
    :param series: pandas.Series
    :return: pandas.Series
    """
    vowels = ["a", "e", "i", "o", "u", "y"]
    return series.fillna("").astype(str).apply(lambda x: variety_characters(x, vowels))
