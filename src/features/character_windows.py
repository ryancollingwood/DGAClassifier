from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from .window import window


def binarize_character_pairs_in_column_as_df(
        series: pd.Series, prefix: str = None, subset_to_pairs: List[str] = None
) -> pd.DataFrame:
    """
    For the given series return a sparsely populated DataFrame denoting which character pairs
    exist in the series.

    It is expected that every value in the series is present

    :param series: pandas.Series
    :param prefix: str - str to prefix the columns of the returned DataFrame
    :param subset_to_pairs: List[str] - Optionally subset the result to only the specified character pairs
        e.g. ["ea", "oo", ee"]
    :return: pandas.DataFrame
    """
    series_windowed = series.fillna("").astype(str).apply(lambda x: ["".join(w) for w in window(x, 2)])
    mlb = MultiLabelBinarizer()

    result_df = pd.DataFrame(
        mlb.fit_transform(series_windowed),
        columns=mlb.classes_,
        index=series.index
    )

    if subset_to_pairs is not None:
        pairs_present = [x for x in list(result_df.columns) if x in subset_to_pairs]
        # ensure the pairs that present in `series` still are present in the output
        for col in [x for x in subset_to_pairs if x not in pairs_present]:
            result_df[col] = np.int32(0)

        result_df = result_df[subset_to_pairs]

    if prefix:
        prefix_columns = [f"{prefix}_{x}" for x in result_df.columns]
        result_df.columns = prefix_columns

    new_columns = sorted(result_df.columns)

    return result_df[new_columns]


def count_character_pairs_in_column_as_df(
        series: pd.Series, prefix: str = None,
        subset_to_pairs: List[str] = None
) -> pd.DataFrame:
    """
    For the given series return a sparsely populated DataFrame denoting the count of character pairs which
    exist in the series.

    It is expected that every value in the series is present

    :param series: pandas.Series
    :param prefix: str - str to prefix the columns of the returned DataFrame
    :param subset_to_pairs: List[str] - Optionally subset the result to only the specified character pairs
        e.g. ["ea", "oo", ee"]
    :return: pandas.DataFrame
    """
    series_windowed_corpus = series.fillna("").astype(str).apply(
        lambda x: " ".join(["".join(w) for w in window(x, 2)])
    ).to_numpy()

    count_vectorizer = CountVectorizer(
        analyzer='word', lowercase=False, tokenizer=None, ngram_range=(1, 1),
    )

    count_windowed = count_vectorizer.fit_transform(series_windowed_corpus)

    result_df = pd.DataFrame(count_windowed.toarray(), columns=count_vectorizer.get_feature_names())

    if subset_to_pairs is not None:
        pairs_present = [x for x in list(result_df.columns) if x in subset_to_pairs]
        # ensure the pairs that present in `series` still are present in the output
        for col in [x for x in subset_to_pairs if x not in pairs_present]:
            result_df[col] = np.int32(0)

        result_df = result_df[subset_to_pairs]

    if prefix:
        prefix_columns = [f"{prefix}_{x}" for x in result_df.columns]
        result_df.columns = prefix_columns

    new_columns = sorted(result_df.columns)

    return result_df[new_columns].fillna(np.int32(0))
