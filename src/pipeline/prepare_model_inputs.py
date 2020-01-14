from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from src.pipeline.steps import preprocess
from src.pipeline.steps import post_process_cleanup
from src.pipeline.steps import feature_generation
from src.pipeline.steps import rescale


def pipeline_prepare_model_inputs(columns: List[str]):

    return [
        feature_generation(columns),
        rescale(),
    ]


def preprocess_model_inputs(df: pd.DataFrame, x_cols: List[str], y_col: str, y_col_replacements: Dict[str, object] = None)\
        -> [pd.DataFrame, List[str]]:
    """
    Clean-up messy incomming data

    :param df:
    :param x_cols:
    :param y_col:
    :param y_col_replacements:
    :return: pd.DataFrame, List[str]
    """
    df = df[x_cols + [y_col]]
    X = df[x_cols]
    y = df[y_col]
    if y_col_replacements is not None:
        y = y.replace(y_col_replacements)
    cleanup_pipeline = Pipeline([
        preprocess(),
    ])

    X = cleanup_pipeline.transform(X)
    new_X_cols = cleanup_pipeline["preprocess"].get_feature_names()
    df = pd.DataFrame(X, columns=new_X_cols)
    df[y_col] = y
    df = post_process_cleanup(df)

    return df, new_X_cols


def prepare_model_inputs(
        df: pd.DataFrame, x_cols: List[str], y_col: str,
        y_col_replacements: Dict[str, object] = None,
        test_size: float = 0.3, random_state_split: float = None,
        do_preprocess: bool = True
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For the given DataFrame apply the transformations and replacements
    returning the feature namse and the test/train splits for X and y
        feature_names, X_train, X_test, y_train, y_test

    :param df: Pandas.DataFrame
    :param x_cols: List[str] - column names of the X variables
    :param y_col: str - column name of the y variable
    :param y_col_replacements:
    :param test_size:
    :param random_state_split:
    :param do_preprocess:
    :return: feature_names, X_train, X_test, y_train, y_test
    """

    if do_preprocess:
        df, new_X_cols = preprocess_model_inputs(df, x_cols, y_col, y_col_replacements)
    else:
        new_X_cols = x_cols

    pipeline = Pipeline(pipeline_prepare_model_inputs(new_X_cols))

    X = pipeline.fit_transform(df[new_X_cols], df[y_col])
    y = df[y_col].to_numpy()
    output_X_cols = pipeline["feature_generation"].get_feature_names()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = random_state_split,
        stratify = y,
    )

    return output_X_cols, X_train, X_test, y_train, y_test



