from typing import Tuple
import pandas as pd
import numpy as np
from src.preprocessing import NormaliseTextColumnsTransformer


def pipeline_step_preprocess() -> Tuple[str, NormaliseTextColumnsTransformer]:
    """
    Return a tuple for the step pre-processing step in a Scikit Learn Pipeline

    :return: tuple(str,  NormaliseTextColumnsTransformer)
    """
    return ("preprocess",
            NormaliseTextColumnsTransformer("normed", ["domain"],)
            )


def post_process_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up  after pre-processing step as we my have through the application of transformations:
        - introduced empty values
        - now have duplicated observations

    This will modify the passed in DataFrame

    :param df:
    :return:
    """
    result_df = df.copy(True)
    result_df = result_df.replace({"": np.nan}).dropna(how="any")
    result_df = result_df.drop_duplicates()
    return result_df
