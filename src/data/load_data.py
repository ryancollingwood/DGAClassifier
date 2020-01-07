from typing import List
import pandas as pd


def load_data(csv_file_path: str, subset_columns: List[str] = None) -> pd.DataFrame:
    """
    Load CSV Data and optionally subset to the specified columns

    :param csv_file_path:
    :param subset_columns:
    :return: pandas.DataFrame
    """
    df = pd.read_csv(csv_file_path)

    if subset_columns is not None:
        return df[subset_columns]

    return df

