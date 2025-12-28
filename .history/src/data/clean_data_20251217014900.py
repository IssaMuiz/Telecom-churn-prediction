import pandas as pd
import numpy as np


def data_info(df: pd.DataFrame):
    """Display basic information about a pandas DataFrame.

      Args:
          df (pd.DataFrame): The DataFrame to inspect.
      """

    return df.info()


def drop_unused_columns(df: pd.DataFrame, columns: list[str]):
    """Drop specified unused columns from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which to drop columns.
        columns (list[str]): A list of column names to drop.
    """

    return df.drop(columns=columns)


def replace_empty_string(df: pd.DataFrame):
    """Replace empty strings in a pandas DataFrame with NaN values.

    Args:
        df (pd.DataFrame): The DataFrame in which to replace empty strings.
    """

    return df.replace('^\s*$', np.nan, regex=True)
