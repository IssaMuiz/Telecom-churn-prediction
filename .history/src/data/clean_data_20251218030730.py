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

    existing_cols = [col for col in columns if col in df.columns]

    df = df.drop(columns=existing_cols)
    return df


def replace_empty_string(df: pd.DataFrame):
    """Replace empty strings in a pandas DataFrame with NaN values.

    Args:
        df (pd.DataFrame): The DataFrame in which to replace empty strings.
    """

    return df.replace(r'^\s*$', np.nan, regex=True)


def duplicated_rows(df: pd.DataFrame):
    """Check for duplicated rows in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check for duplicated rows.
    """

    return df.duplicated().sum()


def drop_duplicated_rows(df: pd.DataFrame):
    """Drop duplicated rows from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which to drop duplicated rows.
    """

    return df.drop_duplicates()


def checking_missing_values(df: pd.DataFrame):
    """Check for missing values in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check for missing values.
    """

    return df.isnull().sum()
