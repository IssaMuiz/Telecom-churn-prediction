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


def check_missing_values(df: pd.DataFrame):
    """Check for missing values in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check for missing values.
    """

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No missing values found in the DataFrame.")
        return None

    return missing_values


def change_to_numeric(df: pd.DataFrame, column: str):
    """ Convert a non numeric column to a numeric column

    Args:
        df (pd.DataFrame): The DataFrame in which to convert to numeric value
        column: The non numeric column to convert to numeric column

    """
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df[column].info()


def check_value_counts(df: pd.DataFrame):
    """Check the value counts in for the categorical columns to check imbalance data

    Args:
        df (pd.DataFrame): The DataFrame to check for missing values.
    """

    for i in df.select_dtypes(include=['object']).columns:
        counts = df[i].value_counts()
        print(counts)
        print("****" * 10)
