import pandas as pd


def data_info(df: pd.DataFrame):
    """Display basic information about a pandas DataFrame.

      Args:
          df (pd.DataFrame): The DataFrame to inspect.
      """

    return df.info()
