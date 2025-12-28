import pandas as pd


def data_info(df: pd.DataFrame):
    """Display basic information about a pandas DataFrame.

      Args:
          df (pd.DataFrame): The DataFrame to inspect.
      """

    print("DataFrame Info:")
    print("-" * 40)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe(include='all'))
