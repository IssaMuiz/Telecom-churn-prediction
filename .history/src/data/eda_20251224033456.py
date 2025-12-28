import pandas as pd


def data_summary(df: DataFrame):
    """Generate a summary of the DataFrame including data types, 
    non-null counts, and basic statistics.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
    """

    summary = df.describe()

    return summary
