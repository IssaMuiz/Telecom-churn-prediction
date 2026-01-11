import pandas as pd


def add_is_long_tenure(df: pd.DataFrame, threshold: int = 24):
    """Add a binary column 'Is Long Tenure' to indicate if 'Tenure Months' exceeds the threshold.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        threshold (int): The tenure months threshold to classify long tenure.
    """

    df['Is Long Tenure'] = (df['Tenure Months'] > threshold).astype(int)
    return df
