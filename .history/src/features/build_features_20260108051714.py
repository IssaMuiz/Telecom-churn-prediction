import pandas as pd


def long_tenure_low_charge(df: pd.DataFrame, tenure_threshold: int = 24, monthly_charge_threshold: int = 70.0, total_charge_threshold: float = 2000.0):
    """Add a binary column 'Is Long Tenure' to indicate if 'Tenure Months' exceeds the threshold.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        threshold (int): The tenure months threshold to classify long tenure.
    """

    df['Is Long Tenure'] = (df['Tenure Months'] > tenure_threshold & (
        df['Monthly Charges'] < monthly_charge_threshold)).astype(int)
    return df
