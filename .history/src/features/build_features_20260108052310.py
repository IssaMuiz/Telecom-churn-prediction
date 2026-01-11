import pandas as pd


def long_tenure_low_charge(df: pd.DataFrame, tenure_threshold: float = 24.0, monthly_charge_threshold: float = 50.0, ):
    """Add a binary column 'Is Long Tenure' to indicate if 'Tenure Months' exceeds the threshold.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        tenure_threshold (float): The tenure months threshold to classify long tenure.
        monthly_charge_threshold (float): The monthly charges threshold to classify low charge.
    """

    df['Long Tenure Low Charge'] = (df['Tenure Months'] > tenure_threshold &
                                    df['Monthly Charges'] < monthly_charge_threshold).astype(float)
    return df
