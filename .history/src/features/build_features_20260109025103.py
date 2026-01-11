import pandas as pd


def long_tenure_low_charge(df: pd.DataFrame, tenure_threshold:  int = 24, monthly_charge_threshold: int = 50, ):
    """Add a binary column 'Long Tenure Low Charge' to indicate if 'Tenure Months' exceeds the threshold.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        tenure_threshold (int): The tenure months threshold to classify long tenure.
        monthly_charge_threshold (int): The monthly charges threshold to classify low charge.
    """

    df['Long Tenure Low Charge'] = ((df['Tenure Months'] > tenure_threshold) &
                                    (df['Monthly Charges'] < monthly_charge_threshold)).astype(int)
    return df
