import pandas as pd
import numpy as np


def charges_per_months(df: pd.DataFrame):
    """
    Docstring for charges_per_months

    :param df: Description
    :type df: pd.DataFrame
    """
    df['charges_per_months'] = df['Total_Charges_log'] / \
        (df['Tenure_months_log'] + 1)

    return df.head()


def high_monthly_charge(df: pd.DataFrame, threshold: float = np.log1p(5.5)):
    """
    Docstring for high_monthly_charge

    :param df: Description
    :type df: pd.DataFrame
    :param threshold: Description
    :type threshold: float
    """
    df['high_monthly_charge'] = (
        df['charges_per_months'] > threshold).astype(int)

    return df.head()


def change_ratio(df: pd.DataFrame):
    """
    Docstring for change_ratio

    :param df: Description
    :type df: pd.DataFrame
    """
    df['change_ration'] = df['Monthly Charges'] / \
        (df['charges_per_months'] + 1)

    return df.head()


def is_month_to_month(df: pd.DataFrame):
    """Check if a customer has a month-to-month contract."""
    df['is_month_to_month'] = (df['Contract_Month-to-Month'] == 1).astype(int)
    return df.head()
