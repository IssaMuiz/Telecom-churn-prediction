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

    return df


def high_monthly_charge(df: pd.DataFrame, threshold: float = 70):
    """
    Docstring for high_monthly_charge

    :param df: Description
    :type df: pd.DataFrame
    :param threshold: Description
    :type threshold: float
    """
    df['high_monthly_charge'] = ((df['Contract_Month-to-month'] == 1) &
                                 (df['Monthly Charges'] > threshold)).astype(int)

    return df


def change_ratio(df: pd.DataFrame):
    """
    Docstring for change_ratio

    :param df: Description
    :type df: pd.DataFrame
    """
    df['change_ration'] = df['Monthly Charges'] / \
        (df['charges_per_months'] + 1)

    return df


def is_month_to_month(df: pd.DataFrame):
    """Check if a customer has a month-to-month contract."""
    df['is_month_to_month'] = (df['Contract_Month-to-month'] == 1).astype(int)
    return df


def single_short_tenure(df: pd.DataFrame):
    """Combine single customer with short tenure"""

    df['single_short_tenure'] = ((df['Dependents_No'] == 1) & (
        df['Tenure_months_log'] < 1.3)).astype(int)

    return df


def fiber_monthly(df: pd.DataFrame):
    """month-to-month contract customer that uses fiber internet services"""

    df['fiber_monthly'] = ((df['Internet Service_Fiber optic'] == 1) & (
        df['Contract_Month-to-month'] == 1)).astype(int)
    return df


def no_family(df: pd.DataFrame):
    """Customer feature (no_family) with dependents and no partner."""

    if 'Dependents' in df.columns and 'Partner' in df.columns:
        df['no_family'] = ((df['Dependents_No'] == 0) &
                           (df['Partner_No'] == 0)).astype(int)
    return df


def tenure_months_log(df: pd.DataFrame):
    """Log transformation of Tenure Months to reduce skewness."""
    return np.log1p(df['Tenure Months'])


def total_charges_log(df: pd.DataFrame):
    """Log transformation of Total Charges to reduce skewness."""
    return np.log1p(df['Total Charges'])
