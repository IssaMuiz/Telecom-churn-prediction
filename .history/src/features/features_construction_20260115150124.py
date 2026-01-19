import pandas as pd


def charges_per_months(df: pd.DataFrame):
    """
    Docstring for charges_per_months

    :param df: Description
    :type df: pd.DataFrame
    """
    df['charges_per_month'] = df['Total_Charges_log'] / \
        (df['Tenure_months_log'] + 1)

    return df.head()


def high_monthly_charge(df: pd.DataFrame, threshold: float = 85):
    """
    Docstring for high_monthly_charge

    :param df: Description
    :type df: pd.DataFrame
    :param threshold: Description
    :type threshold: float
    """
    df['high_monthly_charge'] = (
        (df['Total_Charges_log'] / (df['Tenure_months_log'] + 1)) > threshold).astype(int)

    return df.head()


def change_ratio(df: pd.DataFrame):
    """
    Docstring for change_ratio

    :param df: Description
    :type df: pd.DataFrame
    """
    df['change_ration'] = df['Monthly Charges'] / \
        (df['Total Charges'] / (df['Tenure_months_log'] + 1))

    return df.head()


def price_increase_ratio(df: pd.DataFrame):
    """Calculate the ratio of price increase."""
    df['price_increase_ratio'] = df['Monthly Charges'] / \
        (df['charges_per_month'] + 1)
    return df.head()
