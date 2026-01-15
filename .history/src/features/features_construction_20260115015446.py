import pandas as pd


def charges_per_months(df: pd.DataFrame):
    """
    Docstring for charges_per_months

    :param df: Description
    :type df: pd.DataFrame
    """
    df['charges_per_month'] = df['Total Charges'] / (df['Tenure Months'] + 1)

    return df.head()


def high_monthly_charge(df: pd.DataFrame, threshold: float = 70):
    """
    Docstring for high_monthly_charge

    :param df: Description
    :type df: pd.DataFrame
    :param threshold: Description
    :type threshold: float
    """
    df['high_monthly_charge'] = (
        df['charges_per_month'] > threshold).astype(int)

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
