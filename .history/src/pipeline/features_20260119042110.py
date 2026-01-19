import pandas as pd
from src.features.features_construction import (
    tenure_months_log, total_charges_log)

num_cols = ['Tenure Months',
            'Monthly Charges',
            'Total Charges',]


cat_cols = ['Gender',
            'Senior Citizen',
            'Partner',
            'Dependents',
            'Phone Service',
            'Multiple Lines',
            'Internet Service',
            'Online Security',
            'Online Backup',
            'Device Protection',
            'Tech Support',
            'Streaming TV',
            'Streaming Movies',
            'Contract',
            'Paperless Billing',
            'Payment Method']


def add_engineered_features(df: pd.DataFrame):
    """Add all engineered features to the DataFrame."""

    df['Tenure_months_log'] = tenure_months_log(df)
    df['Total_Charges_log'] = total_charges_log(df)
    if 'Dependents' in df.columns and 'Partner' in df.columns:
        df['no_family'] = ((df['Dependents_No'] == 0) &
                           (df['Partner_No'] == 0)).astype(int))

            return df
