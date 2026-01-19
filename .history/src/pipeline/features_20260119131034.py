import pandas as pd
import numpy as np

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

    df['Tenure_months_log'] = np.log1p(df['Tenure Months'])
    df['Total_Charges_log'] = np.log1p(df['Total Charges'])
    df['no_family'] = ((df['Dependents'] == 'No') &
                       (df['Partner'] == 'No')).astype(int)

    return df
