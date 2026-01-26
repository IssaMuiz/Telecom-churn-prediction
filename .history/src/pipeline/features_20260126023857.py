import pandas as pd
import numpy as np


# Define numerical and categorical columns
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


# Define engineered features
def add_engineered_features(df: pd.DataFrame):
    """Add all engineered features to the DataFrame."""

    # Log transformation of Tenure Months
    df['Tenure_months_log'] = np.log1p(df['Tenure Months'])
    # Log transformation of Total Charges
    df['Total_Charges_log'] = np.log1p(df['Total Charges'])
    df['no_family'] = ((df['Dependents'] == 'No') &
                       # Create no_family feature
                       (df['Partner'] == 'No')).astype(int)

    # Drop Tenure Months and Total Charges original columns
    df.drop(columns=['Tenure Months', 'Total Charges'])

    return df  # Return the modified DataFrame
