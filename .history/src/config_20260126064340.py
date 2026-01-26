MODEL_CONFIG = {
    "model_type": "LogisticRegression",
    "threshold": 0.3,
    "random_state": 42,
    "features": {
        "numeric": ["Monthly Charges", "Tenure_months_log", "Total_Charges_log", "no_family"],
        "categorical": ['Gender',
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
    }
}
MODEL_CONFIG_V2 = {
    "model_type": "LogisticRegression",
    "threshold": 0.3,
    "random_state": 42,
    'C': 10,
    'class_weight': 'balanced',
    'penalty': 'l1',
    'solver': 'liblinear',
    "features": {
        "numeric": ["Monthly Charges", "Tenure_months_log", "Total_Charges_log", "no_family"],
        "categorical": ['Gender',
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
    }
}
