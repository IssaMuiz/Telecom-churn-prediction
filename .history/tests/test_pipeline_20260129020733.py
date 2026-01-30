import sys
import pandas as pd
import numpy as np
import pytest
from src.pipeline.train import run_pipeline
from src.data_exploratory.split_data import split_features_target, split_data

sys.path.append("..")


@pytest.fixture
def raw_data():
    """
    Docstring for raw_data
    Provides a sample raw DataFrame for testing.
    Returns: pd.DataFrame
    """
    return pd.DataFrame({
        "Tenure Months": [2, 23, 8, 9, 4, 9, 3, 21, 12, 26, 45, 12, 38, 34, 11, 51,
                          60, 15, 17, 39],
        "Gender": ["Male", "Female"] * 10,
        "Senior Citizen": ['Yes', 'No'] * 10,
        "Monthly Charges": [50, 60, 70, 80, 90, 100, 55, 65, 75, 85,
                            95, 105, 52, 62, 72, 82, 92, 102, 57, 67],
        "Total Charges": [1000, 2000, 3000, 4000, 5000, 6000,
                          1100, 2100, 3100, 4100,
                          5100, 6100, 1200, 2200, 3200, 4200,
                          5200, 6200, 1300, 2300],
        "Contract": ['Month-to-Month', 'One Year', 'Two Year', 'One Year', 'Month-to-Month'] * 4,
        "Partner": ['Yes', 'No'] * 10,
        "Dependents": ['Yes', 'No'] * 10,
        "Phone Service": ['Yes', 'No'] * 10,
        "Multiple Lines": ['Yes', 'No'] * 10,
        "Internet Service": ['No', 'DSL', 'Fiber Optic', 'Cable'] * 5,
        "Online Security": ['Yes', 'No'] * 10,
        "Online Backup": ['Yes', 'No'] * 10,
        "Device Protection": ['Yes', 'No'] * 10,
        "Tech Support": ['Yes', 'No'] * 10,
        "Streaming TV": ['Yes', 'No'] * 10,
        "Streaming Movies": ['Yes', 'No'] * 10,
        "Paperless Billing": ['Yes', 'No'] * 10,
        "Payment Method": ['Bank Withdrawal', 'Credit Card', 'Mailed Check', 'Mailed Check'] * 5,
        "Churn Value": [0, 1] * 10
    })


def test_run_pipeline(raw_data):
    """
    Docstring for running pipeline
    """
    X, y, z = split_data(raw_data)
    X_train, y_train = split_features_target(X)
    X_val, y_val = split_features_target(y)

    model = run_pipeline(X_train, y_train)
    pred = model.predict(X_val, y_val)
    assert len(pred) == model(pred)
