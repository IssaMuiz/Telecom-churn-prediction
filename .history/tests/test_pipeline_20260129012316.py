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
        "Tenure Months": [np.random.randint(0, 21)],
        "Gender": ["Male", "Female"] * 10,
        "Senior Citizen": [0, 1] * 10,
        "Monthly Charges": [50, 60, 70, 80, 90, 100, 55, 65, 75, 85,
                            95, 105, 52, 62, 72, 82, 92, 102, 57, 67],
        "Total Charges": ["1000", "2000", "3000", "4000", "5000", "6000",
                          "1100", "2100", "3100", "4100",
                          "5100", "6100", "1200", "2200", "3200", "4200",
                          "5200", "6200", "1300", "2300"],
        "Contract": ['Month-to-Month', 'One Year', 'Two Year', 'One Year', 'Month-to-Month'] * 4,
        "Partner": ['Yes', 'No'],
        "Dependents": ['Yes', 'No'],
        "Phone Service": ['Yes', 'No'],
        "Multiple Lines": ['Yes', 'No'],
        "Internet Service": ['No', 'DSL', 'Fiber Optic', 'Cable'],
        "Online Security": ['Yes', 'No'],
        "Online Backup": ['Yes', 'No'],
        "Device Protection": ['Yes', 'No'],
        "Tech Support": ['Yes', 'No'],
        "Streaming TV": ['Yes', 'No'],
        "Streaming Movies": ['Yes', 'No'],
        "Paperless Billing": ['Yes', 'No'],
        "Payment Method": ['Bank Withdrawal', 'Credit Card', 'Mailed Check'],
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
