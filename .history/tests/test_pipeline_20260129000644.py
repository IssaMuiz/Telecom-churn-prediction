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
        "gender": ["Male", "Female"] * 10,
        "SeniorCitizen": [0, 1] * 10,
        "MonthlyCharges": [50, 60, 70, 80, 90, 100, 55, 65, 75, 85,
                           95, 105, 52, 62, 72, 82, 92, 102, 57, 67],
        "TotalCharges": ["1000", "2000", "3000", "4000", "5000", "6000",
                         "1100", "2100", "3100", "4100",
                         "5100", "6100", "1200", "2200", "3200", "4200",
                         "5200", "6200", "1300", "2300"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month",
                     "One year"] * 4,
        "Churn": [0, 1] * 10
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
